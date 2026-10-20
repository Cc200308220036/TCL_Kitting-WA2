"""单层放置：在一个循环内完成接触锁存、定距退让及自动张爪。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math
import time
import numpy as np
import rospy
from naviai_controller import ArmGroup
from .common.runtime import input
from .common.gripper import DualGripperWorker, poll_gripper_completion
from .observability.timing import CycleTiming, state as trace_state
from .algorithms.moving_z_admittance import MovingZAxis, MovingZParameters
from .stage3_common import ARMS, CONTROL_HZ, SERVO_PERIOD_S, SINGLE_STAGE3_CYCLE_GAP_LIMIT_S, Stage3Place, next_z_velocity


CONTACT_RETREAT_DISTANCE_M = 0.0005  # 单层/多层初始接触共用上退距离(m)


@dataclass
class ContactDescentState:
    """接触找平状态，供单层和多层初始下降共用。"""

    offsets: dict = field(default_factory=lambda: {arm: 0.0 for arm in ARMS})
    velocities: dict = field(default_factory=lambda: {arm: 0.0 for arm in ARMS})
    reached: dict = field(default_factory=lambda: {arm: False for arm in ARMS})
    dual_velocity: float = 0.0
    held_arm: Optional[ArmGroup] = None
    retreat_axis: Optional[MovingZAxis] = None
    mode: str = "DUAL_DESCENT"


def update_contact_descent(args, motion, support, dt, retreat_params):
    """共用接触找平：先接触侧定距上退，另一侧继续下降，双侧达到过即保持。"""
    if (
        not all(math.isfinite(support[a]) for a in ARMS)
        or not 0 < dt <= 0.1
    ):
        raise ValueError("接触找平支撑力或控制周期无效")
    for arm in ARMS:
        motion.reached[arm] |= support[arm] >= args.target_force
    if all(motion.reached.values()):
        motion.mode = "BOTH_FORCE_READY"
        motion.dual_velocity = 0.0
        motion.velocities = {arm: 0.0 for arm in ARMS}
    elif not any(motion.reached.values()):
        motion.dual_velocity = max(
            0.0,
            next_z_velocity(
                args,
                motion.dual_velocity,
                sum(support.values()) / len(ARMS),
                args.target_force + args.control_margin,
                args.max_free_speed,
                dt,
            ),
        )
        for arm in ARMS:
            motion.offsets[arm] += motion.dual_velocity * dt
            motion.velocities[arm] = motion.dual_velocity
    else:
        if motion.held_arm is None:
            motion.held_arm = next(a for a in ARMS if motion.reached[a])
            start = motion.offsets[motion.held_arm]
            motion.retreat_axis = MovingZAxis(
                displacement=start,
                retreat_active=True,
                retreat_start=start,
                retreat_travel=CONTACT_RETREAT_DISTANCE_M,
            )
        held = motion.held_arm
        moving = next(a for a in ARMS if a != held)
        if motion.retreat_axis.retreat_active:
            motion.offsets[held] = motion.retreat_axis.step_retreat(
                dt, retreat_params
            )
            motion.velocities[held] = motion.retreat_axis.velocity
        else:
            motion.velocities[held] = 0.0
        # 另一侧继承共同下降速度，不在模式切换时归零或缩放。
        motion.velocities[moving] = max(
            0.0,
            next_z_velocity(
                args,
                motion.velocities[moving],
                support[moving],
                args.target_force + args.control_margin,
                args.max_free_speed,
                dt,
            ),
        )
        motion.offsets[moving] += motion.velocities[moving] * dt
        motion.mode = "{}_{}_{}_DESCENT".format(
            held.name,
            "RETREAT" if motion.retreat_axis.retreat_active else "HOLD",
            moving.name,
        )


class SingleStage3Place(Stage3Place):
    """双侧接触即释放；先接触侧退让，另一侧沿用共同下降速度。"""

    def __init__(self, args):
        super().__init__(args)
        self.descent = ContactDescentState()

    def update_descent(self, support, dt, retreat_params):
        """调用共用找平接口，保持单层的接触锁存和运动行为。"""
        update_contact_descent(self.args, self.descent, support, dt, retreat_params)

    def run_compliant_descent_and_release(self) -> None:
        args, motion = self.args, self.descent
        retreat_params = MovingZParameters(
            target_force=args.target_force,
            mass=args.mass,
            damping=args.damping,
            speed_limit=args.max_free_speed,
            acceleration_limit=args.max_acceleration,
            retreat_distance=CONTACT_RETREAT_DISTANCE_M,
            travel_limit=None,
            equilibrium_limit=None,
        )
        trace_state("WAIT_DESCENT_CONFIRM")
        rospy.logwarn(
            "单层Stage3：目标力%.2fN，先接触侧上退%gmm；下降限速%.1fmm/s，加速度%.1fmm/s²",
            args.target_force,
            1000 * CONTACT_RETREAT_DISTANCE_M,
            1000 * args.max_free_speed,
            1000 * args.max_acceleration,
        )
        input("确认下方放置区域安全后按 Enter 开始下降（Ctrl+C取消）：")
        trace_state("Z_DESCENT")
        self.start_servo()
        previous = time.monotonic()
        accepted_at = None
        rate = rospy.Rate(CONTROL_HZ)
        timing = CycleTiming(SERVO_PERIOD_S)

        while not rospy.is_shutdown():
            now = time.monotonic()
            raw_dt, previous = now - previous, now
            if self.servo_started and raw_dt > SINGLE_STAGE3_CYCLE_GAP_LIMIT_S:
                raise RuntimeError(
                    "控制循环阻塞{:.3f}秒（上限{:.3f}秒）；{}".format(
                        raw_dt, SINGLE_STAGE3_CYCLE_GAP_LIMIT_S, timing.detail(raw_dt)
                    )
                )
            dt = float(np.clip(raw_dt, 0.005, 0.04))
            timing.begin(now)
            support = {}
            for arm in ARMS:
                with timing.section("tcp"):
                    self.current_pose_checked(arm)
                with timing.section("wrench"):
                    support[arm] = self.read_wrench_sample(arm, dt).support
            # 单层的全部接触/运动分支在此处更新；张爪后保持最终目标。
            old_mode = motion.mode
            if self.hand_worker is None:
                self.update_descent(support, dt, retreat_params)
            trace_state("OPENING" if self.hand_worker is not None else motion.mode)
            if old_mode != motion.mode:
                rospy.loginfo(
                    "SINGLE_Z_MODE：%s -> %s；接触锁存L/R=%s/%s",
                    old_mode,
                    motion.mode,
                    motion.reached[ArmGroup.LEFT],
                    motion.reached[ArmGroup.RIGHT],
                )
            commands = {
                arm: self.z_command_pose(arm, motion.offsets[arm]) for arm in ARMS
            }
            if self.servo_started:
                with timing.section("servo"):
                    self.robot.servol(
                        np.concatenate([commands[arm] for arm in ARMS]).tolist(),
                        ArmGroup.DUAL,
                    )
            self.last_commands = {arm: pose.copy() for arm, pose in commands.items()}

            if all(motion.reached.values()) and self.hand_worker is None:
                trace_state("OPENING")
                # 先退出伺服，避免张爪服务影响控制周期调度。
                self.exit_servo()
                self.hand_worker = DualGripperWorker(self.grippers, action="open")
                self.hand_worker.start()
            if self.hand_worker is not None:
                previous_accepted = accepted_at
                accepted_at, settled = poll_gripper_completion(
                    self.hand_worker, now, accepted_at, args.hand_open_settle_seconds,
                    error_prefix="张开夹爪失败: ")
                if previous_accepted is None and accepted_at is not None:
                    rospy.loginfo("HAND_OPEN_ACCEPTED：左右手张开服务均成功")
                if settled:
                    rospy.loginfo("STAGE3_PLACE_COMPLETE：放置与张爪完成")
                    return
            timing.finish(rate)

    def run(self) -> None:
        trace_state("WAIT_READY")
        self.wait_ready()
        trace_state("BASELINE_CAPTURE")
        self.capture_baseline()
        if not self.args.execute:
            return
        try:
            self.run_compliant_descent_and_release()
            if rospy.is_shutdown():
                raise RuntimeError("ROS已关闭，禁止继续放置后运动")
            self.exit_servo()
            self.retreat()
            trace_state("COMPLETED")
        except BaseException:
            trace_state("ABORTED")
            raise
