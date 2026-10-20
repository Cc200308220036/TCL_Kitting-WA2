#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage3公共反馈、导纳、伺服、张爪及撤离能力。"""

from __future__ import annotations

from .common.feedback import ThresholdLatch, wait_feedback_ready, collect_wrench_baseline, WrenchBuffer, ScalarLowPass, read_valid_tcp, WrenchSample
from .common.runtime import input
from .common.state import require_positive, pose7, horizontal_tcp_axis, quaternion_error_rad
from .observability.force_recording import mark as trace_mark, state as trace_state, deactivate

import argparse
from dataclasses import dataclass
import math
import time
from typing import Dict, Optional

import numpy as np
import rospy

from naviai_controller import ArmGroup, NaviController, RobotModel

from .common.gripper import DualGripperWorker, JodellDualGripper, JodellGripperConfig


ARMS = (ArmGroup.LEFT, ArmGroup.RIGHT)
SERVO_PERIOD_S = 0.02
SERVO_GAIN = 800
CONTROL_HZ = 50.0
SINGLE_STAGE3_CYCLE_GAP_LIMIT_S = 0.15


def outward_axes_from_tcp_line(
    tcp_poses: Dict[ArmGroup, np.ndarray]
) -> Dict[ArmGroup, np.ndarray]:
    """根据左右TCP当前水平连线，返回两端分别远离对侧的单位向量。"""
    left_to_right = horizontal_tcp_axis(
        tcp_poses[ArmGroup.LEFT], tcp_poses[ArmGroup.RIGHT], purpose="计算连线撤离方向"
    )
    return {
        ArmGroup.LEFT: -left_to_right,
        ArmGroup.RIGHT: left_to_right,
    }


# 导纳模型与力阈值适配
class TargetForceLatch(ThresholdLatch):
    """多层目标力的迟滞锁存，保持各自进入和退出去抖时间。"""

    def __init__(self, args):
        super().__init__(args.target_force, args.target_force - args.force_hold_band,
                         args.target_reach_debounce, args.target_release_debounce)

def next_z_velocity(
    args,
    current_velocity: float,
    measured_force: float,
    desired_force: float,
    speed_limit: float,
    dt: float,
    spring_force: float = 0.0,
) -> float:
    acceleration = (
        desired_force - measured_force - args.damping * current_velocity - spring_force
    ) / args.mass
    acceleration = float(
        np.clip(
            acceleration,
            -args.max_acceleration,
            args.max_acceleration,
        )
    )
    return float(
        np.clip(
            current_velocity + acceleration * dt,
            -speed_limit,
            speed_limit,
        )
    )


class SwitchingZAdmittance:
    """双臂共同下降与单臂找平下降共用的Z向速度导纳。"""

    DUAL = "DUAL_DESCENT"
    LEFT_ONLY = "LEFT_ONLY_DESCENT"
    RIGHT_ONLY = "RIGHT_ONLY_DESCENT"
    HOLD_BOTH = "BOTH_FORCE_READY"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mode = self.DUAL
        self.offsets = {arm: 0.0 for arm in ARMS}
        self.dual_velocity = 0.0
        self.single_velocity = {arm: 0.0 for arm in ARMS}
        self.held_arm: Optional[ArmGroup] = None
        self.held_anchor = 0.0
        self.held_equilibrium = 0.0
        self.hold_paused = False

    def set_mode(self, mode: str) -> bool:
        if mode == self.mode:
            return False
        self.mode = mode
        # 单侧模式中，原先“保持”的手也要能随负载变化向上/向下微调。
        held_arm = (
            ArmGroup.RIGHT
            if mode == self.LEFT_ONLY
            else ArmGroup.LEFT if mode == self.RIGHT_ONLY else None
        )
        if held_arm is not None and held_arm != self.held_arm:
            self.held_arm = held_arm
            self.held_anchor = self.offsets[held_arm]
            self.held_equilibrium = self.held_anchor
        self.hold_paused = False
        # 切换时速度归零，防止上一模式的积分速度带入新模式。
        self.dual_velocity = 0.0
        self.single_velocity = {arm: 0.0 for arm in ARMS}
        return True

    def step(
        self,
        forces: Dict[ArmGroup, float],
        desired_force: float,
        contact_candidate: bool,
        dt: float,
    ) -> Dict[ArmGroup, float]:
        if self.mode == self.HOLD_BOTH:
            self.dual_velocity = 0.0
            self.single_velocity = {arm: 0.0 for arm in ARMS}
            return self.offsets.copy()

        if self.mode == self.DUAL:
            measured_mean = 0.5 * (forces[ArmGroup.LEFT] + forces[ArmGroup.RIGHT])
            speed_limit = (
                self.args.max_contact_speed
                if contact_candidate
                else self.args.max_free_speed
            )
            self.dual_velocity = next_z_velocity(
                self.args,
                self.dual_velocity,
                measured_mean,
                desired_force,
                speed_limit,
                dt,
            )
            increment = self.dual_velocity * dt
            self.offsets[ArmGroup.LEFT] += increment
            self.offsets[ArmGroup.RIGHT] += increment
        else:
            moving_arm = (
                ArmGroup.LEFT if self.mode == self.LEFT_ONLY else ArmGroup.RIGHT
            )
            held_arm = ArmGroup.RIGHT if moving_arm == ArmGroup.LEFT else ArmGroup.LEFT
            # 虚拟弹簧：M*x'' + D*x' + K*(x-x_eq) = F_target-F_measured。
            # 平衡点缓慢积分力误差，避免固定弹簧留下恒定静差。
            force_error = self.args.target_force - forces[held_arm]
            correction = self.args.hold_max_correction
            self.held_equilibrium = float(
                np.clip(
                    self.held_equilibrium
                    + self.args.hold_integral_gain * force_error * dt,
                    self.held_anchor - correction,
                    self.held_anchor + correction,
                )
            )
            held_velocity = next_z_velocity(
                self.args,
                self.single_velocity[held_arm],
                forces[held_arm],
                self.args.target_force,
                self.args.hold_max_speed,
                dt,
                spring_force=self.args.hold_spring_k
                * (self.offsets[held_arm] - self.held_equilibrium),
            )
            held_target = float(
                np.clip(
                    self.offsets[held_arm] + held_velocity * dt,
                    self.held_anchor - correction,
                    self.held_anchor + correction,
                )
            )
            if held_target != self.offsets[held_arm] + held_velocity * dt:
                held_velocity = 0.0
            self.offsets[held_arm] = held_target
            self.single_velocity[held_arm] = held_velocity

            error_abs = abs(force_error)
            if error_abs > self.args.hold_force_band:
                self.hold_paused = True
            elif error_abs <= 0.8 * self.args.hold_force_band:
                self.hold_paused = False
            if self.hold_paused:
                self.single_velocity[moving_arm] = 0.0
            else:
                velocity = next_z_velocity(
                    self.args,
                    self.single_velocity[moving_arm],
                    forces[moving_arm],
                    desired_force,
                    self.args.max_contact_speed,
                    dt,
                )
                self.single_velocity[moving_arm] = velocity
                self.offsets[moving_arm] += velocity * dt

        return self.offsets.copy()


@dataclass
class RetreatFeedback:
    reached: bool
    position_errors: list
    raw_wrenches: dict


class Stage3Place:
    """公共执行器能力；单层和多层分别保留自身的控制分支。"""

    # 初始化与反馈准备
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.force_recorder = None
        self.force_log_path = None
        self.robot = NaviController(auto_spin=False, model=RobotModel.WA2)
        self.grippers = JodellDualGripper(make_gripper_config(args))
        self.wrenches = WrenchBuffer()
        self.filters = {arm: ScalarLowPass(args.filter_cutoff) for arm in ARMS}
        self.baseline: Dict[ArmGroup, np.ndarray] = {}
        self.anchor_poses: Dict[ArmGroup, np.ndarray] = {}
        self.last_commands: Dict[ArmGroup, np.ndarray] = {}
        self.hand_worker: Optional[DualGripperWorker] = None
        self.servo_started = False
        self.movel_started = False
        rospy.logwarn(
            "Stage3实验配置：腕力用于导纳控制；"
            "不以合力或力矩范数终止下降、张爪和撤离"
        )

    def wait_ready(self, timeout_s=8.0):
        poses = wait_feedback_ready(self.robot, self.wrenches, self.args.stale_timeout,
                                    timeout_s, error_message="双臂TCP、腕力或机器人状态在8秒内未就绪")
        self.anchor_poses = {arm: poses[arm].copy() for arm in ARMS}
        self.last_commands = {arm: poses[arm].copy() for arm in ARMS}

    def capture_baseline(self, *, confirm: bool = True) -> None:
        rospy.logwarn(
            "即将记录抓住tray时的双腕负载基线%.2f秒；"
            "tray不能接触放置面，也不要人为施力",
            self.args.baseline_seconds,
        )
        if confirm:
            input("确认tray悬空且静止后按 Enter 开始记录（Ctrl+C取消）：")
        stats = collect_wrench_baseline(self.wrenches, self.args.baseline_seconds,
                                       label="负载基线", stale_timeout=self.args.stale_timeout)
        for arm, (mean, std, count) in stats.items():
            self.baseline[arm] = mean
            if self.args.multi_layer:
                rospy.loginfo("%s baseline=%s std=%s samples=%d", arm.name,
                              np.round(mean, 4), np.round(std, 4), count)

    def z_sign(self, arm: ArmGroup) -> float:
        return (
            self.args.left_z_force_sign
            if arm == ArmGroup.LEFT
            else self.args.right_z_force_sign
        )

    def read_wrench_sample(
        self,
        arm: ArmGroup,
        dt: float,
    ) -> WrenchSample:
        wrench, stamp, _ = self.wrenches.snapshot(arm)
        if wrench is None or stamp is None:
            raise RuntimeError("{}腕力未就绪".format(arm.name))
        if time.monotonic() - stamp > self.args.stale_timeout:
            raise RuntimeError("{}腕力数据超时".format(arm.name))
        support_raw = self.z_sign(arm) * float(wrench[2] - self.baseline[arm][2])
        support = self.filters[arm].step(support_raw, dt)
        return WrenchSample(raw=wrench, support=support)

    def current_pose_checked(self, arm: ArmGroup) -> np.ndarray:
        pose = read_valid_tcp(self.robot, arm, self.args.stale_timeout)
        tracking_error = float(np.linalg.norm(pose[:3] - self.last_commands[arm][:3]))
        if tracking_error > self.args.tracking_error_limit:
            raise RuntimeError(
                "{} TCP跟踪误差{:.3f} mm超过{:.3f} mm".format(
                    arm.name,
                    1000.0 * tracking_error,
                    1000.0 * self.args.tracking_error_limit,
                )
            )
        return pose

    # 双臂伺服、到位等待与撤离
    def z_command_pose(self, arm: ArmGroup, downward_offset_m: float) -> np.ndarray:
        command = self.anchor_poses[arm].copy()
        # 每只手可有独立Z位移；未运动的一端持续发送其保持位姿。
        command[2] -= downward_offset_m
        return command

    def start_servo(self) -> None:
        initial_dual = np.concatenate(
            [
                self.anchor_poses[ArmGroup.LEFT],
                self.anchor_poses[ArmGroup.RIGHT],
            ]
        )
        if not self.robot.set_servo_params(
            SERVO_PERIOD_S, SERVO_GAIN, arm=ArmGroup.DUAL
        ):
            raise RuntimeError("set_servo_params调用失败")
        self.servo_started = True
        self.robot.servol(initial_dual.tolist(), ArmGroup.DUAL)

    def exit_servo(self) -> None:
        if not self.servo_started:
            return
        self.robot.stop()
        self.robot.clear_servo_params()
        self.servo_started = False
        rospy.sleep(0.1)

    def read_retreat_feedback(self, targets, position_tolerance):
        """读取一轮撤离反馈，沿用原奇异/保护/腕力检查及缺失TCP处理。"""
        reached = True
        errors = []
        retreat_raw = {}
        for arm in ARMS:
            if self.robot.get_is_singular(arm):
                raise RuntimeError("{}臂撤离时进入奇异位形".format(arm.name))
            if self.robot.get_cmd_num() == 15:
                raise RuntimeError("机器人撤离时进入PROTECTED状态")
            actual_raw = self.robot.get_tcp_rt(arm)
            if actual_raw is None:
                reached = False
                continue
            actual = pose7(actual_raw, "{} TCP".format(arm.name))
            wrench, stamp, _ = self.wrenches.snapshot(arm)
            if (
                wrench is None
                or stamp is None
                or time.monotonic() - stamp > self.args.stale_timeout
            ):
                raise RuntimeError("{}腕力在撤离时超时".format(arm.name))
            if self.force_recorder is not None:
                retreat_raw["left" if arm == ArmGroup.LEFT else "right"] = (
                    wrench.copy()
                )
            position_error = float(np.linalg.norm(actual[:3] - targets[arm][:3]))
            angle_error = quaternion_error_rad(actual[3:], targets[arm][3:])
            errors.append(position_error)
            reached = (
                reached
                and position_error <= position_tolerance
                and angle_error <= math.radians(3.0)
            )
        return RetreatFeedback(reached, errors, retreat_raw)

    def wait_movel_target(
        self,
        targets: Dict[ArmGroup, np.ndarray],
        motion_name: str,
        position_tolerance: float = 0.005,
    ) -> None:
        stable_since: Optional[float] = None
        deadline = time.monotonic() + 20.0
        rate = rospy.Rate(50.0)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            feedback = self.read_retreat_feedback(targets, position_tolerance)
            if len(feedback.raw_wrenches) == 2:
                self.record_force_sample(feedback.raw_wrenches)
            if self.args.multi_layer and len(feedback.position_errors) == 2:
                rospy.loginfo_throttle(
                    0.2,
                    "RETREAT TCP error L/R=%.2f/%.2f mm",
                    1000.0 * feedback.position_errors[0],
                    1000.0 * feedback.position_errors[1],
                )
            if feedback.reached:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= 0.2:
                    rospy.loginfo("STAGE3_%s_COMPLETE", motion_name)
                    trace_mark("motion_complete", motion_name)
                    return
            else:
                stable_since = None
            rate.sleep()
        raise RuntimeError("双臂{}未在规定时间内完成".format(motion_name))

    def retreat(self) -> None:
        """撤离：依次下探、沿双TCP水平连线外撤、同步上抬。"""
        # 直接修改下面三行即可调整各段动作；距离均填正数。
        # 每行：阶段、距离(m)、速度(m/s)、加速度(m/s²)、到位容差(m)。
        # 例如距离0.02为20mm，速度0.08为80mm/s，加速度0.1为100mm/s²。
        steps = (
            ("POST_RELEASE_LOWER", 0.0013, 0.01, 0.02, 0.0005),  # 下探1.3mm
            ("OUTWARD_RETREAT", 0.02, 0.08, 0.1, 0.005),  # 每侧外撤20mm
            ("POST_RELEASE_LIFT", 0.03, 0.08, 0.1, 0.005),  # 上抬30mm
        )
        for stage, distance, velocity, acceleration, tolerance in steps:
            trace_state("UPWARD_RETREAT" if stage == "POST_RELEASE_LIFT" else stage)
            targets = {
                arm: pose7(self.robot.get_tcp_rt(arm), arm.name + " TCP")
                for arm in ARMS
            }
            if stage == "OUTWARD_RETREAT":
                for arm, axis in outward_axes_from_tcp_line(targets).items():
                    targets[arm][:3] += axis * distance
            else:
                for pose in targets.values():
                    pose[2] += -distance if stage == "POST_RELEASE_LOWER" else distance
            self.movel_started = True
            if not self.robot.movel(
                np.concatenate([targets[arm] for arm in ARMS]).tolist(),
                ArmGroup.DUAL,
                v=velocity,
                acc=acceleration,
                is_async=True,
            ):
                raise RuntimeError(stage + " MoveL服务返回失败")
            self.wait_movel_target(targets, stage, position_tolerance=tolerance)
            self.movel_started = False

    # 记录与资源清理
    def close_force_log(self):
        if self.force_recorder is not None:
            try:
                self.force_recorder.close()
            finally:
                deactivate(self.force_recorder)
                self.force_recorder = None

    def mark_force_abort(self, error):
        if self.force_recorder is not None:
            self.force_recorder.mark("abort", str(error))
            self.force_recorder.set_state("ABORTED")

    def record_force_sample(self, raw, support=None):
        if self.force_recorder is not None:
            self.force_recorder.sample(raw, support)

    def cleanup(self) -> None:
        if self.servo_started or self.movel_started:
            try:
                self.robot.stop()
            except Exception as exc:  # pragma: no cover - 实机异常兜底
                rospy.logerr("停止机器人失败: %s", exc)
        if self.servo_started:
            try:
                self.robot.clear_servo_params()
            except Exception as exc:  # pragma: no cover - 实机伺服参数清理兜底
                rospy.logerr("清理ServoL参数失败: %s", exc)
        self.servo_started = False
        self.movel_started = False
        try:
            if self.hand_worker is not None:
                self.hand_worker.join()
        finally:
            self.close_force_log()



STAGE3_POSITIVE_PARAMETERS = (
    "target_force",
    "control_margin",
    "mass",
    "damping",
    "max_free_speed",
    "max_acceleration",
    "baseline_seconds",
    "stale_timeout",
    "tracking_error_limit",
    "hand_open_settle_seconds",
)
MULTI_STAGE3_PARAMETERS = (
    "contact_threshold",
    "force_hold_band",
    "hold_force_band",
    "hold_spring_k",
    "hold_integral_gain",
    "hold_max_correction",
    "hold_max_speed",
    "target_reach_debounce",
    "target_release_debounce",
    "imbalance_limit",
    "stable_seconds",
    "max_contact_speed",
)


def validate_stage3_config(args: argparse.Namespace) -> None:
    """校验阶段3配置，供 CLI 和外部 Python API 共同使用。"""
    positive = STAGE3_POSITIVE_PARAMETERS + (MULTI_STAGE3_PARAMETERS if args.multi_layer else ())
    for name in positive:
        require_positive(getattr(args, name), "--" + name.replace("_", "-"))
    if args.filter_cutoff < 0.0 or not math.isfinite(args.filter_cutoff):
        raise ValueError("--filter-cutoff必须是有限非负数")
    if args.multi_layer:
        if args.contact_threshold >= args.target_force:
            raise ValueError("接触阈值必须小于目标放置力")
        if args.force_hold_band >= args.target_force:
            raise ValueError("保持力带必须小于目标放置力")
        if args.max_contact_speed > args.max_free_speed:
            raise ValueError("接触后速度不能大于自由速度")
    make_gripper_config(args).validate()


def make_gripper_config(args: argparse.Namespace) -> JodellGripperConfig:
    return JodellGripperConfig(
        left_service=args.left_gripper_service,
        right_service=args.right_gripper_service,
        profile=int(args.gripper_profile),
        open_position_mm=float(args.gripper_open_position_mm),
        force_percent=float(args.gripper_force_percent),
        speed_percent=float(args.gripper_speed_percent),
        wait_timeout=float(args.gripper_wait_timeout),
    )
