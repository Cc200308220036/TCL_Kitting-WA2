"""Stage1：沿预抓取双TCP的水平连线，双侧导纳靠拢并闭爪。

调用顺序：prepare_grippers → wait_ready → calibrate_bias → run_approach。
闭爪期间继续导纳，服务成功并保持指定时间后返回；调用方负责finally清理。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional

import numpy as np
import rospy
from naviai_controller import ArmGroup, NaviController, RobotModel

from .common.gripper import poll_gripper_completion, DualGripperWorker, JodellDualGripper, JodellGripperConfig
from .common.feedback import ThresholdLatch, wait_feedback_ready, collect_wrench_baseline, WrenchBuffer, ScalarLowPass, read_valid_tcp
from .common.runtime import input
from .common.state import validate_pose7, pose7, horizontal_tcp_axis
from .observability.timing import state as timing_state

ARMS = (ArmGroup.LEFT, ArmGroup.RIGHT)
SERVO_PERIOD_S = 0.02
SERVO_GAIN = 800
CONTROL_HZ = 50.0


def tcp_line_inward_axes(pregrasp_targets) -> Dict[ArmGroup, np.ndarray]:
    """由 Stage 0 左右 TCP 目标连线计算两臂各自指向对侧的单位向量。"""
    if not isinstance(pregrasp_targets, dict):
        raise RuntimeError("状态文件缺少stage0_pregrasp_targets")
    left = np.asarray(
        validate_pose7(pregrasp_targets.get("left", []), "Stage0 LEFT target"),
        dtype=np.float64,
    )
    right = np.asarray(
        validate_pose7(pregrasp_targets.get("right", []), "Stage0 RIGHT target"),
        dtype=np.float64,
    )
    # 只允许水平靠拢；Z 高度差不应转化为 Stage 1 的垂直运动。
    left_inward = horizontal_tcp_axis(
        left, right, description="Stage0左右预抓取TCP", purpose="确定靠拢轴"
    )
    return {
        ArmGroup.LEFT: left_inward,
        ArmGroup.RIGHT: -left_inward,
    }


def read_tcp_pair(robot: NaviController) -> Dict[str, List[float]]:
    result = {}
    for side, arm in (("left", ArmGroup.LEFT), ("right", ArmGroup.RIGHT)):
        pose = robot.get_tcp_rt(arm)
        age = robot.get_tcp_age(arm)
        if pose is None or age is None or age > 0.20:
            raise RuntimeError("{} TCP未就绪或超时".format(side.upper()))
        singular = robot.get_is_singular(arm)
        if singular is None:
            raise RuntimeError("{}臂奇异位状态未就绪".format(side.upper()))
        if singular:
            raise RuntimeError("{}臂处于奇异位形".format(side.upper()))
        result[side] = validate_pose7(pose, "{} TCP".format(side))
    if robot.get_cmd_num() == 15:
        raise RuntimeError("机器人处于PROTECTED状态(cmd_num=15)")
    return result


@dataclass(frozen=True)
class Stage1Config:
    """抓取参数；位移m、时间s、力N，运动速度m/s。"""

    hand_settle_seconds: float = 1.5
    target_force_n: float = 0.7
    contact_threshold_n: float = 0.5
    contact_debounce_s: float = 0.1
    contact_release_debounce_s: float = 0.3
    force_band_n: float = 0.1
    imbalance_band_n: float = 0.5
    virtual_mass_kg: float = 2.0
    damping_ns_m: float = 80.0
    max_free_speed_m_s: float = 0.006
    max_contact_speed_m_s: float = 0.001
    max_acceleration_m_s2: float = 0.01
    max_inward_distance_m: float = 0.02
    max_retreat_distance_m: float = 0.003
    target_ramp_seconds: float = 1.0
    stable_seconds: float = 0.2
    duration_s: float = 60.0
    bias_seconds: float = 1.0
    filter_cutoff_hz: float = 5.0
    stale_timeout_s: float = 0.2
    tracking_error_limit_m: float = 0.015
    left_gripper_service: str = "/zj_humanoid/jodell_rag_left/grasp"
    right_gripper_service: str = "/zj_humanoid/jodell_rag_right/grasp"
    gripper_profile: int = 0
    gripper_initial_position_mm: float = 35.0
    gripper_close_position_mm: float = 17.0
    gripper_force_percent: float = 0.0
    gripper_speed_percent: float = 0.0
    gripper_wait_timeout_s: float = 5.0

    def validate(self) -> None:
        """校验导纳、接触、运行边界及夹爪参数。"""
        nonpositive_allowed = {
            "filter_cutoff_hz",
            "gripper_profile",
            "gripper_initial_position_mm",
            "gripper_close_position_mm",
            "gripper_force_percent",
            "gripper_speed_percent",
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.endswith("_service") or field.name in nonpositive_allowed:
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError("{} 必须是有限正数".format(field.name))
        if self.target_force_n < self.contact_threshold_n:
            raise ValueError("目标力不能小于接触阈值")
        if self.force_band_n >= self.target_force_n:
            raise ValueError("闭爪触发裕量必须小于目标力")
        if not math.isfinite(self.filter_cutoff_hz) or self.filter_cutoff_hz < 0:
            raise ValueError("filter_cutoff_hz 必须是有限非负数")
        if self.max_contact_speed_m_s > self.max_free_speed_m_s:
            raise ValueError("接触后速度不能大于自由靠拢速度")
        if self.max_free_speed_m_s > 0.010 or self.max_contact_speed_m_s > 0.002:
            raise ValueError("自由/接触靠拢速度不得超过10/2 mm/s")
        if (
            not math.isfinite(self.gripper_initial_position_mm)
            or not 0 <= self.gripper_initial_position_mm <= 100
        ):
            raise ValueError("夹爪初始开度必须在[0,100]mm")
        self.make_gripper_config().validate()

    def make_gripper_config(self) -> JodellGripperConfig:
        return JodellGripperConfig(
            left_service=self.left_gripper_service,
            right_service=self.right_gripper_service,
            profile=self.gripper_profile,
            close_position_mm=self.gripper_close_position_mm,
            force_percent=self.gripper_force_percent,
            speed_percent=self.gripper_speed_percent,
            wait_timeout=self.gripper_wait_timeout_s,
        )


def make_stage1_config(**overrides: Any) -> Stage1Config:
    """合并默认值与本次配置，拒绝已移除或拼错的参数。"""
    unknown = set(overrides) - {field.name for field in fields(Stage1Config)}
    if unknown:
        raise TypeError("未知Stage1参数: {}".format(sorted(unknown)))
    config = Stage1Config(**overrides)
    config.validate()
    return config


class ForceApproachAxis:
    """单臂无刚度M-D力寻位，状态为相对预抓取位置的向内位移。"""

    def __init__(self, config: Stage1Config) -> None:
        self.config = config
        self.position_m = 0.0
        self.velocity_m_s = 0.0

    def step(
        self, measured_force_n: float, target_force_n: float, contacted: bool, dt: float
    ) -> float:
        acceleration = (
            target_force_n
            - measured_force_n
            - self.config.damping_ns_m * self.velocity_m_s
        ) / self.config.virtual_mass_kg
        acceleration = float(
            np.clip(
                acceleration,
                -self.config.max_acceleration_m_s2,
                self.config.max_acceleration_m_s2,
            )
        )
        self.velocity_m_s += acceleration * dt
        speed_limit = (
            self.config.max_contact_speed_m_s
            if contacted
            else self.config.max_free_speed_m_s
        )
        # 超过目标力时允许按同一速度上限向外退让。
        self.velocity_m_s = float(np.clip(self.velocity_m_s, -speed_limit, speed_limit))
        self.position_m += self.velocity_m_s * dt

        lower = -self.config.max_retreat_distance_m
        upper = self.config.max_inward_distance_m
        if self.position_m <= lower and self.velocity_m_s < 0.0:
            self.position_m = lower
            self.velocity_m_s = 0.0
        elif self.position_m >= upper and self.velocity_m_s > 0.0:
            self.position_m = upper
            self.velocity_m_s = 0.0
        else:
            self.position_m = float(np.clip(self.position_m, lower, upper))
        return self.position_m


class ContactDetector(ThresholdLatch):
    """抓取接触检测；候选阶段在达到退出阈值时提前降速。"""

    def __init__(self, threshold_n, debounce_s, release_debounce_s):
        super().__init__(threshold_n, 0.7 * threshold_n, debounce_s, release_debounce_s)

    def is_contact_candidate(self, force_n):
        return self.active or force_n >= self.exit_threshold

@dataclass
class ApproachCycle:
    now: float
    dt: float
    elapsed: float
    measured: dict
    contacts: dict


@dataclass
class ApproachFlow:
    stable_since: Optional[float] = None
    one_side_active: bool = False
    hand_accepted_at: Optional[float] = None


class TrayApproach:
    """双TCP水平连线上的双侧导纳抓取控制器。"""

    def __init__(self, config: Stage1Config, pregrasp_targets) -> None:
        self.inward_axes = tcp_line_inward_axes(pregrasp_targets)
        self.reaction_axes = {arm: -axis for arm, axis in self.inward_axes.items()}
        self.config = config
        self.robot = NaviController(auto_spin=False, model=RobotModel.WA2)
        self.grippers = JodellDualGripper(config.make_gripper_config())
        self.wrenches = WrenchBuffer()
        self.bias = {arm: np.zeros(6, dtype=np.float64) for arm in ARMS}
        self.force_filters = {
            arm: ScalarLowPass(config.filter_cutoff_hz) for arm in ARMS
        }
        self.models = {arm: ForceApproachAxis(config) for arm in ARMS}
        self.contact_detectors = {
            arm: ContactDetector(
                config.contact_threshold_n,
                config.contact_debounce_s,
                config.contact_release_debounce_s,
            )
            for arm in ARMS
        }
        self.anchor_poses: Dict[ArmGroup, np.ndarray] = {}
        self.last_commands: Dict[ArmGroup, np.ndarray] = {}
        self.hand_worker: Optional[DualGripperWorker] = None
        self.servo_started = False
        self.cleaned = False

    def prepare_grippers(self) -> None:
        """真实执行前将左右夹爪并行移动到阶段1初始开度。"""
        target = self.config.gripper_initial_position_mm
        rospy.logwarn("设置左右夹爪初始开度为 %.1f mm", target)
        worker = DualGripperWorker(self.grippers, action="close", target_position_mm=target)
        worker.start()
        deadline = time.monotonic() + self.config.gripper_wait_timeout_s + 30.0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            completed, error = worker.status()
            if error is not None:
                raise RuntimeError("夹爪初始位置设置失败: " + error)
            if completed:
                worker.join()
                rospy.loginfo("左右夹爪已到达阶段1初始开度 %.1f mm", target)
                return
            rospy.sleep(0.02)
        raise RuntimeError("左右夹爪未在规定时间内完成初始位置设置")

    def wait_ready(self, timeout_s=8.0):
        poses = wait_feedback_ready(self.robot, self.wrenches, self.config.stale_timeout_s,
                                    timeout_s, error_message="TCP、上肢状态或双腕力在8秒内未就绪")
        self.anchor_poses = {arm: poses[arm].copy() for arm in ARMS}
        self.last_commands = {arm: poses[arm].copy() for arm in ARMS}
    def calibrate_bias(self) -> None:
        rospy.logwarn(
            "即将标定双腕零偏%.2f秒。双夹爪和tray必须完全无接触。",
            self.config.bias_seconds,
        )
        input("确认无接触后按 Enter 开始零偏标定（Ctrl+C取消）：")

        stats = collect_wrench_baseline(self.wrenches, self.config.bias_seconds, label="零偏")
        for arm, (mean, std, count) in stats.items():
            self.bias[arm] = mean
            rospy.loginfo("%s bias=%s std=%s samples=%d", arm.name,
                          np.round(mean, 4), np.round(std, 4), count)


    def read_force(self, arm: ArmGroup, dt: float) -> float:
        raw, stamp, _ = self.wrenches.snapshot(arm)
        if raw is None or stamp is None:
            raise RuntimeError("{}腕力尚未收到".format(arm.name))
        if time.monotonic() - stamp > self.config.stale_timeout_s:
            raise RuntimeError("{}腕力数据超时".format(arm.name))
        corrected = raw - self.bias[arm]
        projected = float(np.dot(corrected[:3], self.reaction_axes[arm]))
        compression = self.force_filters[arm].step(projected, dt)
        return compression

    def current_pose_checked(self, arm: ArmGroup) -> np.ndarray:
        pose = read_valid_tcp(self.robot, arm, self.config.stale_timeout_s)
        tracking_error = float(np.linalg.norm(pose[:3] - self.last_commands[arm][:3]))
        if tracking_error > self.config.tracking_error_limit_m:
            raise RuntimeError(
                "{} TCP跟踪误差{:.4f} m超过{:.4f} m".format(
                    arm.name, tracking_error, self.config.tracking_error_limit_m
                )
            )
        return pose

    def command_pose(self, arm: ArmGroup, inward_offset_m: float) -> np.ndarray:
        command = self.anchor_poses[arm].copy()
        command[:3] += self.inward_axes[arm] * inward_offset_m
        return command

    def log_stage1_summary(self, f_left: float, f_right: float) -> None:
        """记录实际位移与压缩力；方向始终取当前实例的TCP连线。"""
        inward = {}
        for arm in ARMS:
            actual = pose7(self.robot.get_tcp_rt(arm), "{} final TCP".format(arm.name))
            delta = actual[:3] - self.anchor_poses[arm][:3]
            inward[arm] = float(np.dot(delta, self.inward_axes[arm]))
            rospy.loginfo(
                "%s delta_xyz/mm=%s，向内位移=%.2fmm",
                arm.name,
                np.round(1000 * delta, 2),
                1000 * inward[arm],
            )
        rospy.loginfo(
            "STAGE1_GRASP_COMPLETE：总靠拢=%.2fmm，压缩力L/R=%.3f/%.3fN",
            1000 * sum(inward.values()),
            f_left,
            f_right,
        )

    def run_approach(self) -> None:
        target_force = float(self.config.target_force_n)
        grasp_trigger_force = max(
            self.config.contact_threshold_n,
            target_force - self.config.force_band_n,
        )
        rospy.logwarn(
            "沿预抓取双TCP水平连线向内靠拢；" "target=%.3f N/侧，max distance=%.1f mm",
            target_force,
            1000.0 * self.config.max_inward_distance_m,
        )
        rospy.logwarn(
            "双侧力稳定后自动闭合左右夹爪。",
        )
        rospy.logwarn(
            "闭爪触发：fL/fR均>=%.3f N，|fL-fR|<=%.3f N，持续%.2f秒",
            grasp_trigger_force,
            2.0 * self.config.imbalance_band_n,
            self.config.stable_seconds,
        )
        input("确认tray位置、力符号和安全上限后按 Enter " "开始靠拢（Ctrl+C取消）：")

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

        started = time.monotonic()
        previous = started
        flow = ApproachFlow()
        rate = rospy.Rate(CONTROL_HZ)

        while not rospy.is_shutdown():
            now = time.monotonic()
            raw_dt = now - previous
            previous = now
            if raw_dt > 0.15:
                raise RuntimeError("控制循环阻塞{:.3f}秒（上限0.150秒）".format(raw_dt))
            dt = float(np.clip(raw_dt, 0.005, 0.04))
            if now - started > self.config.duration_s:
                raise RuntimeError(
                    "柔顺靠拢超过最大运行时间{:.1f}秒；"
                    "请检查预抓取间隙及duration_s配置".format(self.config.duration_s)
                )

            cycle = self.read_approach_cycle(now, dt, now - started)
            commands = self.update_approach_motion(cycle)
            if self.update_grasp_completion(cycle, flow, grasp_trigger_force):
                return
            self.publish_approach_commands(cycle, commands)
            rate.sleep()

    def read_approach_cycle(self, now, dt, elapsed):
        """一次读取各侧TCP、压缩力并更新接触检测。"""
        measured: Dict[ArmGroup, float] = {}
        contacts: Dict[ArmGroup, bool] = {}

        for arm in ARMS:
            self.current_pose_checked(arm)
            compression = self.read_force(arm, dt)
            measured[arm] = compression
            contacts[arm] = self.contact_detectors[arm].update(compression, now)

        return ApproachCycle(now, dt, elapsed, measured, contacts)

    def update_approach_motion(self, cycle):
        """按力目标渐升和各侧接触状态计算向内位移。"""
        dt = cycle.dt
        commands = {}
        elapsed = cycle.elapsed
        ramp = float(np.clip(elapsed / self.config.target_ramp_seconds, 0.0, 1.0))
        ramped_target = self.config.target_force_n * ramp
        for arm in ARMS:
            offset = self.models[arm].step(
                cycle.measured[arm],
                ramped_target,
                self.contact_detectors[arm].is_contact_candidate(cycle.measured[arm]),
                dt,
            )
            commands[arm] = self.command_pose(arm, offset)
            if (
                offset >= self.config.max_inward_distance_m - 1e-6
                and not cycle.contacts[arm]
            ):
                raise RuntimeError(
                    "{}达到最大向内搜索距离仍未接触tray".format(arm.name)
                )

        return commands

    def update_grasp_completion(self, cycle, flow, grasp_trigger_force):
        """力稳定后启动后台闭爪，期间导纳继续运行。"""
        now, measured, contacts = cycle.now, cycle.measured, cycle.contacts
        left_contact = contacts[ArmGroup.LEFT]
        right_contact = contacts[ArmGroup.RIGHT]
        if left_contact != right_contact:
            if not flow.one_side_active:
                flow.one_side_active = True
                rospy.logwarn(
                    "单侧先接触：LEFT=%s RIGHT=%s，各侧按自身接触状态调速",
                    left_contact,
                    right_contact,
                )
        else:
            flow.one_side_active = False

        f_left = measured[ArmGroup.LEFT]
        f_right = measured[ArmGroup.RIGHT]
        f_mean = 0.5 * (f_left + f_right)
        imbalance = 0.5 * (f_left - f_right)
        target_stable = (
            left_contact
            and right_contact
            and f_left >= grasp_trigger_force
            and f_right >= grasp_trigger_force
            and abs(imbalance) <= self.config.imbalance_band_n
            and abs(self.models[ArmGroup.LEFT].velocity_m_s)
            <= self.config.max_contact_speed_m_s
            and abs(self.models[ArmGroup.RIGHT].velocity_m_s)
            <= self.config.max_contact_speed_m_s
        )
        timing_state(
            "闭爪及力保持"
            if self.hand_worker is not None
            else (
                "双侧力稳定等待"
                if target_stable
                else "接触导纳靠拢" if left_contact or right_contact else "自由靠拢"
            )
        )
        if target_stable:
            if flow.stable_since is None:
                flow.stable_since = now
            elif (
                now - flow.stable_since >= self.config.stable_seconds
                and self.hand_worker is None
            ):
                rospy.loginfo(
                    "APPROACH_FORCE_READY: fL=%.3f fR=%.3f "
                    "mean=%.3f imbalance=%+.3f N",
                    f_left,
                    f_right,
                    f_mean,
                    imbalance,
                )
                # ServoL继续运行；左右手服务在后台并行调用。
                timing_state("闭爪及力保持")
                self.hand_worker = DualGripperWorker(self.grippers, action="close")
                self.hand_worker.start()
                rospy.loginfo(
                    "开始异步闭合左右夹爪：target=%.1f mm "
                    "force=%.1f%% speed=%.1f%%",
                    self.config.gripper_close_position_mm,
                    self.config.gripper_force_percent,
                    self.config.gripper_speed_percent,
                )
        else:
            flow.stable_since = None

        if self.hand_worker is not None:
            previous_accepted = flow.hand_accepted_at
            flow.hand_accepted_at, settled = poll_gripper_completion(
                self.hand_worker, now, flow.hand_accepted_at, self.config.hand_settle_seconds,
                error_prefix="夹爪闭合失败: ")
            if previous_accepted is None and flow.hand_accepted_at is not None:
                rospy.loginfo(
                    "HAND_COMMAND_ACCEPTED：左右手闭合服务均成功；"
                    "不要求关节达到空载目标"
                )
            if settled:
                rospy.loginfo(
                    "GRASP_READY：夹爪闭合后已保持%.2f秒，" "连线方向导纳阶段完成",
                    self.config.hand_settle_seconds,
                )
                self.log_stage1_summary(f_left, f_right)
                return True

        return False

    def publish_approach_commands(self, cycle, commands):
        """每周期发布一次双臂指令并打印原靠拢诊断。"""
        f_left, f_right = cycle.measured[ArmGroup.LEFT], cycle.measured[ArmGroup.RIGHT]
        left_contact, right_contact = cycle.contacts[ArmGroup.LEFT], cycle.contacts[ArmGroup.RIGHT]
        f_mean, imbalance = 0.5 * (f_left + f_right), 0.5 * (f_left - f_right)
        dual_command = np.concatenate(
            [
                commands[ArmGroup.LEFT],
                commands[ArmGroup.RIGHT],
            ]
        )
        self.robot.servol(dual_command.tolist(), ArmGroup.DUAL)
        self.last_commands = {arm: commands[arm].copy() for arm in ARMS}
        rospy.loginfo_throttle(
            0.2,
            "APPROACH fL=%+.3f fR=%+.3f mean=%.3f imbalance=%+.3f N | "
            "contact L/R=%s/%s | offset L/R=%.2f/%.2f mm",
            f_left,
            f_right,
            f_mean,
            imbalance,
            left_contact,
            right_contact,
            1000.0 * self.models[ArmGroup.LEFT].position_m,
            1000.0 * self.models[ArmGroup.RIGHT].position_m,
        )

    def cleanup(self) -> None:
        if self.cleaned:
            return
        self.cleaned = True
        if self.servo_started:
            try:
                self.robot.stop()
            except Exception as exc:  # pragma: no cover - 仅用于实机兜底
                rospy.logerr("stop失败: %s", exc)
            try:
                self.robot.clear_servo_params()
            except Exception as exc:  # pragma: no cover - 仅用于实机兜底
                rospy.logerr("clear_servo_params失败: %s", exc)
            self.servo_started = False
        if self.hand_worker is not None:
            self.hand_worker.join()
