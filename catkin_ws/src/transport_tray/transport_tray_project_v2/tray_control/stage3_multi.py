"""多层竖直接触、水平寻槽、释放与撤离，以及顶面位姿记录。"""

from __future__ import annotations
from .common.runtime import input
from .common.gripper import DualGripperWorker, poll_gripper_completion
from .observability.force_recording import mark as trace_mark, state as trace_state
import argparse
from dataclasses import dataclass, field, replace
import math
import time
from typing import Dict, List, Optional
from types import SimpleNamespace
import numpy as np
import rospy
from scipy.spatial.transform import Rotation
from naviai_controller import ArmGroup
from .algorithms.moving_z_admittance import MovingZAxis, MovingZParameters, step_common_z
from .observability.stage3_logging import start_multilayer_log, write_multilayer_sample
from .stage3 import CONTACT_RETREAT_DISTANCE_M, ContactDescentState, update_contact_descent
from .algorithms.search_path import make_search_path
from .observability.pose_reporting import print_place_result
from .stage3_common import ARMS, CONTROL_HZ, Stage3Place, SwitchingZAdmittance, TargetForceLatch, outward_axes_from_tcp_line, quaternion_error_rad
from .common.state import estimate_tray_fast, prepare_grasp_components, load_state, require_grasp_transforms, require_pose, update_state, wrap_angle, yaw_from_pose7
from .common.vision import estimate_tray_pose

STAGE3_CYCLE_GAP_LIMIT_S = 0.15


def capture_top_tray_pose(args) -> List[float]:
    """记录唯一顶层位姿；识别最多三次，由公共视觉入口统一重试。"""
    return estimate_tray_pose(args, "T_base_top")


def direct_seating_geometry(
    error: Dict[str, float],
    left_down: float,
    right_down: float,
    distance_limit: float,
    z_difference_limit: float,
):
    """检查托盘中心的三维距离，并独立检查左右竖直位移是否对齐。"""
    distance = math.sqrt(error["x"] ** 2 + error["y"] ** 2 + error["z"] ** 2)
    z_difference = abs(left_down - right_down)
    return (
        distance < distance_limit and z_difference <= z_difference_limit,
        distance,
        z_difference,
    )


@dataclass
class SearchHistory:
    """历史最大下沉及其对应的TCP、XY偏移和托盘位姿。"""

    poses: Optional[dict] = None
    drop: float = float("-inf")
    offset: Optional[np.ndarray] = None
    tray_pose: Optional[list] = None


@dataclass
class ReturnMotion:
    """最低点回程及随后Yaw阶段共享的Z导纳/指令状态。"""
    commands: Optional[dict] = None
    settled_since: Optional[float] = None
    z_axis: Optional[MovingZAxis] = None
    anchor_offsets: Optional[dict] = None


@dataclass
class YawSearch:
    center: Optional[np.ndarray] = None
    offset: float = 0.0
    segment: int = 0
    started: Optional[float] = None
    capture_since: Optional[float] = None
    start_z: Optional[float] = None
    settled_since: Optional[float] = None


@dataclass
class MultilayerFlow:
    """跨周期保留的接触、寻槽、落座与张爪状态。"""

    stage_started: float
    both_force_since: Optional[float] = None
    seated_since: Optional[float] = None
    search_start_support: dict = field(default_factory=dict)
    search_center_correction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hand_accepted_at: Optional[float] = None


@dataclass
class Stage3Cycle:
    """本周期反馈快照；阶段函数不自行读取或发布机器人指令。"""

    now: float
    dt: float
    poses: Optional[dict]
    tray_estimate: list
    error: dict
    support: dict
    raw_wrenches: dict
    reached: dict
    both_at_target: bool
    observation: Optional[SimpleNamespace] = None


class MultilayerStage3(Stage3Place):

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        rospy.logwarn(
            "多层Stage3：初始接触上退%gmm；腕力用于导纳、捕获判断及记录，不以合力或力矩范数终止",
            1000 * CONTACT_RETREAT_DISTANCE_M,
        )
        state = load_state(args.state_file)
        if not bool(state.get("stage2_complete", False)):
            raise RuntimeError("状态文件未确认Stage2已到达P_pre")
        self.place_pose = require_pose(state, "place_tray_target")
        self.tray_tcp = require_grasp_transforms(state)
        self.grasp_components = prepare_grasp_components(self.tray_tcp)
        self.search_offset_base = np.zeros(2, dtype=np.float64)
        self.stage_name = "Z_DESCENT"
        self.xy_force_captured = False
        self.slot_yaw_fy_captured = False
        self.direct_seating_selected = False
        self.released = False
        self.radius_limit_release = False
        self.unconfirmed_release_reason = None
        self.history = SearchHistory()
        self.return_motion = ReturnMotion()
        self.yaw_search = YawSearch()
        self.final_search_offset = np.zeros(2, dtype=np.float64)
        self.final_drop_z = 0.0
        # 保留原恒力找平模型；初始接触改用与单层相同的定距退让接口。
        self.z_model = SwitchingZAdmittance(args)
        self.contact_descent = ContactDescentState()
        self.z_model.offsets = self.contact_descent.offsets
        self.contact_retreat_params = MovingZParameters(
            target_force=args.target_force, mass=args.mass, damping=args.damping,
            speed_limit=args.max_free_speed, acceleration_limit=args.max_acceleration,
            retreat_distance=CONTACT_RETREAT_DISTANCE_M, travel_limit=None, equilibrium_limit=None,
        )
        self.target_latches = {arm: TargetForceLatch(args) for arm in ARMS}
        self.force_log_path = None
        self.line_outward_axes = None
        self.descent_start_tray = None
        self.search_start_z = None
        self.moving_z_axis = None
        self.moving_z_anchor_offsets = None
        self.moving_z_params = None
        self.search_xy_paused = False
        self.search_xy_pause_started = None
        self.search_xy_pause_total = 0.0
        self.search_xy_pause_seconds = 0.0
        self.search_start_raw_fx = {}
        self.search_start_raw_fy = {}
        self.xy_path = make_search_path(args)

    def start_moving_z_search(self, support: Dict[ArmGroup, float]) -> None:
        """保留持物基线和双侧Z锚点，建立唯一共同Z导纳。"""
        defaults = MovingZParameters()
        self.search_xy_paused = False
        self.search_xy_pause_started = None
        self.search_xy_pause_total = 0.0
        self.search_xy_pause_seconds = 0.0
        self.moving_z_params = replace(
            defaults,
            travel_limit=None,
            equilibrium_limit=None,
            target_force=self.args.search_z_target_force,
            recovery_hysteresis=self.args.search_z_recovery_hysteresis,
            retreat_distance=self.args.search_z_retreat_distance,
            retreat_trigger_cycles=self.args.search_z_retreat_trigger_cycles,
            speed_limit=self.args.search_z_speed_limit,
            acceleration_limit=self.args.search_z_acceleration,
        )
        self.moving_z_descent_params = replace(
            self.moving_z_params,
            speed_limit=self.args.search_z_down_speed,
            acceleration_limit=self.args.search_z_down_acceleration,
        )
        self.moving_z_axis = MovingZAxis()
        self.moving_z_anchor_offsets = self.z_model.offsets.copy()
        rospy.logwarn(
            "SEARCH_MDK_START：共同Z；寻槽起点支撑力L/R=%.3f/%.3f N仅记录、不减去；持物基线相对目标力=%.2f N；M=%.2f D=%.2f K=%.2f，平衡点不设行程上限，Z退让限速=%.1f mm/s，加速度上限=%.1f mm/s²，力死区=±%.2f N，平衡点增益=%.6f，common连续超力%d周期上退%.2f mm后恢复下降导纳；重新触发迟滞=%.2f N",
            support[ArmGroup.LEFT],
            support[ArmGroup.RIGHT],
            self.moving_z_params.target_force,
            self.moving_z_params.mass,
            self.moving_z_params.damping,
            self.moving_z_params.stiffness,
            1000 * self.moving_z_params.speed_limit,
            1000 * self.moving_z_params.acceleration_limit,
            self.moving_z_params.force_deadband,
            self.moving_z_params.equilibrium_gain,
            self.moving_z_params.retreat_trigger_cycles,
            1000 * self.moving_z_params.retreat_distance,
            self.moving_z_params.recovery_hysteresis,
        )
        rospy.logwarn(
            "SEARCH_Z_DESCENT：下降限速=%.1f mm/s，加速度=%.1f mm/s²；定距退让沿用原速度和加速度",
            1000 * self.moving_z_descent_params.speed_limit,
            1000 * self.moving_z_descent_params.acceleration_limit,
        )

    def step_moving_z_search(
        self, support: Dict[ArmGroup, float], dt: float
    ) -> Dict[ArmGroup, float]:
        """以持物基线相对支撑力计算共同Z增量，保留双侧原有高度差。"""
        axis = self.moving_z_axis
        was_retreating = axis.retreat_active
        delta = step_common_z(
            axis,
            [support[arm] for arm in ARMS],
            dt,
            self.moving_z_params,
            self.moving_z_descent_params,
        )
        if axis.retreat_active != was_retreating:
            rospy.logwarn(
                "SEARCH_Z_RETREAT：%s；共同Z增量=%.3f mm，支撑力L/R=%.3f/%.3f N",
                (
                    "连续超力触发定距上退"
                    if axis.retreat_active
                    else "定距上退完成，恢复下降导纳"
                ),
                1000 * delta,
                support[ArmGroup.LEFT],
                support[ArmGroup.RIGHT],
            )
        result = {arm: self.moving_z_anchor_offsets[arm] + delta for arm in ARMS}
        self.z_model.offsets.update(result)
        return result

    def pause_search_xy_for_retreat(self, support, now):
        """退让触发前冻结XY及路径计时，退让指令完成后恢复。"""
        axis = self.moving_z_axis
        params = self.moving_z_params
        will_retreat = (
            axis.retreat_armed
            and max(support.values()) > params.target_force
            and axis.overload_cycles + 1 >= params.retreat_trigger_cycles
        )
        paused = axis.retreat_active or will_retreat
        if paused and (not self.search_xy_paused):
            self.search_xy_pause_started = now
            rospy.logwarn("SEARCH_XY_PAUSE：冻结XY目标和路径进度，开始共同Z定距退让")
        elif self.search_xy_paused and (not paused):
            self.search_xy_pause_total += now - self.search_xy_pause_started
            self.search_xy_pause_started = None
            rospy.logwarn("SEARCH_XY_RESUME：Z退让指令已到位，沿暂停前的路径继续寻槽")
        self.search_xy_paused = paused
        self.search_xy_pause_seconds = self.search_xy_pause_total + (
            now - self.search_xy_pause_started if paused else 0.0
        )
        return paused

    def calculate_search_observation(self, cycle):
        """同一帧计算观测：XY减搜索起点，Yaw减持物基线，Z仍使用原滤波支撑力。"""
        raw = cycle.raw_wrenches or {}
        xy_fx = {
            arm: float(raw[arm][0] - value)
            for arm, value in getattr(self, "search_start_raw_fx", {}).items()
            if arm in raw
        }
        xy_fy = {
            arm: float(raw[arm][1] - value)
            for arm, value in getattr(self, "search_start_raw_fy", {}).items()
            if arm in raw
        }
        yaw_fy = {arm: float(values[1] - self.baseline[arm][1]) for arm, values in raw.items()}
        estimate = cycle.tray_estimate
        start_z = getattr(cycle, "search_start_z", getattr(self, "search_start_z", None))
        drop = start_z - estimate[2] if start_z is not None and estimate is not None else None
        yaw_drop = (
            self.yaw_search.start_z - estimate[2]
            if self.yaw_search.start_z is not None and estimate is not None
            else None
        )
        return SimpleNamespace(
            xy_fx=xy_fx,
            xy_fy=xy_fy,
            yaw_fy=yaw_fy,
            drop=drop,
            yaw_drop=yaw_drop,
            required_drop=max(
                self.args.slot_min_descent, self.history.drop + self.args.radius_yaw_extra_descent
            ),
        )

    def start_force_log(self) -> None:
        start_multilayer_log(self)

    def write_force_sample(
        self,
        elapsed,
        support,
        raw_wrenches,
        search_start_support,
        error,
        pose_source,
        search_descent=None,
        observation=None,
    ) -> None:
        if observation is None:
            observation = self.calculate_search_observation(
                SimpleNamespace(raw_wrenches=raw_wrenches, tray_estimate=None)
            )
        write_multilayer_sample(
            self,
            elapsed,
            support,
            raw_wrenches,
            search_start_support,
            error,
            pose_source,
            search_descent,
            observation=observation,
        )

    def tcp_pair_checked(self) -> Dict[ArmGroup, np.ndarray]:
        left = self.current_pose_checked(ArmGroup.LEFT)
        right = self.current_pose_checked(ArmGroup.RIGHT)
        return {ArmGroup.LEFT: left, ArmGroup.RIGHT: right}

    def commanded_descent_pose(self):
        """受力尚未达标时，根据已下发指令预测高度，用于进度日志。"""
        if self.descent_start_tray is None:
            raise RuntimeError("Stage3下降起始tray位姿尚未记录")
        estimate = self.descent_start_tray.copy()
        estimate[2] -= 0.5 * (
            self.z_model.offsets[ArmGroup.LEFT] + self.z_model.offsets[ArmGroup.RIGHT]
        )
        return estimate, self.tray_error(estimate)

    def check_robot_state_without_tcp(self) -> None:
        """仅依据受力反馈执行竖直下降时，仍保留机器人状态检查。"""
        state_age = self.robot.get_uplimb_state_age()
        if state_age is None or state_age > self.args.stale_timeout:
            raise RuntimeError("UplimbState数据超时")
        for arm in ARMS:
            if self.robot.get_is_singular(arm):
                raise RuntimeError("{}臂进入奇异位形".format(arm.name))
        if self.robot.get_cmd_num() == 15:
            raise RuntimeError("机器人进入PROTECTED状态(cmd_num=15)")

    def estimate_tray(self, poses: Dict[ArmGroup, np.ndarray]):
        estimate, disagreement = estimate_tray_fast(
            poses[ArmGroup.LEFT], poses[ArmGroup.RIGHT], self.grasp_components
        )
        expected_z_difference = (
            self.z_model.offsets[ArmGroup.RIGHT] - self.z_model.offsets[ArmGroup.LEFT]
        )
        z_residual = disagreement["position_z_signed_m"] - expected_z_difference
        residual = math.hypot(disagreement["position_xy_m"], z_residual)
        disagreement["position_residual_m"] = residual
        if residual > self.args.tray_fusion_position_limit:
            raise RuntimeError(
                "左右TCP反算tray未预期位置差{:.2f} mm超过上限{:.2f} mm（原始差{:.2f} mm，指令Z差{:.2f} mm）".format(
                    1000.0 * residual,
                    1000.0 * self.args.tray_fusion_position_limit,
                    1000.0 * disagreement["position_m"],
                    1000.0 * expected_z_difference,
                )
            )
        if disagreement["yaw_rad"] > math.radians(self.args.tray_fusion_yaw_limit_deg):
            raise RuntimeError(
                "左右TCP反算tray yaw差{:.2f} deg超过上限".format(
                    math.degrees(disagreement["yaw_rad"])
                )
            )
        return (estimate, disagreement)

    def tray_error(self, estimate: List[float]) -> Dict[str, float]:
        return {
            "x": estimate[0] - self.place_pose[0],
            "y": estimate[1] - self.place_pose[1],
            "z": estimate[2] - self.place_pose[2],
            "yaw": wrap_angle(
                yaw_from_pose7(estimate) - yaw_from_pose7(self.place_pose)
            ),
        }

    def rate_limit_xy(self, target: np.ndarray, dt: float) -> None:
        delta = np.asarray(target, dtype=np.float64) - self.search_offset_base
        distance = float(np.linalg.norm(delta))
        maximum = self.args.search_xy_speed * dt
        if distance > maximum and distance > 1e-12:
            delta *= maximum / distance
        self.search_offset_base += delta

    def command_pose(self, arm: ArmGroup, downward_offset: float) -> np.ndarray:
        command = self.anchor_poses[arm].copy()
        command[0:2] += self.search_offset_base
        command[2] -= downward_offset
        return command

    def choose_z_mode(self, reached: Dict[ArmGroup, bool]) -> str:
        if self.stage_name in ("OPENING", "RETURN_SEARCH_LOWEST"):
            return SwitchingZAdmittance.HOLD_BOTH
        if self.stage_name in ("XY_SEARCH", "FINAL_Z_SEATING"):
            return SwitchingZAdmittance.DUAL
        if reached[ArmGroup.LEFT] and reached[ArmGroup.RIGHT]:
            return SwitchingZAdmittance.DUAL
        if reached[ArmGroup.LEFT]:
            return SwitchingZAdmittance.RIGHT_ONLY
        if reached[ArmGroup.RIGHT]:
            return SwitchingZAdmittance.LEFT_ONLY
        return SwitchingZAdmittance.DUAL

    def run_multilayer_descent(self) -> None:
        """统一50Hz循环：反馈快照、阶段判断、一次指令发布和周期等待。"""
        input("确认下方堆叠区域、卡槽和急停无误后按 Enter 开始（Ctrl+C取消）：")
        starts = self.tcp_pair_checked()
        self.anchor_poses = {arm: starts[arm].copy() for arm in ARMS}
        self.last_commands = {arm: starts[arm].copy() for arm in ARMS}
        self.descent_start_tray, _ = self.estimate_tray(starts)
        self.line_outward_axes = outward_axes_from_tcp_line(starts)
        self.start_force_log()
        self.start_servo()
        started = previous = time.monotonic()
        self.flow = MultilayerFlow(stage_started=started)
        rate = rospy.Rate(CONTROL_HZ)
        while not rospy.is_shutdown():
            now = time.monotonic()
            gap = now - previous
            previous = now
            if gap > STAGE3_CYCLE_GAP_LIMIT_S:
                raise RuntimeError(
                    "Stage3控制循环阻塞{:.3f}秒，上限{:.3f}秒，阶段={}".format(
                        gap, STAGE3_CYCLE_GAP_LIMIT_S, self.stage_name
                    )
                )
            cycle, pose_source = self.read_cycle(now, float(np.clip(gap, 0.005, 0.04)))
            # 保持原采样时序：先记录当前阶段，再执行阶段切换与发布。
            self.write_force_sample(
                now - started,
                cycle.support,
                cycle.raw_wrenches,
                self.flow.search_start_support,
                cycle.error,
                pose_source,
                search_descent=cycle.observation.drop,
                observation=cycle.observation,
            )
            result = self.step_stage(cycle)
            if result == "done":
                return
            if result == "skip":
                continue
            self.publish_cycle(cycle)
            rate.sleep()

    def read_cycle(self, now, dt):
        """读取周期快照；初始自由下降沿用指令预测，稳定接触后读取实测TCP。"""
        measured = self.stage_name != "Z_DESCENT" or (
            self.flow.both_force_since is not None
            and now - self.flow.both_force_since >= self.args.stable_seconds
        )
        poses = None
        if measured:
            poses = self.tcp_pair_checked()
            estimate, _ = self.estimate_tray(poses)
            error = self.tray_error(estimate)
        else:
            self.check_robot_state_without_tcp()
            estimate, error = self.commanded_descent_pose()
        support, raw, reached = {}, {}, {}
        for arm in ARMS:
            sample = self.read_wrench_sample(arm, dt)
            support[arm] = sample.support
            reached[arm] = self.target_latches[arm].update(support[arm], now)
            raw[arm] = sample.raw
        if self.stage_name == "XY_SEARCH":
            self.remember_search_lowest(self.search_start_z - estimate[2], poses)
        imbalance = 0.5 * (support[ArmGroup.LEFT] - support[ArmGroup.RIGHT])
        if self.stage_name == "Z_DESCENT":
            # 先侧上退后可卸载，初始接触按“达到过”锁存，不再等待双侧同时恒力。
            reached = {
                arm: self.contact_descent.reached[arm] or support[arm] >= self.args.target_force
                for arm in ARMS
            }
            both = all(reached.values())
        else:
            both = all(reached.values()) and abs(imbalance) <= self.args.imbalance_limit
        cycle = Stage3Cycle(now, dt, poses, estimate, error, support, raw, reached, both)
        cycle.observation = self.calculate_search_observation(cycle)
        return cycle, ("measured" if measured else "commanded")

    def publish_cycle(self, cycle):
        """按阶段计算Z偏移并发布一次双臂ServoL。"""
        trace_state(self.stage_name)
        if self.stage_name != "Z_DESCENT":
            self.z_model.set_mode(self.choose_z_mode(cycle.reached))
        yaw_motion = self.stage_name in ("RETURN_SEARCH_LOWEST", "YAW_SEARCH_LOWEST")
        if self.stage_name == "Z_DESCENT":
            update_contact_descent(
                self.args, self.contact_descent, cycle.support, cycle.dt, self.contact_retreat_params
            )
            offsets = self.contact_descent.offsets
            self.z_model.offsets = offsets
        elif yaw_motion:
            offsets = self.step_radius_z_admittance(cycle.support, cycle.dt)
        elif self.stage_name == "OPENING":
            offsets = self.z_model.offsets.copy()
        elif self.stage_name == "XY_SEARCH":
            offsets = self.step_moving_z_search(cycle.support, cycle.dt)
        else:
            target = self.args.target_force + (
                self.args.control_margin if self.stage_name == "Z_DESCENT" else 0.0
            )
            offsets = self.z_model.step(
                cycle.support,
                target,
                max(cycle.support.values()) >= 0.7 * self.args.contact_threshold,
                cycle.dt,
            )
        commands = {arm: self.command_pose(arm, offsets[arm]) for arm in ARMS}
        if self.return_motion.commands is not None:
            commands = {arm: self.return_motion.commands[arm].copy() for arm in ARMS}
            if yaw_motion:
                for arm in ARMS:
                    commands[arm][2] = self.anchor_poses[arm][2] - offsets[arm]
                    self.return_motion.commands[arm][2] = commands[arm][2]
        self.robot.servol(
            np.concatenate([commands[arm] for arm in ARMS]).tolist(), ArmGroup.DUAL
        )
        self.last_commands = {arm: commands[arm].copy() for arm in ARMS}

    def remember_search_lowest(self, drop: float, poses) -> None:
        """保存托盘中心下沉量最大时的实测双末端位姿。"""
        if self.history.poses is None or drop > self.history.drop:
            self.history.drop = float(drop)
            self.history.poses = {arm: poses[arm].copy() for arm in ARMS}
            self.history.offset = self.search_offset_base.copy()

    def transition(self, stage, now=None):
        """切换阶段；仅显式传入now时重置阶段计时，其他状态由入口函数设置。"""
        self.stage_name = stage
        if now is not None:
            self.flow.stage_started = now
        trace_state(stage)

    def begin_release(self, now=None):
        """统一启动双爪张开；阶段调用者决定是否重置阶段计时。"""
        self.hand_worker = DualGripperWorker(self.grippers, action="open")
        self.hand_worker.start()
        self.transition("OPENING", now)

    def finish_search_and_release(self, drop, offset, reason=None, now=None):
        """提交搜索结果并启动多层张爪；不改变单层退出伺服的流程。"""
        if reason is not None:
            self.unconfirmed_release_reason = reason
        if drop is not None:
            self.final_drop_z = drop
        self.final_search_offset = offset.copy()
        self.begin_release(now)

    def start_unconfirmed_release(self, reason, tray_estimate, search_start_z) -> None:
        """寻槽判据未满足时在当前位置冻结双TCP，复用张爪和撤离流程。"""
        drop = float(search_start_z - tray_estimate[2]) if search_start_z is not None else None
        self.finish_search_and_release(drop, self.search_offset_base, reason=reason)
        trace_mark("stage3_unconfirmed_release", reason)
        rospy.logwarn(
            "STAGE3_UNCONFIRMED_RELEASE：%s；未满足入槽判据，冻结双TCP并自动张爪撤离，随后视觉记录顶层",
            reason,
        )

    def start_return_to_lowest(self, drop: float) -> None:
        if self.history.poses is None:
            raise RuntimeError("缺少寻槽最低位置记录，禁止自动张爪")
        self.radius_limit_release = True
        self.final_drop_z = self.history.drop
        self.final_search_offset = self.history.offset.copy()
        self.return_motion.commands = {
            arm: self.last_commands[arm].copy() for arm in ARMS
        }
        self.return_motion.settled_since = None
        self.return_motion.z_axis = MovingZAxis()
        self.return_motion.anchor_offsets = {
            arm: float(self.z_model.offsets[arm]) for arm in ARMS
        }
        self.history.tray_pose = estimate_tray_fast(
            self.history.poses[ArmGroup.LEFT],
            self.history.poses[ArmGroup.RIGHT],
            self.grasp_components,
        )[0]
        self.yaw_search.center = np.asarray(self.history.tray_pose[:3], dtype=float)
        self.transition("RETURN_SEARCH_LOWEST")
        trace_mark("radius_limit_return", "最大半径返回寻槽最低位置")
        rospy.logwarn(
            "SEARCH_LOWEST_RETURN：当前下沉=%.2f mm，历史最大下沉=%.2f mm；只返回该最低点，回程Z保持导纳；到位后在±%.1f度内搜索Yaw，未确认入槽",
            1000.0 * drop,
            1000.0 * self.history.drop,
            self.args.radius_yaw_range_deg,
        )

    def step_radius_z_admittance(self, support, dt):
        """回程与Yaw搜索采用共同Z导纳，最大接触侧超目标力时向上退让。"""
        force = max((float(support[arm]) for arm in ARMS))
        axis = self.return_motion.z_axis
        if force > self.args.target_force and axis.velocity > 0.0:
            axis.velocity = 0.0
        speed = (
            self.args.radius_z_retreat_speed
            if force > self.args.target_force
            else self.args.radius_z_down_speed
        )
        params = replace(
            MovingZParameters(),
            target_force=self.args.target_force,
            speed_limit=speed,
            acceleration_limit=self.args.radius_z_acceleration,
            travel_limit=None,
            equilibrium_limit=None,
        )
        previous_displacement = axis.displacement
        axis.step(force, dt, params)
        if force > self.args.target_force and axis.displacement > previous_displacement:
            axis.displacement = previous_displacement
            axis.velocity = 0.0
        result = {
            arm: self.return_motion.anchor_offsets[arm] + axis.displacement for arm in ARMS
        }
        self.z_model.offsets.update(result)
        return result

    def advance_search_return(self, poses, now, dt):
        """只回放最低点XY与姿态；途中Z由导纳退让，不强行追历史高度。"""
        deltas = {
            arm: self.history.poses[arm][:2] - self.return_motion.commands[arm][:2]
            for arm in ARMS
        }
        distance = max((float(np.linalg.norm(delta)) for delta in deltas.values()))
        step = self.args.radius_return_xy_speed * dt
        fraction = min(1.0, step / distance) if distance > 0.0 else 1.0
        for arm in ARMS:
            self.return_motion.commands[arm][:2] += fraction * deltas[arm]
            self.return_motion.commands[arm][3:] = self.history.poses[arm][3:]
        reached = distance <= step and all(
            (
                np.linalg.norm(poses[arm][:2] - self.history.poses[arm][:2]) <= 0.0005
                and quaternion_error_rad(poses[arm][3:], self.history.poses[arm][3:])
                <= math.radians(0.2)
                for arm in ARMS
            )
        )
        if not reached:
            self.return_motion.settled_since = None
            return
        if self.return_motion.settled_since is None:
            self.return_motion.settled_since = now
        elif now - self.return_motion.settled_since >= 0.3:
            self.yaw_search.offset = 0.0
            self.yaw_search.segment = 0
            self.yaw_search.started = None
            self.yaw_search.capture_since = None
            # 沿用原进入Yaw时的稳定时间；后续端点使用独立计时字段。
            self.yaw_search.settled_since = self.return_motion.settled_since
            self.yaw_search.start_z = estimate_tray_fast(
                poses[ArmGroup.LEFT], poses[ArmGroup.RIGHT], self.grasp_components
            )[0][2]
            self.transition("YAW_SEARCH_LOWEST")
            trace_mark("lowest_yaw_start", "唯一最低下沉点开启Z导纳及±3度Yaw搜索")
            rospy.logwarn(
                "SEARCH_LOWEST_YAW_START：XY到达唯一历史最低点；Z下降/退让限速=%.1f/%.1f mm/s，Yaw限幅=±%.1f度、速度=%.2f度/秒",
                1000 * self.args.radius_z_down_speed,
                1000 * self.args.radius_z_retreat_speed,
                self.args.radius_yaw_range_deg,
                self.args.radius_yaw_speed_deg_s,
            )

    def advance_lowest_yaw_search(
        self, poses, tray_estimate, search_start_z, now, dt, raw_wrenches=None, observation=None
    ):
        """先检查释放条件；未触发时再推进Yaw扫描，同周期观测只计算一次。"""
        if observation is None:
            observation = self.calculate_search_observation(
                SimpleNamespace(
                    tray_estimate=tray_estimate,
                    raw_wrenches=raw_wrenches,
                    search_start_z=search_start_z,
                )
            )
        if not self.check_yaw_capture(observation, tray_estimate, search_start_z, now):
            self.advance_yaw_command(poses, tray_estimate, search_start_z, now, dt, observation)

    def check_yaw_capture(self, observation, tray_estimate, search_start_z, now):
        """检查Fy立即捕获、下沉去抖和搜索超时；触发后保持自动释放。"""
        if self.yaw_search.started is None:
            self.yaw_search.started = now
        if now - self.yaw_search.started > self.args.radius_yaw_timeout:
            self.start_unconfirmed_release(
                "最低点Yaw搜索超时", tray_estimate, search_start_z
            )
            return True
        drop, required = observation.drop, observation.required_drop
        yaw_phase_drop, relative_fy = observation.yaw_drop, observation.yaw_fy
        fy_hit = any((abs(value) > 2.5 for value in relative_fy.values()))
        if fy_hit or (
            drop >= required and yaw_phase_drop >= self.args.radius_yaw_extra_descent
        ):
            if fy_hit or (
                self.yaw_search.capture_since is not None
                and now - self.yaw_search.capture_since
                >= self.args.yaw_descent_debounce
            ):
                self.slot_yaw_fy_captured = fy_hit
                self.finish_search_and_release(drop, self.history.offset)
                trace_mark(
                    "lowest_yaw_fy_capture" if fy_hit else "lowest_yaw_capture",
                    (
                        "Yaw寻槽相对持物基线Fy L/R={:+.3f}/{:+.3f}N，任一侧绝对值>2.5N，入槽条件满足".format(
                            relative_fy[ArmGroup.LEFT], relative_fy[ArmGroup.RIGHT]
                        )
                        if fy_hit
                        else "唯一最低点Yaw搜索产生新增下沉"
                    ),
                )
                if fy_hit:
                    rospy.logwarn(
                        "SEARCH_LOWEST_YAW_FY_CAPTURE：相对Fy L/R=%+.3f/%+.3f N；入槽条件满足，自动张爪",
                        relative_fy[ArmGroup.LEFT],
                        relative_fy[ArmGroup.RIGHT],
                    )
                rospy.logwarn(
                    "SEARCH_LOWEST_YAW_CAPTURE：实测下沉=%.2f mm，门槛=%.2f mm；冻结双TCP并自动张爪，未独立证明入槽",
                    1000 * drop,
                    1000 * required,
                )
                return True
            elif self.yaw_search.capture_since is None:
                self.yaw_search.capture_since = now
        else:
            self.yaw_search.capture_since = None
        return False

    def advance_yaw_command(self, poses, tray_estimate, search_start_z, now, dt, observation):
        """按原速度扫描正端点、负端点和中心，逐端点稳定后推进。"""
        drop, required = observation.drop, observation.required_drop
        limit = math.radians(self.args.radius_yaw_range_deg)
        targets = (limit, -limit, 0.0)
        target = targets[self.yaw_search.segment]
        step = math.radians(self.args.radius_yaw_speed_deg_s) * dt
        self.yaw_search.offset += float(
            np.clip(target - self.yaw_search.offset, -step, step)
        )
        rotation = Rotation.from_euler("z", self.yaw_search.offset)
        center = self.yaw_search.center
        for arm in ARMS:
            base = self.history.poses[arm]
            command = self.return_motion.commands[arm]
            command[:2] = (center + rotation.apply(base[:3] - center))[:2]
            command[3:] = (rotation * Rotation.from_quat(base[3:])).as_quat()
        if abs(self.yaw_search.offset - target) > 1e-08:
            self.yaw_search.settled_since = None
            return
        actual, _ = estimate_tray_fast(
            poses[ArmGroup.LEFT], poses[ArmGroup.RIGHT], self.grasp_components
        )
        target_yaw = yaw_from_pose7(self.history.tray_pose) + target
        if abs(wrap_angle(yaw_from_pose7(actual) - target_yaw)) > math.radians(0.2):
            self.yaw_search.settled_since = None
            return
        if self.yaw_search.settled_since is None:
            self.yaw_search.settled_since = now
        elif now - self.yaw_search.settled_since >= 0.3:
            rospy.loginfo(
                "SEARCH_LOWEST_YAW_ENDPOINT：相对最低点Yaw=%+.2f度，当前下沉=%.2f mm，新增下沉门槛=%.2f mm",
                math.degrees(target),
                1000 * drop,
                1000 * required,
            )
            self.yaw_search.segment += 1
            self.yaw_search.settled_since = None
            if self.yaw_search.segment == len(targets):
                self.start_unconfirmed_release(
                    "最低点±{:.1f}度Yaw搜索结束，未观察到持续新增下沉".format(
                        self.args.radius_yaw_range_deg
                    ),
                    tray_estimate,
                    search_start_z,
                )

    def run_experiment(self) -> None:
        trace_state("WAIT_READY")
        self.wait_ready()
        poses = self.tcp_pair_checked()
        estimate, disagreement = self.estimate_tray(poses)
        error = self.tray_error(estimate)
        rospy.loginfo(
            "Stage3启动估计 tray=%s error xyz/mm=[%+.2f,%+.2f,%+.2f] yaw=%+.2fdeg fusion=%.2fmm",
            np.round(estimate, 6),
            1000.0 * error["x"],
            1000.0 * error["y"],
            1000.0 * error["z"],
            math.degrees(error["yaw"]),
            1000.0 * disagreement["position_m"],
        )
        if not self.args.execute:
            rospy.logwarn("未指定--execute，仅完成方法B位姿估计，不发送运动")
            return
        trace_state("BASELINE_CAPTURE")
        self.capture_baseline()
        try:
            self.run_multilayer_descent()
        finally:
            if self.search_start_z is not None:
                update_state(
                    self.args.state_file, stage3_search_start_z_m=self.search_start_z
                )
        if rospy.is_shutdown():
            raise RuntimeError("ROS 已关闭，禁止继续放置后运动")
        self.exit_servo()
        if self.released:
            self.retreat()
        else:
            rospy.logwarn(
                "实验停止点已到达：双爪保持闭合，不执行下探、连线外撤、Z向抬升或视觉记录"
            )
        trace_state("RELEASE_AND_RETREAT_COMPLETE")
        self.save_placement_result()

    def save_placement_result(self):
        """保存当前放置结果和历史最低点，沿用原JSON字段。"""
        update_state(
            self.args.state_file,
            stage3_search_z_mode="common",
            stage3_search_z_force_reference="held_baseline",
            stage3_xy_search_path=self.args.search_path,
            stage3_slot_capture_mode=(
                "unconfirmed_release"
                if self.unconfirmed_release_reason
                else (
                    "radius_limit_release"
                    if self.radius_limit_release
                    else (
                        "near_place_direct"
                        if self.direct_seating_selected
                        else "descent_only"
                    )
                )
            ),
            stage3_search_offset_xy=self.final_search_offset.astype(float).tolist(),
            stage3_raw_fx_captured=self.xy_force_captured,
            stage3_yaw_fy_captured=self.slot_yaw_fy_captured,
            stage3_drop_z=float(self.final_drop_z),
            stage3_search_start_z_m=self.search_start_z,
            stage3_release_and_retreat_complete=bool(self.released),
            stage3_radius_limit_release=self.radius_limit_release,
            stage3_unconfirmed_release_reason=self.unconfirmed_release_reason,
            stage3_best_search_drop_z=(
                self.history.drop if self.history.poses is not None else None
            ),
            stage3_best_search_tcp_poses=(
                {
                    side: self.history.poses[arm].astype(float).tolist()
                    for side, arm in (
                        ("left", ArmGroup.LEFT),
                        ("right", ArmGroup.RIGHT),
                    )
                }
                if self.history.poses is not None
                else None
            ),
        )

    def step_stage(self, cycle):
        """每周期只执行当前阶段；skip沿用原分支的立即换周期行为。"""
        handlers = {
            "Z_DESCENT": self.step_initial_descent,
            "XY_SEARCH": self.step_xy_search,
            "RETURN_SEARCH_LOWEST": self.step_return_to_lowest,
            "YAW_SEARCH_LOWEST": self.step_yaw_search,
            "FINAL_Z_SEATING": self.step_seating,
            "OPENING": self.step_release,
        }
        return handlers[self.stage_name](cycle)

    def step_initial_descent(self, cycle):
        """双侧接触锁存后保持原稳定等待，再选择直接落座或建立XY寻槽起点。"""
        if cycle.both_at_target:
            if self.flow.both_force_since is None:
                self.flow.both_force_since = cycle.now
            elif cycle.now - self.flow.both_force_since >= self.args.stable_seconds:
                direct_seating, place_distance, z_difference = direct_seating_geometry(
                    cycle.error,
                    self.z_model.offsets[ArmGroup.LEFT],
                    self.z_model.offsets[ArmGroup.RIGHT],
                    self.args.direct_seating_distance,
                    self.args.direct_seating_max_z_difference,
                )
                if direct_seating:
                    self.enter_seating(cycle, place_distance, z_difference)
                else:
                    self.enter_xy_search(cycle, place_distance, z_difference)
        else:
            self.flow.both_force_since = None

    def enter_seating(self, cycle, place_distance, z_difference):
        """建立直接落座阶段的结果和计时。"""
        self.direct_seating_selected = True
        self.final_search_offset = self.search_offset_base.copy()
        self.final_drop_z = 0.0
        self.transition("FINAL_Z_SEATING", cycle.now)
        rospy.logwarn(
            "Stage3直接落座：距P_place=%.2f mm < %.2f mm，双侧下降差=%.2f mm <= %.2f mm；跳过XY寻槽，继续检查平均Z支撑力稳定",
            1000.0 * place_distance,
            1000.0 * self.args.direct_seating_distance,
            1000.0 * z_difference,
            1000.0 * self.args.direct_seating_max_z_difference,
        )

    def enter_xy_search(self, cycle, place_distance, z_difference):
        """建立搜索起点、Fx/Fy基线、共同Z模型及路径中心。"""
        self.search_start_z = float(cycle.tray_estimate[2])
        self.remember_search_lowest(0.0, cycle.poses)
        self.flow.search_start_support = cycle.support.copy()
        self.search_start_raw_fx = {
            arm: float(cycle.raw_wrenches[arm][0]) for arm in ARMS
        }
        self.search_start_raw_fy = {
            arm: float(cycle.raw_wrenches[arm][1]) for arm in ARMS
        }
        trace_mark(
            "xy_force_baseline",
            "寻槽起点Fy L/R={:+.3f}/{:+.3f}N".format(
                self.search_start_raw_fy[ArmGroup.LEFT],
                self.search_start_raw_fy[ArmGroup.RIGHT],
            ),
        )
        rospy.loginfo(
            "SEARCH_FX_BASELINE：进入XY寻槽时原始Fx L/R=%+.3f/%+.3f N；后续使用各侧相对该值的Fx增量判断入槽",
            self.search_start_raw_fx[ArmGroup.LEFT],
            self.search_start_raw_fx[ArmGroup.RIGHT],
        )
        self.start_moving_z_search(self.flow.search_start_support)
        self.flow.search_center_correction = np.array(
            [-cycle.error["x"], -cycle.error["y"]], dtype=np.float64
        )
        self.transition("XY_SEARCH", cycle.now)
        rospy.logwarn(
            "Stage3状态 Z_DESCENT -> %s；起点Z力L/R=%.2f/%.2f N；error xyz=[%+.2f,%+.2f,%+.2f] mm yaw=%+.2f deg；距P_place=%.2f mm，双侧下降差=%.2f mm",
            self.stage_name,
            self.flow.search_start_support[ArmGroup.LEFT],
            self.flow.search_start_support[ArmGroup.RIGHT],
            1000.0 * cycle.error["x"],
            1000.0 * cycle.error["y"],
            1000.0 * cycle.error["z"],
            math.degrees(cycle.error["yaw"]),
            1000.0 * place_distance,
            1000.0 * z_difference,
        )

    def step_xy_search(self, cycle):
        """检查力/下沉捕获，再推进XY路径；搜索结束则转入最低点回退。"""
        assert self.search_start_z is not None
        observation = cycle.observation
        drop = observation.drop
        xy_force_hit = any(
            (
                abs(observation.xy_fx[arm])
                > self.args.xy_force_capture_threshold
                or abs(observation.xy_fy[arm])
                > self.args.xy_force_capture_threshold
                for arm in ARMS
            )
        )
        if drop >= self.args.slot_min_descent or xy_force_hit:
            self.xy_force_captured = bool(xy_force_hit)
            if xy_force_hit:
                trace_mark(
                    "xy_fx_fy_capture",
                    "XY任一轴力入槽：相对Fx L/R={:+.3f}/{:+.3f}N，相对Fy L/R={:+.3f}/{:+.3f}N；任一侧任一轴绝对值>{:.1f}N，无Z门槛".format(
                        observation.xy_fx[ArmGroup.LEFT],
                        observation.xy_fx[ArmGroup.RIGHT],
                        observation.xy_fy[ArmGroup.LEFT],
                        observation.xy_fy[ArmGroup.RIGHT],
                        self.args.xy_force_capture_threshold,
                    ),
                )
            self.finish_search_and_release(drop, self.search_offset_base, now=cycle.now)
            trace_mark("moving_z_slot_release", "移动Z导纳寻槽触发后自动张爪")
            rospy.loginfo(
                "XY捕获：触发=%s，下沉=%.2fmm，自动张爪",
                "相对Fx/Fy" if xy_force_hit else "下沉",
                1000.0 * drop,
            )
        else:
            xy_paused = self.pause_search_xy_for_retreat(cycle.support, cycle.now)
            elapsed = cycle.now - self.flow.stage_started - self.search_xy_pause_seconds
            local, path_finished = self.xy_path.target(
                self.search_offset_base - self.flow.search_center_correction,
                elapsed=elapsed,
                yaw=yaw_from_pose7(self.place_pose),
                paused=xy_paused,
            )
            if elapsed > self.args.search_timeout:
                self.start_return_to_lowest(drop)
                self.flow.stage_started = cycle.now
                return "skip"
            if path_finished:
                self.start_return_to_lowest(drop)
                self.flow.stage_started = cycle.now
            elif not xy_paused:
                target = self.flow.search_center_correction + local
                self.rate_limit_xy(target, cycle.dt)

    def step_return_to_lowest(self, cycle):
        """推进最低点回退，不设置回程时间限制。"""
        self.advance_search_return(cycle.poses, cycle.now, cycle.dt)

    def step_yaw_search(self, cycle):
        """在最低点执行一个周期的Yaw扫描及自动释放判断。"""
        self.advance_lowest_yaw_search(
            cycle.poses,
            cycle.tray_estimate,
            self.search_start_z,
            cycle.now,
            cycle.dt,
            cycle.raw_wrenches,
            observation=cycle.observation,
        )

    def step_seating(self, cycle):
        """直接落座：平均支撑力达标并持续稳定后自动张爪。"""
        mean_support = 0.5 * sum(cycle.support.values())
        if mean_support < self.args.target_force:
            self.flow.seated_since = None
        elif self.flow.seated_since is None:
            self.flow.seated_since = cycle.now
        elif cycle.now - self.flow.seated_since >= self.args.seated_stable_seconds:
            trace_mark("direct_seating_release", "近位落座支撑力稳定，自动张爪")
            self.begin_release(cycle.now)
            rospy.loginfo(
                "直接落座：平均Z力=%.2fN达到目标%.2fN并稳定，自动张爪",
                mean_support,
                self.args.target_force,
            )

    def step_release(self, cycle):
        """等待双夹爪服务及开度确认完成，锁存释放成功。"""
        assert self.hand_worker is not None
        self.flow.hand_accepted_at, settled = poll_gripper_completion(
            self.hand_worker, cycle.now, self.flow.hand_accepted_at,
            self.args.hand_open_settle_seconds, error_prefix="张开夹爪失败: ")
        if settled:
            self.released = True
            rospy.loginfo(
                "STAGE3_PLACE_COMPLETE：%s；张爪完成",
                (
                    "未确认入槽（{}）".format(self.unconfirmed_release_reason)
                    if self.unconfirmed_release_reason
                    else (
                        "最大半径搜索结束（未确认入槽）"
                        if self.radius_limit_release
                        else "落座"
                    )
                ),
            )
            return "done"


def record_top_pose(args, expected_pose: Optional[List[float]] = None) -> List[float]:
    """保存实际视觉顶面；P_place仅用于打印记录。"""
    from .common.vision import has_record_frame

    if not has_record_frame():
        rospy.sleep(args.vision_settle)
    pose = capture_top_tray_pose(args)
    state = load_state(args.state_file, required=False)
    if expected_pose is not None:
        print_place_result(
            expected_pose,
            pose,
            release_reason=state.get("stage3_unconfirmed_release_reason"),
            radius_limit=state.get("stage3_radius_limit_release", False),
        )
    layer_index = (
        int(args.layer_index)
        if args.layer_index is not None
        else int(state.get("target_layer_index", state.get("layer_index", 0) + 1))
    )
    update_state(
        args.state_file,
        top_tray_pose=pose,
        layer_index=layer_index,
        top_pose_source="vision_after_release_and_retreat",
        top_pose_backend=args.vision_backend,
        top_pose_label=args.label,
        stage3_complete=True,
    )
    rospy.loginfo("T_base_top已保存，layer_index=%d: %s", layer_index, pose)
    return pose
