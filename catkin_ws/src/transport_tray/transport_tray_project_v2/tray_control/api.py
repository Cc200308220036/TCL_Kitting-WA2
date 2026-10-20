"""单层和多层托盘搬运的同步 Python 调用接口。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import rospy

from . import stage0, stage1, stage2, stage3_multi
from .common import state, vision
from .common.runtime import operation, managed_controller
from .config import DEFAULT_CONFIG, TrayConfig, motion_values
from .stage3 import SingleStage3Place
from .observability.pose_reporting import print_pose6d
from .observability.timing import phase, state as timing_state


@dataclass(frozen=True)
class StageResult:
    step: str
    completed: bool
    state_path: str
    measurements: Dict[str, Any] = field(default_factory=dict)
    targets: Dict[str, Any] = field(default_factory=dict)

class TrayOperationError(RuntimeError):
    """阶段执行失败，或现场物理状态不满足执行前置条件。"""


class _TrayBase:
    _multi_layer = False

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG,
        *,
        state_path: str | None = None,
        confirm: Callable[[str, Dict[str, Any]], bool],
        before_vision: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = TrayConfig(config_path, state_path)
        self.confirm = confirm
        self.before_vision = before_vision

    @property
    def state_path(self) -> str:
        return self.config.state_path

    def _require(self, step: str) -> None:
        current = state.load_state(self.state_path, required=False)
        if not current.get("package_steps", {}).get(step) == "SUCCEEDED":
            raise TrayOperationError("先完成 {}".format(step))

    def _perform(self, name: str, action: Callable[[], Any]) -> StageResult:
        if not rospy.core.is_initialized():
            raise TrayOperationError("调用前必须由主程序初始化一次 rospy 节点")
        with phase(
            {
                "prep": "Stage0",
                "grip": "Stage1",
                "lift": "Stage1",
                "move_to_pre": "Stage3多层",
                "place": "Stage3多层" if self._multi_layer else "Stage3单层",
            }.get(name, "Stage2"),
            name,
        ), operation(self.confirm, self.before_vision):
            try:
                value = action()
                if rospy.is_shutdown():
                    raise TrayOperationError("ROS 已关闭，阶段不得标记完成")
            except BaseException:
                state.set_step_status(self.state_path, name, "FAILED")
                raise
            current = state.set_step_status(self.state_path, name, "SUCCEEDED")
            targets = {}
            for key in ("stage0_pregrasp_targets", "place_tray_target",
                        "preplace_tray_target", "preplace_tcp_targets"):
                if key in current:
                    targets[key] = current[key]
            measurements = value if isinstance(value, dict) else {}
            return StageResult(name, True, self.state_path, measurements, targets)

    def prep(self, *, speed: float | None = None,
             acceleration: float | None = None) -> StageResult:
        mv = motion_values("prep", {"speed": speed, "acceleration": acceleration})
        def action():
            from argparse import Namespace
            try:
                plan = stage0.plan_pregrasp(
                    Namespace(**self.config.base()), self.config.calibration_file()
                )
            except stage0.CalibrationMismatchError as exc:
                raise TrayOperationError(str(exc)) from exc
            state.invalidate_grasp_state(
                self.state_path,
                pickup_tray_pose=plan["tray_pose"],
                pickup_label=self.config.common["vision"]["label"],
                pickup_pose_source=self.config.common["vision"]["vision_backend"],
                stage0_pregrasp_targets=plan["targets"],
                stage0_complete=False,
                package_steps={},
                stage3_search_start_z_m=None,
                stage3_unconfirmed_release_reason=None,
                stage3_radius_limit_release=False,
                stage3_release_and_retreat_complete=False,
            )
            if not stage0.confirm_movel():
                raise PermissionError("用户取消预抓取 MoveL")
            timing_state("预抓取MoveL及到位等待")
            stage0.move_dual_to_pregrasp(plan["targets"], mv["speed"], mv["acceleration"])
            state.update_state(self.state_path, stage0_complete=True)
            return plan
        return self._perform("prep", action)

    def grip(self, *, free_speed: float | None = None,
             contact_speed: float | None = None,
             acceleration: float | None = None,
             gripper_speed_percent: float | None = None) -> StageResult:
        mv = motion_values("grip", {
            "free_speed": free_speed, "contact_speed": contact_speed,
            "acceleration": acceleration,
            "gripper_speed_percent": gripper_speed_percent,
        })
        def action():
            current = state.load_state(self.state_path)
            if not current.get("stage0_complete"):
                raise TrayOperationError("尚未真实到达预抓取位")
            pickup = state.require_pose(current, "pickup_tray_pose")
            base = dict(self.config.common["stage1"])
            base.update(max_free_speed_m_s=mv["free_speed"],
                        max_contact_speed_m_s=mv["contact_speed"],
                        max_acceleration_m_s2=mv["acceleration"],
                        gripper_speed_percent=mv["gripper_speed_percent"])
            cfg = stage1.make_stage1_config(**base)
            controller = stage1.TrayApproach(cfg, current["stage0_pregrasp_targets"])
            state.invalidate_grasp_state(self.state_path)
            with managed_controller(controller):
                timing_state("夹爪准备")
                controller.prepare_grippers()
                timing_state("等待反馈就绪")
                controller.wait_ready()
                timing_state("力基线采集")
                controller.calibrate_bias()
                timing_state("靠拢准备")
                controller.run_approach()
                tcp = stage1.read_tcp_pair(controller.robot)
                state.save_grasp_state(self.state_path, pickup, tcp,
                    state.compute_grasp_transforms(pickup, tcp["left"], tcp["right"]),
                    "stage0_vision_and_gripped_tcp")
            return {"pickup_tray_pose": pickup}
        return self._perform("grip", action)

    def lift(self, *, speed: float | None = None,
             acceleration: float | None = None,
             height_m: float | None = None, robot=None) -> StageResult:
        """按指定高度或配置层高同步抬升，不创建腕力订阅。"""
        self._require("grip")
        movement = motion_values("lift", {"speed": speed, "acceleration": acceleration})
        if height_m is None:
            layer = 1
            if self._multi_layer:
                current = state.load_state(self.state_path)
                layer_index = current.get("layer_index")
                if (not isinstance(layer_index, int) or isinstance(layer_index, bool)
                        or layer_index < 1 or current.get("top_tray_pose") is None):
                    raise TrayOperationError("多层抬升需要有效的顶层位姿和层号")
                layer = layer_index + 1
            height_m = self.config.lift_height_for_layer(layer)

        def action():
            from naviai_controller import NaviController, RobotModel
            active_robot = robot
            if active_robot is None:
                active_robot = NaviController(model=RobotModel.WA2)
                rospy.sleep(1.)
            state.update_state(self.state_path, stage2_lift_complete=False)
            stage2.input("确认tray已抓稳，Enter执行双臂同步抬升：")
            stage2.lift_tray(active_robot, height_m,
                             movement["speed"], movement["acceleration"])
            state.update_state(self.state_path, stage2_lift_complete=True,
                               stage2_lift_height=float(height_m))
            return {"phase": "lift", "lift_height_m": float(height_m)}

        return self._perform("lift", action)


class SingleLayerTray(_TrayBase):
    def place(self, *, robot=None, initial_lower_distance=0.0,
              record_top_runner=None) -> StageResult:
        """单层下降放置、张爪撤离并记录首层；主流程显式传入5cm预下降。"""
        def action():
            if initial_lower_distance:
                from naviai_controller import ArmGroup
                with phase("Stage3单层", "首层双臂同步下降5cm"):
                    if not self.confirm("lower_5cm", {"prompt": "首层转腰完成，双臂同步下降5cm，Enter执行："}):
                        raise PermissionError("用户取消首层下降")
                    if rospy.is_shutdown():
                        raise RuntimeError("ROS已关闭")
                    robot.stop()
                    robot.clear_servo_params()
                    targets = stage1.read_tcp_pair(robot)
                    for pose in targets.values():
                        pose[2] -= initial_lower_distance
                    if not robot.movel(targets["left"] + targets["right"], ArmGroup.DUAL,
                                       v=.01, acc=.02, is_async=False):
                        raise RuntimeError("首层双臂下降5cm失败")
            with managed_controller(
                SingleStage3Place(self.config.stage3_args(False))
            ) as controller:
                controller.run()
            timing_state("RECORD_TOP")
            if not self.confirm("record_top", {"prompt": "确认首层已放置且双臂撤离，Enter记录顶面："}):
                raise PermissionError("用户取消首层记录")
            record = lambda: vision.record_first_layer(self.config)
            pose = record_top_runner(record) if record_top_runner else record()
            print_pose6d("第一层实际放置tray", pose)
            return {"top_tray_pose": pose}

        return self._perform("place", action)


class MultiLayerTray(_TrayBase):
    _multi_layer = True

    def move_to_pre(self, *, robot=None, xy_offset=(0.0, 0.0),
                    held_tray_pose_from_tcp=None) -> StageResult:
        """获取持物状态、规划目标并同步移动到P_pre；默认沿用配置定位方式。"""
        def action():
            from argparse import Namespace
            from naviai_controller import NaviController, RobotModel

            current = state.load_state(self.state_path)
            state.require_pose(current, "top_tray_pose")
            layer = current.get("layer_index")
            if not isinstance(layer, int) or isinstance(layer, bool) or layer < 1:
                raise ValueError("缺少有效上一层记录")
            if current.get("stage3_release_and_retreat_complete") and current.get("target_layer_index", 0) > layer:
                raise RuntimeError("上一层顶面尚未记录，请先补记")
            args = self.config.stage2_args("preplace", multi_layer=True, target_layer_index=layer + 1)
            active_robot = robot
            if active_robot is None:
                active_robot = NaviController(model=RobotModel.WA2)
                rospy.sleep(1.0)
            use_tcp = args.held_tray_pose_from_tcp if held_tray_pose_from_tcp is None else held_tray_pose_from_tcp
            held = stage2.capture_held_tray(active_robot, args, Namespace(**self.config.base()),
                                           current, use_tcp=use_tcp)
            state.save_grasp_state(self.state_path, held.pose, held.tcp, held.grasp, held.source,
                stage2_complete=False, stage3_complete=False, stage3_release_and_retreat_complete=False)
            timing_state("P_pre规划")
            plan = stage2.plan_preplacement(args, held.tcp, held_estimate=held.estimate)
            stage2.move_to_pre(active_robot, args, plan, xy_offset)
            return {"held_tray_pose": plan.held_pose}

        return self._perform("move_to_pre", action)

    def place(self, *, free_speed: float | None = None,
              contact_speed: float | None = None,
              acceleration: float | None = None,
              xy_speed: float | None = None,
              search_angular_speed: float | None = None,
              search_max_radius: float | None = None,
              gripper_speed_percent: float | None = None,
              record_top_runner: Callable | None = None,
              search_path: str = "spiral", ring_step: float | None = None) -> StageResult:
        self._require("move_to_pre")
        movement = dict(free_speed=free_speed, contact_speed=contact_speed,
                        acceleration=acceleration, xy_speed=xy_speed,
                        search_angular_speed=search_angular_speed,
                        gripper_speed_percent=gripper_speed_percent)
        args = self.config.stage3_args(True, movement, search_max_radius,
                                      search_path=search_path, ring_step=ring_step)
        current = state.load_state(self.state_path)
        layer = current.get("target_layer_index", current.get("layer_index", 1) + 1)
        args.xy_force_capture_threshold = self.config.multi["search"]["xy_force_capture_threshold"][layer - 2]
        def action():
            controller = stage3_multi.MultilayerStage3(args)
            with managed_controller(controller, force_log=True):
                controller.run_experiment()
                if not controller.released:
                    raise TrayOperationError("Stage3 在完成张爪前停止")
                if controller.force_recorder is not None:
                    controller.force_recorder.set_state("RECORD_TOP")
                def record():
                    try:
                        return stage3_multi.record_top_pose(
                            args, expected_pose=controller.place_pose,
                        )
                    except (RuntimeError, TimeoutError) as exc:
                        raise TrayOperationError(
                            "多层张爪及撤离已完成，但顶层视觉记录失败；确认相机视野后运行 "
                            "python3 main.py --execute --mode multi --record-top-after-place 补记。原始错误：{}".format(exc)
                        ) from exc
                pose = record_top_runner(record) if record_top_runner else record()
                if controller.force_recorder is not None:
                    controller.force_recorder.set_state("COMPLETED")
                state.update_state(self.state_path, stage3_force_csv_path=controller.force_log_path,
                                   stage3_placed_minus_p_place_z_mm=1000 * (pose[2] - controller.place_pose[2]))
            return {"top_tray_pose": pose, "search_max_radius": args.search_max_radius,
                    "place_tray_target": controller.place_pose,
                    "placed_minus_p_place_z_mm": 1000 * (pose[2] - controller.place_pose[2]),
                    "force_csv_path": controller.force_log_path}
        return self._perform("place", action)
