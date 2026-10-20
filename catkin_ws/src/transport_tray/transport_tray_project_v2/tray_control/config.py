"""v2精简配置：仅四个测试所用运动、视觉、导纳及寻槽参数。"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Any, Dict
import yaml
from . import stage1
from .common import vision
from .stage3_common import validate_stage3_config, MULTI_STAGE3_PARAMETERS
from .common.state import require_positive

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "transport_tray.yaml"
MOTION_DEFAULTS = {
    "prep": {"speed": 0.03, "acceleration": 0.05},
    "grip": {
        "free_speed": 0.006,
        "contact_speed": 0.001,
        "acceleration": 0.01,
        "gripper_speed_percent": 0.0,
    },
    "lift": {"speed": 0.01, "acceleration": 0.02},
    "move_to_pre": {"speed": 0.01, "acceleration": 0.02},
    "place": {
        "free_speed": 0.003,
        "contact_speed": 0.0005,
        "acceleration": 0.01,
        "gripper_speed_percent": 0.0,
    },
    "place_multi": {"xy_speed": 0.002, "search_angular_speed": 0.39},
}
MOTION_LIMITS = {
    "prep": {"speed": 0.05, "acceleration": 0.1},
    "grip": {
        "free_speed": 0.01,
        "contact_speed": 0.002,
        "acceleration": None,
        "gripper_speed_percent": 100.0,
    },
    "lift": {"speed": 0.05, "acceleration": 0.1},
    "move_to_pre": {"speed": 0.05, "acceleration": 0.1},
    "place": {
        "free_speed": None,
        "contact_speed": None,
        "acceleration": None,
        "gripper_speed_percent": 100.0,
    },
    "place_multi": {"xy_speed": None, "search_angular_speed": None},
}
CONFIG_KEYS = {
    "common": {
        "vision": [
            "vision_backend",
            "label",
            "yolo_label",
            "tray_pose_camera_to_head",
            "tray_pose_rgbd_timeout",
        ],
        "paths": ["state_file", "force_log_dir", "calibration_file"],
        "stage0": ["symmetry_enabled"],
        "stage1": [
            "hand_settle_seconds",
            "target_force_n",
            "contact_threshold_n",
            "contact_debounce_s",
            "contact_release_debounce_s",
            "force_band_n",
            "imbalance_band_n",
            "virtual_mass_kg",
            "damping_ns_m",
            "max_retreat_distance_m",
            "target_ramp_seconds",
            "stable_seconds",
            "bias_seconds",
            "filter_cutoff_hz",
            "stale_timeout_s",
            "tracking_error_limit_m",
            "left_gripper_service",
            "right_gripper_service",
            "gripper_profile",
            "gripper_initial_position_mm",
            "gripper_close_position_mm",
            "gripper_force_percent",
            "gripper_wait_timeout_s",
            "max_inward_distance_m",
            "duration_s",
        ],
        "stage2": [
            "orientation_tolerance_deg",
            "lift_heights_by_layer",
            "tcp_tolerance",
            "relative_position_limit",
        ],
        "stage3": [
            "descent_motion",
            "target_force",
            "left_z_force_sign",
            "right_z_force_sign",
            "contact_threshold",
            "control_margin",
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
            "mass",
            "damping",
            "filter_cutoff",
            "baseline_seconds",
            "stale_timeout",
            "tracking_error_limit",
            "hand_open_settle_seconds",
            "left_gripper_service",
            "right_gripper_service",
            "gripper_profile",
            "gripper_open_position_mm",
            "gripper_force_percent",
            "gripper_wait_timeout",
        ],
    },
    "multi_layer": {
        "preplace": [
            "held_tray_pose_from_tcp",
            "stack_pitch",
            "preplace_clearance",
            "max_preplace_translation",
        ],
        "search": [
            "search_paths",
            "search_z_acceleration",
            "search_z_down_speed",
            "search_z_down_acceleration",
            "radius_z_acceleration",
            "search_z_speed_limit",
            "search_z_target_force",
            "search_z_recovery_hysteresis",
            "search_z_retreat_distance",
            "search_z_retreat_trigger_cycles",
            "xy_force_capture_threshold",
            "direct_seating_distance",
            "direct_seating_max_z_difference",
            "radius_return_xy_speed",
            "radius_z_down_speed",
            "radius_z_retreat_speed",
            "radius_yaw_speed_deg_s",
            "radius_yaw_range_deg",
            "radius_yaw_timeout",
            "radius_yaw_extra_descent",
            "vision_settle",
            "search_timeout",
            "tray_fusion_position_limit",
            "tray_fusion_yaw_limit_deg",
            "seated_stable_seconds",
            "slot_min_descent",
            "yaw_descent_debounce",
        ],
    },
}


def _require_keys(values, keys, name):
    if not isinstance(values, dict) or set(values) != keys:
        raise ValueError(name + "字段不完整或含未知字段")


def _positive(value, name):
    require_positive(value, name, strict=True)


def motion_values(name: str, overrides: Dict[str, Any]) -> Dict[str, float]:
    base = MOTION_DEFAULTS[name]
    unknown = set(overrides) - set(base)
    if unknown:
        raise ValueError("未知运动参数 {}: {}".format(name, sorted(unknown)))
    result = dict(base)
    for key, value in overrides.items():
        if value is not None:
            result[key] = float(value)
    for key, value in result.items():
        if (
            not math.isfinite(value)
            or value < 0
            or (value == 0 and key != "gripper_speed_percent")
        ):
            raise ValueError("{} {} 必须是有限正数".format(name, key))
        cap = MOTION_LIMITS[name][key]
        if cap is not None and value > cap:
            raise ValueError("{} {} 超过当前代码上限 {}".format(name, key, cap))
    if name == "grip" and result["contact_speed"] > result["free_speed"]:
        raise ValueError("接触后速度不能大于自由速度")
    return result


class TrayConfig:
    def __init__(self, path=DEFAULT_CONFIG, state_path=None):
        self.path = Path(path).expanduser().resolve()
        with self.path.open(encoding="utf-8") as stream:
            self.data = yaml.safe_load(stream)
        if not isinstance(self.data, dict) or set(self.data) != set(CONFIG_KEYS):
            raise ValueError("v2配置需要且仅包含common/multi_layer")
        for section, groups in CONFIG_KEYS.items():
            if set(self.data[section]) != set(groups):
                raise ValueError(section + "区域字段不完整或含未知字段")
            for name, fields in groups.items():
                if set(self.data[section][name]) != set(fields):
                    raise ValueError(section + "." + name + "字段不完整或含未知字段")
        self.common = self.data["common"]
        self.multi = self.data["multi_layer"]
        paths = self.common["paths"]
        self.state_path = (
            str(Path(state_path).expanduser().resolve())
            if state_path
            else self._path(paths["state_file"])
        )
        self.force_log_dir = self._path(paths["force_log_dir"])
        self._validate()

    @staticmethod
    def _path(value):
        p = Path(value).expanduser()
        return str((p if p.is_absolute() else ROOT / p).resolve())

    def vision_args(self):
        result = dict(self.common["vision"])
        result["tray_pose_camera_to_head"] = self._path(
            result["tray_pose_camera_to_head"]
        )
        vision.validate_vision_arguments(argparse.Namespace(**result))
        return result

    def calibration_file(self):
        return self._path(self.common["paths"]["calibration_file"])

    def base(self):
        return dict(
            **self.vision_args(),
            state_file=self.state_path,
            force_log_dir=self.force_log_dir,
        )

    def lift_height_for_layer(self, layer_index):
        heights = self.common["stage2"]["lift_heights_by_layer"]
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 1 <= layer_index <= len(heights)
        ):
            raise ValueError("目标层号必须在1到{}之间".format(len(heights)))
        return float(heights[layer_index - 1])

    def stage2_args(
        self,
        phase,
        speed=None,
        acceleration=None,
        *,
        multi_layer=False,
        target_layer_index=None,
    ):
        name = {"lift": "lift", "preplace": "move_to_pre"}[phase]
        movement = motion_values(name, {"speed": speed, "acceleration": acceleration})
        values = {
            **self.base(),
            **self.common["stage2"],
            **self.multi["preplace"],
            "phase": phase,
            "execute": True,
            "multi_layer": multi_layer,
        }
        if target_layer_index is None:
            if multi_layer:
                raise ValueError("多层Stage2需要目标层号")
            target_layer_index = 1
        values.update(
            target_layer_index=target_layer_index,
            lift_height=self.lift_height_for_layer(target_layer_index),
            velocity=MOTION_DEFAULTS["lift"]["speed"],
            acceleration=MOTION_DEFAULTS["lift"]["acceleration"],
            preplace_velocity=MOTION_DEFAULTS["move_to_pre"]["speed"],
            preplace_acceleration=MOTION_DEFAULTS["move_to_pre"]["acceleration"],
        )
        if phase == "lift":
            values.update(
                velocity=movement["speed"], acceleration=movement["acceleration"]
            )
        else:
            values.update(preplace_velocity=movement["speed"],
                          preplace_acceleration=movement["acceleration"])
        return argparse.Namespace(**values)

    def _stage3_motion_values(self, multi, overrides):
        """读取下降配置并校验，再应用接口传入的运动覆盖值。"""
        common = dict(self.common["stage3"])
        profiles = common.pop("descent_motion")
        _require_keys(profiles, {"single", "multi"}, "descent_motion")
        for name, profile in profiles.items():
            keys = {"speed", "acceleration"} | (
                {"contact_speed"} if name == "multi" else set()
            )
            _require_keys(profile, keys, "descent_motion." + name)
            for key, value in profile.items():
                _positive(value, "descent_motion." + name + "." + key)
        profile = profiles["multi" if multi else "single"]
        motion = dict(
            free_speed=profile["speed"],
            contact_speed=profile.get("contact_speed", 0.001),
            acceleration=profile["acceleration"],
        )
        motion.update(
            {
                k: v
                for k, v in overrides.items()
                if k in MOTION_DEFAULTS["place"] and v is not None
            }
        )
        mv = motion_values("place", motion)
        return common, mv

    def _stage3_search_args(self, values, overrides, radius, search_path, ring_step):
        """加入多层搜索参数，校验力阈值并配置所选路线。"""
        from .algorithms.search_path import configure_search_path

        extra = motion_values(
            "place_multi",
            {
                k: v
                for k, v in overrides.items()
                if k in MOTION_DEFAULTS["place_multi"]
            },
        )
        values.update(self.multi["search"])
        thresholds = values["xy_force_capture_threshold"]
        if not isinstance(thresholds, list) or len(thresholds) != 2:
            raise ValueError(
                "xy_force_capture_threshold需提供第二层、第三层两个阈值"
            )
        for value in thresholds:
            _positive(value, "xy_force_capture_threshold")
        values.update(
            xy_force_capture_threshold=thresholds[0],
            search_xy_speed=extra["xy_speed"],
            search_angular_speed=extra["search_angular_speed"],
            search_path=search_path,
            layer_index=None,
        )
        args = argparse.Namespace(**values)
        configure_search_path(args, radius=radius, ring_step=ring_step)
        return args

    def stage3_args(
        self,
        multi,
        movement=None,
        radius=None,
        *,
        search_path="spiral",
        ring_step=None,
    ):
        """组装下降与寻槽参数并校验；接口值覆盖配置，不做倍率换算。"""
        overrides = movement or {}
        common, mv = self._stage3_motion_values(multi, overrides)
        values = {
            **self.base(),
            **common,
            "execute": True,
            "multi_layer": multi,
            "max_free_speed": mv["free_speed"],
            "max_contact_speed": mv["contact_speed"],
            "max_acceleration": mv["acceleration"],
            "gripper_speed_percent": mv["gripper_speed_percent"],
        }
        if multi:
            args = self._stage3_search_args(
                values, overrides, radius, search_path, ring_step
            )
        else:
            for key in MULTI_STAGE3_PARAMETERS:
                values.pop(key)
            args = argparse.Namespace(**values)
        validate_stage3_config(args)
        if multi:
            self.validate_multi_stage3(args)
        return args

    @staticmethod
    def validate_multi_stage3(args: argparse.Namespace) -> None:
        if not (
            math.isfinite(args.search_z_target_force)
            and 0 < args.search_z_target_force <= args.target_force
        ):
            raise ValueError("XY目标力须为有限正数且不超过Stage3目标力")
        if not (
            math.isfinite(args.search_z_recovery_hysteresis)
            and 0 <= args.search_z_recovery_hysteresis < args.search_z_target_force
        ):
            raise ValueError("XY恢复迟滞须非负且小于XY目标力")
        if (
            isinstance(args.search_z_retreat_trigger_cycles, bool)
            or not isinstance(args.search_z_retreat_trigger_cycles, int)
            or args.search_z_retreat_trigger_cycles < 1
        ):
            raise ValueError("XY退让触发周期数必须为正整数")
        for key in POSITIVE_MULTI_SEARCH_PARAMETERS:
            _positive(getattr(args, key), key)

    def _validate(self):
        self.vision_args()
        if not isinstance(self.multi["preplace"]["held_tray_pose_from_tcp"], bool):
            raise ValueError("held_tray_pose_from_tcp必须为布尔值")
        if self.common["stage0"]["symmetry_enabled"] is not True:
            raise ValueError("Stage0固定启用180度分支对齐")
        config = dict(self.common["stage1"])
        stage1.make_stage1_config(**config)
        heights = self.common["stage2"]["lift_heights_by_layer"]
        if (
            not isinstance(heights, list)
            or len(heights) != 3
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or not 0 < v <= 0.15
                for v in heights
            )
        ):
            raise ValueError("抬升数组需提供三个(0,0.15]m高度")
        args = self.stage2_args("lift")
        for key in (
            "lift_height",
            "tcp_tolerance",
            "orientation_tolerance_deg",
            "max_preplace_translation",
            "relative_position_limit",
            "stack_pitch",
            "preplace_clearance",
        ):
            v = getattr(args, key)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(key + "必须为有限正数")
        if args.tcp_tolerance > 0.01:
            raise ValueError("Stage2到位容差超出封装上限")
        self.stage3_args(False)
        self.stage3_args(True)


POSITIVE_MULTI_SEARCH_PARAMETERS = (
    "search_z_retreat_distance",
    "search_z_speed_limit",
    "search_z_acceleration",
    "search_z_down_speed",
    "search_z_down_acceleration",
    "radius_z_acceleration",
    "search_max_radius",
    "radius_return_xy_speed",
    "radius_z_down_speed",
    "radius_z_retreat_speed",
    "radius_yaw_speed_deg_s",
    "radius_yaw_range_deg",
    "radius_yaw_timeout",
    "radius_yaw_extra_descent",
    "xy_force_capture_threshold",
    "direct_seating_distance",
    "direct_seating_max_z_difference",
    "search_pitch",
    "search_timeout",
    "yaw_descent_debounce",
    "slot_min_descent",
    "seated_stable_seconds",
    "tray_fusion_position_limit",
    "tray_fusion_yaw_limit_deg",
    "vision_settle",
)
