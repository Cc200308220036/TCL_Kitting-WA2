"""Stage3多层CSV组装；不参与运动与阶段判断。"""

from __future__ import annotations
import math
from typing import Dict, Optional
import numpy as np
import rospy
from naviai_controller import ArmGroup
from ..algorithms.moving_z_admittance import MovingZParameters
from .force_recording import ForceCsvRecorder, WRENCH_COLUMNS, activate, state as trace_state


# CSV列名沿用历史名称，内部属性使用新的XY捕获/Yaw下沉命名。
def start_multilayer_log(place) -> None:
    """真实运动时保存每个控制周期的腕力，供离线分析入槽判据。"""
    model = MovingZParameters()
    place.force_log_metadata = {
        'target_force_n': place.args.target_force,
        'moving_z_search_enabled': True,
        'slot_capture_mode': 'descent_only',
        'slot_raw_fx_enabled': True,
        'slot_raw_fx_threshold_n': place.args.xy_force_capture_threshold,
        'slot_xy_fy_threshold_n': place.args.xy_force_capture_threshold,
        'slot_yaw_fy_threshold_n': 2.5,
        'search_path': place.args.search_path,
        'search_z_mode': 'common',
        'search_z_force_reference': 'held_baseline',
        'search_z_target_force_n': place.args.search_z_target_force,
        'search_z_recovery_hysteresis_n': place.args.search_z_recovery_hysteresis,
        'search_z_retreat_distance_mm': 1000 * place.args.search_z_retreat_distance,
        'search_z_retreat_trigger_cycles': place.args.search_z_retreat_trigger_cycles,
        'search_pause_xy_during_retreat': True,
        'search_z_force_deadband_n': model.force_deadband,
        'search_z_equilibrium_gain': model.equilibrium_gain,
        'slot_min_descent_mm': 1000.0 * place.args.slot_min_descent,
        'slot_force_drop_debounce_s': place.args.yaw_descent_debounce,
        'seated_stable_seconds': place.args.seated_stable_seconds,
        'seating_timeout_s': '',
    }
    columns = [
        "elapsed_s",
        "ros_time_s",
        "stage",
        "pose_source",
        "target_force_n",
        "moving_z_search_enabled",
        "slot_capture_mode",
        "slot_min_descent_mm",
        "slot_raw_fx_enabled",
        "slot_raw_fx_threshold_n",
        "slot_raw_fx_captured",
        "slot_xy_fy_threshold_n",
        "slot_yaw_fy_threshold_n",
        "slot_yaw_fy_captured",
        "search_path",
        "search_z_mode",
        "search_z_force_reference",
        "search_z_target_force_n",
        "search_z_recovery_hysteresis_n",
        "search_z_retreat_distance_mm",
        "search_z_retreat_trigger_cycles",
        "search_z_overload_cycles",
        "search_z_retreat_armed",
        "search_pause_xy_during_retreat",
        "search_xy_paused",
        "search_xy_pause_seconds",
        "search_z_force_deadband_n",
        "search_z_equilibrium_gain",
        "search_z_retreat_active",
        "slot_force_drop_debounce_s",
        "seated_stable_seconds",
        "seating_timeout_s",
        "search_descent_mm",
        "search_start_left_n",
        "search_start_right_n",
    ]
    for side in ("left", "right"):
        columns += [
            "{}_search_fx_baseline_n".format(side),
            "{}_search_relative_fx_n".format(side),
            "{}_search_fy_baseline_n".format(side),
            "{}_search_relative_fy_n".format(side),
        ]
        columns += [
            "{}_search_relative_support_n".format(side),
            "{}_search_z_velocity_mm_s".format(side),
            "{}_search_z_displacement_mm".format(side),
            "{}_search_equilibrium_mm".format(side),
        ]
    columns += [
        "radius_yaw_command_deg",
        "radius_z_velocity_mm_s",
        "tray_error_x_mm",
        "tray_error_y_mm",
        "tray_error_z_mm",
        "xy_cmd_x_mm",
        "xy_cmd_y_mm",
        "left_down_mm",
        "right_down_mm",
        "left_line_force_n",
        "right_line_force_n",
    ]
    place.force_recorder = ForceCsvRecorder(
        place.args.force_log_dir,
        {
            "left": place.baseline[ArmGroup.LEFT],
            "right": place.baseline[ArmGroup.RIGHT],
        },
        prefix="stage3_multi",
        extra_columns=columns,
    )
    place.force_log_path = place.force_recorder.path
    activate(place.force_recorder)
    trace_state(place.stage_name)
    rospy.loginfo("Stage3腕力逐周期CSV：%s", place.force_log_path)


def write_multilayer_sample(
    place,
    elapsed: float,
    support: Dict[ArmGroup, float],
    raw_wrenches: Dict[ArmGroup, np.ndarray],
    search_start_support: Dict[ArmGroup, float],
    error: Dict[str, float],
    pose_source: str,
    search_descent: Optional[float] = None,
    *,
    observation,
) -> None:
    row = {
        **place.force_log_metadata,
        "elapsed_s": elapsed,
        "ros_time_s": rospy.get_time(),
        "stage": place.stage_name,
        "pose_source": pose_source,
        "slot_raw_fx_captured": place.xy_force_captured,
        "slot_yaw_fy_captured": place.slot_yaw_fy_captured,
        "search_xy_paused": place.search_xy_paused,
        "search_xy_pause_seconds": place.search_xy_pause_seconds,
        "search_z_overload_cycles": (
            place.moving_z_axis.overload_cycles
            if place.stage_name == "XY_SEARCH" and place.moving_z_axis is not None
            else ""
        ),
        "search_z_retreat_armed": (
            place.moving_z_axis.retreat_armed
            if place.stage_name == "XY_SEARCH" and place.moving_z_axis is not None
            else ""
        ),
        "search_z_retreat_active": (
            place.moving_z_axis.retreat_active
            if place.stage_name == "XY_SEARCH" and place.moving_z_axis is not None
            else ""
        ),
        "search_descent_mm": 1000.0 * search_descent if search_descent is not None else "",
        "radius_yaw_command_deg": (
            math.degrees(place.yaw_search.offset) if place.stage_name == "YAW_SEARCH_LOWEST" else ""
        ),
        "radius_z_velocity_mm_s": (
            1000.0 * place.return_motion.z_axis.velocity
            if place.return_motion.z_axis is not None
            and place.stage_name in ("RETURN_SEARCH_LOWEST", "YAW_SEARCH_LOWEST")
            else ""
        ),
        "search_start_left_n": search_start_support.get(ArmGroup.LEFT, ""),
        "search_start_right_n": search_start_support.get(ArmGroup.RIGHT, ""),
        "tray_error_x_mm": 1000.0 * error["x"],
        "tray_error_y_mm": 1000.0 * error["y"],
        "tray_error_z_mm": 1000.0 * error["z"],
        "xy_cmd_x_mm": 1000.0 * place.search_offset_base[0],
        "xy_cmd_y_mm": 1000.0 * place.search_offset_base[1],
        "left_down_mm": 1000.0 * place.z_model.offsets[ArmGroup.LEFT],
        "right_down_mm": 1000.0 * place.z_model.offsets[ArmGroup.RIGHT],
    }
    for arm, side in ((ArmGroup.LEFT, "left"), (ArmGroup.RIGHT, "right")):
        raw = raw_wrenches[arm]
        for index, (_, raw_key, _) in enumerate(WRENCH_COLUMNS[side]):
            row[raw_key] = raw[index]
        row["{}_support_fz_n".format(side)] = support[arm]
        if arm in place.search_start_raw_fx:
            row["{}_search_fx_baseline_n".format(side)] = place.search_start_raw_fx[arm]
            row["{}_search_relative_fx_n".format(side)] = (
                observation.xy_fx[arm]
            )
        if arm in place.search_start_raw_fy:
            row["{}_search_fy_baseline_n".format(side)] = place.search_start_raw_fy[arm]
            row["{}_search_relative_fy_n".format(side)] = (
                observation.xy_fy[arm]
            )
        if place.stage_name == "XY_SEARCH" and place.moving_z_axis is not None:
            axis = place.moving_z_axis
            row["{}_search_relative_support_n".format(side)] = support[arm]
            row["{}_search_z_velocity_mm_s".format(side)] = -1000.0 * axis.velocity
            row["{}_search_z_displacement_mm".format(side)] = (
                -1000.0 * axis.displacement
            )
            row["{}_search_equilibrium_mm".format(side)] = -1000.0 * axis.equilibrium
        row["{}_line_force_n".format(side)] = (
            float(np.dot(raw[:3], place.line_outward_axes[arm]))
            if place.line_outward_axes is not None
            else ""
        )
    place.force_recorder.writerow(row)
