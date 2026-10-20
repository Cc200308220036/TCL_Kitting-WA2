#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段0：抓取标定、托盘对称姿态对齐及双臂预抓取运动。"""

from __future__ import annotations

from .common.runtime import input
from .common import vision
from .observability.timing import state as timing_state

import json
import math
import os
import time
from typing import Dict, List

import numpy as np
import rospy
from scipy.spatial.transform import Rotation

from naviai_controller import ArmGroup, NaviController, RobotModel

from .common.state import update_state, validate_pose7


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


class CalibrationMismatchError(RuntimeError):
    """视觉入口与抓取标定的后端或标签不一致。"""


def plan_pregrasp(args, calibration_path):
    """视觉定位、读取标定、对称姿态对齐并计算双TCP预抓取目标。"""
    timing_state("视觉识别")
    measured = validate_xyzquat(vision.estimate_tray_pose(args, "T_base_tray"))
    timing_state("标定加载/目标换算")
    calibration = load_calibration(calibration_path)
    if (
        calibration.get("vision_backend") != args.vision_backend
        or calibration.get("vision_label") != args.label
    ):
        raise CalibrationMismatchError("视觉后端或标签与 Stage0 标定不一致")
    aligned, raw_delta, aligned_delta, flipped = align_tray_half_turn_symmetry(
        measured, calibration["reference_tray_xyzquat"]
    )
    return dict(
        tray_pose=aligned,
        targets=compute_pregrasp_xyzquats(aligned, calibration),
        raw_yaw_delta=raw_delta,
        aligned_yaw_delta=aligned_delta,
        symmetry_flipped=flipped,
    )


def validate_xyzquat(values) -> List[float]:
    # Stage0保留输入四元数数值，由后续Rotation运算处理归一化。
    try:
        return validate_pose7(values, "tray视觉位姿", normalize=False)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def load_calibration(path: str) -> Dict[str, object]:
    """读取左右夹爪分别示教生成的Stage 0标定文件。"""
    if not os.path.isfile(path):
        raise RuntimeError(
            "Stage 0标定文件不存在: {}\n"
            "请在V2目录运行 python3 tray_control/stage0_calibrate.py --calibrate".format(path)
        )
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise RuntimeError("Stage 0标定文件根节点必须是字典")
    validate_xyzquat(data.get("reference_tray_xyzquat"))
    tcp_data = data.get("reference_tcp_xyzquats")
    if not isinstance(tcp_data, dict):
        raise RuntimeError("标定文件缺少 reference_tcp_xyzquats")
    for side in ("left", "right"):
        validate_xyzquat(tcp_data.get(side))
    return data


def save_calibration(
    path: str,
    tray_xyzquat,
    tcp_xyzquats: Dict[str, List[float]],
    vision_backend: str = "unknown",
    label: str = "",
) -> None:
    """原子更新标定文件，避免留下不完整JSON。"""
    payload = {
        "version": 1,
        "assumption": "tray_planar_yaw_only",
        "vision_backend": str(vision_backend),
        "vision_label": str(label),
        "reference_tray_xyzquat": validate_xyzquat(tray_xyzquat),
        "reference_tcp_xyzquats": {
            side: validate_xyzquat(tcp_xyzquats[side])
            for side in ("left", "right")
        },
        "note": (
            "实际夹爪末端中心已由人工对准tray两侧中心时，"
            "读取的tray和左右电机TCP位姿"
        ),
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary_path = path + ".tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def align_tray_half_turn_symmetry(
    tray_xyzquat, reference_xyzquat
) -> tuple[List[float], float, float, bool]:
    """把180度对称tray的视觉姿态对齐到标定姿态的最近等价分支。"""
    tray_pose = np.asarray(validate_xyzquat(tray_xyzquat), dtype=np.float64)
    reference_pose = np.asarray(
        validate_xyzquat(reference_xyzquat), dtype=np.float64
    )
    tray_rotation = Rotation.from_quat(tray_pose[3:])
    reference_yaw = float(
        Rotation.from_quat(reference_pose[3:]).as_euler("xyz")[2]
    )
    raw_yaw = float(tray_rotation.as_euler("xyz")[2])
    raw_delta = math.atan2(
        math.sin(raw_yaw - reference_yaw),
        math.cos(raw_yaw - reference_yaw),
    )

    aligned_pose = tray_pose.copy()
    flipped = abs(raw_delta) > 0.5 * math.pi
    if flipped:
        # 在tray自身坐标系绕Z轴转180度，保持位置和视觉测得的法向倾斜。
        aligned_rotation = tray_rotation * Rotation.from_euler("z", math.pi)
        aligned_pose[3:] = aligned_rotation.as_quat()

    aligned_yaw = float(
        Rotation.from_quat(aligned_pose[3:]).as_euler("xyz")[2]
    )
    aligned_delta = math.atan2(
        math.sin(aligned_yaw - reference_yaw),
        math.cos(aligned_yaw - reference_yaw),
    )
    return (
        aligned_pose.astype(float).tolist(),
        raw_delta,
        aligned_delta,
        flipped,
    )


def compute_pregrasp_xyzquats(
    tray_xyzquat,
    calibration: Dict[str, object],
) -> Dict[str, List[float]]:
    """用同步标定关系计算左右TCP，仅跟随tray平移和平面yaw。"""
    tray_pose = np.asarray(validate_xyzquat(tray_xyzquat), dtype=np.float64)
    reference_tray_pose = np.asarray(
        validate_xyzquat(calibration["reference_tray_xyzquat"]),
        dtype=np.float64,
    )
    reference_tcp_xyzquats = calibration["reference_tcp_xyzquats"]
    if not isinstance(reference_tcp_xyzquats, dict):
        raise RuntimeError("标定数据中的TCP位姿格式错误")
    tray_yaw = float(
        Rotation.from_quat(tray_pose[3:]).as_euler("xyz", degrees=False)[2]
    )
    reference_tray_yaw = float(
        Rotation.from_quat(reference_tray_pose[3:]).as_euler(
            "xyz", degrees=False
        )[2]
    )
    yaw_rotation = Rotation.from_euler("z", tray_yaw, degrees=False)
    reference_yaw_rotation = Rotation.from_euler(
        "z", reference_tray_yaw, degrees=False
    )
    yaw_delta = math.atan2(
        math.sin(tray_yaw - reference_tray_yaw),
        math.cos(tray_yaw - reference_tray_yaw),
    )
    yaw_delta_rotation = Rotation.from_euler("z", yaw_delta, degrees=False)

    # 左右分别保留tray -> 电机TCP关系，间接包含两侧各自的
    # TCP -> 夹爪工作点安装偏移；不对电机TCP的X/Z求平均。
    reference_tcp_poses = {
        side: np.asarray(
            validate_xyzquat(reference_tcp_xyzquats[side]), dtype=np.float64
        )
        for side in ("left", "right")
    }
    calibrated_offsets_tray = {
        side: reference_yaw_rotation.inv().apply(
            reference_tcp_poses[side][:3] - reference_tray_pose[:3]
        )
        for side in ("left", "right")
    }

    targets: Dict[str, List[float]] = {}
    for side in ("left", "right"):
        reference_tcp_pose = reference_tcp_poses[side]
        position_base = tray_pose[:3] + yaw_rotation.apply(
            calibrated_offsets_tray[side]
        )

        # 完整保留手工标定的TCP姿态，只叠加tray的yaw变化。
        reference_tcp_rotation = Rotation.from_quat(
            reference_tcp_pose[3:]
        )
        tcp_rotation_base = yaw_delta_rotation * reference_tcp_rotation
        quaternion = tcp_rotation_base.as_quat()
        targets[side] = [
            *position_base.astype(float).tolist(),
            *quaternion.astype(float).tolist(),
        ]

    separation = float(np.linalg.norm(
        np.asarray(targets["left"][:3])
        - np.asarray(targets["right"][:3])
    ))
    if not math.isfinite(separation) or separation < 0.10:
        raise RuntimeError("左右TCP标定目标异常重合")
    return targets


def move_dual_to_pregrasp(
    targets: Dict[str, List[float]],
    velocity: float,
    acceleration: float,
) -> None:
    """同步控制左右TCP到达预抓取位姿。"""
    if not math.isfinite(velocity) or not 0.0 < velocity <= 0.05:
        raise ValueError("velocity 必须在 (0, 0.05] m/s")
    if not math.isfinite(acceleration) or not 0.0 < acceleration <= 0.10:
        raise ValueError("acceleration 必须在 (0, 0.10] m/s^2")

    left = validate_xyzquat(targets["left"])
    right = validate_xyzquat(targets["right"])
    dual_target = left + right

    robot = NaviController(auto_spin=False, model=RobotModel.WA2)
    rospy.sleep(1.0)
    robot.stop()
    robot.clear_servo_params()
    try:
        success = robot.movel(
            dual_target,
            ArmGroup.DUAL,
            v=velocity,
            acc=acceleration,
            is_async=False,
        )
        if not success:
            raise RuntimeError("双臂预抓取MoveL服务返回失败")
    except Exception:
        robot.stop()
        raise

    rospy.loginfo(
        "STAGE0_PREGRASP_COMPLETE: 双臂已到达视觉预抓取目标"
    )


def confirm_movel() -> bool:
    """按 Enter 执行；取消由调用方的确认回调处理。"""
    try:
        answer = input(
            "确认左右目标、双臂直线路径无障碍且急停可用后按 Enter 执行；"
            "输入其他内容或按 Ctrl+C 取消："
        )
    except KeyboardInterrupt:
        print("\n已取消双臂 MoveL")
        return False
    return answer.strip() == ""
