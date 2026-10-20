#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多层 tray 实验脚本共用的位姿变换和 JSON 状态文件。

本模块只保存实验数据，不初始化 ROS，也不发送机器人指令。
位姿统一采用 [x, y, z, qx, qy, qz, qw]，变换记号为 T_A_B。
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_STATE_FILE = os.path.join(SCRIPT_DIR, "runtime", "tray_state.json")


def validate_pose7(
    values: Iterable[float],
    name: str = "pose",
    *,
    as_array: bool = False,
    normalize: bool = True,
) -> List[float] | np.ndarray:
    """统一校验七维位姿；默认返回归一化列表，控制循环可直接取数组。"""
    try:
        pose = np.asarray(list(values), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError("{}必须是7个数值".format(name))
    if pose.size != 7 or not np.all(np.isfinite(pose)):
        raise ValueError("{}必须是7个有限数值".format(name))
    norm = float(np.linalg.norm(pose[3:]))
    if norm < 1e-9:
        raise ValueError("{}的四元数不能为零".format(name))
    if normalize:
        pose[3:] /= norm
    return pose if as_array else pose.tolist()


def require_positive(value, name, *, strict=False):
    """严格配置拒绝布尔/非数值；运行参数沿用float转换后的有限正数检查。"""
    if strict and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ValueError(name + "必须为有限正数")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(name + "必须为有限正数")


def vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size != size or not np.all(np.isfinite(result)):
        raise ValueError("{}必须包含{}个有限数值".format(name, size))
    return result.copy()


def pose7(values: Sequence[float], name: str) -> np.ndarray:
    return validate_pose7(values, name, as_array=True)


def quaternion_error_rad(actual: np.ndarray, target: np.ndarray) -> float:
    qa = actual / np.linalg.norm(actual)
    qt = target / np.linalg.norm(target)
    dot = float(np.clip(abs(np.dot(qa, qt)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def horizontal_tcp_axis(left, right, *, description="左右TCP", purpose="计算水平连线方向"):
    """返回左TCP指向右TCP的水平单位轴；向内/向外符号由阶段决定。"""
    direction = np.asarray(right[:3], dtype=np.float64) - np.asarray(left[:3], dtype=np.float64)
    direction[2] = 0.0
    span = float(np.linalg.norm(direction))
    if not math.isfinite(span) or span < 0.10:
        raise RuntimeError("{}水平间距仅{:.1f} mm，无法可靠{}".format(
            description, 1000.0 * span, purpose))
    return direction / span


def pose7_to_matrix(values: Iterable[float]) -> np.ndarray:
    pose = np.asarray(validate_pose7(values), dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def matrix_to_pose7(transform: np.ndarray) -> List[float]:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("transform必须是有限的4x4矩阵")
    quaternion = Rotation.from_matrix(value[:3, :3]).as_quat()
    return validate_pose7([*value[:3, 3], *quaternion])


def yaw_from_pose7(values: Iterable[float]) -> float:
    pose = np.asarray(validate_pose7(values), dtype=np.float64)
    return float(Rotation.from_quat(pose[3:]).as_euler("xyz")[2])


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def pose7_from_xyz_yaw(x: float, y: float, z: float, yaw: float) -> List[float]:
    quaternion = Rotation.from_euler("z", float(yaw)).as_quat()
    return validate_pose7([x, y, z, *quaternion])


def compute_grasp_transforms(
    tray_pose: Iterable[float],
    left_tcp_pose: Iterable[float],
    right_tcp_pose: Iterable[float],
) -> Dict[str, List[float]]:
    """计算 T_tray_tcp_L/R。"""
    t_base_tray = pose7_to_matrix(tray_pose)
    inverse_tray = np.linalg.inv(t_base_tray)
    return {
        "left": matrix_to_pose7(inverse_tray @ pose7_to_matrix(left_tcp_pose)),
        "right": matrix_to_pose7(inverse_tray @ pose7_to_matrix(right_tcp_pose)),
    }


def prepare_grasp_components(tray_tcp):
    """一次性校验抓取变换，缓存平移及逆旋转四元数供实时反算使用。"""
    components = {}
    for side in ("left", "right"):
        pose = validate_pose7(tray_tcp[side], "tray_tcp." + side)
        components[side] = (tuple(pose[:3]), tuple(-v for v in pose[3:6]) + (pose[6],))
    return components


def _tray_xyz_yaw_from_tcp(tcp, grasp_position, inverse_grasp_quat):
    """使用单位四元数计算托盘位姿：T_base_tray = T_base_tcp * T_tray_tcp的逆。"""
    x, y, z, qx, qy, qz, qw = (float(value) for value in tcp)
    ix, iy, iz, iw = inverse_grasp_quat
    tx = qw * ix + qx * iw + qy * iz - qz * iy
    ty = qw * iy - qx * iz + qy * iw + qz * ix
    tz = qw * iz + qx * iy - qy * ix + qz * iw
    tw = qw * iw - qx * ix - qy * iy - qz * iz
    px, py, pz = grasp_position
    cx = 2.0 * (ty * pz - tz * py)
    cy = 2.0 * (tz * px - tx * pz)
    cz = 2.0 * (tx * py - ty * px)
    rx = px + tw * cx + ty * cz - tz * cy
    ry = py + tw * cy + tz * cx - tx * cz
    rz = pz + tw * cz + tx * cy - ty * cx
    yaw = math.atan2(2.0 * (tw * tz + tx * ty), 1.0 - 2.0 * (ty * ty + tz * tz))
    return (x - rx, y - ry, z - rz, yaw)


def estimate_tray_fast(left_tcp, right_tcp, grasp_components):
    """按方法B融合位姿，输入须已校验归一化，抓取变换仅预计算一次。"""
    left = _tray_xyz_yaw_from_tcp(left_tcp, *grasp_components["left"])
    right = _tray_xyz_yaw_from_tcp(right_tcp, *grasp_components["right"])
    yaw = math.atan2(
        math.sin(left[3]) + math.sin(right[3]), math.cos(left[3]) + math.cos(right[3])
    )
    x = 0.5 * (left[0] + right[0])
    y = 0.5 * (left[1] + right[1])
    z = 0.5 * (left[2] + right[2])
    half_yaw = 0.5 * yaw
    estimate = [x, y, z, 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)]
    dx, dy, dz = (left[i] - right[i] for i in range(3))
    disagreement = {
        "position_m": math.sqrt(dx * dx + dy * dy + dz * dz),
        "position_xy_m": math.hypot(dx, dy),
        "position_z_signed_m": dz,
        "yaw_rad": abs(wrap_angle(left[3] - right[3])),
    }
    return (estimate, disagreement)


def estimate_tray_from_tcp(
    left_tcp_pose: Iterable[float],
    right_tcp_pose: Iterable[float],
    tray_tcp: Dict[str, Iterable[float]],
) -> Tuple[List[float], Dict[str, float]]:
    """普通入口负责校验和预计算，与Stage3复用同一套反算及融合算法。"""
    return estimate_tray_fast(
        validate_pose7(left_tcp_pose, "left TCP"),
        validate_pose7(right_tcp_pose, "right TCP"),
        prepare_grasp_components(tray_tcp),
    )


def compute_place_and_pre(
    top_pose: Iterable[float],
    stack_pitch: float,
    clearance: float,
) -> Tuple[List[float], List[float]]:
    top = validate_pose7(top_pose, "T_base_top")
    if not math.isfinite(stack_pitch) or stack_pitch <= 0.0:
        raise ValueError("stack_pitch必须是有限正数")
    if not math.isfinite(clearance) or clearance <= 0.0:
        raise ValueError("clearance必须是有限正数")
    yaw = yaw_from_pose7(top)
    place = pose7_from_xyz_yaw(top[0], top[1], top[2] + stack_pitch, yaw)
    pre = pose7_from_xyz_yaw(top[0], top[1], top[2] + stack_pitch + clearance, yaw)
    return place, pre


def load_state(path: str = DEFAULT_STATE_FILE, required: bool = True) -> Dict[str, Any]:
    if not os.path.isfile(path):
        if required:
            raise RuntimeError("多层tray状态文件不存在: {}".format(path))
        return {"version": 1}
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise RuntimeError("多层tray状态文件根节点必须是对象")
    return data


def save_state(data: Dict[str, Any], path: str = DEFAULT_STATE_FILE) -> None:
    if not isinstance(data, dict):
        raise TypeError("state必须是字典")
    payload = dict(data)
    payload["version"] = 1
    payload["updated_unix"] = time.time()
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def update_state(path: str = DEFAULT_STATE_FILE, **entries: Any) -> Dict[str, Any]:
    state = load_state(path, required=False)
    state.update(entries)
    save_state(state, path)
    return state


def set_step_status(path, name, status):
    """只更新一个业务步骤，保留其他步骤和状态内容。"""
    current = load_state(path, required=False)
    steps = dict(current.get("package_steps") or {})
    steps[name] = status
    return update_state(path, package_steps=steps)


def invalidate_grasp_state(path, **entries):
    """使抓取及后续运动失效；保留上一层顶面，额外重置由调用阶段指定。"""
    values = dict(
        stage2_lift_complete=False,
        stage1_complete=False,
        grasp_tray_pose=None,
        grasp_pose_source=None,
        grasp_tcp_poses=None,
        tray_tcp_grasp=None,
        stage2_complete=False,
        stage3_complete=False,
    )
    values.update(entries)
    return update_state(path, **values)


def save_grasp_state(path, pose, tcp, grasp, source, **entries):
    """保存同一次定位的托盘、TCP及抓取关系，不隐式修改其他完成标记。"""
    return update_state(
        path,
        grasp_tray_pose=pose,
        grasp_tcp_poses=tcp,
        tray_tcp_grasp=grasp,
        grasp_pose_source=source,
        stage1_complete=True,
        **entries,
    )


def require_pose(state: Dict[str, Any], key: str) -> List[float]:
    if key not in state:
        raise RuntimeError("状态文件缺少{}".format(key))
    return validate_pose7(state[key], key)


def require_grasp_transforms(state: Dict[str, Any]) -> Dict[str, List[float]]:
    value = state.get("tray_tcp_grasp")
    if not isinstance(value, dict):
        raise RuntimeError("状态文件缺少tray_tcp_grasp")
    return {
        side: validate_pose7(value.get(side), "tray_tcp_grasp.{}".format(side))
        for side in ("left", "right")
    }


__all__ = [
    "DEFAULT_STATE_FILE",
    "vector",
    "require_positive",
    "pose7",
    "quaternion_error_rad",
    "horizontal_tcp_axis",
    "compute_grasp_transforms",
    "compute_place_and_pre",
    "estimate_tray_from_tcp",
    "estimate_tray_fast",
    "prepare_grasp_components",
    "load_state",
    "matrix_to_pose7",
    "pose7_from_xyz_yaw",
    "pose7_to_matrix",
    "require_grasp_transforms",
    "require_pose",
    "save_state",
    "update_state",
    "set_step_status",
    "invalidate_grasp_state",
    "save_grasp_state",
    "validate_pose7",
    "wrap_angle",
    "yaw_from_pose7",
]
