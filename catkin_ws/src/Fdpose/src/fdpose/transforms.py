"""通用相对位姿计算，仅依赖 NumPy，不管理任务、机器人或关系文件。

约定 T_A_B 将 B 坐标系中的点变换到 A 坐标系。
调用方负责提供同一参考坐标系下、同一时刻的物体和目标位姿。
"""

import numpy as np


def validate_transform(name: str, transform) -> np.ndarray:
    """检查有限的 4x4 齐次矩阵；沿用原关系工具的输入检查。"""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN/Inf")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")
    return matrix


def compute_relative_pose(T_reference_object, T_reference_target) -> np.ndarray:
    """计算 T_object_target = inv(T_reference_object) @ T_reference_target。

    target 可以是另一个物体、工具或任意目标坐标系，不绑定 TCP 命名。
    """
    object_pose = validate_transform("T_reference_object", T_reference_object)
    target_pose = validate_transform("T_reference_target", T_reference_target)
    return validate_transform("T_object_target", np.linalg.inv(object_pose) @ target_pose)


def apply_relative_pose(T_reference_object, T_object_target) -> np.ndarray:
    """应用固定关系，返回 T_reference_target。"""
    object_pose = validate_transform("T_reference_object", T_reference_object)
    relative_pose = validate_transform("T_object_target", T_object_target)
    return validate_transform("T_reference_target", object_pose @ relative_pose)
