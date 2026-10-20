"""V1/V2 共用的无状态工具；不持有节点、网络或服务生命周期状态。"""

import os

import numpy as np
import yaml


def matrix_to_ros_pose(matrix):
    """
    Input: 4x4 homogeneous transformation matrix.
    Return: geometry_msgs/Pose.
    """
    from geometry_msgs.msg import Pose
    from scipy.spatial.transform import Rotation as R

    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    rotation = R.from_matrix(matrix[:3, :3]).as_quat()
    position = matrix[:3, 3]

    ros_pose = Pose()
    ros_pose.position.x = float(position[0])
    ros_pose.position.y = float(position[1])
    ros_pose.position.z = float(position[2])
    ros_pose.orientation.x = float(rotation[0])
    ros_pose.orientation.y = float(rotation[1])
    ros_pose.orientation.z = float(rotation[2])
    ros_pose.orientation.w = float(rotation[3])
    return ros_pose


def read_mesh_config(mesh_registry_path, logerr):
    """读取并校验注册表；文件级错误返回空表，坏条目不影响其他 Label。"""
    if not os.path.exists(mesh_registry_path):
        logerr(f"Mesh registry YAML not found at {mesh_registry_path}")
        return {}
    try:
        with open(mesh_registry_path, "r", encoding="utf-8") as f:
            mesh_config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        logerr(f"Failed to read mesh registry {mesh_registry_path}: {exc}")
        return {}
    if mesh_config is None:
        return {}
    if not isinstance(mesh_config, dict) or not isinstance(mesh_config.get("meshes"), dict):
        logerr(f"Mesh registry must contain a 'meshes' mapping: {mesh_registry_path}")
        return {}
    meshes = {}
    for label, info in mesh_config["meshes"].items():
        if (not isinstance(label, str) or not label.strip()
                or not isinstance(info, dict)
                or not isinstance(info.get("path"), str) or not info["path"].strip()):
            logerr(f"Invalid mesh registry entry {label!r}: expected a label and nonempty path")
            continue
        meshes[label] = info
    return meshes


def depth_mono16_to_meters(depth_image):
    """V2 消息协议固定为 mono16 毫米，不根据图像内容猜测单位。"""
    depth = depth_image.astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return depth * np.float32(0.001)


def legacy_depth_to_meters(depth_image):
    """仅供 V1 使用：保留旧版基于中位数的兼容转换。"""
    depth = depth_image.astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero = depth[depth > 0]
    # mono16 from client is expected to be millimeters.
    if nonzero.size > 0 and float(np.nanmedian(nonzero)) > 20.0:
        depth *= 0.001
    return depth.astype(np.float32, copy=False)


def crop_by_mask(color, depth, mask, K, *, use_roi, roi_margin, min_mask_pixels):
    """Crop current image by current mask ROI and shift camera intrinsics."""
    if not use_roi:
        h, w = mask.shape[:2]
        return color, depth, mask, K, (0, 0, w, h)

    ys, xs = np.where(mask)
    if xs.size < min_mask_pixels:
        raise ValueError(f"mask too small: {xs.size} pixels")

    h, w = mask.shape[:2]
    margin = max(0, int(roi_margin))
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(w, int(xs.max()) + margin + 1)
    y2 = min(h, int(ys.max()) + margin + 1)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid ROI: {(x1, y1, x2, y2)}")

    color_roi = np.ascontiguousarray(color[y1:y2, x1:x2])
    depth_roi = np.ascontiguousarray(depth[y1:y2, x1:x2])
    mask_roi = np.ascontiguousarray(mask[y1:y2, x1:x2])

    K_roi = K.copy()
    K_roi[0, 2] -= float(x1)
    K_roi[1, 2] -= float(y1)

    return color_roi, depth_roi, mask_roi, K_roi, (x1, y1, x2, y2)


def draw_pose_visualization(color_rgb_full, pose, K, bbox, *, axis_scale, axis_thickness):
    """只绘制相机系 Pose；输入完整 RGB 图像、完整图内参及原 Mesh 包围框。"""
    from fdpose.core.utils import draw_posed_3d_box, draw_xyz_axis

    ob_in_cam = np.asarray(pose, dtype=np.float32).reshape(4, 4)
    vis = draw_posed_3d_box(
        K,
        img=color_rgb_full.copy(),
        ob_in_cam=ob_in_cam,
        bbox=bbox,
    )
    return draw_xyz_axis(
        vis,
        ob_in_cam=ob_in_cam,
        scale=axis_scale,
        K=K,
        thickness=axis_thickness,
        transparency=0,
        is_input_rgb=True,
    )
