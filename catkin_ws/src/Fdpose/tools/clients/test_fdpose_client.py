#!/usr/bin/env python3
# 运行：python tools/clients/test_fdpose_client.py [--label 标签 --rgb 图片 --depth 深度 --mask 掩码]。
# 实际调用：src/fdpose/ros/client.py 的 FdposeClient → 已启动的 V2 Service；不自行创建模型。
# 配置：本脚本使用命令行参数；Mesh 由服务端查 assets/meshes/registry.yaml。
# 服务端需另运行 scripts/pose_estimator_track_node_roi_v2.py，读取 config/fdpose_server_v2.yaml。
"""Manually test FDPose V2 with local RGB, depth, and mask files (no YOLO).

This client still uses ROS and the FDPose server, but it does not start or call
the segmentation service.  Message conversion and service calls are delegated
to the reusable FdposeClient. Integer depth is millimeters; float depth is meters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2
import numpy as np
import rospy
from scipy.spatial.transform import Rotation

from fdpose.paths import SAMPLES_DIR
from fdpose.ros.client import (
    DEFAULT_MESH_SERVICE,
    DEFAULT_POSE_SERVICE,
    FdposeClient,
    FdposeClientError,
)


DEFAULT_SAMPLE_DIR = SAMPLES_DIR / "current"


def load_local_images(rgb_path: Path, depth_path: Path, mask_path: Path):
    color_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    missing = []
    if color_bgr is None:
        missing.append(str(rgb_path))
    if depth is None:
        missing.append(str(depth_path))
    if mask is None:
        missing.append(str(mask_path))
    if missing:
        raise FileNotFoundError(f"Cannot read local input files: {missing}")

    image_size = color_bgr.shape[:2]
    if depth.shape[:2] != image_size or mask.shape[:2] != image_size:
        raise ValueError(
            "RGB/depth/mask size mismatch: "
            f"rgb={image_size}, depth={depth.shape[:2]}, mask={mask.shape[:2]}"
        )
    if not np.any(mask > 0):
        raise ValueError(f"Mask is empty: {mask_path}")

    return color_bgr, depth, mask


def run_local_test(args) -> int:
    rgb_path = Path(args.rgb).expanduser().resolve()
    depth_path = Path(args.depth).expanduser().resolve()
    mask_path = Path(args.mask).expanduser().resolve()
    color_bgr, depth, mask = load_local_images(rgb_path, depth_path, mask_path)

    client = FdposeClient(
        mesh_service=args.load_mesh_service,
        pose_service=args.pose_service,
        timeout=args.timeout,
    )
    rospy.loginfo(
        "Calling FDPose from local files: label=%s rgb=%s depth=%s mask=%s",
        args.label, rgb_path, depth_path, mask_path,
    )
    try:
        client.load_mesh(args.label)
        result = client.estimate_one(
            label=args.label,
            color_bgr=color_bgr,
            depth=depth,
            mask=mask,
        )
    except FdposeClientError as exc:
        rospy.logerr("FDPose request failed: %s", exc)
        return 1

    xyz = result.T_object[:3, 3]
    quat = Rotation.from_matrix(result.T_object[:3, :3]).as_quat()
    print("returned poses: 1")
    print(f"[0] label={result.label}")
    print("  xyz: " + " ".join(f"{value:.6f}" for value in xyz))
    print("  quat: " + " ".join(f"{value:.6f}" for value in quat))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manually test FDPose V2 from local RGB/depth/mask files."
    )
    parser.add_argument("--label", default="connector_mod")
    parser.add_argument(
        "--rgb",
        default=str(DEFAULT_SAMPLE_DIR / "rgb/0.png"),
    )
    parser.add_argument(
        "--depth",
        default=str(DEFAULT_SAMPLE_DIR / "depth/0.png"),
    )
    parser.add_argument(
        "--mask",
        default=str(DEFAULT_SAMPLE_DIR / "masks/0.png"),
    )
    parser.add_argument("--load-mesh-service", default=DEFAULT_MESH_SERVICE)
    parser.add_argument("--pose-service", default=DEFAULT_POSE_SERVICE)
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rospy.init_node("fdpose_local_test_client", anonymous=True)
    return run_local_test(args)


if __name__ == "__main__":
    raise SystemExit(main())
