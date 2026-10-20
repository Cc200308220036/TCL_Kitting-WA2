#!/usr/bin/env python3
# 运行：python tools/clients/test_pipeline.py --rgb 图片 --depth 深度 --labels 标签。
# 实际调用：src/fdpose/pipeline/pipeline.py 的 FdposePipeline → 已启动的 YOLO/FDPose Service。
# 配置：config/fdpose_pipeline.yaml（--config 可覆盖）；不读取服务进程部署配置。
"""本地 RGB-D → 新版 YOLO → FDPose V2；只调用公共 Pipeline，不实现第二套流程。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2
import rospy

from fdpose.pipeline import FdposePipeline
from fdpose.pipeline.pipeline import DEFAULT_PIPELINE_CONFIG
from fdpose.ros.client import FdposeClientError


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", required=True, help="本地 BGR 彩色图像路径")
    parser.add_argument("--depth", required=True, help="本地 uint16 毫米深度图路径")
    parser.add_argument("--labels", nargs="+", required=True, help="YOLO 类别及注册 Mesh 标签")
    parser.add_argument("--config", default=str(DEFAULT_PIPELINE_CONFIG), help="Pipeline 连接 YAML")
    parser.add_argument("--timeout", type=float, help="覆盖 YAML 的服务就绪等待秒数")
    return parser


def run_local_test(args):
    color = cv2.imread(str(Path(args.rgb).expanduser()), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(Path(args.depth).expanduser()), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        raise ValueError(f"无法读取 RGB-D: {args.rgb}, {args.depth}")
    pipeline = FdposePipeline.from_ros_services(config_path=args.config, timeout=args.timeout)
    poses = pipeline.process_rgbd(color=color, depth=depth, labels=args.labels)
    print(f"returned poses: {len(poses)}")
    for pose in poses:
        print(f"label={pose.label}\nT_camera_object:\n{pose.T_object}")
    return 0


def main():
    args = build_parser().parse_args(rospy.myargv()[1:])
    rospy.init_node("fdpose_pipeline_local_test", anonymous=True)
    try:
        return run_local_test(args)
    except (ValueError, OSError, FdposeClientError, rospy.ROSException, rospy.ServiceException) as exc:
        rospy.logerr("Pipeline test failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
