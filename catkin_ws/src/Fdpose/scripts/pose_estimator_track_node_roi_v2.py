#!/usr/bin/env python3
# 运行：python scripts/pose_estimator_track_node_roi_v2.py（推荐 V2 服务入口）。
# 实际执行：src/fdpose/ros/server_v2.py::main；配置由 server_config.py 加载。
# 配置：config/fdpose_server_v2.yaml；可用 _config_path:=/绝对路径/config.yaml 覆盖。
"""FDPose V2.3 ROS server entry point.

Differences from ``pose_estimator_track_node_roi.py``:

- rejects requests with insufficient valid depth inside the object mask;
- debug and visualization save failures do not invalidate a successful pose;
- all loaded labels share one ScorePredictor and PoseRefinePredictor;
- supports YAML defaults and an optional fixed RGB-D startup warmup;
- supports registered, external-path, and selectively unloaded Mesh labels;
- uses independent V2 service names and ``runtime/debug/<timestamp>_<id>/`` raw/vis outputs.

FoundationPose inference still comes from the unchanged V1 ``estimator.py``.
"""

from _fdpose_bootstrap import add_src_to_path

add_src_to_path()

from fdpose.ros.server_v2 import main  # noqa: E402


if __name__ == "__main__":
    main()
