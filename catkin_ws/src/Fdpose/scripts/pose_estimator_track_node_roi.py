#!/usr/bin/env python3
# 运行：python scripts/pose_estimator_track_node_roi.py（保留的 V1，不是默认入口）。
# 实际执行：src/fdpose/ros/server.py::main；参数来自 ROS 私有参数。
# 资产配置：assets/meshes/registry.yaml、config/camera/；不读取 V2 的 YAML。
"""Historical FDPose V1 ROS server entry point (not the default).

This file only preserves the historical command name after the source migration.
The implementation lives in ``fdpose.ros.server`` and exposes
``/fdpose_load_mesh`` and ``/pose_estimation_service``.  Unlike the legacy
server, every request estimates from the request's current RGB-D image and mask.
"""

from _fdpose_bootstrap import add_src_to_path

add_src_to_path()

from fdpose.ros.server import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    main()
