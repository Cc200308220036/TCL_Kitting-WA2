#!/usr/bin/env python3
# 运行：python scripts/pose_estimator_node.py（兼容旧安装名称，实际启动 V2）。
# 实际执行：src/fdpose/ros/server_v2.py::main；新使用者优先运行 pose_estimator_track_node_roi_v2.py。
# 配置：config/fdpose_server_v2.yaml；可用 _config_path:=/绝对路径/config.yaml 覆盖。
"""Historical installed entry name, now forwarding to FDPose V2.

The legacy implementation has been removed. This thin entry remains because
the unchanged CMakeLists.txt installs it; it no longer provides legacy behavior
or legacy service names. Prefer pose_estimator_track_node_roi_v2.py explicitly.
"""

import sys

from _fdpose_bootstrap import add_src_to_path

add_src_to_path()

from fdpose.ros.server_v2 import main  # noqa: E402


if __name__ == "__main__":
    print(
        "pose_estimator_node.py now starts FDPose V2, not Legacy. "
        "Default services: /fdpose_v2/load_mesh, /fdpose_v2/pose_estimation. "
        "Prefer scripts/pose_estimator_track_node_roi_v2.py.",
        file=sys.stderr,
    )
    main()
