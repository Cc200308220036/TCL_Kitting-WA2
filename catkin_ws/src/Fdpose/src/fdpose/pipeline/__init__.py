"""FDPose 前置感知流程编排。

公共 FdposePipeline 对接新版 YOLO SegServiceClient，不创建相机或实现推理。
from_ros_services 默认读取项目 config/fdpose_pipeline.yaml 的连接参数。
"""

from .pipeline import FdposePipeline

__all__ = ["FdposePipeline"]
