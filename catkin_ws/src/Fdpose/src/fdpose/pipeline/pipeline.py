#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组合分割、可选深度优化和 FDPose 的轻量流程。

职责边界：

- 相机和 ROS 节点的生命周期由主程序管理；
- 分割与深度优化由各自模块提供 Client；
- 本模块只排列调用顺序，并把分割结果转换为 FDPose 输入。

分割 Client 需要提供::

    segment(color, depth=None, labels=None)

并返回含 ``have_objs`` / ``items`` 的响应（新版 YOLO SegServiceClient）。
RGB-D 由本流程保留，不从分割响应取回。深度优化 Client（可选）需要提供
``refine(color_bgr, depth)``，返回同尺寸深度，uint16 表示毫米，float32 表示米。
外部 Client 不应原地修改输入图像；同一 Pipeline 实例由主程序串行调用。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence
import rospy

import numpy as np

from fdpose.paths import CONFIG_DIR


DEFAULT_PIPELINE_CONFIG = CONFIG_DIR / "fdpose_pipeline.yaml"


def _load_connection_config(config_path):
    """只读取连接参数，不根据配置动态导入代码；None 表示不用 YAML。"""
    config = {
        "mesh_service": "/fdpose_v2/load_mesh",
        "pose_service": "/fdpose_v2/pose_estimation",
        "segmentation_service": "/instance_segmentation_service",
        "timeout": 20.0,
    }
    if config_path is not None:
        import yaml

        with Path(config_path).expanduser().open(encoding="utf-8") as handle:
            try:
                values = yaml.safe_load(handle)
            except yaml.YAMLError as exc:
                raise ValueError(f"Invalid Pipeline YAML: {config_path}") from exc
        if values is None:
            values = {}
        if not isinstance(values, dict):
            raise ValueError("Pipeline config must be a YAML mapping")
        unknown = values.keys() - config.keys()
        if unknown:
            raise ValueError(f"Unknown Pipeline config keys: {list(unknown)}")
        config.update(values)
    return config


class FdposePipeline:
    """FDPose 所需前置感知步骤的编排器。"""

    def __init__(
        self,
        fdpose_client: Any,
        segmentation_client: Any,
        depth_refine_client: Optional[Any] = None,
    ):
        """方式一：复用调用方创建的 Client 实例。

        适用：主程序已持有 Client、需要独立配置各模块的超时，或需要在
        Pipeline 外直接管理 Mesh；测试时也可注入满足相同接口的替身。
        此处不创建 Client、不连接服务，ROS 初始化由主程序负责。
        分割实例须提供 segment()，FDPose 须提供 load_meshes()/estimate()，
        可选深度实例须提供 refine()；具体输入输出约定见模块说明。
        """
        if fdpose_client is None:
            raise ValueError("fdpose_client must not be None")
        if segmentation_client is None:
            raise ValueError("segmentation_client must not be None")

        self.fdpose = fdpose_client
        self.segmentation = segmentation_client
        self.depth_refine = depth_refine_client

    @classmethod
    def from_ros_services(
        cls,
        *,
        # FDPose：与构造函数的 Client 顺序一致。
        mesh_service: Optional[str] = None,
        pose_service: Optional[str] = None,
        # YOLO：服务名在前，可选的工厂和额外构造参数随后。
        segmentation_service: Optional[str] = None,
        segmentation_client_factory: Optional[Callable[..., Any]] = None,
        segmentation_client_kwargs: Optional[Dict[str, Any]] = None,
        # 深度补全：不需要时整组省略；服务名与工厂必须成对提供。
        depth_refine_service: Optional[str] = None,
        depth_refine_client_factory: Optional[Callable[..., Any]] = None,
        depth_refine_client_kwargs: Optional[Dict[str, Any]] = None,
        # 连接配置默认位于项目 config/；传 None 跳过文件读取。
        config_path: Optional[str] = str(DEFAULT_PIPELINE_CONFIG),
        # 显式传参优先于 YAML；None 使用配置值，不是无限等待。
        timeout: Optional[float] = None,
    ) -> "FdposePipeline":
        """方式二：按服务名创建新 Client，再复用相同的构造函数和处理流程。

        适用：调用方尚未创建 Client，希望用已知服务名快速组装 Pipeline，
        且各 Client 的构造接口符合下述约定。调用前由主程序初始化 ROS；
        此方法可能等待外部服务就绪，不会启动服务或创建相机。
        分割默认延迟导入新版 YOLO SegServiceClient，接受 service_name 和
        wait_timeout_sec。可选深度工厂沿用 service_name / timeout_sec 约定；
        若需复用实例、独立设置超时或外部构造接口不同，使用方式一。
        timeout 是服务就绪等待时间，不是整条 Pipeline 的执行时限。
        全部参数仅支持关键字传递；参数分组不改变实际初始化或推理顺序。
        默认读取 config/fdpose_pipeline.yaml；显式传参 > YAML > 内置默认值。
        配置仅含 FDPose/YOLO 服务名及 timeout，不包含 Client 导入路径。
        config_path=None 跳过 YAML，指定文件不存在或键错误时直接报错。
        """

        config = _load_connection_config(config_path)
        for key, value in (("mesh_service", mesh_service), ("pose_service", pose_service),
                           ("segmentation_service", segmentation_service), ("timeout", timeout)):
            if value is not None:
                config[key] = value
        for key in ("mesh_service", "pose_service", "segmentation_service"):
            value = config[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string")
            config[key] = value.strip()
        timeout = config["timeout"]
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not np.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be a finite positive number of seconds")
        timeout = float(timeout)

        from fdpose.ros.client import FdposeClient

        if (depth_refine_client_factory is None) != (
            depth_refine_service is None
        ):
            raise ValueError(
                "depth_refine_client_factory and depth_refine_service "
                "must be provided together"
            )
        if segmentation_client_factory is None:
            from naviai_yolo.ros.seg_service_client import SegServiceClient

            segmentation_client_factory = SegServiceClient
        if not callable(segmentation_client_factory):
            raise TypeError("segmentation_client_factory must be callable")

        seg_kwargs = dict(segmentation_client_kwargs or {})
        if {"service_name", "wait_timeout_sec", "timeout_sec"} & seg_kwargs.keys():
            raise ValueError(
                "segmentation_client_kwargs must not override "
                "service_name or wait_timeout_sec (use timeout)"
            )
        segmentation_client = segmentation_client_factory(
            service_name=config["segmentation_service"],
            wait_timeout_sec=float(timeout),
            **seg_kwargs,
        )

        depth_refine_client = None
        if depth_refine_client_factory is not None:
            depth_kwargs = dict(depth_refine_client_kwargs or {})
            if "service_name" in depth_kwargs or "timeout_sec" in depth_kwargs:
                raise ValueError(
                    "depth_refine_client_kwargs must not override "
                    "service_name or timeout_sec"
                )
            depth_refine_client = depth_refine_client_factory(
                service_name=str(depth_refine_service),
                timeout_sec=float(timeout),
                **depth_kwargs,
            )

        return cls(
            fdpose_client=FdposeClient(
                mesh_service=config["mesh_service"],
                pose_service=config["pose_service"],
                timeout=timeout,
            ),
            segmentation_client=segmentation_client,
            depth_refine_client=depth_refine_client,
        )

    @staticmethod
    def _normalize_labels(labels: Sequence[str]):
        if isinstance(labels, str):
            labels = [labels]
        if labels is None:
            raise ValueError("labels must contain at least one non-empty label")

        normalized = []
        seen = set()
        for label in labels:
            if not isinstance(label, str) or not label.strip():
                raise ValueError("labels must contain only non-empty strings")
            value = label.strip()
            if value not in seen:
                normalized.append(value)
                seen.add(value)
        if not normalized:
            raise ValueError("labels must contain at least one non-empty label")
        return normalized

    @staticmethod
    def _to_detection_inputs(items):
        from fdpose.ros.client import DetectionInput

        detections = []
        for item in items:
            if isinstance(item, DetectionInput):
                detections.append(item)
            else:
                detections.append(DetectionInput.from_yolo_item(item))
        return detections

    @staticmethod
    def _to_bgr(color, color_order: str):
        color = np.asarray(color)
        if (color.ndim != 3 or color.shape[2] != 3 or color.size == 0
                or color.dtype != np.uint8):
            raise ValueError("color must be a nonempty HxWx3 uint8 image")

        order = str(color_order).lower()
        if order == "bgr":
            return np.ascontiguousarray(color)
        if order == "rgb":
            return color[:, :, ::-1].copy()
        raise ValueError("color_order must be 'rgb' or 'bgr'")

    @staticmethod
    def _check_depth(depth, image_shape):
        depth = np.asarray(depth)
        if depth.shape != image_shape or depth.dtype not in (np.uint16, np.float32):
            raise ValueError(
                "depth must match color HxW and use uint16 (mm) or float32 (m)"
            )
        return np.ascontiguousarray(depth)

    def process_rgbd(
        self,
        color,
        depth,
        labels: Sequence[str],
        *,
        color_order: str = "bgr",
        load_mesh: bool = True,
        run_depth_refine: Optional[bool] = None,
        T_target_camera=None,
    ):
        """串行处理一帧 RGB-D，返回 PoseResult 列表；调用异常向上传播。"""
        return self._process_rgbd(
            color, depth, labels, color_order=color_order, load_mesh=load_mesh,
            run_depth_refine=run_depth_refine, parallel=False,
            T_target_camera=T_target_camera,
        )

    def process_rgbd_parallel(
        self,
        color,
        depth,
        labels: Sequence[str],
        *,
        color_order: str = "bgr",
        load_mesh: bool = True,
        T_target_camera=None,
    ):
        """并行执行分割和深度补全，再调用 FDPose。

        只有分割和深度补全互相并行；Mesh 加载在它们之前完成，FDPose
        在两个结果都返回后执行。没有注入深度补全 Client 时，自动退化为
        不做深度补全的普通 RGB-D 流程。
        """

        return self._process_rgbd(
            color, depth, labels, color_order=color_order, load_mesh=load_mesh,
            run_depth_refine=None, parallel=True,
            T_target_camera=T_target_camera,
        )

    def _process_rgbd(
        self, color, depth, labels, *, color_order, load_mesh,
        run_depth_refine, parallel, T_target_camera,
    ):
        # 在任何服务调用前校验参数；不自动缩放图像，也不推测深度单位。
        labels = self._normalize_labels(labels)
        color_bgr = self._to_bgr(color, color_order)
        depth = self._check_depth(depth, color_bgr.shape[:2])
        if run_depth_refine is None:
            run_depth_refine = self.depth_refine is not None
        if run_depth_refine and self.depth_refine is None:
            raise RuntimeError(
                "run_depth_refine=True but no depth_refine_client was provided"
            )
        if load_mesh:
            self.fdpose.load_meshes(labels)
            rospy.loginfo(f"Loaded meshes: {labels}")

        # 并行模式下
        if parallel and run_depth_refine:
            # 两个服务读取同一帧；线程只并行调用独立 Client，不并行执行 Pose。
            # 离开上下文前等待两个调用结束；异常不伪装成“未检出”。
            with ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="fdpose_pipeline",
            ) as executor:
                segmentation_future = executor.submit(
                    self.segmentation.segment,
                    color=color_bgr, depth=depth, labels=labels,
                )
                depth_future = executor.submit(self.depth_refine.refine, color_bgr, depth)
                response = segmentation_future.result()
                depth_used = depth_future.result()
        else:
            response = self.segmentation.segment(color=color_bgr, depth=depth, labels=labels)
            depth_used = depth

        if not response.have_objs or not response.items:
            return []

        detections = self._to_detection_inputs(response.items)
        for detection in detections:
            if detection.mask.shape != color_bgr.shape[:2]:
                raise ValueError(f"mask for {detection.label!r} must match color HxW")
        # 串行模式下，深度补全在 FDPose 之前执行
        if run_depth_refine and not parallel:
            depth_used = self.depth_refine.refine(color_bgr, depth)
        depth_used = self._check_depth(depth_used, color_bgr.shape[:2])
        if not np.any(np.isfinite(depth_used) & (depth_used > 0)):
            raise ValueError("depth has no finite positive values for pose estimation")

        return self.fdpose.estimate(
            color_bgr=color_bgr,
            depth=depth_used,
            detections=detections,
            T_target_camera=T_target_camera,
        )

    def process_from_camera(
        self,
        frame_provider: Callable[[], Any],
        labels: Sequence[str],
        **process_kwargs,
    ):
        """向外部相机对象取一帧，然后复用 :meth:`process_rgbd`。

        ``frame_provider`` 由主程序传入，例如 ``camera.get_rgbd``，返回值
        必须是 ``(color, depth)``。本类不创建相机、不订阅 Topic，也不管理
        相机线程。
        """

        if not callable(frame_provider):
            raise TypeError("frame_provider must be callable")
        frame = frame_provider()
        if not isinstance(frame, (tuple, list)) or len(frame) != 2:
            raise ValueError("frame_provider must return (color, depth)")
        color, depth = frame
        return self.process_rgbd(
            color=color,
            depth=depth,
            labels=labels,
            **process_kwargs,
        )


__all__ = ["FdposePipeline"]
