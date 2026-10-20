#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable Python client for the FDPose ROS services.

The public API uses NumPy arrays instead of ROS image and pose messages:

- :class:`DetectionInput` is the common detection format produced after YOLO,
  SAM, or local-mask conversion.
- :class:`PoseResult` contains the returned object pose matrix.
- :class:`FdposeClient` owns ROS service proxies and message conversion.

The caller still owns ROS node initialization, image acquisition,
segmentation, and task/TCP calculations.  An optional caller-provided
``T_target_camera`` can transform the FDPose result out of the camera frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Optional, Sequence, Tuple

import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

from naviai_manip_srvs.msg import DetItem as FdposeDetItem
from naviai_manip_srvs.srv import (
    FdposeLoadMesh,
    FdposeLoadMeshRequest,
    PoseEst,
    PoseEstRequest,
)

from fdpose.paths import MESH_REGISTRY_PATH, MESHES_DIR


DEFAULT_MESH_SERVICE = "/fdpose_v2/load_mesh"
DEFAULT_POSE_SERVICE = "/fdpose_v2/pose_estimation"
_MESH_LABEL_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


class FdposeClientError(RuntimeError):
    """Raised when an FDPose ROS service operation cannot be completed."""


@dataclass(frozen=True)
class DetectionInput:
    """ROS-independent detection input consumed by :class:`FdposeClient`.

    ``label`` identifies the Mesh; no suffix encodes state or symmetry.
    ``mask`` must be a two-dimensional NumPy array.  Every value greater than
    zero is treated as object foreground.  ``bbox`` follows
    ``(x_min, y_min, x_max, y_max)`` and is computed from the mask when omitted.
    """

    label: str
    mask: np.ndarray
    confidence: float = 1.0
    bbox: Optional[Tuple[int, int, int, int]] = None

    def __post_init__(self):
        label = str(self.label).strip()
        if not label:
            raise ValueError("DetectionInput.label must not be empty")

        mask = np.asarray(self.mask)
        if mask.ndim != 2:
            raise ValueError(
                f"DetectionInput.mask must be 2D, got shape={mask.shape}"
            )
        if mask.size == 0 or not np.any(mask > 0):
            raise ValueError("DetectionInput.mask must contain foreground pixels")

        confidence = float(self.confidence)
        if not np.isfinite(confidence):
            raise ValueError("DetectionInput.confidence must be finite")

        bbox = self.bbox
        if bbox is not None:
            if len(bbox) != 4:
                raise ValueError("DetectionInput.bbox must contain four values")
            bbox = tuple(int(value) for value in bbox)

        object.__setattr__(self, "label", label)
        object.__setattr__(self, "mask", np.ascontiguousarray(mask))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "bbox", bbox)

    @classmethod
    def from_yolo_item(cls, item, bridge: Optional[CvBridge] = None):
        """Convert a YOLO/segmentation ROS ``DetItem`` to ``DetectionInput``.

        The conversion intentionally relies only on the common ``label``,
        ``confidence``, ``bbox``, and ``mask`` fields, so it works with the
        existing ``navi_types/DetItem`` segmentation response.
        """

        bridge = bridge or CvBridge()
        mask_value = getattr(item, "mask", None)
        if mask_value is None:
            raise ValueError("YOLO item has no mask field")

        if isinstance(mask_value, np.ndarray):
            mask = mask_value
        else:
            mask = bridge.imgmsg_to_cv2(mask_value, desired_encoding="passthrough")

        raw_bbox = getattr(item, "bbox", None)
        bbox = None
        if raw_bbox is not None and len(raw_bbox) == 4:
            bbox = tuple(int(value) for value in raw_bbox)

        return cls(
            label=str(getattr(item, "label", "")),
            mask=np.asarray(mask),
            confidence=float(getattr(item, "confidence", 1.0)),
            bbox=bbox,
        )


@dataclass(frozen=True)
class PoseResult:
    """One FDPose result in the output frame selected by the caller.

    ``T_object`` is ``T_camera_object`` when ``T_target_camera`` is omitted
    from the estimation call, otherwise it is ``T_target_object``.
    """

    label: str
    T_object: np.ndarray

    def __post_init__(self):
        label = str(self.label).strip()
        matrix = np.asarray(self.T_object, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"T_object must be 4x4, got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("T_object contains NaN/Inf")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError("T_object last row must be [0, 0, 0, 1]")

        object.__setattr__(self, "label", label)
        object.__setattr__(
            self,
            "T_object",
            np.ascontiguousarray(matrix),
        )


class FdposeClient:
    """Reusable client for mesh loading and one-shot FDPose estimation.

    The caller must call ``rospy.init_node`` before using service methods.  The
    client does not acquire camera images or call a segmentation service.
    """

    def __init__(
        self,
        mesh_service: str = DEFAULT_MESH_SERVICE,
        pose_service: str = DEFAULT_POSE_SERVICE,
        timeout: float = 20.0,
    ):
        self.mesh_service = str(mesh_service)
        self.pose_service = str(pose_service)
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        self.bridge = CvBridge()
        self._load_mesh_proxy = None
        self._pose_proxy = None

    def _require_ros_node(self) -> None:
        if not rospy.core.is_initialized():
            raise FdposeClientError(
                "ROS node is not initialized; call rospy.init_node(...) first"
            )

    def wait_until_ready(self, timeout: Optional[float] = None) -> None:
        """Wait for both FDPose services and initialize their proxies."""

        self._require_ros_node()
        wait_timeout = self.timeout if timeout is None else float(timeout)
        if wait_timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        try:
            rospy.wait_for_service(self.mesh_service, timeout=wait_timeout)
            rospy.wait_for_service(self.pose_service, timeout=wait_timeout)
        except rospy.ROSException as exc:
            raise FdposeClientError(
                "FDPose services are not ready: "
                f"mesh={self.mesh_service}, pose={self.pose_service}"
            ) from exc

        # create service proxies
        self._load_mesh_proxy = rospy.ServiceProxy(
            self.mesh_service,
            FdposeLoadMesh,
            persistent=False,
        )
        self._pose_proxy = rospy.ServiceProxy(
            self.pose_service,
            PoseEst,
            persistent=False,
        )

    def _ensure_ready(self) -> None:
        if self._load_mesh_proxy is None or self._pose_proxy is None:
            self.wait_until_ready()

    @staticmethod
    def _normalize_labels(labels: Sequence[str]):
        if isinstance(labels, str):
            labels = [labels]

        normalized = []
        seen = set()
        for label in labels:
            if not isinstance(label, str) or not label.strip():
                raise ValueError("labels must contain non-empty strings")
            value = str(label).strip()
            if value in seen:
                continue
            normalized.append(value)
            seen.add(value)
        return normalized

    @staticmethod
    def _normalize_mesh_paths(
        mesh_paths: Optional[Sequence[Optional[str]]],
        expected_count: int,
    ):
        if mesh_paths is None:
            return []
        if isinstance(mesh_paths, (str, Path)):
            mesh_paths = [mesh_paths]

        paths = list(mesh_paths)
        if len(paths) != expected_count:
            raise ValueError(
                "mesh_paths must match labels: "
                f"labels={expected_count}, mesh_paths={len(paths)}"
            )

        normalized = []
        for path in paths:
            if path is None or not str(path).strip():
                normalized.append("")
                continue
            resolved = Path(path).expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"Mesh file not found: {resolved}")
            normalized.append(str(resolved))
        return normalized

    @staticmethod
    def _normalize_registry_label(label: str) -> str:
        value = str(label).strip()
        if not _MESH_LABEL_PATTERN.fullmatch(value):
            raise ValueError(
                f"Invalid mesh label={value!r}; use lowercase letters, "
                "numbers, and underscores"
            )
        return value

    @staticmethod
    def _read_mesh_registry():
        registry_path = Path(MESH_REGISTRY_PATH)
        if not registry_path.is_file():
            raise FileNotFoundError(f"Mesh registry not found: {registry_path}")

        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        if not isinstance(registry, dict) or set(registry) != {"meshes"} or not isinstance(
            registry.get("meshes"), dict
        ):
            raise ValueError(
                f"Unsupported mesh registry structure: {registry_path}"
            )
        return registry

    @staticmethod
    def _write_mesh_registry(registry) -> None:
        """调用方持有注册锁；结构化更新并原子替换，避免依赖 YAML 排版。"""
        registry_path = Path(MESH_REGISTRY_PATH)
        updated = yaml.safe_dump(registry, allow_unicode=True, sort_keys=False)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(registry_path.parent),
                prefix=f".{registry_path.name}.",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(updated)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, registry_path.stat().st_mode)
            os.replace(str(temp_path), str(registry_path))
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def register_mesh(self, label: str, mesh_path: str) -> str:
        """复制单个 Mesh 并写磁盘注册表，返回保存路径；不调用 ROS 服务。

        label 已注册或目标目录已存在时直接报错，不覆盖、不删除旧资产。
        不管理临时关联，也不复制外部材质/纹理；加载需显式调用 load_mesh。
        """
        label = self._normalize_registry_label(label)
        source_path = self._normalize_mesh_paths([mesh_path], 1)[0]
        if not source_path:
            raise ValueError("mesh_path must not be empty")
        source = Path(source_path)

        registry_path = Path(MESH_REGISTRY_PATH)
        if not registry_path.is_file():
            raise FileNotFoundError(f"Mesh registry not found: {registry_path}")
        # 使用固定锁文件，不能锁原子替换后会换 inode 的 YAML 本身。
        with registry_path.with_name(f".{registry_path.name}.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            registry = self._read_mesh_registry()
            if label in registry["meshes"]:
                raise ValueError(f"Mesh label is already registered: {label}")

            target_dir = Path(MESHES_DIR) / "objects" / label
            if target_dir.exists():
                raise FileExistsError(f"Mesh target already exists: {target_dir}")
            target_dir.mkdir(parents=True)
            target_path = target_dir / source.name
            try:
                shutil.copy2(str(source), str(target_path))
                registry["meshes"][label] = {
                    "path": target_path.relative_to(MESHES_DIR).as_posix(),
                }
                self._write_mesh_registry(registry)
            except Exception:
                # 仅清理本次新建的未注册副本，已有资产和源文件不动。
                target_path.unlink(missing_ok=True)
                target_dir.rmdir()
                raise
            return str(target_path.resolve())

    def load_mesh(
        self,
        label: str,
        mesh_path: Optional[str] = None,
        *,
        reload: bool = False,
    ) -> None:
        """加载注册项或直接加载外部路径；不复制、不写注册表、不记住路径。

        reload 显式卸载该标签后重新加载，两次服务请求
        不是原子操作，应在暂停相关推理时执行。V2 拒绝同标签不同来源；相同
        路径内容更新仍需 reload=True，普通加载不检查文件内容变化。
        """

        label = self._normalize_registry_label(label)
        if mesh_path is not None and not str(mesh_path).strip():
            raise ValueError("mesh_path must not be empty")
        self.load_meshes(
            [label], mesh_paths=None if mesh_path is None else [mesh_path], reload=reload
        )

    def load_meshes(
        self,
        labels: Sequence[str],
        clear_existing: bool = False,
        mesh_paths: Optional[Sequence[Optional[str]]] = None,
        *,
        reload: bool = False,
    ) -> None:
        """批量加载显式路径或服务端注册表标签，不复制、不注册、不记住路径。

        mesh_paths 中的空路径表示查服务端注册表。reload 仅卸载本批标签；
        clear_existing 卸载全部。所有路径/消息校验在卸载前完成。
        """

        normalized = self._normalize_labels(labels)
        if not normalized:
            raise ValueError("labels must contain at least one non-empty label")
        normalized_paths = self._normalize_mesh_paths(
            mesh_paths,
            expected_count=len(normalized),
        )
        if not any(normalized_paths):
            normalized_paths = []

        request = FdposeLoadMeshRequest()
        request.cancel = False
        request.goods = normalized
        if normalized_paths:
            if not hasattr(request, "mesh_paths"):
                raise FdposeClientError(
                    "FdposeLoadMeshRequest has no mesh_paths field; add "
                    "string[] mesh_paths to FdposeLoadMesh.srv and rebuild "
                    "naviai_manip_srvs"
                )
            request.mesh_paths = normalized_paths
        elif hasattr(request, "mesh_paths"):
            request.mesh_paths = []

        # 请求字段及本地参数验证完成后，才允许清除已有 Mesh。
        self._ensure_ready()
        if clear_existing:
            self.clear_meshes()
        elif reload:
            self.clear_meshes(normalized)

        try:
            response = self._load_mesh_proxy(request)
        except rospy.ServiceException as exc:
            raise FdposeClientError(
                f"Mesh service call failed for labels={normalized}"
            ) from exc
        if not bool(response.success):
            raise FdposeClientError(
                f"Failed to load Mesh labels: {normalized}; see server log. "
                "If a loaded Mesh source conflicts, use reload=True."
            )

    def clear_meshes(
        self,
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        """仅卸载服务端缓存，不修改磁盘资产或注册表。

        None 或空列表表示全部；非空输入中的空白/非字符串标签报错，避免误清空。
        """

        normalized = [] if labels is None else self._normalize_labels(labels)
        self._ensure_ready()
        request = FdposeLoadMeshRequest()
        request.cancel = True
        request.goods = normalized
        if hasattr(request, "mesh_paths"):
            request.mesh_paths = []
        try:
            response = self._load_mesh_proxy(request)
        except rospy.ServiceException as exc:
            target = normalized or "all"
            raise FdposeClientError(
                f"Mesh clear service call failed for labels={target}"
            ) from exc
        if not bool(response.success):
            target = normalized or "all"
            raise FdposeClientError(
                f"FDPose server failed to clear Mesh labels={target}"
            )

    @staticmethod
    def _validate_color_depth(color_bgr, depth):
        color = np.asarray(color_bgr)
        depth_array = np.asarray(depth)

        if color.ndim != 3 or color.shape[2] != 3:
            raise ValueError(f"color_bgr must have shape HxWx3, got {color.shape}")
        if color.dtype != np.uint8:
            raise ValueError(f"color_bgr must be uint8, got {color.dtype}")
        if depth_array.ndim != 2:
            raise ValueError(f"depth must have shape HxW, got {depth_array.shape}")
        if depth_array.shape != color.shape[:2]:
            raise ValueError(
                "RGB/depth size mismatch: "
                f"rgb={color.shape[:2]}, depth={depth_array.shape}"
            )

        return np.ascontiguousarray(color), np.ascontiguousarray(depth_array)

    @staticmethod
    def _depth_to_mono16(depth: np.ndarray) -> np.ndarray:
        """Convert uint depth in millimeters or float depth in meters to mono16."""

        depth = np.asarray(depth)
        if depth.dtype == np.bool_:
            raise ValueError("depth must not be boolean")

        if np.issubdtype(depth.dtype, np.floating):
            depth_value = np.nan_to_num(
                depth.astype(np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            depth_value *= 1000.0
        elif np.issubdtype(depth.dtype, np.integer):
            depth_value = depth.astype(np.float32)
        else:
            raise ValueError(f"unsupported depth dtype: {depth.dtype}")

        return np.ascontiguousarray(
            np.clip(depth_value, 0.0, 65535.0).astype(np.uint16)
        )

    @staticmethod
    def _bbox_from_mask(mask_bool: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask_bool)
        if xs.size == 0:
            raise ValueError("mask must contain foreground pixels")
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def _build_det_item(
        self,
        detection: DetectionInput,
        image_shape: Tuple[int, int],
    ):
        if not isinstance(detection, DetectionInput):
            raise TypeError(
                "detections must contain DetectionInput instances; "
                "use DetectionInput.from_yolo_item(...) for YOLO results"
            )
        if detection.mask.shape != image_shape:
            raise ValueError(
                f"Mask size mismatch for label={detection.label}: "
                f"mask={detection.mask.shape}, image={image_shape}"
            )

        mask_bool = np.asarray(detection.mask > 0, dtype=bool)
        bbox = detection.bbox or self._bbox_from_mask(mask_bool)

        item = FdposeDetItem()
        item.label = detection.label
        item.confidence = detection.confidence
        if hasattr(item, "bbox"):
            item.bbox = list(bbox)
        item.mask = self.bridge.cv2_to_imgmsg(
            mask_bool.astype(np.uint16) * 65535,
            encoding="mono16",
        )
        return item

    @staticmethod
    def _pose_msg_to_matrix(pose_msg) -> np.ndarray:
        quaternion = np.asarray(
            [
                pose_msg.orientation.x,
                pose_msg.orientation.y,
                pose_msg.orientation.z,
                pose_msg.orientation.w,
            ],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-12:
            raise FdposeClientError("FDPose returned a zero-norm quaternion")
        quaternion /= norm

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        matrix[:3, 3] = [
            float(pose_msg.position.x),
            float(pose_msg.position.y),
            float(pose_msg.position.z),
        ]
        return matrix

    @staticmethod
    def _validate_target_camera(T_target_camera) -> Optional[np.ndarray]:
        if T_target_camera is None:
            return None

        matrix = np.asarray(T_target_camera, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"T_target_camera must be 4x4, got {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("T_target_camera contains NaN/Inf")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError(
                "T_target_camera last row must be [0, 0, 0, 1]"
            )
        return matrix

    def estimate(
        self,
        color_bgr: np.ndarray,
        depth: np.ndarray,
        detections: Sequence[DetectionInput],
        T_target_camera: Optional[np.ndarray] = None,
    ):
        """Estimate object poses, optionally expressed in a target frame.

        When ``T_target_camera`` is provided, each service result is converted
        with ``T_target_object = T_target_camera @ T_camera_object``.
        """

        color, depth_array = self._validate_color_depth(color_bgr, depth)
        T_target_camera = self._validate_target_camera(T_target_camera)
        detections = list(detections)
        if not detections:
            raise ValueError("detections must not be empty")

        request = PoseEstRequest()
        request.color_image = self.bridge.cv2_to_imgmsg(color, encoding="bgr8")
        request.depth_image = self.bridge.cv2_to_imgmsg(
            self._depth_to_mono16(depth_array),
            encoding="mono16",
        )
        request.items = [
            self._build_det_item(detection, color.shape[:2])
            for detection in detections
        ]

        self._ensure_ready()
        try:
            response = self._pose_proxy(request)
        except rospy.ServiceException as exc:
            raise FdposeClientError("Pose estimation service call failed") from exc

        results = []
        for obj_pose in response.obj_poses:
            T_object = self._pose_msg_to_matrix(obj_pose.pose)
            if T_target_camera is not None:
                T_object = T_target_camera @ T_object
            results.append(
                PoseResult(
                    label=str(obj_pose.label),
                    T_object=T_object,
                )
            )
        return results

    def estimate_one(
        self,
        label: str,
        color_bgr: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        confidence: float = 1.0,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        T_target_camera: Optional[np.ndarray] = None,
    ) -> PoseResult:
        """Estimate one object, optionally expressed in a target frame."""

        detection = DetectionInput(
            label=label,
            mask=mask,
            confidence=confidence,
            bbox=bbox,
        )
        results = self.estimate(
            color_bgr,
            depth,
            [detection],
            T_target_camera=T_target_camera,
        )

        expected_label = str(label).strip()
        for result in results:
            if result.label == expected_label:
                return result

        raise FdposeClientError(
            f"FDPose returned no pose for label={expected_label!r}; "
            f"returned={[result.label for result in results]}"
        )


__all__ = [
    "DEFAULT_MESH_SERVICE",
    "DEFAULT_POSE_SERVICE",
    "DetectionInput",
    "FdposeClient",
    "FdposeClientError",
    "PoseResult",
]
