"""YOLO 实例分割结果的公共后处理。"""

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


DEFAULT_MASK_THRESHOLD = 0.5
MONO16_FOREGROUND = np.uint16(65535)
ClassNames = Union[Mapping[int, str], Sequence[str]]


@dataclass(frozen=True)
class SegDetection:
    """与 ROS 消息无关的单个实例分割结果。"""

    source_index: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray


def _scale_mask_to_image(mask: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    """移除推理时的 letterbox padding，并恢复到原始图像尺寸。"""
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")

    target_h, target_w = int(image_shape[0]), int(image_shape[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError("image height and width must be positive")

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[:, :, 0]

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask.shape}")

    source_h, source_w = mask.shape
    if (source_h, source_w) == (target_h, target_w):
        return mask

    gain = min(source_h / target_h, source_w / target_w)
    # Mirror LetterBox: round the resized dimensions before splitting padding.
    resized_w, resized_h = round(target_w * gain), round(target_h * gain)
    pad_w = (source_w - resized_w) / 2
    pad_h = (source_h - resized_h) / 2
    top, left = round(pad_h - 0.1), round(pad_w - 0.1)
    bottom, right = top + resized_h, left + resized_w
    cropped = mask[top:bottom, left:right]
    if cropped.size == 0:
        raise ValueError(
            f"mask crop is empty: mask_shape={mask.shape}, image_shape={(target_h, target_w)}"
        )

    if cropped.dtype == np.bool_:
        cropped = cropped.astype(np.uint8)
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def postprocess_segmentation(
    boxes: Optional[np.ndarray],
    masks: Optional[np.ndarray],
    names: Optional[ClassNames],
    image_shape: Sequence[int],
    target_labels: Optional[Iterable[str]] = None,
    mask_threshold: float = DEFAULT_MASK_THRESHOLD,
) -> List[SegDetection]:
    """生成原图尺寸的二值分割结果；target_labels 为 None 或空时不限制类别。"""
    if boxes is None:
        return []

    boxes_array = np.asarray(boxes)
    if boxes_array.size == 0:
        return []
    if boxes_array.ndim != 2 or boxes_array.shape[1] < 6:
        raise ValueError(f"boxes must have shape (N, >=6), got {boxes_array.shape}")
    if masks is None or len(masks) < len(boxes_array):
        raise ValueError("masks must contain one mask for every box")
    if names is None:
        raise ValueError("class names are missing")

    label_filter = None if target_labels is None else {str(label) for label in target_labels}
    detections = []

    for index, box in enumerate(boxes_array):
        x1, y1, x2, y2, confidence, class_value = map(float, box[:6])
        class_id = int(class_value)
        try:
            label = str(names[class_id])
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"unknown class id: {class_id}") from exc

        if label_filter and label not in label_filter:
            continue

        mask = _scale_mask_to_image(np.asarray(masks[index]), image_shape)
        if np.issubdtype(mask.dtype, np.floating):
            mask_binary = (mask > mask_threshold).astype(np.uint8)
        else:
            mask_binary = (mask > 0).astype(np.uint8)

        if not mask_binary.any():
            continue

        detections.append(
            SegDetection(
                source_index=index,
                label=label,
                confidence=float(confidence),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                mask=np.ascontiguousarray(mask_binary),
            )
        )

    return detections


def postprocess_mask_file(
    mask: np.ndarray, image_shape: Sequence[int], label: str
) -> List[SegDetection]:
    """Validate an image-aligned static mask and derive its bounding box."""
    if mask.ndim != 2:
        raise ValueError("mask file must be a single-channel image")
    if mask.shape != tuple(image_shape[:2]):
        raise ValueError(
            f"mask size {mask.shape} does not match color image {tuple(image_shape[:2])}"
        )
    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return []
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return [SegDetection(0, str(label), 1.0, bbox, binary)]


def mask_to_mono16(mask_binary: np.ndarray) -> np.ndarray:
    """将 0/1 Mask 转换为 ROS mono16 使用的 0/65535。"""
    return np.where(mask_binary > 0, MONO16_FOREGROUND, np.uint16(0)).astype(np.uint16)


def mask_to_uint8(mask_binary: np.ndarray) -> np.ndarray:
    """将 0/1 Mask 转换为保存 PNG 使用的 0/255。"""
    return np.where(mask_binary > 0, np.uint8(255), np.uint8(0)).astype(np.uint8)
