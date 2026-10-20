"""Decode RGB-D inputs without silently changing depth units."""

import numpy as np


def decode_color(bridge, message):
    color = bridge.imgmsg_to_cv2(message, 'bgr8')
    if (color is None or color.size == 0 or color.ndim != 3
            or color.shape[2] != 3 or color.dtype != np.uint8):
        raise ValueError('color must decode to a nonempty HxWx3 BGR uint8 image')
    return color


def decode_depth(bridge, message, color_shape):
    encoding = getattr(message, 'encoding', None)
    dtype = {'16UC1': np.uint16, '32FC1': np.float32}.get(encoding)
    if dtype is None:
        raise ValueError('depth encoding must be 16UC1 (mm) or 32FC1 (m)')
    depth = bridge.imgmsg_to_cv2(message, 'passthrough')
    if depth is None or depth.shape != tuple(color_shape[:2]) or depth.dtype != dtype:
        raise ValueError('depth must match the color image size and its declared encoding')
    return depth
