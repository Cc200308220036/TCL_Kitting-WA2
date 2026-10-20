"""Read an existing local mask without ROS, model inference or debug output."""

from pathlib import Path
from typing import List, Sequence, Union

import cv2

from naviai_yolo.postprocess import SegDetection, postprocess_mask_file


def load_mask_file(
    mask_path: Union[str, Path], image_shape: Sequence[int], label: str
) -> List[SegDetection]:
    """Load one image-aligned mask; return one detection or [] for a black mask.

    image_shape is the corresponding color image's (height, width[, channels]).
    All nonzero pixels belong to one instance, with the supplied label and a
    fixed confidence of 1.0. Invalid files, channels or sizes raise ValueError.
    Relative paths are resolved against the caller's working directory.
    """
    path = Path(mask_path).expanduser()
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f'Unable to read mask file: {path}')
    return postprocess_mask_file(mask, image_shape, label)
