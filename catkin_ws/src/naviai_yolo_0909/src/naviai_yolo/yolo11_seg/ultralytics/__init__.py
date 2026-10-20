# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.3.21"

import os
import sys

# ultralytics 包的父目录（.../yolo11_seg），保证 `from ultralytics...` 能解析
yolo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if yolo_path not in sys.path:
    sys.path.insert(0, yolo_path)


# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
)
