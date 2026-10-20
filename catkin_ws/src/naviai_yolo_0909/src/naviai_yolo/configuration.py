"""Locate configuration, model resources and writable outputs in source/install spaces."""

import math
import os
from pathlib import Path

import yaml


PACKAGE_NAME = 'naviai_manip_seg'


def validate_timeout(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f'{name} must be a finite positive number of seconds')


def validate_model_config(config):
    path = config.get('seg_model_path')
    if not isinstance(path, str) or not path.strip():
        raise ValueError('seg_model_path must be a nonempty path string')
    conf = config.get('conf')
    if (isinstance(conf, bool) or not isinstance(conf, (int, float))
            or not math.isfinite(conf) or not 0 <= conf <= 1):
        raise ValueError('conf must be a finite number between 0 and 1')
    imgsz = config.get('imgsz')
    sizes = imgsz if isinstance(imgsz, list) else [imgsz]
    if ((isinstance(imgsz, list) and len(imgsz) != 2)
            or any(type(size) is not int or size <= 0 for size in sizes)):
        raise ValueError('imgsz must be a positive integer or [height, width] of positive integers')


def package_root():
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / 'package.xml').is_file():
        return source_root

    import rospkg
    return Path(rospkg.RosPack().get_path(PACKAGE_NAME))


def resolve_package_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else package_root() / path


def resolve_output_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root = package_root()
    if (root / 'src/naviai_yolo').is_dir():
        return root / path
    ros_home = Path(os.environ.get('ROS_HOME', '~/.ros')).expanduser()
    return ros_home / PACKAGE_NAME / path


def load_config(filename, config_path=None):
    path = resolve_package_path(config_path) if config_path else package_root() / 'config' / filename
    with path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f'Configuration must be a mapping: {path}')
    return config
