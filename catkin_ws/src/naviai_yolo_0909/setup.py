#!/usr/bin/env python3
"""Install the Python library and its bundled YOLO runtime through catkin."""

from setuptools import find_packages, setup
from catkin_pkg.python_setup import generate_distutils_setup


setup(**generate_distutils_setup(
    packages=find_packages('src', exclude=['naviai_yolo.yolo11_seg.tests*']),
    package_dir={'': 'src'},
    package_data={
        'naviai_yolo.yolo11_seg': ['LICENSE'],
        'naviai_yolo.yolo11_seg.ultralytics': ['**/*.yaml', 'assets/*.jpg'],
    },
))
