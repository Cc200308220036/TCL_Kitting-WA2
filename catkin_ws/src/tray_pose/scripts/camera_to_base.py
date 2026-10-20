#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相机系位姿转到 BASE：T_base_object = T_base_head @ T_head_camera @ T_camera_object。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rospy
import tf
from scipy.spatial.transform import Rotation


DEFAULT_CAMERA_TO_HEAD = Path(
    "/root/catkin_ws/src/Fdpose/config/camera/extrinsic.npy"
)
BASE_FRAME = "BASE"
HEAD_FRAME = "HEAD"
TF_TIMEOUT = 2.0


def trans_quat_to_matrix(translation, quaternion):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    T[0, 3] = float(translation[0])
    T[1, 3] = float(translation[1])
    T[2, 3] = float(translation[2])
    return T


class CameraToBase:
    """查 TF BASE<-HEAD，再乘 HEAD<-camera 外参。"""

    def __init__(
        self,
        camera_to_head_path=DEFAULT_CAMERA_TO_HEAD,
        base_frame=BASE_FRAME,
        head_frame=HEAD_FRAME,
        tf_timeout=TF_TIMEOUT,
    ):
        T = np.asarray(np.load(str(camera_to_head_path)), dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(
                "T_head_camera must be 4x4, got %s from %s"
                % (T.shape, camera_to_head_path)
            )
        self.T_head_camera = T
        self.base_frame = base_frame
        self.head_frame = head_frame
        self.tf_timeout = float(tf_timeout)
        self._listener = tf.TransformListener()

    def lookup_T_base_head(self):
        self._listener.waitForTransform(
            self.base_frame,
            self.head_frame,
            rospy.Time(0),
            rospy.Duration(self.tf_timeout),
        )
        trans, quat = self._listener.lookupTransform(
            self.base_frame,
            self.head_frame,
            rospy.Time(0),
        )
        return trans_quat_to_matrix(trans, quat)

    def T_base_camera(self):
        """4x4, BASE <- camera。"""
        return self.lookup_T_base_head() @ self.T_head_camera

    def transform(self, T_camera_object):
        """T_camera_object -> T_base_object。"""
        T = np.asarray(T_camera_object, dtype=np.float64).reshape(4, 4)
        return self.T_base_camera() @ T
