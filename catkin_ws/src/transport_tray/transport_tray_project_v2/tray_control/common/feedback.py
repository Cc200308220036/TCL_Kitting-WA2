"""Stage1/Stage3共用的腕力缓存、滤波和TCP反馈读取。"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import WrenchStamped
from naviai_controller import ArmGroup
from .state import pose7

ARMS = (ArmGroup.LEFT, ArmGroup.RIGHT)


class WrenchBuffer:
    """保存双腕最新六维力和本机接收时间。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: Dict[ArmGroup, Optional[np.ndarray]] = {arm: None for arm in ARMS}
        self._stamps: Dict[ArmGroup, Optional[float]] = {arm: None for arm in ARMS}
        self._sequences = {arm: 0 for arm in ARMS}
        self._subscribers = [
            rospy.Subscriber(
                "/wrist_force_control/left_arm_compensated_force",
                WrenchStamped,
                lambda msg: self._update(ArmGroup.LEFT, msg),
                queue_size=1,
                tcp_nodelay=True,
            ),
            rospy.Subscriber(
                "/wrist_force_control/right_arm_compensated_force",
                WrenchStamped,
                lambda msg: self._update(ArmGroup.RIGHT, msg),
                queue_size=1,
                tcp_nodelay=True,
            ),
        ]

    def _update(self, arm: ArmGroup, message: WrenchStamped) -> None:
        wrench = message.wrench
        value = np.array(
            [
                wrench.force.x,
                wrench.force.y,
                wrench.force.z,
                wrench.torque.x,
                wrench.torque.y,
                wrench.torque.z,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(value)):
            return
        with self._lock:
            self._values[arm] = value
            self._stamps[arm] = time.monotonic()
            self._sequences[arm] += 1

    def snapshot(
        self, arm: ArmGroup
    ) -> Tuple[Optional[np.ndarray], Optional[float], int]:
        with self._lock:
            value = self._values[arm]
            return (
                None if value is None else value.copy(),
                self._stamps[arm],
                self._sequences[arm],
            )


class ScalarLowPass:
    def __init__(self, cutoff_hz: float) -> None:
        self.cutoff_hz = float(cutoff_hz)
        self.initialized = False
        self.value = 0.0

    def step(self, sample: float, dt: float) -> float:
        if self.cutoff_hz <= 0.0 or not self.initialized:
            self.value = float(sample)
            self.initialized = True
            return self.value
        tau = 1.0 / (2.0 * math.pi * self.cutoff_hz)
        alpha = float(np.clip(dt / (tau + dt), 0.0, 1.0))
        self.value += alpha * (float(sample) - self.value)
        return self.value


@dataclass(frozen=True)
class WrenchSample:
    raw: np.ndarray
    support: float


def read_valid_tcp(robot, arm, stale_timeout):
    """检查TCP/机器人状态反馈，阶段自身继续检查跟踪误差。"""
    pose_raw = robot.get_tcp_rt(arm)
    tcp_age = robot.get_tcp_age(arm)
    state_age = robot.get_uplimb_state_age()
    if pose_raw is None or tcp_age is None or tcp_age > stale_timeout:
        raise RuntimeError("{} TCP数据超时".format(arm.name))
    if state_age is None or state_age > stale_timeout:
        raise RuntimeError("UplimbState数据超时")
    if robot.get_is_singular(arm):
        raise RuntimeError("{}臂进入奇异位形".format(arm.name))
    if robot.get_cmd_num() == 15:
        raise RuntimeError("机器人进入PROTECTED状态(cmd_num=15)")
    return pose7(pose_raw, "{} TCP".format(arm.name))


class ThresholdLatch:
    """带迟滞和双向时间去抖的阈值状态；不用于单层一次性接触锁存。"""

    def __init__(self, enter_threshold, exit_threshold, enter_seconds, exit_seconds):
        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = float(exit_threshold)
        self.enter_seconds = float(enter_seconds)
        self.exit_seconds = float(exit_seconds)
        self.active = False
        self.above_since = None
        self.below_since = None

    def update(self, value, now):
        if self.active:
            if value < self.exit_threshold:
                if self.below_since is None:
                    self.below_since = now
                elif now - self.below_since >= self.exit_seconds:
                    self.active = False
                    self.above_since = self.below_since = None
            else:
                self.below_since = None
        elif value >= self.enter_threshold:
            if self.above_since is None:
                self.above_since = now
            elif now - self.above_since >= self.enter_seconds:
                self.active = True
                self.below_since = None
        else:
            self.above_since = None
        return self.active


def wait_feedback_ready(robot, wrenches, stale_timeout, timeout_s=8.0, *, error_message):
    deadline = time.monotonic() + timeout_s
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        now = time.monotonic()
        ready = True
        poses: Dict[ArmGroup, np.ndarray] = {}
        for arm in ARMS:
            pose_raw = robot.get_tcp_rt(arm)
            tcp_age = robot.get_tcp_age(arm)
            wrench, stamp, _ = wrenches.snapshot(arm)
            singular = robot.get_is_singular(arm)
            if (
                pose_raw is None
                or tcp_age is None
                or tcp_age > stale_timeout
                or wrench is None
                or stamp is None
                or now - stamp > stale_timeout
                or singular is None
            ):
                ready = False
                break
            if singular:
                raise RuntimeError("{}臂处于奇异位形".format(arm.name))
            poses[arm] = pose7(pose_raw, "{} TCP".format(arm.name))
        state_age = robot.get_uplimb_state_age()
        if state_age is None or state_age > stale_timeout:
            ready = False
        if robot.get_cmd_num() == 15:
            raise RuntimeError("机器人处于PROTECTED状态(cmd_num=15)")
        if ready:
            return poses
        rospy.sleep(0.02)
    raise RuntimeError(error_message)


def collect_wrench_baseline(wrenches, seconds, *, label, stale_timeout=None):
    """按消息序号去重，返回各侧(均值、标准差、样本数)。None保持不检查新鲜度。"""
    samples = {arm: [] for arm in ARMS}
    last_sequence = {arm: -1 for arm in ARMS}
    deadline = time.monotonic() + seconds
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        for arm in ARMS:
            value, stamp, sequence = wrenches.snapshot(arm)
            fresh = stale_timeout is None or (
                value is not None
                and stamp is not None
                and time.monotonic() - stamp <= stale_timeout
            )
            if value is not None and fresh and sequence != last_sequence[arm]:
                samples[arm].append(value)
                last_sequence[arm] = sequence
        rospy.sleep(0.002)
    result = {}
    for arm, values in samples.items():
        if len(values) < 20:
            raise RuntimeError("{}腕{}样本不足".format(arm.name, label))
        data = np.stack(values, axis=0)
        result[arm] = (np.mean(data, axis=0), np.std(data, axis=0), len(values))
    return result
