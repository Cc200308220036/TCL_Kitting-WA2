#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jodell RAG 左右夹爪的进程内 Python 适配器。"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Dict

import rospy
import rosservice


LEFT = "left"
RIGHT = "right"
SIDES = (LEFT, RIGHT)


@dataclass(frozen=True)
class JodellGripperConfig:
    """夹爪服务与开合参数；位置单位 mm，力和速度单位为百分比。"""

    left_service: str = "/zj_humanoid/jodell_rag_left/grasp"
    right_service: str = "/zj_humanoid/jodell_rag_right/grasp"
    profile: int = 0
    open_position_mm: float = 100.0
    close_position_mm: float = 17.0
    force_percent: float = 0.0
    speed_percent: float = 0.0
    wait_timeout: float = 5.0

    def validate(self) -> None:
        if not self.left_service or not self.right_service:
            raise ValueError("左右夹爪服务名不能为空")
        if int(self.profile) not in (0, 1, 2):
            raise ValueError("profile 只能是 0(SOFT)、1(NORMAL) 或 2(FIRM)")
        for name in ("open_position_mm", "close_position_mm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise ValueError("{} 必须在 [0, 100] mm".format(name))
        for name in ("force_percent", "speed_percent"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise ValueError("{} 必须在 [0, 100]".format(name))
        if not math.isfinite(self.wait_timeout) or self.wait_timeout <= 0.0:
            raise ValueError("wait_timeout 必须是有限正数")


class JodellDualGripper:
    """通过 ``jodell_rag_msgs/grasp`` 服务控制左右夹爪。"""

    def __init__(self, config: JodellGripperConfig) -> None:
        config.validate()
        self.config = config
        self._services: Dict[str, str] = {
            LEFT: config.left_service,
            RIGHT: config.right_service,
        }

    def command(self, side: str, target_position_mm: float):
        if side not in SIDES:
            raise ValueError("side 必须是 left 或 right")
        service_name = self._services[side]
        rospy.wait_for_service(service_name, timeout=self.config.wait_timeout)
        service_class = rosservice.get_service_class_by_name(service_name)
        if service_class is None:
            raise RuntimeError("无法解析夹爪服务类型: {}".format(service_name))
        proxy = rospy.ServiceProxy(service_name, service_class, persistent=False)
        response = proxy(
            profile=int(self.config.profile),
            target_position_mm=float(target_position_mm),
            force_percent=float(self.config.force_percent),
            speed_percent=float(self.config.speed_percent),
        )
        if not bool(response.success):
            raise RuntimeError(
                "{}夹爪服务失败: {}".format(side.upper(), response.message)
            )
        rospy.loginfo(
            "%s gripper accepted: target=%.1fmm final=%.1fmm "
            "force_feedback=%.1f%% speed_feedback=%.1f%% object=%s message=%s",
            side.upper(),
            float(response.applied_target_position_mm),
            float(response.final_position_mm),
            float(response.force_feedback_percent),
            float(response.speed_feedback_percent),
            bool(response.object_detected),
            response.message,
        )
        return response

    def open(self, side: str):
        return self.command(side, self.config.open_position_mm)

    def close(self, side: str):
        return self.command(side, self.config.close_position_mm)


__all__ = [
    "JodellDualGripper",
    "JodellGripperConfig",
    "LEFT",
    "RIGHT",
]


class DualGripperWorker:
    """后台并行开/闭双爪；闭爪允许被物体阻挡，张爪校验最终开度。"""

    def __init__(self, grippers, *, action, target_position_mm=None):
        if action not in ("open", "close"):
            raise ValueError("夹爪动作必须为open或close")
        self.grippers = grippers
        self.action = action
        self.target_position_mm = target_position_mm
        self._lock = threading.Lock()
        self._completed = {side: False for side in SIDES}
        self._errors = []
        self._threads = []

    def start(self):
        if self._threads:
            raise RuntimeError("夹爪任务不能重复启动")
        for side in SIDES:
            thread = threading.Thread(target=self._call, args=(side,),
                                      name="{}_{}_gripper".format(self.action, side), daemon=True)
            self._threads.append(thread)
            thread.start()

    def _call(self, side):
        try:
            if self.action == "open":
                response = self.grippers.open(side)
                position = float(response.final_position_mm)
                target = self.grippers.config.open_position_mm
                if not math.isfinite(position) or abs(position - target) > 2.0:
                    raise RuntimeError(
                        "{}夹爪张开后位置{:.1f} mm未到目标{:.1f} mm的2 mm范围内".format(
                            side.upper(), position, target))
            elif self.target_position_mm is None:
                self.grippers.close(side)
            else:
                self.grippers.command(side, self.target_position_mm)
            with self._lock:
                self._completed[side] = True
        except Exception as exc:
            with self._lock:
                self._errors.append("{}: {}".format(side.upper(), exc))

    def status(self):
        with self._lock:
            return all(self._completed.values()), "; ".join(self._errors) if self._errors else None

    def join(self, timeout_s=0.2):
        for thread in self._threads:
            thread.join(timeout=timeout_s)


def poll_gripper_completion(worker, now, accepted_at, settle_seconds, *, error_prefix):
    """返回首次完成时间与稳定完成标志；每次只查询一次，不阻塞控制循环。"""
    accepted, error = worker.status()
    if error is not None:
        raise RuntimeError(error_prefix + error)
    if accepted and accepted_at is None:
        accepted_at = now
    return accepted_at, accepted_at is not None and now - accepted_at >= settle_seconds
