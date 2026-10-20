#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同步抬升、三步转腰，以及多层预放置平移。"""

from __future__ import annotations

from .observability.timing import phase, state as timing_state
from .common.runtime import input
from . import stage1
from .common import vision

from dataclasses import dataclass
import math

import numpy as np
import rospy

from naviai_controller import ArmGroup

from .common.state import vector, quaternion_error_rad, compute_grasp_transforms, compute_place_and_pre, load_state, estimate_tray_from_tcp, require_grasp_transforms, require_pose, update_state


WAIST_SPEED = 0.5
WAIST_ACCELERATION = 0.1
WAIST_STEPS = (("抬腰", 3, 0.2), ("转腰", 2, 1.5), ("下腰", 3, 0.5))


def lift_tray(robot, height_m, speed=0.01, acceleration=0.02):
    """同步沿BASE +Z抬升双臂；完成判断交给底层同步服务。"""
    if not math.isfinite(height_m) or not 0 < height_m <= 0.15:
        raise ValueError("抬升高度必须在(0,0.15]m内")
    try:
        if rospy.is_shutdown():
            raise RuntimeError("ROS已关闭")
        robot.stop()
        robot.clear_servo_params()
        if not robot.movel_relative_base(
            [0.0, 0.0, height_m, 0.0, 0.0, height_m],
            ArmGroup.DUAL,
            v=speed,
            acc=acceleration,
            is_async=False,
        ):
            raise RuntimeError("双臂同步抬升失败")
        if rospy.is_shutdown():
            raise RuntimeError("ROS已关闭，抬升不得标记成功")
    except BaseException:
        robot.stop()
        raise


def turn_waist(robot, speed=WAIST_SPEED, acceleration=WAIST_ACCELERATION):
    """逐步同步MoveJ，每步只修改一个腰关节，不订阅或检查腕力。"""
    try:
        robot.stop()
        robot.clear_servo_params()
        for name, index, value in WAIST_STEPS:
            input("{}：q{}={}rad，Enter执行：".format(name, index + 1, value))
            with phase("Stage2", name):
                if rospy.is_shutdown():
                    raise RuntimeError("ROS已关闭")
                target = vector(robot.get_joints(ArmGroup.WAIST), 4, "waist joints")
                target[index] = value
                if not robot.movej(
                    target.tolist(),
                    ArmGroup.WAIST,
                    v=speed,
                    acc=acceleration,
                    is_async=False,
                ):
                    raise RuntimeError(name + "失败")
                if rospy.is_shutdown():
                    raise RuntimeError("ROS已关闭，禁止下一步转腰")
    except BaseException:
        robot.stop()
        raise


@dataclass
class HeldTrayState:
    """同次定位得到的托盘/TCP/抓取变换，反算结果供预放置直接复用。"""

    pose: list
    tcp: dict
    grasp: dict
    source: str
    estimate: object = None


def capture_held_tray(robot, args, vision_args, current, *, use_tcp):
    """按入口指定的TCP或视觉方式定位，视觉期间仍检查TCP静止。"""
    estimate = None
    if use_tcp:
        timing_state("手中tray双TCP反算")
        tcp = stage1.read_tcp_pair(robot)
        grasp = require_grasp_transforms(current)
        estimate = estimate_tray_from_tcp(tcp["left"], tcp["right"], grasp)
        pose, source = estimate[0], "dual_tcp"
    else:
        timing_state("手中tray视觉识别")
        before = stage1.read_tcp_pair(robot)
        pose = vision.estimate_tray_pose(vision_args, "当前手中 T_base_tray")
        tcp = stage1.read_tcp_pair(robot)
        for side in ("left", "right"):
            if np.linalg.norm(
                np.asarray(tcp[side][:3]) - before[side][:3]
            ) > args.tcp_tolerance or quaternion_error_rad(
                np.asarray(tcp[side][3:]), np.asarray(before[side][3:])
            ) > math.radians(
                args.orientation_tolerance_deg
            ):
                raise RuntimeError("视觉采集期间TCP移动，拒绝建立抓取变换")
        grasp = compute_grasp_transforms(pose, tcp["left"], tcp["right"])
        source = vision_args.vision_backend
    return HeldTrayState(pose, tcp, grasp, source, estimate)


@dataclass
class PreplacementPlan:
    """一次TCP快照生成的平移计划；双臂姿态始终保持不变。"""

    held_pose: list
    place_pose: list
    pre_pose: list
    xy_targets: dict
    pre_targets: dict
    skip_z: bool
    target_layer_index: int


def plan_preplacement(args, tcp_poses, *, held_estimate=None):
    """复用同次TCP反算结果，计算P_place/P_pre及先XY后Z的双TCP目标。"""
    saved = load_state(args.state_file)
    if not saved.get("stage1_complete"):
        raise RuntimeError("尚未建立抓取变换")
    layer = saved.get("layer_index")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 1:
        raise ValueError("缺少有效上一层层号")
    grasp = require_grasp_transforms(saved)
    held, disagreement = held_estimate if held_estimate is not None else estimate_tray_from_tcp(
        tcp_poses["left"], tcp_poses["right"], grasp
    )
    if disagreement["position_m"] > args.relative_position_limit:
        raise RuntimeError("双TCP反算tray中心不一致")
    place, pre = compute_place_and_pre(
        require_pose(saved, "top_tray_pose"), args.stack_pitch, args.preplace_clearance
    )
    # state的融合位姿为水平姿态；目标元数据沿用抓取姿态，运动只改XYZ。
    place[3:] = pre[3:] = require_pose(saved, "grasp_tray_pose")[3:]
    delta = np.asarray(pre[:3]) - held[:3]
    skip_z = bool(delta[2] > 0)
    if (
        max(float(np.linalg.norm(delta[:2])), 0.0 if skip_z else abs(delta[2]))
        > args.max_preplace_translation
    ):
        raise ValueError("到P_pre的分段平移超过行程上限")
    xy_targets, pre_targets = {}, {}
    for side in ("left", "right"):
        xy = vector(tcp_poses[side], 7, side + " TCP")
        xy[:2] += delta[:2]
        final = xy.copy()
        if not skip_z:
            final[2] += delta[2]
        xy_targets[side], pre_targets[side] = xy.tolist(), final.tolist()
    return PreplacementPlan(
        held, place, pre, xy_targets, pre_targets, skip_z, layer + 1
    )


def move_to_pre(robot, args, plan, xy_offset=(0.0, 0.0)):
    """同步MoveL依次执行XY、非低位Z和可选共同XY偏置，不采集力或轮询到位。"""
    offset = vector(xy_offset, 2, "XY偏置")
    if np.linalg.norm(offset) > args.max_preplace_translation:
        raise ValueError("XY偏置超过行程上限")
    offset_targets = {}
    for side, values in plan.pre_targets.items():
        target = np.asarray(values).copy()
        target[:2] += offset
        offset_targets[side] = target.tolist()
    dz = plan.held_pose[2] - plan.pre_pose[2]
    update_state(
        args.state_file,
        stage2_complete=False,
        stage3_complete=False,
        stage3_release_and_retreat_complete=False,
        place_tray_target=plan.place_pose,
        preplace_tray_target=plan.pre_pose,
        preplace_tcp_targets=plan.pre_targets,
        target_layer_index=plan.target_layer_index,
        stack_pitch=float(args.stack_pitch),
        preplace_clearance=float(args.preplace_clearance),
        test_stage3_held_minus_preplace_z_mm=1000 * dz,
        test_stage3_below_preplace_mm=max(0.0, -1000 * dz),
        test_stage3_preplace_z_motion_skipped=plan.skip_z,
        test_stage3_preplace_xy_reached=False,
        test_stage3_preplace_target_reached=False,
        test_stage3_xy_offset_m=offset.tolist(),
        test_stage3_xy_offset_complete=False,
    )
    input(
        "低于P_pre：保持Z，仅对齐XY，Enter执行："
        if plan.skip_z
        else "确认先XY、后Z移动到P_pre的路径，Enter执行："
    )
    segments = [("P_pre XY", plan.xy_targets)]
    if not plan.skip_z:
        segments.append(("P_pre Z", plan.pre_targets))
    if np.any(offset):
        segments.append(("P_pre后XY偏置", offset_targets))
    try:
        robot.stop()
        robot.clear_servo_params()
        for label, targets in segments:
            if rospy.is_shutdown():
                raise RuntimeError("ROS已关闭，禁止继续P_pre运动")
            timing_state(label)
            if not robot.movel(
                targets["left"] + targets["right"],
                ArmGroup.DUAL,
                v=args.preplace_velocity,
                acc=args.preplace_acceleration,
                is_async=False,
            ):
                raise RuntimeError(label + "同步MoveL失败")
            if rospy.is_shutdown():
                raise RuntimeError("ROS已关闭，P_pre不得标记成功")
    except BaseException:
        robot.stop()
        raise
    update_state(
        args.state_file,
        stage2_complete=True,
        test_stage3_preplace_xy_reached=True,
        test_stage3_preplace_target_reached=not plan.skip_z,
        test_stage3_xy_offset_complete=True,
        test_stage3_xy_offset_tcp_targets=offset_targets,
    )
