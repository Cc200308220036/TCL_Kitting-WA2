"""执行实际Yaw方法，离线验证持物基线Fy条件和原下沉防抖。"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from stage3_probe import search_controller, bind_methods
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation


class YawFyTests(TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "tray_control/stage3_multi.py"
        method = next(n for n in ast.walk(ast.parse(path.read_text()))
                      if isinstance(n, ast.FunctionDef) and n.name == "advance_lowest_yaw_search")
        self.worker = Mock()
        self.mark = Mock()
        scope = dict(math=math, np=np, Rotation=Rotation, ARMS=("left", "right"),
                     ArmGroup=SimpleNamespace(LEFT="left", RIGHT="right"),
                     DualGripperWorker=Mock(return_value=self.worker), trace_state=Mock(), trace_mark=self.mark,
                     rospy=SimpleNamespace(logwarn=Mock(), loginfo=Mock()),
                     yaw_from_pose7=lambda pose: 0., wrap_angle=lambda angle: angle,
                     estimate_tray_fast=lambda *args: ([0., 0., 0., 0., 0., 0., 1.], None))
        release = next(n for n in ast.walk(ast.parse(path.read_text()))
                       if isinstance(n, ast.FunctionDef) and n.name == "begin_release")
        exec(compile(ast.fix_missing_locations(ast.Module([release, method], [])), str(path), "exec"), scope)
        self.advance = scope["advance_lowest_yaw_search"]
        poses = {side: np.array([0., 0., 0., 0., 0., 0., 1.]) for side in ("left", "right")}
        self.controller = search_controller(
            args=SimpleNamespace(radius_yaw_timeout=50., slot_min_descent=.005,
                                 radius_yaw_extra_descent=.0005, yaw_descent_debounce=.2,
                                 radius_yaw_range_deg=3., radius_yaw_speed_deg_s=.3),
            yaw_search=SimpleNamespace(started=0., capture_since=None, start_z=.01,
                segment=0, offset=0., center=np.zeros(3), settled_since=None),
            return_motion=SimpleNamespace(commands={s: p.copy() for s, p in poses.items()},
                settled_since=None, z_axis=None, anchor_offsets=None),
            history=SimpleNamespace(drop=.001, offset=np.zeros(2), poses=poses, tray_pose=poses["left"]),
            grasp_components={},
            baseline={s: np.array([0., 4., 0., 0., 0., 0.]) for s in poses},
            stage_name="YAW_SEARCH_LOWEST", grippers=object(), slot_yaw_fy_captured=False,
            start_unconfirmed_release=Mock())
        bind_methods(self.controller, ast.parse(path.read_text()), scope,
                     ['transition', 'begin_release', 'check_yaw_capture', 'advance_yaw_command',
                      'calculate_search_observation', 'finish_search_and_release'])
        self.poses = poses

    def cycle(self, delta=0., side="left", now=1., z=.01):
        raw = {s: baseline.copy() for s, baseline in self.controller.baseline.items()}
        raw[side][1] += delta
        self.advance(self.controller, self.poses, [0., 0., z], .01, now, .02, raw)

    def test_either_side_and_sign_triggers_immediately_without_descent(self):
        for side in ("left", "right"):
            for delta in (-2.51, 2.51):
                with self.subTest(side=side, delta=delta):
                    self.setUp()
                    self.cycle(delta, side)
                    self.assertEqual(self.controller.stage_name, "OPENING")
                    self.assertTrue(self.controller.slot_yaw_fy_captured)
                    self.worker.start.assert_called_once()
                    self.assertEqual(self.mark.call_args.args[0], "lowest_yaw_fy_capture")

    def test_threshold_is_strict_and_raw_force_alone_does_not_trigger(self):
        for delta in (0., 2.5, -2.5):
            with self.subTest(delta=delta):
                self.setUp()
                self.cycle(delta)
                self.assertEqual(self.controller.stage_name, "YAW_SEARCH_LOWEST")
                self.worker.start.assert_not_called()

    def test_original_descent_debounce_keeps_timer_until_capture(self):
        self.cycle(now=1., z=.004)
        self.cycle(now=1.1, z=.004)
        self.assertEqual(self.controller.yaw_search.capture_since, 1.)
        self.worker.start.assert_not_called()
        self.cycle(now=1.3, z=.004)
        self.assertEqual(self.controller.stage_name, "OPENING")
        self.assertFalse(self.controller.slot_yaw_fy_captured)
        self.assertEqual(self.mark.call_args.args[0], "lowest_yaw_capture")
