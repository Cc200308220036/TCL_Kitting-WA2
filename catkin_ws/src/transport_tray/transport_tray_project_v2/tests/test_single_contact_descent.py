"""单层真实生产模型的离线接触、定距退让及速度检查。"""
import ast
from enum import Enum
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import TestCase

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1] / "tray_control"
spec = importlib.util.spec_from_file_location("_single_retreat_math", ROOT / "algorithms/moving_z_admittance.py")
moving = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = moving
spec.loader.exec_module(moving)


class Arm(Enum):
    LEFT = "left"
    RIGHT = "right"


ARMS = (Arm.LEFT, Arm.RIGHT)
tree = ast.parse((ROOT / "stage3.py").read_text())
common_tree = ast.parse((ROOT / "stage3_common.py").read_text())
velocity = next(n for n in common_tree.body if isinstance(n, ast.FunctionDef) and n.name == "next_z_velocity")
update = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "update_contact_descent")
function = ast.parse("def advance(motion, args, support, dt, retreat_params):\n    pass").body[0]
# 直接运行生产运动更新，跳过说明，直接使用共用函数体。
function.body = update.body[1:] + ast.parse("return motion.offsets.copy()").body
scope = dict(np=np, math=math, ARMS=ARMS, ArmGroup=Arm,
             MovingZAxis=moving.MovingZAxis, MovingZParameters=moving.MovingZParameters)
constant = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CONTACT_RETREAT_DISTANCE_M" for t in n.targets))
exec(compile(ast.fix_missing_locations(ast.Module(
    ast.parse("from __future__ import annotations").body + [constant, velocity, function], [])), "single_control", "exec"), scope)


class SingleContactTests(TestCase):
    def setUp(self):
        config = yaml.safe_load((ROOT / "config/transport_tray.yaml").read_text())["common"]["stage3"]
        self.args = SimpleNamespace(**config, max_free_speed=.006,
                                    max_contact_speed=.001, max_acceleration=.02)
        self.model = SimpleNamespace(offsets={a: 0. for a in ARMS}, velocities={a: 0. for a in ARMS},
            reached={a: False for a in ARMS}, dual_velocity=0., held_arm=None, retreat_axis=None,
            mode="DUAL_DESCENT")
        params = moving.MovingZParameters(target_force=self.args.target_force, mass=self.args.mass,
            damping=self.args.damping, speed_limit=.006, acceleration_limit=.02,
            retreat_distance=scope["CONTACT_RETREAT_DISTANCE_M"], travel_limit=None, equilibrium_limit=None)
        self.model.step = lambda forces, desired, candidate, dt: scope["advance"](
            self.model, self.args, forces, dt, params)

    def step(self, left, right, dt=.02):
        return self.model.step({Arm.LEFT: left, Arm.RIGHT: right}, 2.1, True, dt)

    def test_both_reach_threshold_same_sample_freezes_immediately(self):
        self.model.dual_velocity = .006
        self.step(2., 2.)
        self.assertTrue(all(self.model.reached.values()))
        self.assertEqual(self.model.mode, "BOTH_FORCE_READY")
        self.assertIsNone(self.model.retreat_axis)
        self.assertEqual(self.model.offsets, {a: 0. for a in ARMS})
        self.assertEqual(tuple(self.model.velocities[a] for a in ARMS), (0., 0.))

    def test_either_side_retreats_once_by_half_mm_and_holds(self):
        for held in ARMS:
            self.setUp()
            forces = {a: 0. for a in ARMS}
            forces[held] = 2.
            self.model.offsets = {a: .01 for a in ARMS}
            self.model.step(forces, 2.1, True, .02)
            # 力因上退归零也保留接触锁存，并完成定距上退。
            for _ in range(100):
                self.step(0., 0.)
            self.assertAlmostEqual(self.model.offsets[held], .0095)
            self.assertTrue(self.model.reached[held])
            self.assertFalse(self.model.retreat_axis.retreat_active)
            self.assertEqual(self.model.velocities[held], 0.)
            for _ in range(50):
                self.model.step(forces, 2.1, True, .02)
            self.assertAlmostEqual(self.model.offsets[held], .0095)

    def test_second_side_contact_releases_even_if_first_side_has_unloaded(self):
        self.step(2., 0.)
        before = self.model.offsets.copy()
        self.step(0., 2.)
        self.assertTrue(all(self.model.reached.values()))
        self.assertEqual(self.model.mode, "BOTH_FORCE_READY")
        self.assertEqual(self.model.offsets, before)

    def test_below_threshold_does_not_latch(self):
        self.step(1.999, 1.999)
        self.assertFalse(any(self.model.reached.values()))

    def test_no_contact_candidate_slowdown_and_acceleration_is_doubled(self):
        self.step(.5, .5)
        self.assertAlmostEqual(self.model.dual_velocity, .0004)
        for _ in range(30):
            self.step(.5, .5)
        self.assertAlmostEqual(self.model.dual_velocity, .006)
        self.assertAlmostEqual(self.model.offsets[Arm.LEFT], self.model.offsets[Arm.RIGHT])

    def test_other_side_inherits_speed_without_hold_force_scaling(self):
        for _ in range(30):
            self.step(0., 0.)
        for held_force in (2., 0., 4.):
            before = self.model.offsets[Arm.RIGHT]
            self.step(held_force, 0.)
            self.assertAlmostEqual(self.model.velocities[Arm.RIGHT], .006)
            self.assertAlmostEqual(self.model.offsets[Arm.RIGHT] - before, .006 * .02)

    def test_retreat_profile_respects_speed_acceleration_and_fixed_distance(self):
        self.model.offsets = {a: .01 for a in ARMS}
        self.step(2., 0.)
        for _ in range(100):
            previous = self.model.retreat_axis.velocity
            self.step(0., 0.)
            self.assertLessEqual(abs(self.model.retreat_axis.velocity), .006)
            self.assertLessEqual(abs(self.model.retreat_axis.velocity - previous), .02 * .02 + 1e-12)
        self.assertAlmostEqual(self.model.offsets[Arm.LEFT], .0095)
        self.setUp()
        self.model.offsets = {a: -.0028 for a in ARMS}
        for _ in range(100):
            self.step(2., 0.)
        self.assertAlmostEqual(self.model.offsets[Arm.LEFT], -.0033)

    def test_nonfinite_force_or_invalid_cycle_rejected_before_latching(self):
        for left, right, dt in ((float('nan'), 2., .02), (2., float('inf'), .02), (2., 2., 0.)):
            with self.assertRaises(ValueError):
                self.step(left, right, dt)
            self.assertFalse(any(self.model.reached.values()))

    def test_descent_and_side_difference_can_exceed_former_limits(self):
        self.model.offsets = {Arm.LEFT: .16, Arm.RIGHT: .23}
        for _ in range(100):
            self.step(0., 0.)
        self.assertGreater(self.model.offsets[Arm.LEFT], .16)
        self.assertGreater(self.model.offsets[Arm.RIGHT], .23)
        self.assertGreater(abs(self.model.offsets[Arm.LEFT] - self.model.offsets[Arm.RIGHT]), .05)
