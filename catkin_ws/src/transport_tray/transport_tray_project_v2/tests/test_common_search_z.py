"""直接执行生产Z导纳方法，验证单基线、共同增量、退让及独立模式。"""
import ast
from dataclasses import replace
from enum import Enum
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict
from unittest import TestCase
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1] / "tray_control"
spec = importlib.util.spec_from_file_location("_common_z_math", ROOT / "algorithms/moving_z_admittance.py")
math_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = math_module
spec.loader.exec_module(math_module)


class ArmGroup(Enum):
    LEFT = "left"
    RIGHT = "right"


ARMS = (ArmGroup.LEFT, ArmGroup.RIGHT)
TREE = ast.parse((ROOT / "stage3_multi.py").read_text())
METHODS = [n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef)
           and n.name in ("start_moving_z_search", "step_moving_z_search", "pause_search_xy_for_retreat")]
scope = dict(Dict=Dict, ArmGroup=ArmGroup, ARMS=ARMS, replace=replace,
             MovingZParameters=math_module.MovingZParameters,
             MovingZAxis=math_module.MovingZAxis, step_common_z=math_module.step_common_z,
             rospy=SimpleNamespace(logwarn=Mock()))
exec(compile(ast.fix_missing_locations(ast.Module(METHODS, [])), str(ROOT / "stage3_multi.py"), "exec"), scope)


class CommonSearchZTests(TestCase):
    def setup_controller(self):
        self.controller = SimpleNamespace(
            args=SimpleNamespace(target_force=2., search_z_target_force=2.,
                                 search_z_retreat_distance=.0005, search_z_retreat_trigger_cycles=3,
                                 search_z_recovery_hysteresis=.2, search_z_speed_limit=.003,
                                 max_acceleration=.01, search_z_acceleration=.01,
                                 search_z_down_speed=.0075, search_z_down_acceleration=.02),
            z_model=SimpleNamespace(offsets={ArmGroup.LEFT: .01, ArmGroup.RIGHT: .015}))
        # 入口已经为2N：不二次归零时，此时不应再继续下压。
        self.start_support = {arm: 2. for arm in ARMS}
        scope["start_moving_z_search"](self.controller, self.start_support)
        return self.controller

    def step(self, left, right):
        return scope["step_moving_z_search"](
            self.controller, {ArmGroup.LEFT: left, ArmGroup.RIGHT: right}, .02)

    def pause_xy(self, left, right, now):
        return scope["pause_search_xy_for_retreat"](
            self.controller, {ArmGroup.LEFT: left, ArmGroup.RIGHT: right}, now)

    def test_pause_starts_before_third_overload_z_update_and_ends_after_retreat(self):
        c = self.setup_controller()
        for i in range(2):
            self.assertFalse(self.pause_xy(3., 0., i * .02))
            self.step(3., 0.)
        self.assertTrue(self.pause_xy(3., 0., .04))  # 当周期Z尚未更新，XY必须先停。
        self.step(3., 0.)
        axis = c.moving_z_axis
        now = .04
        while axis.retreat_active:
            now += .02
            self.assertTrue(self.pause_xy(0., 0., now))
            self.step(0., 0.)
        self.assertAlmostEqual(axis.displacement, -.0005)
        now += .02
        self.assertFalse(self.pause_xy(0., 0., now))
        self.assertAlmostEqual(c.search_xy_pause_seconds, now - .04)
        pause_total = c.search_xy_pause_seconds
        self.assertFalse(self.pause_xy(0., 0., now + 2.))
        self.assertEqual(c.search_xy_pause_seconds, pause_total)


    def test_disarmed_overload_does_not_pause_xy(self):
        c = self.setup_controller()
        c.moving_z_axis.retreat_armed = False
        c.moving_z_axis.overload_cycles = 2
        self.assertFalse(self.pause_xy(3., 0., 1.))

    def test_common_z_uses_held_baseline_without_second_zero(self):
        c = self.setup_controller()
        self.assertEqual(self.step(2., 2.), {ArmGroup.LEFT: .01, ArmGroup.RIGHT: .015})

    def test_common_increments_preserve_entry_height_difference_and_integrate_once(self):
        c = self.setup_controller()
        result = self.step(1., 1.5)
        self.assertAlmostEqual(result[ArmGroup.LEFT] - .01, .000008)
        for _ in range(100):
            result = self.step(1., 1.5)
            self.assertAlmostEqual(result[ArmGroup.RIGHT] - result[ArmGroup.LEFT], .005)

    def test_high_side_retreats_both_even_when_mean_is_below_target(self):
        self.setup_controller()
        for _ in range(2):
            self.assertEqual(self.step(3., 0.), {ArmGroup.LEFT: .01, ArmGroup.RIGHT: .015})
        result = self.step(3., 0.)  # 均值1.5N，但连续三个周期左侧3N触发共同上退。
        self.assertLess(result[ArmGroup.LEFT], .01)
        self.assertLess(result[ArmGroup.RIGHT], .015)
        self.assertAlmostEqual(result[ArmGroup.LEFT] - .01, result[ArmGroup.RIGHT] - .015)

    def test_overload_stops_downward_inertia_and_spring_continuation(self):
        c = self.setup_controller()
        axis = c.moving_z_axis
        axis.velocity = .003
        axis.equilibrium = .01
        result = self.step(3., 0.)
        self.assertEqual(result[ArmGroup.LEFT], .01)
        self.assertEqual(axis.velocity, 0.)
        self.step(3., 0.)
        result = self.step(3., 0.)
        self.assertLess(result[ArmGroup.LEFT], .01)
        self.assertLess(axis.velocity, 0.)


    def test_common_descent_uses_explicit_limits_and_preserves_retreat_profile(self):
        c = self.setup_controller()
        self.assertEqual(c.moving_z_descent_params.speed_limit, .0075)
        self.assertEqual(c.moving_z_descent_params.acceleration_limit, .02)
        self.assertEqual(c.moving_z_params.speed_limit, .003)
        self.assertEqual(c.moving_z_params.acceleration_limit, .01)
        legacy_axis = math_module.MovingZAxis()
        for _ in range(100):
            math_module.step_common_z(legacy_axis, [3., 0.], .02, c.moving_z_params)
            self.step(3., 0.)
            self.assertEqual(c.moving_z_axis.displacement, legacy_axis.displacement)
            self.assertEqual(c.moving_z_axis.velocity, legacy_axis.velocity)


class MovingZDeadbandTests(TestCase):
    def test_defaults_and_deadband_can_be_disabled(self):
        p = math_module.MovingZParameters()
        self.assertEqual(p.force_deadband, .2)
        self.assertEqual(p.equilibrium_gain, .0005)
        replace(p, force_deadband=0.)
        for bad in (-.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                replace(p, force_deadband=bad)

    def test_noise_inside_deadband_does_not_integrate_or_move_at_rest(self):
        p = math_module.MovingZParameters()
        for mode in ("independent", "common"):
            axis = math_module.MovingZAxis()
            for i in range(500):
                force = 1.85 if i % 2 else 2.15
                if mode == "common":
                    math_module.step_common_z(axis, [force, force], .02, p)
                else:
                    axis.step(force, .02, p)
            self.assertEqual(axis.displacement, 0.)
            self.assertEqual(axis.velocity, 0.)
            self.assertEqual(axis.equilibrium, 0.)

    def test_outside_deadband_uses_excess_error_and_half_gain(self):
        p = math_module.MovingZParameters()
        for force, expected_sign in ((1.5, 1), (2.5, -1)):
            axis = math_module.MovingZAxis()
            axis.step(force, .02, p)
            self.assertAlmostEqual(axis.equilibrium, expected_sign * .0005 * .3 * .02)
            self.assertGreater(expected_sign * axis.velocity, 0.)

    def test_common_high_side_still_prevents_downward_motion_inside_deadband(self):
        axis = math_module.MovingZAxis(velocity=.003, equilibrium=.01)
        math_module.step_common_z(axis, [2.1, 0.], .02, math_module.MovingZParameters())
        self.assertLessEqual(axis.displacement, 0.)
        self.assertLessEqual(axis.velocity, 0.)


class CommonFixedRetreatTests(TestCase):
    def trigger(self, axis, p):
        for _ in range(p.retreat_trigger_cycles):
            math_module.step_common_z(axis, [2., .5], .02, p)
        self.assertTrue(axis.retreat_active)

    def finish(self, axis, p, forces=(2., .5), dt=.02):
        for _ in range(1000):
            before, velocity = axis.displacement, axis.velocity
            math_module.step_common_z(axis, forces, dt, p)
            self.assertLessEqual(axis.displacement, before)
            self.assertLessEqual(abs(axis.velocity), p.speed_limit + 1e-12)
            self.assertLessEqual(abs(axis.velocity - velocity), p.acceleration_limit * dt + 1e-12)
            self.assertLessEqual(abs(axis.displacement - before), p.speed_limit * dt + 1e-12)
            self.assertGreaterEqual(axis.displacement, -p.travel_limit)
            if not axis.retreat_active:
                return
        self.fail("定距退让未完成")

    def test_three_consecutive_cycles_and_single_side_peak(self):
        p = math_module.MovingZParameters(target_force=1.5, recovery_hysteresis=.2)
        axis = math_module.MovingZAxis()
        # 高侧可在左右之间变化，只要每个周期max(force)>target就算连续。
        for forces in ([2., .5], [.5, 2.]):
            math_module.step_common_z(axis, forces, .02, p)
            self.assertFalse(axis.retreat_active)
            self.assertEqual(axis.displacement, 0.)
        math_module.step_common_z(axis, [2., .5], .02, p)
        self.assertTrue(axis.retreat_active)
        self.finish(axis, p)
        self.assertAlmostEqual(axis.displacement, -.0005)

    def test_non_overload_cycle_clears_count_and_equality_does_not_trigger(self):
        p = math_module.MovingZParameters(target_force=1.5)
        axis = math_module.MovingZAxis()
        for high in (2., 2., 1.5, 2., 1.5, 2., 2.):
            math_module.step_common_z(axis, [high, high], .02, p)
            self.assertFalse(axis.retreat_active)
        self.assertEqual(axis.overload_cycles, 2)
        math_module.step_common_z(axis, [2., 2.], .02, p)
        self.assertTrue(axis.retreat_active)

    def test_fixed_distance_finishes_despite_continuing_overload_and_does_not_repeat(self):
        p = math_module.MovingZParameters(target_force=1.5, recovery_hysteresis=.2)
        axis = math_module.MovingZAxis(displacement=.007, equilibrium=.02, velocity=.003)
        self.trigger(axis, p)
        self.finish(axis, p)
        self.assertAlmostEqual(axis.displacement, .0065)
        self.assertEqual(axis.velocity, 0.)
        self.assertEqual(axis.equilibrium, axis.displacement)
        for _ in range(300):
            math_module.step_common_z(axis, [2., .5], .02, p)
            self.assertFalse(axis.retreat_active)
            self.assertAlmostEqual(axis.displacement, .0065)
        self.assertFalse(axis.retreat_armed)

    def test_resume_descent_before_rearm_threshold_and_next_contact_can_retreat(self):
        p = math_module.MovingZParameters(target_force=1.5, recovery_hysteresis=.2)
        axis = math_module.MovingZAxis()
        self.trigger(axis, p)
        self.finish(axis, p, forces=(0., 0.))  # 途中卸载也必须完成0.5mm。
        before = axis.displacement
        math_module.step_common_z(axis, [1.4, .5], .02, p)
        self.assertGreater(axis.displacement, before)
        self.assertFalse(axis.retreat_armed)  # 1.3N门槛只影响下一次退让触发。
        math_module.step_common_z(axis, [1.3, .5], .02, p)
        self.assertTrue(axis.retreat_armed)
        before = axis.displacement
        self.trigger(axis, p)
        self.finish(axis, p)
        self.assertAlmostEqual(axis.displacement, before - .0005)

    def test_triangular_and_trapezoidal_profiles_obey_limits_and_irregular_dt(self):
        for speed in (.003, .001):
            for dt in (.005, .02, .037, .1):
                with self.subTest(speed=speed, dt=dt):
                    p = math_module.MovingZParameters(target_force=1.5, speed_limit=speed,
                                                     acceleration_limit=.01)
                    axis = math_module.MovingZAxis()
                    self.trigger(axis, p)
                    self.finish(axis, p, dt=dt)
                    self.assertAlmostEqual(axis.displacement, -.0005)

    def test_retreat_respects_upper_travel_boundary(self):
        p = math_module.MovingZParameters(target_force=1.5)
        for position in (-p.travel_limit, -p.travel_limit + .0001):
            axis = math_module.MovingZAxis(displacement=position)
            for _ in range(3):
                math_module.step_common_z(axis, [2., .5], .02, p)
            if axis.retreat_active:
                self.finish(axis, p)
            self.assertAlmostEqual(axis.displacement, -p.travel_limit)
            self.assertFalse(axis.retreat_active)

    def test_common_descent_never_retreats_outside_fixed_retreat_state(self):
        p = math_module.MovingZParameters(target_force=1.5)
        axis = math_module.MovingZAxis(displacement=.004, equilibrium=-.01, velocity=-.003)
        for _ in range(100):
            before = axis.displacement
            math_module.step_common_z(axis, [1.4, 1.4], .02, p)
            self.assertGreaterEqual(axis.displacement, before)
            self.assertGreaterEqual(axis.velocity, 0.)

    def test_invalid_input_rejected_during_retreat_and_before_counting(self):
        p = math_module.MovingZParameters(target_force=1.5)
        for active in (False, True):
            axis = math_module.MovingZAxis()
            if active:
                self.trigger(axis, p)
            for forces, dt in (([], .02), ([float('nan'), 1.], .02), ([1., float('inf')], .02),
                               ([2., 2.], 0.), ([2., 2.], .11)):
                before = vars(axis).copy()
                with self.assertRaises(ValueError):
                    math_module.step_common_z(axis, forces, dt, p)
                self.assertEqual(vars(axis), before)

    def test_retreat_parameters_validate(self):
        for bad in (0, -1, 1.5, True, float('nan')):
            with self.assertRaises(ValueError):
                math_module.MovingZParameters(retreat_trigger_cycles=bad)
        for bad in (0., -.001, float('nan'), float('inf'), .06):
            with self.assertRaises(ValueError):
                math_module.MovingZParameters(retreat_distance=bad)

    def test_explicit_xy_settings_do_not_change_stage3_target(self):
        c = SimpleNamespace(args=SimpleNamespace(target_force=2., search_z_target_force=1.5, search_z_recovery_hysteresis=.2,
                                 search_z_retreat_distance=.0005, search_z_retreat_trigger_cycles=3,
                                 search_z_speed_limit=.003, max_acceleration=.01, search_z_acceleration=.01,
                                 search_z_down_speed=.0075, search_z_down_acceleration=.02),
                            z_model=SimpleNamespace(offsets={arm:0. for arm in ARMS}))
        scope["start_moving_z_search"](c, {arm:2. for arm in ARMS})
        self.assertEqual(c.moving_z_params.target_force, 1.5)
        self.assertEqual(c.moving_z_params.recovery_hysteresis, .2)
        self.assertEqual(c.moving_z_params.retreat_distance, .0005)
        self.assertEqual(c.moving_z_params.retreat_trigger_cycles, 3)
        self.assertEqual(c.args.target_force, 2.)


class UnboundedMovingZTests(TestCase):
    def test_unbounded_axis_crosses_old_travel_and_equilibrium_limits(self):
        p = math_module.MovingZParameters(travel_limit=None, equilibrium_limit=None)
        axis = math_module.MovingZAxis(displacement=.16, equilibrium=.16, velocity=.003)
        for _ in range(100):
            axis.step(0., .02, p)
        self.assertGreater(axis.displacement, .16)
        self.assertGreater(axis.equilibrium, .16)
        axis = math_module.MovingZAxis(displacement=-.1, equilibrium=-.1)
        for _ in range(100):
            axis.step(5., .02, p)
        self.assertLess(axis.displacement, -.1)

    def test_common_fixed_retreat_remains_exact_without_global_travel_limit(self):
        p = math_module.MovingZParameters(travel_limit=None, equilibrium_limit=None)
        axis = math_module.MovingZAxis(displacement=-.1, equilibrium=-.1)
        for _ in range(3):
            math_module.step_common_z(axis, [3., 0.], .02, p)
        while axis.retreat_active:
            math_module.step_common_z(axis, [3., 0.], .02, p)
        self.assertAlmostEqual(axis.displacement, -.1 - p.retreat_distance)
