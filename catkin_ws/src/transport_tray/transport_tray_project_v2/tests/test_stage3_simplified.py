"""Run the actual Stage3 state branches offline with robot/ROS boundaries stubbed.

AST extraction avoids importing hardware clients. Branch statements are compiled
from production source, rather than duplicating the decision rules in the tests.
"""
import ast
import copy
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from stage3_probe import search_controller, bind_methods
from typing import Dict, Optional, Tuple
from unittest import TestCase
from unittest.mock import Mock
import math
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation
import yaml
import importlib.util

ROOT = Path(__file__).resolve().parents[1] / 'tray_control'
TREE = ast.parse((ROOT / 'stage3_multi.py').read_text())

_spec = importlib.util.spec_from_file_location('_stage3_branch_paths', ROOT / 'algorithms/search_path.py')
paths = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paths)


class ArmGroup(Enum):
    LEFT = 'left'
    RIGHT = 'right'


ARMS = tuple(ArmGroup)


def state_branch(state):
    method_name = {'XY_SEARCH': 'step_xy_search',
                   'FINAL_Z_SEATING': 'step_seating'}[state]
    method = copy.deepcopy(next(n for n in ast.walk(TREE)
                  if isinstance(n, ast.FunctionDef) and n.name == method_name))
    # Expose the new snapshot/state attributes as the existing probe's local inputs.
    class ProbeLocals(ast.NodeTransformer):
        def visit_Subscript(self, node):
            value = ast.unparse(node)
            if value in ('cycle.support[ArmGroup.LEFT]', 'cycle.support[ArmGroup.RIGHT]'):
                return ast.Name('f_left' if 'LEFT' in value else 'f_right', node.ctx)
            return self.generic_visit(node)

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'cycle':
                return ast.Name(node.attr, node.ctx)
            if ast.unparse(node.value) == 'self.flow':
                return ast.Name(node.attr, node.ctx)
            if ast.unparse(node) == 'self.search_start_z':
                return ast.Name('search_start_z', node.ctx)
            return self.generic_visit(node)

    branch = SimpleNamespace(body=ProbeLocals().visit(method).body)
    # Each call returns the locals modified by the actual branch to carry timers.
    function = ast.parse('def cycle(**values):\n    pass').body[0]
    names = ['self', 'now', 'stage_started', 'seated_since', 'search_start_z',
             'tray_estimate', 'slot_candidate_since', 'search_center_correction',
             'search_start_support', 'support', 'force_drop_since',
             'force_drop_latched', 'force_drop_peak', 'dt', 'f_left', 'f_right', 'raw_wrenches', 'observation']
    class ExtractedCycle(ast.NodeTransformer):
        def __init__(self):
            self.nested_loop = 0

        def visit_For(self, node):
            self.nested_loop += 1
            self.generic_visit(node)
            self.nested_loop -= 1
            return node

        def visit_Return(self, node):
            return ast.Return(ast.Call(ast.Name('locals', ast.Load()), [], []))

        def visit_Continue(self, node):
            # Production branch continues the 50 Hz loop. The extracted
            # single-cycle function must return its updated locals instead.
            if self.nested_loop:
                return node
            return ast.Return(ast.Call(ast.Name('locals', ast.Load()), [], []))

    function.body = [ast.Assign([ast.Name(name, ast.Store())],
                     ast.Subscript(ast.Name('values', ast.Load()), ast.Constant(name), ast.Load()))
                     for name in names] + [ExtractedCycle().visit(stmt) for stmt in copy.deepcopy(branch.body)] + ast.parse('return locals()').body
    module = ast.fix_missing_locations(ast.Module([function], []))
    scope = dict(np=np, math=math, ArmGroup=ArmGroup, ARMS=ARMS,
                 DualGripperWorker=Mock(),
                 rospy=SimpleNamespace(
                     loginfo=Mock(side_effect=lambda fmt, *args: fmt % args if args else fmt),
                     logwarn=Mock(side_effect=lambda fmt, *args: fmt % args if args else fmt)),
                 trace_state=Mock(), trace_mark=Mock(), yaw_from_pose7=lambda pose: 0.)
    exec(compile(module, str(ROOT / 'stage3_multi.py'), 'exec'), scope)
    return scope['cycle']


SEARCH = state_branch('XY_SEARCH')
SEATING = state_branch('FINAL_Z_SEATING')


class SimplifiedStage3Tests(TestCase):
    def setUp(self):
        SEARCH.__globals__['DualGripperWorker'].reset_mock(return_value=True)
        search = yaml.safe_load((ROOT / 'config/transport_tray.yaml').read_text())['multi_layer']['search']
        geometry = search['search_paths']['spiral']
        search.update(search_max_radius=geometry['radius'], search_pitch=geometry['pitch'],
                      search_path='spiral', square_ring_step=.002, circle_ring_step=.001)
        search['xy_force_capture_threshold'] = search['xy_force_capture_threshold'][1]
        args = SimpleNamespace(**search, search_angular_speed=1., target_force=2.)
        args.slot_min_descent = .002
        self.controller = search_controller(args=args, stage_name='XY_SEARCH',
            search_offset_base=np.array([.001, -.001]), place_pose=[0., 0., 0.],
            search_start_raw_fx={arm: 0. for arm in ARMS}, search_start_raw_fy={arm: 0. for arm in ARMS},
            baseline={arm: np.zeros(6) for arm in ARMS}, rate_limit_xy=Mock(),
            pause_search_xy_for_retreat=Mock(return_value=False), search_xy_pause_seconds=0.,
            direct_seating_selected=False, slot_captured=False, xy_force_captured=False,
            hand_worker=None, grippers=object())
        release = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == 'begin_release')
        release_scope = dict(DualGripperWorker=SEARCH.__globals__['DualGripperWorker'], trace_state=Mock())
        exec(compile(ast.fix_missing_locations(ast.Module([release], [])), 'release_helper', 'exec'), release_scope)
        self.controller.flow = SimpleNamespace(stage_started=0.)
        self.controller.xy_path = paths.make_search_path(args)
        bind_methods(self.controller, TREE, release_scope, ['transition', 'begin_release', 'finish_search_and_release', 'calculate_search_observation'])
        self.controller.history = SimpleNamespace(poses=None, drop=float('-inf'), offset=None, tray_pose=None)
        self.controller.start_unconfirmed_release = Mock(side_effect=lambda *a: setattr(self.controller, 'stage_name', 'OPENING'))
        self.values = dict(self=self.controller, now=1., stage_started=0., seated_since=None,
            search_start_z=0., tray_estimate=[0., 0., -.002], slot_candidate_since=None,
            search_center_correction=np.zeros(2), search_start_support={a: 2. for a in ARMS},
            support={a: 2. for a in ARMS}, force_drop_since={a: None for a in ARMS},
            force_drop_latched={a: False for a in ARMS}, force_drop_peak={a: 0. for a in ARMS},
            dt=.02, f_left=2., f_right=2., raw_wrenches={a: np.zeros(6) for a in ARMS})

    def cycle(self, function, **updates):
        self.values.update(updates)
        self.values["support"] = {ArmGroup.LEFT: self.values["f_left"], ArmGroup.RIGHT: self.values["f_right"]}
        if isinstance(self.controller.xy_path, paths.SpiralSearch):
            self.controller.xy_path = paths.make_search_path(self.controller.args)
        self.values['observation'] = self.controller.calculate_search_observation(SimpleNamespace(
            raw_wrenches=self.values['raw_wrenches'], tray_estimate=self.values['tray_estimate'],
            search_start_z=self.values['search_start_z']))
        result = function(**self.values)
        if result is not None:
            self.values.update({key: result[key] for key in self.values})

    def test_threshold_freezes_without_latches_even_at_search_timeout(self):
        self.cycle(SEARCH, now=100.)
        self.assertEqual(self.controller.stage_name, 'OPENING')
        self.controller.rate_limit_xy.assert_not_called()
        self.assertFalse(any(self.values['force_drop_latched'].values()))
        np.testing.assert_array_equal(self.controller.final_search_offset, [.001, -.001])

    def test_below_threshold_continues_search(self):
        self.cycle(SEARCH, tray_estimate=[0., 0., -.0019])
        self.assertEqual(self.controller.stage_name, 'XY_SEARCH')
        self.controller.rate_limit_xy.assert_called_once()

    def test_xy_pause_freezes_circle_square_and_spiral_progress(self):
        for path_name in ('circle', 'square', 'spiral'):
            with self.subTest(path=path_name):
                self.setUp()
                c=self.controller
                c.args.search_path=path_name
                c.pause_search_xy_for_retreat=Mock(return_value=True)
                c.xy_path=paths.make_search_path(c.args)
                before=c.search_offset_base.copy()
                index=getattr(c.xy_path, 'index', None)
                self.cycle(SEARCH,tray_estimate=[0.,0.,-.001])
                np.testing.assert_array_equal(c.search_offset_base,before)
                c.rate_limit_xy.assert_not_called()
                self.assertEqual(getattr(c.xy_path, 'index', None),index)
                self.assertEqual(c.stage_name, 'XY_SEARCH')

    def test_resume_continues_circle_from_previous_target(self):
        c=self.controller
        c.args.search_path='circle'
        c.pause_search_xy_for_retreat=Mock(return_value=False)
        c.xy_path=SimpleNamespace(target=Mock(return_value=(np.array([.001,0.]),False)))
        self.cycle(SEARCH,tray_estimate=[0.,0.,-.001])
        c.xy_path.target.assert_called_once()
        c.rate_limit_xy.assert_called_once()

    def test_spiral_clock_does_not_skip_forward_after_xy_pause(self):
        self.controller.search_xy_pause_seconds=10.
        self.cycle(SEARCH,now=11.,tray_estimate=[0.,0.,-.001])
        expected_radius=self.controller.args.search_pitch/(2*math.pi)
        self.assertAlmostEqual(np.linalg.norm(self.controller.rate_limit_xy.call_args.args[0]),expected_radius)
        self.controller.rate_limit_xy.assert_called_once()

    def test_either_raw_fx_sign_can_capture_without_descent(self):
        for arm in ARMS:
            for fx in (-3.51, 3.51):
                with self.subTest(arm=arm, fx=fx):
                    self.setUp()
                    self.values['raw_wrenches'][arm][0] = fx
                    self.cycle(SEARCH, tray_estimate=[0., 0., -.001])
                    self.assertEqual(self.controller.stage_name, 'OPENING')
                    self.assertTrue(self.controller.xy_force_captured)

    def test_raw_fx_threshold_is_strict(self):
        for fx in (-3.5, 3.5):
            self.setUp()
            self.values['raw_wrenches'][ArmGroup.LEFT][0] = fx
            self.cycle(SEARCH, tray_estimate=[0., 0., -.001])
            self.assertEqual(self.controller.stage_name, 'XY_SEARCH')

    def test_circle_reaches_outer_ring_before_finishing_and_reuses_xy_speed_limit(self):
        self.controller.args.search_path = "circle"
        self.controller.args.search_max_radius = .01
        self.controller.args.search_timeout = 234.
        self.controller.xy_path = SimpleNamespace(target=Mock(return_value=(np.array([.01, 0.]), False)))
        self.controller.start_return_to_lowest = Mock()
        self.cycle(SEARCH, now=100., tray_estimate=[0., 0., -.001])
        self.controller.start_return_to_lowest.assert_not_called()
        np.testing.assert_allclose(self.controller.rate_limit_xy.call_args.args[0], [.01, 0.])
        self.controller.xy_path.target.return_value = (np.array([.01, 0.]), True)
        self.cycle(SEARCH, now=101., tray_estimate=[0., 0., -.001])
        self.controller.start_return_to_lowest.assert_called_once_with(.001)

    def test_combined_force_capture_has_no_height_gate(self):
        for height in (-.001, .01):
            for axis in (0, 1):
                self.setUp()
                self.values['raw_wrenches'][ArmGroup.LEFT][axis] = 4.
                self.cycle(SEARCH, tray_estimate=[0., 0., height])
                self.assertEqual(self.controller.stage_name, 'OPENING')

    def test_either_axis_strictly_above_threshold_captures(self):
        for arm in ARMS:
            for fx, fy, capture in ((4., 4., True), (4., -4., True), (-4., 4., True), (-4., -4., True),
                                    (3.5, 4., True), (4., 3.5, True), (4., 0., True), (0., 4., True),
                                    (0., -4., True), (-4., 0., True), (3.5, -3.5, False), (0., 0., False)):
                with self.subTest(arm=arm, fx=fx, fy=fy):
                    self.setUp()
                    self.controller.grippers = object()
                    self.controller.hand_worker = None
                    self.values['raw_wrenches'][arm][:2] = [fx, fy]
                    self.cycle(SEARCH, tray_estimate=[0., 0., -.001])
                    self.assertEqual(self.controller.xy_force_captured, capture)
                    self.assertEqual(self.controller.stage_name, 'OPENING' if capture else 'XY_SEARCH')

    def test_cross_arm_fx_and_fy_capture(self):
        self.controller.grippers = object()
        self.controller.hand_worker = None
        self.values['raw_wrenches'][ArmGroup.LEFT][0] = 4.
        self.values['raw_wrenches'][ArmGroup.RIGHT][1] = 4.
        self.cycle(SEARCH, tray_estimate=[0., 0., -.001])
        self.assertTrue(self.controller.xy_force_captured)
        self.assertEqual(self.controller.stage_name, 'OPENING')

    def test_combined_fy_uses_search_start_not_held_baseline(self):
        self.controller.grippers = object()
        self.controller.hand_worker = None
        self.controller.baseline[ArmGroup.RIGHT][1] = -20.
        self.controller.search_start_raw_fy[ArmGroup.RIGHT] = 10.
        self.values['raw_wrenches'][ArmGroup.RIGHT][:2] = [0., 10.]
        self.cycle(SEARCH, tray_estimate=[0., 0., -.001])
        self.assertEqual(self.controller.stage_name, 'XY_SEARCH')
        self.values['raw_wrenches'][ArmGroup.RIGHT][1] = 14.
        self.cycle(SEARCH, now=1.02, tray_estimate=[0., 0., -.001])
        self.assertEqual(self.controller.stage_name, 'OPENING')
        self.assertTrue(self.controller.xy_force_captured)

    def test_descent_capture_remains_available_above_p_place(self):
        self.controller.grippers = object()
        self.controller.hand_worker = None
        self.values['raw_wrenches'][ArmGroup.RIGHT][0] = 0.
        self.cycle(SEARCH, search_start_z=.008, tray_estimate=[0., 0., .006])
        self.assertEqual(self.controller.stage_name, 'OPENING')
        self.assertFalse(self.controller.xy_force_captured)

    def test_moving_z_mode_releases_on_raw_fx_without_seating_wait(self):
        self.controller.grippers = object()
        self.controller.hand_worker = None
        self.values['raw_wrenches'][ArmGroup.RIGHT][1] = 0.
        self.values['raw_wrenches'][ArmGroup.RIGHT][0] = -3.51
        self.cycle(SEARCH, tray_estimate=[0., 0., -.0005])
        self.assertEqual(self.controller.stage_name, 'OPENING')
        self.assertTrue(self.controller.xy_force_captured)
        self.controller.hand_worker.start.assert_called_once()

    def test_search_fx_uses_entry_baseline_not_absolute_sensor_value(self):
        self.controller.grippers = object()
        self.controller.hand_worker = None
        self.values['raw_wrenches'][ArmGroup.RIGHT][1] = 0.
        self.controller.search_start_raw_fx[ArmGroup.RIGHT] = 4.55
        self.values['raw_wrenches'][ArmGroup.RIGHT][0] = 4.76
        self.cycle(SEARCH, tray_estimate=[0., 0., -.0005])
        self.assertEqual(self.controller.stage_name, 'XY_SEARCH')
        self.assertFalse(self.controller.xy_force_captured)
        self.values['raw_wrenches'][ArmGroup.RIGHT][0] = 7.55
        self.cycle(SEARCH, now=1.02, tray_estimate=[0., 0., -.0005])
        self.assertEqual(self.controller.stage_name, 'XY_SEARCH')
        self.values['raw_wrenches'][ArmGroup.RIGHT][0] = 8.06
        self.cycle(SEARCH, now=1.04, tray_estimate=[0., 0., -.0005])
        self.assertEqual(self.controller.stage_name, 'OPENING')
        self.assertTrue(self.controller.xy_force_captured)

    def test_moving_z_mode_releases_on_descent_without_fx(self):
        self.controller.grippers = object()
        self.controller.hand_worker = None
        self.cycle(SEARCH)
        self.assertEqual(self.controller.stage_name, 'OPENING')
        self.assertFalse(self.controller.xy_force_captured)
        self.controller.hand_worker.start.assert_called_once()

    def test_max_radius_opens_without_descent_or_force_latches(self):
        self.controller.args.search_max_radius = self.controller.args.search_pitch / (2 * math.pi)
        def release(drop):
            self.controller.stage_name = 'RETURN_SEARCH_LOWEST'
        self.controller.start_return_to_lowest = Mock(side_effect=release)
        self.cycle(SEARCH, tray_estimate=[0., 0., -.001])
        self.controller.start_return_to_lowest.assert_called_once()
        self.assertEqual(self.controller.stage_name, 'RETURN_SEARCH_LOWEST')
        self.controller.rate_limit_xy.assert_not_called()
        self.assertFalse(any(self.values['force_drop_latched'].values()))

    def test_spiral_timeout_returns_to_lowest_instead_of_stopping(self):
        def return_lowest(drop):
            self.controller.stage_name = 'RETURN_SEARCH_LOWEST'
        self.controller.start_return_to_lowest = Mock(side_effect=return_lowest)
        self.cycle(SEARCH, now=self.controller.args.search_timeout + 1.,
                   tray_estimate=[0., 0., -.001])
        self.assertEqual(self.controller.stage_name, 'RETURN_SEARCH_LOWEST')
        self.controller.start_return_to_lowest.assert_called_once()





    def test_direct_seating_automatically_releases_after_average_force_stabilizes(self):
        self.controller.stage_name = 'FINAL_Z_SEATING'
        self.cycle(SEATING, now=1., f_left=1., f_right=3.)
        self.assertEqual(self.controller.stage_name, 'FINAL_Z_SEATING')
        self.cycle(SEATING, now=1.21)
        self.assertEqual(self.controller.stage_name, 'OPENING')
        self.controller.hand_worker.start.assert_called_once()

    def test_direct_seating_lost_force_resets_stability_timer(self):
        self.controller.stage_name = 'FINAL_Z_SEATING'
        self.cycle(SEATING, now=1.)
        self.cycle(SEATING, now=1.1, f_left=1.9, f_right=1.9)
        self.assertIsNone(self.values['seated_since'])
        self.cycle(SEATING, now=1.2, f_left=2., f_right=2.)
        self.cycle(SEATING, now=1.3)
        self.assertEqual(self.controller.stage_name, 'FINAL_Z_SEATING')
        self.cycle(SEATING, now=1.41)
        self.assertEqual(self.controller.stage_name, 'OPENING')
        self.controller.rate_limit_xy.assert_not_called()


    def test_yaw_search_exhaustion_releases_without_capture(self):
        method = copy.deepcopy(next(n for n in ast.walk(TREE)
                                    if isinstance(n, ast.FunctionDef)
                                    and n.name == 'advance_lowest_yaw_search'))
        scope = dict(math=math, np=np, Rotation=Rotation, ARMS=ARMS,
                     ArmGroup=ArmGroup, yaw_from_pose7=lambda pose: 0.,
                     wrap_angle=lambda angle: angle,
                     estimate_tray_fast=lambda left, right, grasp: ([0., 0., 0., 0., 0., 0., 1.], None),
                     rospy=SimpleNamespace(loginfo=Mock()))
        exec(compile(ast.fix_missing_locations(ast.Module([method], [])),
                     str(ROOT / 'stage3_multi.py'), 'exec'), scope)
        controller = self.controller
        controller.args.radius_yaw_timeout = 50.
        controller.args.radius_yaw_range_deg = 2.
        controller.args.radius_yaw_speed_deg_s = .3
        controller.args.radius_yaw_extra_descent = .0005
        controller.yaw_search.started = 0.
        controller.yaw_search.capture_since = None
        controller.yaw_search.start_z = 0.
        controller.history.drop = .001
        controller.yaw_search.segment = 2
        controller.yaw_search.offset = 0.
        controller.yaw_search.settled_since = 0.
        controller.history.tray_pose = [0., 0., 0., 0., 0., 0., 1.]
        controller.yaw_search.center = np.zeros(3)
        controller.history.poses = {arm: np.array([0., 0., 0., 0., 0., 0., 1.]) for arm in ARMS}
        controller.return_motion.commands = {arm: pose.copy()
                                             for arm, pose in controller.history.poses.items()}
        controller.grasp_components = {}
        poses = {arm: pose.copy() for arm, pose in controller.history.poses.items()}
        bind_methods(controller, TREE, scope, ['check_yaw_capture', 'advance_yaw_command', 'calculate_search_observation', 'finish_search_and_release'])
        scope['advance_lowest_yaw_search'](controller, poses,
                                           [0., 0., 0.], 0., 1., .02)
        self.assertEqual(controller.stage_name, 'OPENING')
        self.assertEqual(controller.yaw_search.segment, 3)
        controller.start_unconfirmed_release.assert_called_once()
        self.assertIn('未观察到持续新增下沉',
                      controller.start_unconfirmed_release.call_args.args[0])

    def test_square_branch_uses_base_xy_without_tray_yaw_rotation(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('_square_probe', ROOT / 'algorithms/search_path.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.controller.args.search_path = 'square'
        self.controller.xy_path = module.SquareSearch()
        self.controller.search_offset_base = np.array([.004, -.003])
        self.cycle(SEARCH, tray_estimate=[0., 0., -.001],
                   search_center_correction=np.array([.004, -.003]))
        np.testing.assert_allclose(self.controller.rate_limit_xy.call_args.args[0], [.004, -.001])
        self.controller.rate_limit_xy.assert_called_once()

    def test_square_exhaustion_reuses_lowest_point_return(self):
        self.controller.args.search_path = 'square'
        self.controller.xy_path = SimpleNamespace(target=Mock(return_value=(np.array([.01, 0.]), True)))
        self.controller.search_offset_base = np.array([.01, 0.])
        self.controller.start_return_to_lowest = Mock()
        self.cycle(SEARCH, tray_estimate=[0., 0., -.001])
        self.controller.start_return_to_lowest.assert_called_once()
        self.controller.rate_limit_xy.assert_not_called()

    def test_near_place_direct_seating_preserved(self):
        self.controller.stage_name = 'FINAL_Z_SEATING'
        self.controller.direct_seating_selected = True
        self.cycle(SEATING, now=100., tray_estimate=[0., 0., .001])
        self.cycle(SEATING, now=100.21)
        self.assertEqual(self.controller.stage_name, 'OPENING')

    def test_invalid_force_and_speed_rejected_by_actual_validator(self):
        tree = ast.parse((ROOT / 'config.py').read_text())
        validator = copy.deepcopy(next(n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == 'validate_multi_stage3'))
        validator.decorator_list = []
        positive = next(n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'POSITIVE_MULTI_SEARCH_PARAMETERS' for t in n.targets))
        scope = dict(argparse=SimpleNamespace(Namespace=SimpleNamespace), math=math)
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_positive')
        state_tree = ast.parse((ROOT / 'common/state.py').read_text())
        require_positive = next(n for n in state_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'require_positive')
        exec(compile(ast.fix_missing_locations(ast.Module([require_positive, helper, positive, validator], [])), 'config.py', 'exec'), scope)
        validate = scope['validate_multi_stage3']
        self.controller.args.max_free_speed = .005
        validate(self.controller.args)
        for key, value in [('search_z_speed_limit', float('nan')),
                           ('xy_force_capture_threshold', 0.), ('xy_force_capture_threshold', float('nan'))]:
            args = copy.copy(self.controller.args)
            setattr(args, key, value)
            with self.assertRaises(ValueError):
                validate(args)


class MeanAdmittanceTests(TestCase):
    def test_actual_dual_model_uses_average_without_k_and_freezes_when_waiting(self):
        tree = ast.parse((ROOT / 'stage3_common.py').read_text())
        model_class = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                           and n.name == 'SwitchingZAdmittance')
        scope = dict(np=np, Dict=Dict, Optional=Optional, Tuple=Tuple, ArmGroup=ArmGroup,
                     ARMS=ARMS, argparse=SimpleNamespace(Namespace=SimpleNamespace))
        exec(compile(ast.fix_missing_locations(ast.Module([next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "next_z_velocity"), model_class], [])), 'stage3.py', 'exec'), scope)
        model_type = scope['SwitchingZAdmittance']
        # Extract the real mode selector as well: search/seating share DUAL;
        # opening the grippers disables integration.
        choose = copy.deepcopy(next(n for n in ast.walk(TREE)
                                     if isinstance(n, ast.FunctionDef) and n.name == 'choose_z_mode'))
        scope['SwitchingZAdmittance'] = model_type
        exec(compile(ast.fix_missing_locations(ast.Module([choose], [])), 'stage3_multi.py', 'exec'), scope)
        config = yaml.safe_load((ROOT / 'config/transport_tray.yaml').read_text())['common']['stage3']
        args = SimpleNamespace(**config, max_contact_speed=.0005, max_free_speed=.003,
                               max_acceleration=.01)
        for left, right, direction in [(1., 2., 1), (2., 3., -1)]:
            results = []
            for stiffness in [0., 100000.]:
                args.hold_spring_k = stiffness
                model = model_type(args)
                controller = SimpleNamespace(stage_name='XY_SEARCH')
                model.set_mode(scope['choose_z_mode'](controller, {a: False for a in ARMS}))
                offsets = model.step({ArmGroup.LEFT: left, ArmGroup.RIGHT: right}, 2., True, .02)
                self.assertGreater(direction * offsets[ArmGroup.LEFT], 0)
                self.assertEqual(offsets[ArmGroup.LEFT], offsets[ArmGroup.RIGHT])
                results.append(offsets)
                controller.stage_name = 'FINAL_Z_SEATING'
                self.assertEqual(scope['choose_z_mode'](controller, {a: False for a in ARMS}), model.DUAL)
                controller.stage_name = 'OPENING'
                model.set_mode(scope['choose_z_mode'](controller, {a: False for a in ARMS}))
                self.assertEqual(model.step({a: 0. for a in ARMS}, 2., True, .02), offsets)
            self.assertEqual(results[0], results[1])


