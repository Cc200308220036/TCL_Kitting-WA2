"""独立Stage3测试选项：同步XY偏置，不连接机器人。"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("multi_test_probe", ROOT / "test_stage3_multi_z_search_place.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class XYOffsetTests(TestCase):
    def setUp(self):
        self.state = {}
        self.starts = {
            "left": np.array([.4, .3, .5, 0., -.6, 0., .8]),
            "right": np.array([.45, -.3, .51, 0., -.8, 0., .6]),
        }
        self.args = SimpleNamespace(state_file="fake_state", max_preplace_translation=.6,
                                    preplace_velocity=.01, preplace_acceleration=.02,
                                    stack_pitch=.033, preplace_clearance=.02)
        self.robot = Mock()
        self.robot.movel.return_value = True
        targets = {side: pose.tolist() for side, pose in self.starts.items()}
        self.plan = SimpleNamespace(held_pose=[0., 0., .4], pre_pose=[0., 0., .5],
            place_pose=[0., 0., .48], xy_targets=targets, pre_targets=targets,
            skip_z=True, target_layer_index=2)
        tree = ast.parse((ROOT / "tray_control/stage2.py").read_text())
        state_tree = ast.parse((ROOT / "tray_control/common/state.py").read_text())
        functions = [n for n in state_tree.body if isinstance(n, ast.FunctionDef) and n.name == "vector"]
        functions += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "move_to_pre"]
        scope = {"np": np, "Sequence": list,
            "rospy": SimpleNamespace(is_shutdown=lambda: False), "input": Mock(),
            "ArmGroup": SimpleNamespace(DUAL="dual"), "timing_state": Mock(),
            "update_state": lambda path, **kw: self.state.update(kw)}
        exec(compile(ast.fix_missing_locations(ast.Module(functions, [])), "preplace", "exec"), scope)
        self.move = lambda offset: scope["move_to_pre"](self.robot, self.args, self.plan, offset)

    def test_both_arms_move_minus_eight_mm_y_and_keep_z_orientation(self):
        self.move([0., -.008])
        targets = np.array(self.robot.movel.call_args.args[0]).reshape(2, 7)
        for index, side in enumerate(("left", "right")):
            np.testing.assert_allclose(targets[index][:2], self.starts[side][:2] + [0., -.008])
            np.testing.assert_array_equal(targets[index][2:], self.starts[side][2:])
        self.assertTrue(all(call.kwargs['is_async'] is False for call in self.robot.movel.call_args_list))
        self.assertTrue(self.state["stage2_complete"])
        self.assertEqual(self.state["test_stage3_xy_offset_m"], [0., -.008])

    def test_offset_exception_stops_without_marking_ready(self):
        self.robot.movel.side_effect = [True, RuntimeError("MoveL异常")]
        with self.assertRaisesRegex(RuntimeError, "MoveL异常"):
            self.move([0., -.008])
        self.assertFalse(self.state["stage2_complete"])
        self.assertFalse(self.state["test_stage3_xy_offset_complete"])
        self.assertEqual(self.robot.stop.call_count, 2)

    def test_offset_service_failure_stops_without_marking_ready(self):
        self.robot.movel.side_effect = [True, False]
        with self.assertRaisesRegex(RuntimeError, "同步MoveL失败"):
            self.move([0., -.008])
        self.assertEqual(self.robot.movel.call_count, 2)
        self.assertFalse(self.state["stage2_complete"])
        self.assertEqual(self.robot.stop.call_count, 2)

    def test_zero_offset_does_not_send_extra_motion(self):
        self.move([0., 0.])
        self.assertEqual(self.robot.movel.call_count, 1)  # 仅P_pre XY，无偏置段。
        self.assertTrue(self.state["test_stage3_xy_offset_complete"])
