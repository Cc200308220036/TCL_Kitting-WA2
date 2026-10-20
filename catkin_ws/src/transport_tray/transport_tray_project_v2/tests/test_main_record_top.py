"""补记入口：仅视觉、层号幂等、未撤离拒绝，不连接机器人。"""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import test_v2_runtime as runtime_fakes

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_main_record_probe", ROOT / "main.py")
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)
_time_spec = importlib.util.spec_from_file_location("_task_summary_probe", ROOT / "tray_control/observability/timing.py")
summary_timing = importlib.util.module_from_spec(_time_spec)
_time_spec.loader.exec_module(summary_timing)


class TaskSummaryTests(TestCase):
    def test_three_layers_merge_states_and_keep_return_and_overhead(self):
        totals = {("流程", "初始化/其他"): .125}
        for layer in range(1, 4):
            for stage in ("Stage0", "Stage1", "Stage2", "Stage3单层" if layer == 1 else "Stage3多层"):
                totals[("第{}层/{}".format(layer, stage), "准备")] = 1.
                totals[("第{}层/{}".format(layer, stage), "运行")] = 2.
            if layer < 3:
                totals[("第{}层/返回取料".format(layer), "返回双臂")] = 4.
        original = totals.copy()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summary_timing.print_task_summary(totals)
        text = output.getvalue()
        self.assertEqual(totals, original)
        for layer in range(1, 4):
            for task in ("到预抓取位", "抓取tray（含抬升）", "转腰", "放置"):
                self.assertIn("第{}层搬运 / {}：3.000s".format(layer, task), text)
        self.assertEqual(text.count("返回到初始位置：4.000s"), 2)
        self.assertIn("初始化/阶段衔接：0.125s", text)
        self.assertIn("总阶段合计：44.125s", text)

    def test_displayed_rows_add_up_to_total_after_rounding(self):
        from decimal import Decimal
        totals = {("第1层/Stage0", "运行"): .0006,
                  ("第1层/Stage1", "运行"): .0006,
                  ("第1层/流程", "清理"): .0006}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summary_timing.print_task_summary(totals)
        lines = output.getvalue().splitlines()
        rows = [Decimal(line.rsplit("：", 1)[1][:-1]) for line in lines if line.startswith("  ")]
        total = Decimal(lines[-1].rsplit("：", 1)[1][:-1])
        self.assertEqual(sum(rows), total)
        self.assertEqual(total, Decimal("0.002"))


class RecordTopTests(TestCase):
    def setUp(self):
        self.fixture = runtime_fakes.RuntimeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.args = SimpleNamespace(config=str(ROOT / "tray_control/config/transport_tray.yaml"),
                                    state_file=str(self.fixture.path))
        self.fixture.state.save_state(dict(
            top_tray_pose=[0., 0., .4, 0., 0., 0., 1.], layer_index=1,
            target_layer_index=2, stage3_release_and_retreat_complete=True,
            place_tray_target=[0., 0., .433, 0., 0., 0., 1.],
            package_steps={"place": "FAILED"}), str(self.fixture.path))

    def test_legacy_cli_only_records_and_repeat_keeps_target_layer(self):
        pose = [.1, .2, .45, 0., 0., 2**-.5, 2**-.5]
        cli = ["main.py", "--execute", "--mode", "multi", "--record-top-after-place",
               "--state-file", str(self.fixture.path)]
        with patch.object(sys, "argv", cli), patch("terminal.confirm", return_value=True), \
                patch.object(self.fixture.multi, "capture_top_tray_pose", return_value=pose), \
                contextlib.redirect_stdout(io.StringIO()):
            main.main()
            main.main()
        saved = self.fixture.state.load_state(str(self.fixture.path))
        self.assertEqual(saved["top_tray_pose"], pose)
        self.assertEqual(saved["layer_index"], 2)
        self.assertEqual(saved["package_steps"]["place"], "SUCCEEDED")
        self.assertAlmostEqual(saved["stage3_placed_minus_p_place_z_mm"], 17.)
        sys.modules["naviai_controller"].NaviController.assert_not_called()

    def test_main_three_layer_loop_uses_public_preplacement_and_explicit_search_path(self):
        import tray_control
        f = self.fixture
        single = Mock(config=f.Config(state_path=str(f.path)), state_path=str(f.path))
        multi = Mock(config=single.config, state_path=str(f.path))
        events = []
        single.prep.side_effect = lambda **kw: events.append('prep')
        single.grip.side_effect = lambda **kw: events.append('grip')
        single.lift.side_effect = lambda **kw: events.append('lift')
        def first_layer(*args):
            events.append('single_place')
            f.state.update_state(str(f.path), layer_index=1, stage3_complete=True)
        def move_to_pre(**kwargs):
            events.append('move_to_pre')
        def place(**kwargs):
            events.append('multi_place')
            layer = f.state.load_state(str(f.path))['layer_index'] + 1
            f.state.update_state(str(f.path), layer_index=layer, stage3_complete=True)
            return SimpleNamespace(measurements={'force_csv_path': 'force.csv'})
        multi.move_to_pre.side_effect = move_to_pre
        multi.place.side_effect = place
        options = SimpleNamespace(config='unused', state_file=str(f.path), layers=3,
                                  lift_cm=[1., 1., 7.], search_path='circle', xy_offset=(.003, -.004))
        robot = f.Robot()
        with patch.object(tray_control, 'SingleLayerTray', return_value=single), \
                patch.object(tray_control, 'MultiLayerTray', return_value=multi), \
                patch('naviai_controller.NaviController', return_value=robot), \
                patch.object(main, 'stage2_waist', side_effect=lambda r: events.append('waist')), \
                patch.object(main, 'stage3_single', side_effect=first_layer), \
                contextlib.redirect_stdout(io.StringIO()):
            main.run(options)
        self.assertEqual(events, ['prep', 'grip', 'lift', 'waist', 'single_place'] +
                         ['prep', 'grip', 'lift', 'waist', 'move_to_pre', 'multi_place'] * 2)
        self.assertEqual([call.kwargs['height_m'] for call in single.lift.call_args_list], [.01, .01, .07])
        single.prep.assert_called_with(speed=.01, acceleration=.02)
        single.grip.assert_called_with(free_speed=.006, contact_speed=.001, acceleration=.01)
        for call in multi.move_to_pre.call_args_list:
            self.assertIs(call.kwargs['robot'], robot)
            self.assertEqual(call.kwargs['xy_offset'], (.003, -.004))
        for call in multi.place.call_args_list:
            self.assertEqual(call.kwargs['search_path'], 'circle')
        self.assertIsNotNone(multi.place.call_args_list[0].kwargs['record_top_runner'])
        self.assertIsNone(multi.place.call_args_list[1].kwargs['record_top_runner'])
        self.assertEqual(f.state.load_state(str(f.path))['layer_index'], 3)

    def test_missing_retreat_flag_rejects_before_vision_and_keeps_state(self):
        self.fixture.state.update_state(str(self.fixture.path), stage3_release_and_retreat_complete=False)
        before = self.fixture.path.read_bytes()
        with patch.object(self.fixture.multi, "capture_top_tray_pose") as capture:
            with self.assertRaisesRegex(RuntimeError, "撤离完成"):
                main.record_top_after_place(self.args)
        capture.assert_not_called()
        self.assertEqual(before, self.fixture.path.read_bytes())

    def test_vision_failure_keeps_previous_top_and_failed_step(self):
        before = self.fixture.path.read_bytes()
        with patch("terminal.confirm", return_value=True), \
                patch.object(self.fixture.multi, "capture_top_tray_pose", side_effect=RuntimeError("视觉失败")):
            with self.assertRaisesRegex(RuntimeError, "视觉失败"):
                main.record_top_after_place(self.args)
        self.assertEqual(before, self.fixture.path.read_bytes())

    def test_top_recording_runs_in_worker_while_main_thread_returns(self):
        vision=self.fixture.vision
        config=self.fixture.Config(state_path=str(self.fixture.path))
        frame=object()
        started,returned=threading.Event(),threading.Event()
        order=[]
        def record():
            self.assertIsNot(threading.current_thread(),threading.main_thread())
            self.assertIs(vision._RECORD_FRAME.value,frame)
            started.set()
            self.assertTrue(returned.wait(2.))
            order.append('record_done')
            return [0.,0.,.433,0.,0.,0.,1.]
        def restore(robot):
            self.assertIs(threading.current_thread(),threading.main_thread())
            self.assertTrue(started.wait(2.))
            self.assertFalse(vision.has_record_frame())
            order.append('return')
            returned.set()
        with patch('terminal.confirm',return_value=True), \
                patch.object(vision,'capture_record_frame',return_value=frame), \
                patch.object(main,'restore_initial_position',side_effect=restore):
            result=main.record_top_during_return(record,config,object())
        self.assertEqual(result[2],.433)
        self.assertEqual(order,['return','record_done'])
        self.assertFalse(vision.has_record_frame())

    def test_worker_error_is_propagated_after_return(self):
        config=self.fixture.Config(state_path=str(self.fixture.path))
        def fail():
            raise RuntimeError('估姿失败')
        with patch('terminal.confirm',return_value=True), \
                patch.object(self.fixture.vision,'capture_record_frame',return_value=object()), \
                patch.object(main,'restore_initial_position') as restore:
            with self.assertRaisesRegex(RuntimeError,'估姿失败'):
                main.record_top_during_return(fail,config,object())
        restore.assert_called_once()

    def test_frozen_frame_retry_does_not_recapture_or_query_moving_transform(self):
        import numpy as np
        vision=self.fixture.vision
        args=self.fixture.Config().stage3_args(True)
        frame=(np.zeros((2,2,3)),np.ones((2,2)),np.eye(4))
        frame[2][0,3]=.42
        estimator=SimpleNamespace(estimate=Mock(
            side_effect=[RuntimeError('重试'),[SimpleNamespace(label=args.label,T_object=np.eye(4))]]))
        with patch.object(vision,'_TRAY_POSE_ESTIMATOR',estimator), \
                patch.object(vision,'_capture_tray_frame') as capture, \
                patch('tray_control.common.runtime.vision_ready') as ready:
            pose=vision.record_from_frame(lambda:vision.estimate_tray_pose(args,'T_base_top'),frame)
        self.assertEqual(pose,[0.,0.,0.,0.,0.,0.,1.])
        capture.assert_not_called()
        ready.assert_not_called()
        self.assertEqual(estimator.estimate.call_count,2)
        for call in estimator.estimate.call_args_list:
            self.assertIs(call.args[0],frame[0])
            self.assertIs(call.args[1],frame[1])
            self.assertIs(call.kwargs['T_target_camera'],frame[2])
        self.assertFalse(vision.has_record_frame())

    def test_capture_copies_frame_and_transform_before_robot_returns(self):
        import numpy as np
        vision=self.fixture.vision
        args=self.fixture.Config().stage3_args(True)
        color,depth,transform=np.zeros((2,2,3)),np.ones((2,2)),np.eye(4)
        camera=SimpleNamespace(T_base_camera=Mock(return_value=transform))
        module=SimpleNamespace(RgbdPoseEstimator=Mock(),_capture_rgbd=Mock(return_value=(color,depth)))
        with patch.dict(sys.modules,{'estimate_rgbd_pose':module}), \
                patch.object(vision,'_TRAY_POSE_ESTIMATOR',SimpleNamespace(camera_to_base=camera)), \
                patch('tray_control.common.runtime.vision_ready'):
            frame=vision.capture_record_frame(args)
        color[:]=9;depth[:]=9;transform[0,3]=9
        self.assertEqual(frame[0].sum(),0.)
        np.testing.assert_array_equal(frame[1],np.ones((2,2)))
        np.testing.assert_array_equal(frame[2],np.eye(4))
        camera.T_base_camera.assert_called_once()

    def test_preplace_uses_current_dual_tcp_and_saved_grasp_without_vision(self):
        self.check_preplace_pose_source(True)

    def test_preplace_switch_restores_visual_pose_and_grasp(self):
        self.check_preplace_pose_source(False)

    def check_preplace_pose_source(self, from_tcp):
        import numpy as np
        f=self.fixture
        pickup=[.1,0.,.6,0.,0.,0.,1.]
        tcp={side:f.poses[arm].copy() for side,arm in [('left',runtime_fakes.ArmGroup.LEFT),('right',runtime_fakes.ArmGroup.RIGHT)]}
        grasp=f.state.compute_grasp_transforms(pickup,tcp['left'],tcp['right'])
        shift=f.state.pose7_to_matrix([.05,-.02,0.,0.,0.,2**-.5,2**-.5])
        moved={side:f.state.matrix_to_pose7(shift@f.state.pose7_to_matrix(pose)) for side,pose in tcp.items()}
        expected=f.state.matrix_to_pose7(shift@f.state.pose7_to_matrix(pickup))
        f.state.save_state({'top_tray_pose':[0.,0.,.4,0.,0.,0.,1.],'layer_index':1,
                            **({'tray_tcp_grasp':grasp} if from_tcp else {})},str(f.path))
        for side,arm in [('left',runtime_fakes.ArmGroup.LEFT),('right',runtime_fakes.ArmGroup.RIGHT)]:
            f.poses[arm]=np.array(moved[side])
        tray=f.api.MultiLayerTray(state_path=str(f.path),confirm=Mock(return_value=True))
        self.assertTrue(tray.config.multi['preplace']['held_tray_pose_from_tcp'])
        tray.config.multi['preplace']['held_tray_pose_from_tcp']=from_tcp
        if not from_tcp:
            expected=[.12,.03,.6,0.,0.,0.,1.]
        options=SimpleNamespace(search_path='circle')
        original_args = tray.config.stage3_args
        with patch.object(f.state, 'estimate_tray_from_tcp', wraps=f.state.estimate_tray_from_tcp) as estimate, \
                patch.object(f.stage2, 'estimate_tray_from_tcp', estimate), \
                patch.object(f.stage1,'read_tcp_pair',return_value=moved), \
                patch.object(f.vision,'estimate_tray_pose',return_value=expected) as vision, \
                patch.object(tray,'place',return_value=SimpleNamespace(measurements={'force_csv_path':'force.csv'})) as place, \
                patch('terminal.confirm',return_value=True):
            main.stage3_multi(tray,options,robot=f.Robot())
        estimate.assert_called_once()
        place.assert_called_once_with(search_path='circle', record_top_runner=None)
        self.assertEqual(tray.config.stage3_args, original_args)
        if from_tcp:
            vision.assert_not_called()
        else:
            vision.assert_called_once()
        saved=f.state.load_state(str(f.path))
        np.testing.assert_allclose(saved['grasp_tray_pose'],expected,atol=1e-12)
        self.assertEqual(saved['grasp_pose_source'],'dual_tcp' if from_tcp else tray.config.common['vision']['vision_backend'])
        self.assertEqual(saved['tray_tcp_grasp'],grasp if from_tcp else
                         f.state.compute_grasp_transforms(expected,moved['left'],moved['right']))
        self.assertEqual(f.calls, ['stop', 'offset', 'offset'])
        self.assertTrue(saved['stage2_complete'])
