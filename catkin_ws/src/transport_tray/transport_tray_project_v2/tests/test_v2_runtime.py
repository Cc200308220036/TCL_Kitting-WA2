"""四入口的离线调用链与状态检查；只模拟机器人/视觉边界，不发送运动。"""
import contextlib
import csv
import io
import itertools
from enum import Enum
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

class ArmGroup(Enum):
    LEFT='left'; RIGHT='right'; DUAL='dual'; WAIST='waist'

def load_script(name, directory=ROOT):
    spec=importlib.util.spec_from_file_location('v2_'+name,directory/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class RuntimeTests(TestCase):
    def setUp(self):
        ros=ModuleType('rospy');ros.core=SimpleNamespace(is_initialized=lambda:True);ros.is_shutdown=lambda:False
        for name in ('sleep','logwarn','loginfo','logerr','loginfo_throttle','logwarn_throttle','Subscriber','Rate','init_node'):
            setattr(ros,name,Mock())
        nav=ModuleType('naviai_controller');nav.ArmGroup=ArmGroup;nav.RobotModel=SimpleNamespace(WA2='wa2');nav.HandType=SimpleNamespace();nav.NaviController=Mock()
        geo=ModuleType('geometry_msgs');msg=ModuleType('geometry_msgs.msg');msg.WrenchStamped=type('WrenchStamped',(),{});geo.msg=msg
        self.modules=patch.dict(sys.modules,{'rospy':ros,'naviai_controller':nav,'geometry_msgs':geo,'geometry_msgs.msg':msg,'rosservice':ModuleType('rosservice')})
        self.modules.start();self.addCleanup(self.modules.stop)
        from tray_control import api, stage0, stage1, stage2, stage3, stage3_multi, stage3_common
        from tray_control.common import state, vision
        from tray_control.config import TrayConfig
        self.api,self.state,self.stage0,self.stage1,self.stage2,self.stage3,self.multi,self.vision=api,state,stage0,stage1,stage2,stage3,stage3_multi,vision
        self.common=stage3_common
        self.Config=TrayConfig
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'state.json'
        self.calls=[]
        self.poses={ArmGroup.LEFT:np.array([.1,.2,.7,0.,0.,0.,1.]),ArmGroup.RIGHT:np.array([.1,-.2,.71,0.,0.,0.,1.])}
        self.joints=[.02,.01,.1,.3]
        outer=self
        class Robot:
            def stop(self):outer.calls.append('stop')
            def clear_servo_params(self):pass
            def get_joints(self,arm):return list(outer.joints)
            def movej(self,target,arm,**kw):outer.calls.append(('waist',list(target),kw));outer.joints[:]=target;return True
            def movel(self,target,arm,**kw):
                outer.calls.append('offset');outer.poses[ArmGroup.LEFT]=np.array(target[:7]);outer.poses[ArmGroup.RIGHT]=np.array(target[7:]);return True
            def movel_relative_base(self,delta,arm,**kw):
                outer.calls.append(('lift',delta[2]))
                for group,start in ((ArmGroup.LEFT,0),(ArmGroup.RIGHT,3)):
                    outer.poses[group][:3] += delta[start:start+3]
                return True
        self.Robot=Robot

    def test_config_assets_and_runtime_are_local_and_pruned(self):
        c=self.Config()
        self.assertTrue(Path(c.calibration_file()).is_file())
        self.assertTrue(Path(c.vision_args()['tray_pose_camera_to_head']).is_file())
        for p in (c.state_path,c.force_log_dir,c.calibration_file()):self.assertTrue(Path(p).is_relative_to(ROOT))
        self.assertNotIn('single_layer',c.data);self.assertNotIn('light_lift',c.multi)
        self.assertFalse(any('alignment' in k for k in c.common['stage3']))
        a=c.stage3_args(True);self.assertEqual(a.search_z_speed_limit,.003)
        self.assertEqual(a.search_z_retreat_distance,c.multi['search']['search_z_retreat_distance'])
        self.assertEqual(a.search_z_retreat_trigger_cycles,3)
        self.assertEqual((a.radius_z_down_speed,a.radius_z_retreat_speed),(.002,.001))
        single=c.stage3_args(False)
        self.assertEqual((single.max_free_speed,single.max_acceleration), (.006,.02))
        self.assertFalse(hasattr(single, "max_contact_speed"))
        self.assertEqual((a.max_free_speed,a.max_contact_speed,a.max_acceleration),
            tuple(c.common["stage3"]["descent_motion"]["multi"][k] for k in ("speed", "contact_speed", "acceleration")))

    def test_shared_pose_validation_preserves_boundary_types_and_input_values(self):
        raw = np.array([.1, .2, .3, 0., 0., 0., 2.])
        expected = [.1, .2, .3, 0., 0., 0., 1.]
        self.assertEqual(self.state.validate_pose7(iter(raw)), expected)
        np.testing.assert_array_equal(self.stage0.validate_xyzquat(raw), raw)
        for validate in (self.stage1.pose7, self.common.pose7):
            result = validate(raw, 'TCP')
            self.assertIsInstance(result, np.ndarray)
            np.testing.assert_array_equal(result, expected)
            result[0] = 99.
            self.assertEqual(raw[0], .1)
        visual = self.vision._retry_tray_pose(lambda *a: raw, SimpleNamespace(), 'tray')
        self.assertIsInstance(visual, list)
        self.assertEqual(visual, expected)
        self.assertEqual(raw[-1], 2.)
        invalid = ([0.] * 6, [0.] * 7, [float('nan'), 0., 0., 0., 0., 0., 1.])
        for pose in invalid:
            with self.subTest(pose=pose):
                with self.assertRaises(ValueError):
                    self.state.validate_pose7(pose)
                for validate in (self.stage1.pose7, self.common.pose7):
                    with self.assertRaises(ValueError):
                        validate(pose, 'TCP')
                with self.assertRaises(RuntimeError):
                    self.stage0.validate_xyzquat(pose)
                with self.assertRaises(RuntimeError):
                    self.vision._retry_tray_pose(lambda *a: pose, SimpleNamespace(), 'tray')

    def test_shared_tray_estimator_preserves_geometry_and_cached_fast_path(self):
        from scipy.spatial.transform import Rotation
        tray_pose = [.4, -.1, .5, *Rotation.from_euler('xyz', [.04, -.03, 3.12]).as_quat()]
        grasp = {
            'left': [.02, .2, .01, *Rotation.from_euler('xyz', [.5, -.3, .2]).as_quat()],
            'right': [-.01, -.2, -.02, *Rotation.from_euler('xyz', [-.2, .4, -.3]).as_quat()],
        }
        base = self.state.pose7_to_matrix(tray_pose)
        tcp = {side: self.state.matrix_to_pose7(base @ self.state.pose7_to_matrix(pose))
               for side, pose in grasp.items()}
        expected = [.4, -.1, .5, *Rotation.from_euler('z', 3.12).as_quat()]
        pose, disagreement = self.state.estimate_tray_from_tcp(tcp['left'], tcp['right'], grasp)
        np.testing.assert_allclose(pose, expected, atol=1e-12)
        self.assertLess(disagreement['position_m'], 1e-12)
        self.assertLess(disagreement['yaw_rad'], 1e-12)
        components = self.state.prepare_grasp_components(grasp)
        with patch.object(self.state, 'prepare_grasp_components', side_effect=AssertionError('重复预计算')), \
                patch.object(self.state, 'validate_pose7', side_effect=AssertionError('实时循环重复校验')), \
                patch.object(np.linalg, 'inv', side_effect=AssertionError('实时循环重复求逆')):
            for _ in range(10):
                fast, delta = self.state.estimate_tray_fast(tcp['left'], tcp['right'], components)
                np.testing.assert_array_equal(fast, pose)
                self.assertEqual(delta, disagreement)
        self.assertIs(self.multi.estimate_tray_fast, self.state.estimate_tray_fast)

    def test_stage0_calibration_entry_writes_v2_schema_without_motion_or_task_state(self):
        script = load_script('stage0_calibrate', ROOT / 'tray_control')
        destination = self.temp.name + '/calibration.json'
        original = dict(layer_index=2, stage3_complete=True)
        self.state.save_state(original, str(self.path))
        before = self.path.read_bytes()
        pose = [0., 0., .4, 0., 0., 0., 1.]
        tcp = {side: self.poses[arm].tolist() for side, arm in
               (('left', ArmGroup.LEFT), ('right', ArmGroup.RIGHT))}
        with patch.object(self.vision, 'estimate_tray_pose', return_value=pose), \
                patch.object(script, 'read_current_tcp_xyzquats', return_value=tcp), \
                patch('terminal.confirm', return_value=True), \
                contextlib.redirect_stdout(io.StringIO()):
            script.main(['--calibrate', '--calibration-file', destination])
        saved = self.stage0.load_calibration(destination)
        self.assertEqual(saved['vision_backend'], 'tray-pose-yolo')
        self.assertEqual(saved['vision_label'], 'white_tray')
        self.assertEqual(saved['reference_tray_xyzquat'], pose)
        self.assertEqual(saved['reference_tcp_xyzquats'], tcp)
        self.assertEqual(self.path.read_bytes(), before)
        sys.modules['naviai_controller'].NaviController.assert_not_called()

    def test_stage0_calibration_cancel_preserves_existing_file(self):
        script = load_script('stage0_calibrate', ROOT / 'tray_control')
        destination = Path(self.temp.name) / 'calibration.json'
        destination.write_text('existing calibration')
        with patch.object(self.vision, 'estimate_tray_pose', return_value=[0.,0.,.4,0.,0.,0.,1.]), \
                patch.object(script, 'read_current_tcp_xyzquats') as read_tcp, \
                patch('terminal.confirm', return_value=False), \
                contextlib.redirect_stdout(io.StringIO()):
            script.main(['--calibrate', '--calibration-file', str(destination)])
        self.assertEqual(destination.read_text(), 'existing calibration')
        read_tcp.assert_not_called()

    def test_stage0_calibration_preview_uses_current_planner_and_separate_state(self):
        script = load_script('stage0_calibrate', ROOT / 'tray_control')
        preview = Path(self.temp.name) / 'preview.json'
        with patch.object(self.vision, 'estimate_tray_pose', return_value=[0.,0.,.4,0.,0.,0.,1.]), \
                patch.object(script, 'read_current_tcp_xyzquats') as read_tcp, \
                patch('terminal.confirm', return_value=True), \
                contextlib.redirect_stdout(io.StringIO()):
            script.main(['--preview', '--state-file', str(preview)])
        saved = self.state.load_state(str(preview))
        self.assertFalse(saved['stage0_complete'])
        self.assertEqual(set(saved['stage0_pregrasp_targets']), {'left','right'})
        read_tcp.assert_not_called()
        sys.modules['naviai_controller'].NaviController.assert_not_called()

    def test_stage_result_remains_public_frozen_and_has_independent_defaults(self):
        from dataclasses import FrozenInstanceError
        from tray_control import StageResult
        self.assertIs(StageResult, self.api.StageResult)
        first = StageResult('prep', True, str(self.path))
        second = StageResult('prep', True, str(self.path))
        first.measurements['test'] = 1
        self.assertEqual(second.measurements, {})
        first.targets['test'] = 2
        self.assertEqual(second.targets, {})
        with self.assertRaises(FrozenInstanceError):
            first.completed = False

    def test_default_state_path_matches_configuration_and_custom_override(self):
        self.assertEqual(self.state.DEFAULT_STATE_FILE, self.Config().state_path)
        self.assertEqual(Path(self.state.DEFAULT_STATE_FILE), ROOT / 'tray_control/runtime/tray_state.json')
        self.assertEqual(self.Config(state_path=str(self.path)).state_path, str(self.path))

    def test_dual_gripper_worker_preserves_open_and_close_completion_rules(self):
        from tray_control.common.gripper import DualGripperWorker

        grippers = Mock()
        grippers.config.open_position_mm = 35.
        # 闭爪被物体挡住时仍成功；自定义准备开度也沿用闭爪任务。
        grippers.close.return_value = SimpleNamespace(final_position_mm=25.)
        for target in (None, 80.):
            worker = DualGripperWorker(grippers, action="close", target_position_mm=target)
            worker.start()
            worker.join(1.)
            self.assertEqual(worker.status(), (True, None))
            with self.assertRaisesRegex(RuntimeError, '重复启动'):
                worker.start()
        self.assertEqual(grippers.close.call_count, 2)
        self.assertEqual(grippers.command.call_count, 2)
        for position, accepted in ((35., True), (33., True), (32.9, False), (float('nan'), False)):
            with self.subTest(open_position=position):
                grippers.open.return_value = SimpleNamespace(final_position_mm=position)
                worker = DualGripperWorker(grippers, action="open")
                worker.start()
                worker.join(1.)
                done, error = worker.status()
                self.assertEqual(done, accepted)
                self.assertEqual(error is None, accepted)
        grippers.close.side_effect = RuntimeError('服务失败')
        worker = DualGripperWorker(grippers, action="close")
        worker.start()
        worker.join(1.)
        done, error = worker.status()
        self.assertFalse(done)
        self.assertIn('服务失败', error)

    def test_multilayer_cycle_uses_one_wrench_snapshot_for_control_and_logging(self):
        args = self.Config(state_path=str(self.path)).stage3_args(True)
        self.state.save_state(dict(stage2_complete=True,
            place_tray_target=[0., 0., .4, 0., 0., 0., 1.],
            tray_tcp_grasp={'left': [0., .2, 0., 0., 0., 0., 1.],
                            'right': [0., -.2, 0., 0., 0., 0., 1.]}), str(self.path))
        c = self.multi.MultilayerStage3(args)
        c.flow = SimpleNamespace(both_force_since=None)
        c.baseline = {arm: np.zeros(6) for arm in self.poses}
        c.check_robot_state_without_tcp = Mock()
        c.commanded_descent_pose = Mock(return_value=([0., 0., .5, 0., 0., 0., 1.], {}))
        c.wrenches = Mock()
        frames = {arm: np.array([100., 200., 1. + index, 10., 20., 30.])
                  for index, arm in enumerate(self.poses)}
        reads = {arm: 0 for arm in self.poses}
        def snapshot(arm):
            reads[arm] += 1
            # 若再次取帧，则返回完全不同的数据。
            return frames[arm] * reads[arm], 10., reads[arm]
        c.wrenches.snapshot.side_effect = snapshot
        with patch.object(self.common.time, 'monotonic', return_value=10.):
            cycle, source = c.read_cycle(10., .02)
        self.assertEqual(source, 'commanded')
        self.assertEqual(reads, {arm: 1 for arm in self.poses})
        for arm in self.poses:
            np.testing.assert_array_equal(cycle.raw_wrenches[arm], frames[arm])
            self.assertEqual(cycle.support[arm], c.z_sign(arm) * frames[arm][2])
        c.cleanup()

    def test_shared_feedback_waits_and_baselines_keep_distinct_freshness_rules(self):
        from tray_control.common import feedback
        clock = SimpleNamespace(now=10.)
        robot, wrenches = Mock(), Mock()
        robot.get_tcp_rt.side_effect = lambda arm: self.poses[arm].copy()
        robot.get_tcp_age.return_value = 0.
        robot.get_uplimb_state_age.return_value = 0.
        robot.get_cmd_num.return_value = 0
        robot.get_is_singular.side_effect = lambda arm: None if clock.now == 10. else False
        wrenches.snapshot.side_effect = lambda arm: (np.ones(6), clock.now, 1)
        with patch.object(feedback.time, 'monotonic', side_effect=lambda: clock.now), \
                patch.object(feedback.rospy, 'sleep', side_effect=lambda dt: setattr(clock, 'now', clock.now + dt)):
            poses = feedback.wait_feedback_ready(robot, wrenches, .2, error_message='未就绪')
            self.assertGreater(clock.now, 10.)
            for arm in self.poses:
                np.testing.assert_array_equal(poses[arm], self.poses[arm])
            # 模拟一条消息被轮询两次，且接收时间已过期。
            for stale_timeout in (None, .2):
                clock.now = 10.
                wrenches.snapshot.side_effect = lambda arm: (
                    np.full(6, 3.), clock.now - 1., int(round((clock.now - 10.) / .004)))
                if stale_timeout is None:
                    stats = feedback.collect_wrench_baseline(wrenches, .12, label='零偏')
                    for mean, std, count in stats.values():
                        np.testing.assert_array_equal(mean, np.full(6, 3.))
                        np.testing.assert_array_equal(std, np.zeros(6))
                        self.assertGreaterEqual(count, 20)
                        self.assertLess(count, 40)  # 序号去重，不能把60次轮询当作60帧。
                else:
                    with self.assertRaisesRegex(RuntimeError, '负载基线样本不足'):
                        feedback.collect_wrench_baseline(wrenches, .12, label='负载基线', stale_timeout=stale_timeout)
            robot.get_cmd_num.return_value = 15
            with self.assertRaisesRegex(RuntimeError, 'PROTECTED'):
                feedback.wait_feedback_ready(robot, wrenches, .2, error_message='未就绪')

    def test_managed_controller_cleans_every_boundary_after_exception(self):
        from tray_control.common.runtime import managed_controller
        events = []
        controller = Mock()
        subscriber = Mock()
        subscriber.unregister.side_effect = lambda: events.append('unsubscribe')
        controller.wrenches = SimpleNamespace(_subscribers=[subscriber])
        controller.mark_force_abort.side_effect = lambda exc: events.append('abort')
        controller.close_force_log.side_effect = lambda: events.append('close')
        controller.cleanup.side_effect = lambda: events.append('cleanup')
        with self.assertRaisesRegex(RuntimeError, '运动失败'):
            with managed_controller(controller, force_log=True):
                raise RuntimeError('运动失败')
        self.assertEqual(events, ['abort', 'close', 'cleanup', 'unsubscribe'])
        events.clear()
        controller.close_force_log.side_effect = RuntimeError('日志失败')
        with self.assertRaisesRegex(RuntimeError, '日志失败'):
            with managed_controller(controller, force_log=True):
                pass
        self.assertEqual(events, ['cleanup', 'unsubscribe'])

    def test_single_public_place_preserves_main_lowering_recording_and_cleanup(self):
        import tray_control
        script = load_script('main')
        tray = self.api.SingleLayerTray(state_path=str(self.path), confirm=Mock(return_value=True))
        controller = Mock()
        subscriber = Mock()
        controller.wrenches = SimpleNamespace(_subscribers=[subscriber])
        tcp = {side: self.poses[arm].tolist() for side, arm in
               (('left', ArmGroup.LEFT), ('right', ArmGroup.RIGHT))}
        pose = [0., 0., .4, 0., 0., 0., 1.]
        runner = Mock(side_effect=lambda record: record())
        with patch.object(self.api, 'SingleStage3Place', return_value=controller), \
                patch.object(self.stage1, 'read_tcp_pair', return_value=tcp), \
                patch.object(self.vision, 'estimate_tray_pose', return_value=pose):
            result = script.stage3_single(tray, self.Robot(), runner)
        self.assertTrue(result.completed)
        runner.assert_called_once()
        controller.run.assert_called_once()
        controller.cleanup.assert_called_once()
        subscriber.unregister.assert_called_once()
        saved = self.state.load_state(str(self.path))
        self.assertEqual(saved['layer_index'], 1)
        self.assertTrue(saved['stage3_complete'])
        self.assertEqual(saved['package_steps']['place'], 'SUCCEEDED')
        self.assertEqual(saved['top_tray_pose'], pose)
        self.assertAlmostEqual(self.poses[ArmGroup.LEFT][2], .65)
        self.assertAlmostEqual(self.poses[ArmGroup.RIGHT][2], .66)
        # 取消预下降时不创建控制器、不发送MoveL，也不识别顶面。
        tray.confirm.return_value = False
        with patch.object(self.api, 'SingleStage3Place') as factory, \
                patch.object(self.vision, 'record_first_layer') as record:
            with self.assertRaises(PermissionError):
                script.stage3_single(tray, self.Robot())
        factory.assert_not_called()
        record.assert_not_called()

    def test_business_state_helpers_preserve_previous_top_and_stage_boundaries(self):
        top = [0., 0., .4, 0., 0., 0., 1.]
        self.state.save_state(dict(top_tray_pose=top, layer_index=2,
            package_steps={'prep': 'SUCCEEDED'}, stage0_complete=True,
            stage1_complete=True, stage2_complete=True, stage2_lift_complete=True,
            stage3_complete=True, stage3_release_and_retreat_complete=True,
            grasp_tray_pose=top, grasp_tcp_poses={}, tray_tcp_grasp={}), str(self.path))
        cleared = self.state.invalidate_grasp_state(str(self.path))
        self.assertEqual(cleared['top_tray_pose'], top)
        self.assertEqual(cleared['layer_index'], 2)
        self.assertTrue(cleared['stage0_complete'])
        self.assertTrue(cleared['stage3_release_and_retreat_complete'])
        self.assertFalse(cleared['stage1_complete'])
        self.assertFalse(cleared['stage2_lift_complete'])
        self.assertIsNone(cleared['tray_tcp_grasp'])
        failed = self.state.set_step_status(str(self.path), 'grip', 'FAILED')
        self.assertEqual(failed['package_steps'], {'prep': 'SUCCEEDED', 'grip': 'FAILED'})
        tcp = {side: self.poses[arm].tolist() for side, arm in
               (('left', ArmGroup.LEFT), ('right', ArmGroup.RIGHT))}
        grasp = self.state.compute_grasp_transforms(top, tcp['left'], tcp['right'])
        saved = self.state.save_grasp_state(str(self.path), top, tcp, grasp, 'dual_tcp',
                                            stage3_release_and_retreat_complete=False)
        self.assertTrue(saved['stage1_complete'])
        self.assertEqual(saved['top_tray_pose'], top)
        self.assertEqual(saved['grasp_tcp_poses'], tcp)
        self.assertFalse(saved['stage3_release_and_retreat_complete'])
        self.assertFalse(saved['stage3_complete'])

    def test_stage0_plan_keeps_public_calibration_error_and_failure_status(self):
        tray = self.api.SingleLayerTray(state_path=str(self.path), confirm=Mock(return_value=True))
        with patch.object(self.vision, 'estimate_tray_pose', return_value=[0.,0.,.5,0.,0.,0.,1.]), \
                patch.object(self.stage0, 'load_calibration', return_value={
                    'vision_backend': 'tray-pose-yolo', 'vision_label': 'wrong'}), \
                patch.object(self.stage0, 'move_dual_to_pregrasp') as move:
            with self.assertRaisesRegex(self.api.TrayOperationError, '标定不一致'):
                tray.prep()
        move.assert_not_called()
        self.assertEqual(self.state.load_state(str(self.path))['package_steps']['prep'], 'FAILED')

    def test_search_observation_uses_distinct_xy_and_held_baselines(self):
        c = self.multi.MultilayerStage3.__new__(self.multi.MultilayerStage3)
        c.args = SimpleNamespace(slot_min_descent=.003, radius_yaw_extra_descent=.0005)
        c.search_start_z = .5
        c.yaw_search = self.multi.YawSearch(start_z=.49)
        c.history = self.multi.SearchHistory(drop=.004)
        c.baseline = {arm: np.array([0., 2., 3., 0., 0., 0.]) for arm in self.poses}
        c.search_start_raw_fx = {arm: 10. for arm in self.poses}
        c.search_start_raw_fy = {arm: 20. for arm in self.poses}
        raw = {arm: np.array([11., 22., 30., 0., 0., 0.]) for arm in self.poses}
        support = {arm: 7. for arm in self.poses}
        observation = c.calculate_search_observation(SimpleNamespace(
            raw_wrenches=raw, tray_estimate=[0.,0.,.48,0.,0.,0.,1.], support=support))
        for arm in self.poses:
            self.assertEqual(observation.xy_fx[arm], 1.)
            self.assertEqual(observation.xy_fy[arm], 2.)
            self.assertEqual(observation.yaw_fy[arm], 20.)
            self.assertEqual(support[arm], 7.)
            np.testing.assert_array_equal(raw[arm], [11.,22.,30.,0.,0.,0.])
        self.assertAlmostEqual(observation.drop, .02)
        self.assertAlmostEqual(observation.yaw_drop, .01)
        self.assertAlmostEqual(observation.required_drop, .0045)

    def test_retreat_feedback_preserves_missing_tcp_and_wrench_check_order(self):
        c = self.stage3.SingleStage3Place(self.Config().stage3_args(False))
        c.robot = Mock()
        c.robot.get_is_singular.return_value = False
        c.robot.get_cmd_num.return_value = 0
        c.robot.get_tcp_rt.side_effect = lambda arm: self.poses[arm].copy()
        c.wrenches = Mock()
        c.wrenches.snapshot.return_value = (np.zeros(6), 10., 1)
        c.force_recorder = Mock()
        with patch.object(self.common.time, 'monotonic', return_value=10.):
            result = c.read_retreat_feedback(self.poses, .005)
            self.assertTrue(result.reached)
            self.assertEqual(result.position_errors, [0., 0.])
            self.assertEqual(set(result.raw_wrenches), {'left', 'right'})
            c.robot.get_tcp_age.assert_not_called()
            c.wrenches.reset_mock()
            c.robot.get_tcp_rt.side_effect = lambda arm: None if arm == ArmGroup.LEFT else self.poses[arm].copy()
            result = c.read_retreat_feedback(self.poses, .005)
            self.assertFalse(result.reached)
            self.assertEqual(set(result.raw_wrenches), {'right'})
            c.wrenches.snapshot.assert_called_once_with(ArmGroup.RIGHT)
            c.robot.get_tcp_rt.side_effect = lambda arm: self.poses[arm].copy()
            c.wrenches.snapshot.return_value = (np.zeros(6), 9., 1)
            with self.assertRaisesRegex(RuntimeError, '腕力在撤离时超时'):
                c.read_retreat_feedback(self.poses, .005)
            c.robot.get_is_singular.return_value = True
            with self.assertRaisesRegex(RuntimeError, '撤离时进入奇异'):
                c.read_retreat_feedback(self.poses, .005)
        c.cleanup()

    def test_stage1_projects_force_without_resultant_or_torque_limits(self):
        import time
        common = self.stage1
        config_values = dict(self.Config().common['stage1'])
        config = common.make_stage1_config(**config_values)
        self.assertFalse(hasattr(config, 'force_limit_n'))
        self.assertFalse(hasattr(config, 'torque_limit_nm'))
        arm = self.stage1.ArmGroup.LEFT
        controller = common.TrayApproach.__new__(common.TrayApproach)
        controller.config = config
        controller.bias = {arm: np.array([10., 20., 30., 1., 2., 3.])}
        controller.reaction_axes = {arm: np.array([0., 1., 0.])}
        controller.force_filters = {arm: common.ScalarLowPass(5.)}
        controller.wrenches = Mock()
        raw = controller.bias[arm] + np.array([500., .7, -600., 100., -200., 300.])
        controller.wrenches.snapshot.return_value = (raw, time.monotonic(), 1)
        compression = controller.read_force(arm, .02)
        self.assertAlmostEqual(compression, .7)
        controller.wrenches.snapshot.return_value = (raw, time.monotonic() - 1., 1)
        with self.assertRaisesRegex(RuntimeError, '腕力数据超时'):
            controller.read_force(arm, .02)

    def test_stage1_closes_and_cleans_up_with_large_off_axis_loads(self):
        common = self.stage1
        config_values = dict(self.Config().common['stage1'])
        config = common.make_stage1_config(**config_values)
        targets = {side: self.poses[arm].tolist() for side, arm in
                   (('left', self.stage1.ArmGroup.LEFT), ('right', self.stage1.ArmGroup.RIGHT))}
        controller = self.stage1.TrayApproach(config, targets)
        controller.robot = Mock()
        controller.anchor_poses = {arm: pose.copy() for arm, pose in self.poses.items()}
        controller.last_commands = {arm: pose.copy() for arm, pose in self.poses.items()}
        controller.current_pose_checked = Mock()
        controller.log_stage1_summary = Mock()
        controller.wrenches = Mock()
        clock = SimpleNamespace(now=10.)
        def snapshot(arm):
            force = np.array([500., 0., -600.]) + .7 * controller.reaction_axes[arm]
            return np.r_[force, [100., -200., 300.]], clock.now, 1
        controller.wrenches.snapshot.side_effect = snapshot
        rate = Mock()
        rate.sleep.side_effect = lambda: setattr(clock, 'now', clock.now + .02)
        worker = Mock()
        worker.status.return_value = (True, None)
        with patch.object(common.time, 'monotonic', side_effect=lambda: clock.now), \
                patch.object(common.rospy, 'is_shutdown', return_value=False), \
                patch.object(common.rospy, 'Rate', return_value=rate), \
                patch.object(common, 'input', return_value=''), \
                patch.object(common, 'DualGripperWorker', return_value=worker):
            try:
                controller.run_approach()
            finally:
                controller.cleanup()
        worker.start.assert_called_once()
        worker.join.assert_called_once()
        controller.log_stage1_summary.assert_called_once()
        self.assertTrue(controller.robot.servol.called)
        controller.robot.stop.assert_called_once()
        controller.robot.clear_servo_params.assert_called_once()
        self.assertFalse(controller.servo_started)

    def test_stage1_tcp_line_is_horizontal_and_isolated_between_instances(self):
        stage1 = self.stage1
        config = stage1.make_stage1_config(**self.Config().common['stage1'])
        first_targets = {
            'left': [.3, .2, .7, 0., 0., 0., 1.],
            'right': [-.1, -.2, .9, 0., 0., 0., 1.],
        }
        first = stage1.TrayApproach(config, first_targets)
        original_axes = {arm: axis.copy() for arm, axis in first.inward_axes.items()}
        second = stage1.TrayApproach(config, {
            'left': [0., .3, .7, 0., 0., 0., 1.],
            'right': [0., -.3, .7, 0., 0., 0., 1.],
        })
        self.assertFalse(np.allclose(first.inward_axes[ArmGroup.LEFT],
                                     second.inward_axes[ArmGroup.LEFT]))
        for side, arm in (('left', ArmGroup.LEFT), ('right', ArmGroup.RIGHT)):
            np.testing.assert_array_equal(first.inward_axes[arm], original_axes[arm])
            first.anchor_poses[arm] = np.array(first_targets[side])
            command = first.command_pose(arm, .01)
            np.testing.assert_allclose(command[:2],
                                       np.array(first_targets[side][:2]) + .01 * original_axes[arm][:2])
            np.testing.assert_array_equal(command[2:], first_targets[side][2:])
            # Large force orthogonal to the diagonal TCP line must not change its projection.
            axis = first.reaction_axes[arm]
            orthogonal = np.array([-axis[1], axis[0], 0.])
            raw = np.r_[.7 * axis + 500 * orthogonal, [100., -200., 300.]]
            first.wrenches = Mock()
            first.wrenches.snapshot.return_value = (raw, stage1.time.monotonic(), 1)
            self.assertAlmostEqual(first.read_force(arm, .02), .7)

    def test_stage1_removed_options_are_rejected(self):
        for key, value in (('mode', 'measure'), ('execute', False), ('close_hands', False),
                           ('left_force_sign', 1.), ('right_force_sign', -1.),
                           ('measure_seconds', 8.), ('approach_axis', 'tray-y')):
            with self.subTest(option=key), self.assertRaises(TypeError):
                self.stage1.make_stage1_config(**{key: value})

    def test_single_contact_loop_latches_once_and_opens_without_confirmation(self):
        import itertools
        args=self.Config(state_path=str(self.path)).stage3_args(False)
        args.hand_open_settle_seconds=.001
        controller=self.stage3.SingleStage3Place(args)
        controller.anchor_poses={a:p.copy() for a,p in self.poses.items()}
        forces=iter([(2.1,0.),(0.,2.1)])
        current=[(0.,0.)]
        def read(arm,*a,**kw):
            if arm==ArmGroup.LEFT:
                current[0]=next(forces,(0.,2.1))
            return SimpleNamespace(support=current[0][0 if arm==ArmGroup.LEFT else 1], raw=np.zeros(6))
        worker=Mock();worker.status.return_value=(True,None)
        with patch.object(controller,'start_servo'),patch.object(controller,'exit_servo'), \
                patch.object(controller,'current_pose_checked'),patch.object(controller,'record_force_sample'), \
                patch.object(controller,'read_wrench_sample',side_effect=read), \
                patch.object(self.stage3,'DualGripperWorker',return_value=worker), \
                patch.object(self.stage3,'input',return_value=''), \
                patch.object(self.stage3.time,'monotonic',side_effect=itertools.count(1.,.002)):
            controller.run_compliant_descent_and_release()
        self.assertTrue(all(controller.descent.reached.values()))
        worker.start.assert_called_once()
        self.assertFalse(hasattr(controller, "confirmation"))

    def test_stage3_explicit_motion_defaults_preserve_current_values(self):
        config = self.Config()
        single, multi = config.stage3_args(False), config.stage3_args(True)
        self.assertEqual((single.max_free_speed, single.max_acceleration), (.006, .02))
        self.assertEqual((multi.max_free_speed, multi.max_contact_speed, multi.max_acceleration),
            tuple(config.common["stage3"]["descent_motion"]["multi"][k] for k in ("speed", "contact_speed", "acceleration")))
        c = self.multi.MultilayerStage3.__new__(self.multi.MultilayerStage3)
        c.return_motion = self.multi.ReturnMotion()
        c.yaw_search = self.multi.YawSearch()
        c.args = multi
        c.z_model = SimpleNamespace(offsets={ArmGroup.LEFT: 0., ArmGroup.RIGHT: 0.})
        c.start_moving_z_search({ArmGroup.LEFT: 0., ArmGroup.RIGHT: 0.})
        self.assertEqual((c.moving_z_params.speed_limit, c.moving_z_params.acceleration_limit), (.003, .01))
        self.assertEqual((c.moving_z_descent_params.speed_limit, c.moving_z_descent_params.acceleration_limit), (.0075, .02))
        # Changing the base limit must not multiply or overwrite the separately configured descent limit.
        config.multi['search'].update(search_z_speed_limit=.004, search_z_acceleration=.03)
        c.args = config.stage3_args(True)
        c.start_moving_z_search({ArmGroup.LEFT: 0., ArmGroup.RIGHT: 0.})
        self.assertEqual((c.moving_z_params.speed_limit, c.moving_z_params.acceleration_limit), (.004, .03))
        self.assertEqual((c.moving_z_descent_params.speed_limit, c.moving_z_descent_params.acceleration_limit), (.0075, .02))
        config.common['stage3']['descent_motion']['single'].update(speed=.012, acceleration=.03)
        args = config.stage3_args(False)
        self.assertEqual((args.max_free_speed, args.max_acceleration), (.012, .03))

    def test_stage3_accepts_explicit_limits_above_old_empirical_ceilings(self):
        config = self.Config()
        args = config.stage3_args(True, movement={'free_speed': .02, 'contact_speed': .002})
        changes = dict(hold_spring_k=1500., hold_integral_gain=.003,
            hold_max_correction=.02, hold_max_speed=.002, hold_force_band=3.,
            search_z_speed_limit=.004, search_z_acceleration=.03,
            search_z_retreat_distance=.004, radius_return_xy_speed=.004,
            radius_yaw_speed_deg_s=2., radius_yaw_range_deg=5.,
            search_max_radius=.04, direct_seating_distance=.006,
            direct_seating_max_z_difference=.02, slot_min_descent=.01,
            radius_z_down_speed=.004, radius_yaw_extra_descent=.02)
        for key, value in changes.items():
            setattr(args, key, value)
        self.common.validate_stage3_config(args)
        config.validate_multi_stage3(args)
        self.assertTrue(all(getattr(args, key) == value for key, value in changes.items()))
        for key, invalid in [('max_free_speed', float('nan')), ('mass', 0.)]:
            old = getattr(args, key)
            setattr(args, key, invalid)
            with self.assertRaises(ValueError):
                self.common.validate_stage3_config(args)
            setattr(args, key, old)

    def test_search_geometry_uses_configuration_and_explicit_overrides(self):
        config = self.Config()
        defaults = {'square': .01, 'circle': .01, 'spiral': .005}
        for route, radius in defaults.items():
            args = config.stage3_args(True, search_path=route)
            self.assertEqual(args.search_max_radius, radius)
            self.assertFalse(hasattr(args, "duration"))
        config.multi['search']['search_paths']['square'].update(radius=.012, ring_step=.003)
        config.multi['search']['radius_yaw_range_deg'] = 1.5
        args = config.stage3_args(True, search_path='square')
        self.assertEqual((args.search_max_radius, args.square_ring_step, args.radius_yaw_range_deg), (.012, .003, 1.5))
        override = config.stage3_args(True, radius=.025, search_path='square', ring_step=.004)
        self.assertEqual((override.search_max_radius, override.square_ring_step), (.025, .004))
        self.assertEqual(config.multi['search']['search_paths']['square']['radius'], .012)
        config.multi['search']['search_paths']['circle'].update(radius=.012, ring_step=.002)
        circle = config.stage3_args(True, search_path='circle')
        self.assertEqual((circle.search_max_radius, circle.circle_ring_step), (.012, .002))

    def test_main_preserves_configured_search_geometry_and_yaw(self):
        script = load_script('main')
        tray = self.api.MultiLayerTray(state_path=str(self.path), confirm=Mock(return_value=True))
        tray.config.multi['search']['search_paths']['square'].update(radius=.012, ring_step=.003)
        tray.config.multi['search']['radius_yaw_range_deg'] = 1.5
        captured = []
        def place(**kwargs):
            captured.append(tray.config.stage3_args(True, search_path=kwargs['search_path']))
            return SimpleNamespace(measurements={'force_csv_path': 'test.csv'})
        with patch.object(tray, 'move_to_pre'), patch.object(tray, 'place', side_effect=place):
            script.stage3_multi(tray, SimpleNamespace(search_path='square'), robot=Mock())
        args = captured[0]
        self.assertEqual((args.search_max_radius, args.square_ring_step, args.radius_yaw_range_deg), (.012, .003, 1.5))

    def test_yaw_z_acceleration_uses_its_own_configuration(self):
        args = self.Config().stage3_args(True)
        args.radius_z_acceleration = .035
        c = self.multi.MultilayerStage3.__new__(self.multi.MultilayerStage3)
        c.return_motion = self.multi.ReturnMotion()
        c.yaw_search = self.multi.YawSearch()
        c.args = args
        c.return_motion.z_axis = SimpleNamespace(displacement=0., velocity=0., step=Mock())
        c.return_motion.anchor_offsets = {arm: 0. for arm in self.poses}
        c.z_model = SimpleNamespace(offsets={arm: 0. for arm in self.poses})
        c.step_radius_z_admittance({arm: 0. for arm in self.poses}, .02)
        params = c.return_motion.z_axis.step.call_args.args[2]
        self.assertEqual((params.speed_limit, params.acceleration_limit), (.002, .035))

    def test_config_rejects_invalid_common_retreat_settings(self):
        for name, bad_values in (
            ('search_z_retreat_distance', (0., -.001, float('nan'))),
            ('search_z_retreat_trigger_cycles', (0, -1, 1.5, True)),
        ):
            for bad in bad_values:
                with self.subTest(name=name, value=bad):
                    args=self.Config().stage3_args(True)
                    setattr(args,name,bad)
                    with self.assertRaises(ValueError):
                        self.Config.validate_multi_stage3(args)

    def test_prep_grip_lift_and_post_vision_execute_with_v2_api(self):
        script=load_script('test_stage0_stage1_pose_record')
        poses={'left':self.poses[ArmGroup.LEFT].tolist(),'right':self.poses[ArmGroup.RIGHT].tolist()}
        gripper=Mock();gripper.wrenches=SimpleNamespace(_subscribers=[])
        gripper.run_approach.side_effect=lambda:self.calls.append('grip')
        with patch('terminal.confirm',return_value=True),patch.object(self.stage0,'move_dual_to_pregrasp',side_effect=lambda *a:self.calls.append('prep')),patch.object(self.vision,'estimate_tray_pose',return_value=[.1,0.,.6,0.,0.,0.,1.]),patch.object(self.stage1,'TrayApproach',return_value=gripper),patch.object(self.stage1,'read_tcp_pair',return_value=poses),patch('naviai_controller.NaviController',return_value=self.Robot()):
            report={'state_path':str(self.path)}
            script.execute(SimpleNamespace(config=str(ROOT/'tray_control/config/transport_tray.yaml')),report)
        self.assertTrue(report['completed']);self.assertEqual(self.calls,['prep','grip','stop',('lift',script.LIFT_HEIGHT_M)])
        self.assertEqual(self.state.load_state(str(self.path))['package_steps']['lift'],'SUCCEEDED')
        saved=self.state.load_state(str(self.path))
        self.assertEqual(saved['grasp_pose_source'],'stage0_vision_and_gripped_tcp')
        estimated,_=self.state.estimate_tray_from_tcp(poses['left'],poses['right'],saved['tray_tcp_grasp'])
        np.testing.assert_allclose(self.state.pose7_to_matrix(estimated),
                                   self.state.pose7_to_matrix(saved['pickup_tray_pose']),atol=1e-12)

    def test_waist_three_steps_changes_one_joint_and_uses_snapshot_speed(self):
        script=load_script('test_stage2_waist_three_steps')
        with patch('naviai_controller.NaviController',return_value=self.Robot()),patch('builtins.input',return_value=''),patch.object(sys,'argv',['test','--execute']):script.main()
        moves=[c for c in self.calls if isinstance(c,tuple) and c[0]=='waist']
        self.assertEqual(len(moves),3)
        old=[.02,.01,.1,.3]
        for move,index,value in zip(moves,[3,2,3],[.2,1.5,.5]):
            expected=list(old);expected[index]=value;self.assertEqual(move[1],expected)
            self.assertEqual(move[2]['v'],script.WAIST_SPEED);self.assertEqual(move[2]['acc'],script.WAIST_ACCELERATION);old=expected
            self.assertFalse(move[2]['is_async'])

    def test_lift_uses_requested_layer_heights_without_force_monitoring(self):
        tray = self.api.SingleLayerTray(state_path=str(self.path), confirm=Mock(return_value=True))
        self.state.update_state(str(self.path), package_steps={'grip': 'SUCCEEDED'})
        original_heights = list(tray.config.common['stage2']['lift_heights_by_layer'])
        robot = Mock()
        robot.movel_relative_base.return_value = True
        with patch.object(self.stage2.rospy, 'Subscriber', side_effect=AssertionError('force subscription')), \
                patch.object(self.stage2.rospy, 'Rate', side_effect=AssertionError('feedback loop')):
            for height in (.01, .01, .07):
                tray.lift(height_m=height, robot=robot, speed=.01, acceleration=.02)
                robot.movel_relative_base.assert_called_with(
                    [0., 0., height, 0., 0., height], ArmGroup.DUAL,
                    v=.01, acc=.02, is_async=False)
        self.assertEqual(tray.config.common['stage2']['lift_heights_by_layer'], original_heights)
        saved = self.state.load_state(str(self.path))
        self.assertTrue(saved['stage2_lift_complete'])
        self.assertEqual(saved['stage2_lift_height'], .07)
        self.assertEqual(saved['package_steps']['lift'], 'SUCCEEDED')

    def test_preplace_sync_segments_preserve_tcp_geometry_and_skip_low_z(self):
        # A rotated grasp with unequal TCP heights must translate both arms identically.
        held = [.1, -.02, .6, 0., 0., 2**-.5, 2**-.5]
        tcp = {side: self.poses[arm].tolist() for side, arm in
               (("left", ArmGroup.LEFT), ("right", ArmGroup.RIGHT))}
        grasp = self.state.compute_grasp_transforms(held, tcp["left"], tcp["right"])
        args = self.Config(state_path=str(self.path)).stage2_args("preplace", multi_layer=True, target_layer_index=2)
        for top_z in (.4, .65):
            with self.subTest(top_z=top_z):
                self.state.save_state(dict(top_tray_pose=[0., 0., top_z, 0., 0., 0., 1.],
                    layer_index=1, stage1_complete=True, grasp_tray_pose=held,
                    tray_tcp_grasp=grasp), str(self.path))
                robot = Mock()
                robot.movel.return_value = True
                with patch.object(self.stage2, "input", return_value=""), \
                        patch.object(self.stage2.rospy, "Subscriber", side_effect=AssertionError("force subscription")), \
                        patch.object(self.stage2.rospy, "Rate", side_effect=AssertionError("arrival polling")), \
                        patch.object(self.state, "estimate_tray_from_tcp", wraps=self.state.estimate_tray_from_tcp) as estimate:
                    # stage2 imports the shared state function directly.
                    with patch.object(self.stage2, "estimate_tray_from_tcp", estimate):
                        plan = self.stage2.plan_preplacement(args, tcp)
                    self.stage2.move_to_pre(robot, args, plan, (.003, -.004))
                estimate.assert_called_once()
                skip_z = top_z > held[2]
                self.assertEqual(plan.skip_z, skip_z)
                self.assertEqual(robot.movel.call_count, 2 if skip_z else 3)
                for call in robot.movel.call_args_list:
                    self.assertIs(call.kwargs['is_async'], False)
                    left, right = np.array(call.args[0][:7]), np.array(call.args[0][7:])
                    np.testing.assert_allclose(left[3:], tcp['left'][3:])
                    np.testing.assert_allclose(right[3:], tcp['right'][3:])
                    np.testing.assert_allclose(right[:3] - left[:3],
                        np.array(tcp['right'][:3]) - tcp['left'][:3])
                final = robot.movel.call_args.args[0]
                reconstructed, _ = self.state.estimate_tray_from_tcp(final[:7], final[7:], grasp)
                np.testing.assert_allclose(reconstructed[:2], [.003, -.004], atol=1e-12)
                self.assertAlmostEqual(reconstructed[2], held[2] if skip_z else plan.pre_pose[2])
                saved = self.state.load_state(str(self.path))
                self.assertTrue(saved['stage2_complete'])
                self.assertEqual(saved['test_stage3_preplace_target_reached'], not skip_z)

    def test_preplace_failure_stops_without_marking_success(self):
        held = [.1, 0., .6, 0., 0., 0., 1.]
        tcp = {side: self.poses[arm].tolist() for side, arm in
               (("left", ArmGroup.LEFT), ("right", ArmGroup.RIGHT))}
        grasp = self.state.compute_grasp_transforms(held, tcp['left'], tcp['right'])
        self.state.save_state(dict(top_tray_pose=[0., 0., .4, 0., 0., 0., 1.], layer_index=1,
            stage1_complete=True, grasp_tray_pose=held, tray_tcp_grasp=grasp), str(self.path))
        args = self.Config(state_path=str(self.path)).stage2_args('preplace', multi_layer=True, target_layer_index=2)
        plan = self.stage2.plan_preplacement(args, tcp)
        robot = Mock()
        robot.movel.side_effect = [True, False]
        with patch.object(self.stage2, 'input', return_value=''), self.assertRaisesRegex(RuntimeError, '同步MoveL失败'):
            self.stage2.move_to_pre(robot, args, plan, (.01, 0.))
        self.assertEqual(robot.movel.call_count, 2)
        self.assertEqual(robot.stop.call_count, 2)
        saved = self.state.load_state(str(self.path))
        self.assertFalse(saved['stage2_complete'])
        self.assertFalse(saved['test_stage3_xy_offset_complete'])

    def test_failed_lift_stops_and_does_not_mark_complete(self):
        tray = self.api.SingleLayerTray(state_path=str(self.path), confirm=Mock(return_value=True))
        self.state.update_state(str(self.path), package_steps={'grip': 'SUCCEEDED'})
        robot = Mock()
        robot.movel_relative_base.return_value = False
        with self.assertRaisesRegex(RuntimeError, '抬升失败'):
            tray.lift(height_m=.01, robot=robot)
        self.assertEqual(robot.stop.call_count, 2)
        saved = self.state.load_state(str(self.path))
        self.assertFalse(saved['stage2_lift_complete'])
        self.assertEqual(saved['package_steps']['lift'], 'FAILED')

    def test_main_waist_reuses_robot_without_force_or_feedback_loop(self):
        script = load_script('main')
        with patch('terminal.confirm', return_value=True), \
                patch.object(self.stage2.rospy, 'Subscriber', side_effect=AssertionError('force subscription')), \
                patch.object(self.stage2.rospy, 'Rate', side_effect=AssertionError('feedback loop')):
            script.stage2_waist(self.Robot())
        moves = [c for c in self.calls if isinstance(c, tuple) and c[0] == 'waist']
        self.assertEqual(len(moves), 3)
        self.assertEqual(moves[-1][1], [.02, .01, 1.5, .5])
        self.assertTrue(all(c[2]['is_async'] is False for c in moves))

    def test_waist_failure_prevents_next_step(self):
        robot = Mock()
        robot.get_joints.side_effect = ([.02, .01, .1, .3], [.02, .01, .1, .2])
        robot.movej.side_effect = (True, False)
        with patch.object(self.stage2, 'input', return_value=''), \
                self.assertRaisesRegex(RuntimeError, '转腰失败'):
            self.stage2.turn_waist(robot)
        self.assertEqual(robot.movej.call_count, 2)
        self.assertEqual(robot.stop.call_count, 2)

    def test_main_passes_three_layer_heights_to_shared_robot(self):
        import tray_control
        script = load_script('main')
        tray = Mock(state_path=str(self.path))
        robot = Mock()
        options = SimpleNamespace(config='unused', state_file=str(self.path),
                                  layers=3, lift_cm=[1., 1., 7.])
        records = [{'layer_index': i, 'stage3_complete': True} for i in (1, 2, 3)]
        with patch.object(tray_control, 'SingleLayerTray', return_value=tray), \
                patch.object(tray_control, 'MultiLayerTray', return_value=Mock()), \
                patch('naviai_controller.NaviController', return_value=robot), \
                patch.object(script, 'stage2_waist') as waist, \
                patch.object(script, 'stage3_single'), patch.object(script, 'stage3_multi'), \
                patch.object(self.state, 'load_state', side_effect=records), \
                contextlib.redirect_stdout(io.StringIO()):
            script.run(options)
        self.assertEqual([c.kwargs['height_m'] for c in tray.lift.call_args_list], [.01, .01, .07])
        self.assertTrue(all(c.kwargs['robot'] is robot for c in tray.lift.call_args_list))
        self.assertEqual(waist.call_count, 3)
        self.assertTrue(all(c.args == (robot,) for c in waist.call_args_list))

    def test_single_stage3_then_fresh_first_layer_without_backup(self):
        script=load_script('test_stage3_single_z_place')
        self.state.save_state({'layer_index':1,'target_layer_index':2,'stage3_release_and_retreat_complete':True},str(self.path))
        controller=Mock();controller.wrenches=SimpleNamespace(_subscribers=[])
        with patch.object(self.stage3,'SingleStage3Place',return_value=controller),patch('terminal.confirm',return_value=True),patch.object(self.vision,'estimate_tray_pose',return_value=[0.,0.,.4,0.,0.,0.,1.]),patch.object(sys,'argv',['test','--execute','--state-file',str(self.path)]):script.main()
        controller.run.assert_called_once();s=self.state.load_state(str(self.path))
        self.assertEqual(s['layer_index'],1);self.assertNotIn('target_layer_index',s)
        self.assertEqual(list(Path(self.temp.name).iterdir()),[self.path])

    def test_multi_preplace_offset_stage3_and_new_top_in_order(self):
        self.exercise_multi()

    def test_multi_place_uses_top_record_runner(self):
        self.check_multi_place_threshold(2, 2.0)

    def test_third_layer_place_uses_its_force_threshold(self):
        self.check_multi_place_threshold(3, 3.5)

    def check_multi_place_threshold(self, layer, threshold, **place_options):
        self.state.save_state({'target_layer_index':layer,
                              'package_steps':{'move_to_pre':'SUCCEEDED'}},str(self.path))
        controller=Mock(released=True,force_recorder=None,force_log_path='force.csv',
                        radius_limit_release=False,unconfirmed_release_reason=None,
                        place_pose=np.array([0.,0.,.433,0.,0.,0.,1.]))
        controller.wrenches=SimpleNamespace(_subscribers=[])
        pose=[0.,0.,.435,0.,0.,0.,1.]
        runner=Mock(side_effect=lambda record:record())
        tray=self.api.MultiLayerTray(state_path=str(self.path),confirm=Mock(return_value=True))
        with patch.object(self.multi,'MultilayerStage3',return_value=controller) as factory, \
                patch.object(self.multi,'record_top_pose',return_value=pose) as record:
            result=tray.place(record_top_runner=runner, **place_options)
        runner.assert_called_once()
        self.assertEqual(factory.call_args.args[0].xy_force_capture_threshold, threshold)
        record.assert_called_once()
        self.assertEqual(result.measurements['top_tray_pose'],pose)
        self.assertEqual(self.state.load_state(str(self.path))['package_steps']['place'],'SUCCEEDED')
        return result, factory.call_args.args[0]

    def test_search_named_api_parameters_reach_controller_and_result(self):
        result, args = self.check_multi_place_threshold(2, 2.0,
            search_angular_speed=.61, search_max_radius=.012, xy_speed=.004)
        self.assertEqual((args.search_angular_speed, args.search_max_radius, args.search_xy_speed),
                         (.61, .012, .004))
        self.assertEqual(result.measurements['search_max_radius'], .012)
        self.assertNotIn('spiral_max_radius', result.measurements)

    def test_multi_unconfirmed_yaw_release_reports_actual_z_even_outside_normal_z_limit(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            self.exercise_multi(unconfirmed=True,actual_z=.47)
        expected_mm=1000*(.47-(.4+self.Config().multi['preplace']['stack_pitch']))
        self.assertIn("{:+.3f}mm".format(expected_mm),output.getvalue())
        self.assertIn("未独立确认入槽",output.getvalue())

    def exercise_multi(self,unconfirmed=False,actual_z=.43):
        script=load_script('test_stage3_multi_z_search_place');outer=self
        self.state.save_state({'top_tray_pose':[0.,0.,.4,0.,0.,0.,1.],'layer_index':1},str(self.path))
        class Multi:
            def __init__(self,args):
                self.args=args;self.released=False;self.radius_limit_release=unconfirmed
                self.unconfirmed_release_reason="Yaw搜索未观察到新增下沉" if unconfirmed else None
                self.force_recorder=None;self.force_log_path=None
                self.place_pose=outer.state.load_state(args.state_file)['place_tray_target'];self.wrenches=SimpleNamespace(_subscribers=[])
            def run_experiment(self):
                outer.calls.append('stage3');self.released=True
                assert self.args.search_path=="square"
                assert self.args.search_max_radius==.010
                assert self.args.search_timeout>40.
                assert self.args.radius_yaw_range_deg==3.0
                assert not hasattr(self.args,"slot_fx_height_limit_z")
                outer.state.update_state(self.args.state_file,stage3_release_and_retreat_complete=True,
                    stage3_radius_limit_release=self.radius_limit_release,
                    stage3_unconfirmed_release_reason=self.unconfirmed_release_reason)
            def close_force_log(self):pass
            def mark_force_abort(self,error):pass
            def cleanup(self):pass
        tcp={'left':self.poses[ArmGroup.LEFT].tolist(),'right':self.poses[ArmGroup.RIGHT].tolist()}
        with patch.object(script,'XY_OFFSET_M',[0.,-.005]),patch('naviai_controller.NaviController',return_value=self.Robot()),patch.object(self.multi,'MultilayerStage3',Multi),patch('terminal.confirm',return_value=True),patch.object(self.stage1,'read_tcp_pair',return_value=tcp),patch.object(self.vision,'estimate_tray_pose',return_value=[.1,0.,.6,0.,0.,0.,1.]),patch.object(self.multi,'estimate_tray_pose',return_value=[0.,0.,actual_z,0.,0.,0.,1.]),patch.object(sys,'argv',['test','--execute','--state-file',str(self.path)]):script.main()
        self.assertEqual(self.calls,['stop','offset','offset','offset','stage3'])
        saved=self.state.load_state(str(self.path));self.assertEqual(saved['layer_index'],2)
        self.assertEqual(saved['test_stage3_xy_offset_m'],[0.,-.005])
        self.assertEqual(saved['package_steps']['place'],'SUCCEEDED')
        self.assertAlmostEqual(saved['stage3_placed_minus_p_place_z_mm'],
                               1000*(actual_z-saved['place_tray_target'][2]))

    def test_single_place_keeps_timing_without_creating_force_csv_on_success_or_failure(self):
        from tray_control.observability import timing
        for fails in (False, True):
            with self.subTest(fails=fails):
                args = self.Config(state_path=str(self.path)).stage3_args(False)
                args.force_log_dir = str(Path(self.temp.name) / 'single_force_logs')
                controller = self.stage3.SingleStage3Place(args)
                timer = timing.StateTimer('Stage3单层', clock=itertools.count(1., .1).__next__)
                def descent():
                    self.stage3.trace_state('Z_DESCENT')
                    if fails:
                        raise RuntimeError('实测反馈超时')
                    self.stage3.trace_state('OPENING')
                with patch.object(timing, '_active', timer), \
                        patch.object(controller, 'wait_ready'), patch.object(controller, 'capture_baseline'), \
                        patch.object(controller, 'run_compliant_descent_and_release', side_effect=descent), \
                        patch.object(controller, 'exit_servo'), patch.object(controller, 'retreat'):
                    try:
                        if fails:
                            with self.assertRaisesRegex(RuntimeError, '实测反馈超时'):
                                controller.run()
                        else:
                            controller.run()
                    finally:
                        controller.cleanup()
                    timer.finish()
                self.assertGreater(timer.totals[('Stage3单层', 'Z_DESCENT')], 0.)
                self.assertIn(('Stage3单层', 'ABORTED' if fails else 'COMPLETED'), timer.totals)
                self.assertFalse(Path(args.force_log_dir).exists())
                self.assertIsNone(controller.force_log_path)
                self.assertIsNone(controller.force_recorder)

    def test_multilayer_controller_csv_preserves_diagnostics_and_adds_baseline(self):
        c=self.Config(state_path=str(self.path));args=c.stage3_args(True);args.force_log_dir=self.temp.name
        self.state.save_state({'stage2_complete':True,'place_tray_target':[0.,0.,.43,0.,0.,0.,1.],
                              'tray_tcp_grasp':{'left':[0.,.2,0.,0.,0.,0.,1.],'right':[0.,-.2,0.,0.,0.,0.,1.]}},str(self.path))
        controller=self.multi.MultilayerStage3(args)
        controller.baseline={a:np.array([1.,2.,3.,.1,.2,.3]) for a in (ArmGroup.LEFT,ArmGroup.RIGHT)}
        controller.xy_force_captured = True
        controller.start_force_log()
        try:
            with patch.object(self.multi.rospy,'get_time',return_value=1.,create=True):
                controller.write_force_sample(.02,{a:2. for a in controller.baseline},
                    {a:controller.baseline[a]+1. for a in controller.baseline},{},
                    {'x':0.,'y':0.,'z':.01},'measured')
        finally:
            controller.cleanup()
        with open(controller.force_log_path,newline='') as stream:rows=list(csv.DictReader(stream))
        sample=next(r for r in rows if r['row_type']=='sample')
        self.assertEqual(float(sample['left_fx_baseline_n']),1.)
        self.assertEqual(float(sample['right_tz_delta_nm']),1.)
        self.assertEqual(float(sample['tray_error_z_mm']),10.)
        self.assertEqual(sample['slot_raw_fx_captured'], 'True')
        self.assertEqual(sample['slot_capture_mode'], 'descent_only')
        self.assertEqual(float(sample['slot_force_drop_debounce_s']), args.yaw_descent_debounce)
        self.assertEqual(float(sample['search_z_retreat_distance_mm']),1000*args.search_z_retreat_distance)
        self.assertEqual(int(sample['search_z_retreat_trigger_cycles']),3)

    def test_release_retreat_uses_fixed_distances_and_motion_parameters(self):
        args = self.Config(state_path=str(self.path)).stage3_args(False)
        keys = ('post_release_lower_distance', 'post_release_lower_velocity',
                'post_release_lower_acceleration', 'outward_distance',
                'post_release_lift_height', 'retreat_velocity', 'retreat_acceleration',
                'retreat_force_limit', 'retreat_torque_limit', 'retreat_tcp_tolerance',
                'retreat_orientation_tolerance_deg', 'retreat_timeout')
        self.assertTrue(all(not hasattr(args, key) for key in keys))
        c = self.stage3.SingleStage3Place(args)
        c.robot = self.Robot()
        c.robot.get_tcp_rt = lambda arm: self.poses[arm].copy()
        c.robot.movel = Mock(wraps=c.robot.movel)
        c.wait_movel_target = Mock()
        starts = {arm: pose.copy() for arm, pose in self.poses.items()}
        c.retreat()
        moves = c.robot.movel.call_args_list
        self.assertEqual(len(moves), 3)
        expected = {arm: pose.copy() for arm, pose in starts.items()}
        for index, (speed, acceleration) in enumerate(((.01, .02), (.08, .1), (.08, .1))):
            for arm in expected:
                if index == 0:
                    expected[arm][2] -= .0013
                elif index == 1:
                    expected[arm][1] += .02 if arm == ArmGroup.LEFT else -.02
                else:
                    expected[arm][2] += .03
            np.testing.assert_allclose(moves[index].args[0],
                np.concatenate([expected[ArmGroup.LEFT], expected[ArmGroup.RIGHT]]))
            self.assertEqual(moves[index].kwargs, dict(v=speed, acc=acceleration, is_async=True))
        self.assertEqual(c.wait_movel_target.call_args_list[0].kwargs, {'position_tolerance': .0005})
        self.assertEqual(c.wait_movel_target.call_count, 3)
        self.assertFalse(c.movel_started)

    def test_stage3_removed_limits_do_not_abort_force_or_waiting(self):
        removed = ('torque_delta_limit', 'duration', 'one_side_timeout',
                   'radius_return_timeout', 'release_confirm_timeout')
        for multi in (False, True):
            args = self.Config().stage3_args(multi)
            self.assertTrue(all(not hasattr(args, key) for key in removed))
        c = self.stage3.SingleStage3Place(self.Config().stage3_args(False))
        c.baseline = {a: np.zeros(6) for a in self.poses}
        c.wrenches = Mock()
        c.wrenches.snapshot.return_value = (np.array([0., 0., 1., 10., 0., 0.]), 10., 1)
        with patch.object(self.stage3.time, 'monotonic', return_value=10.):
            sample = c.read_wrench_sample(ArmGroup.LEFT, .02)
        self.assertEqual(sample.support, 1.)
        self.assertEqual(sample.raw[3], 10.)
        c = self.multi.MultilayerStage3.__new__(self.multi.MultilayerStage3)
        c.return_motion = self.multi.ReturnMotion()
        c.yaw_search = self.multi.YawSearch()
        c.args = self.Config().stage3_args(True)
        c.flow = SimpleNamespace(stage_started=0.)
        c.advance_search_return = Mock()
        cycle = SimpleNamespace(now=100000., dt=.02, poses=self.poses)
        c.step_return_to_lowest(cycle)
        c.advance_search_return.assert_called_once_with(self.poses, 100000., .02)
        c.slot_captured = False

    def test_post_release_and_retreat_do_not_abort_on_absolute_load(self):
        c = self.stage3.SingleStage3Place(self.Config().stage3_args(False))
        c.baseline = {a: np.zeros(6) for a in self.poses}
        c.wrenches = Mock()
        raw = np.array([100., 0., 0., 10., 0., 0.])
        c.robot = Mock()
        c.robot.get_is_singular.return_value = False
        c.robot.get_cmd_num.return_value = 0
        c.robot.get_tcp_rt.side_effect = lambda arm: self.poses[arm].copy()
        with patch.object(self.stage3.time, 'monotonic', side_effect=itertools.count(10., .01)):
            c.wrenches.snapshot.side_effect = lambda arm: (raw.copy(), self.stage3.time.monotonic(), 1)
            c.read_wrench_sample(ArmGroup.LEFT, .02)
            c.wait_movel_target(self.poses, 'TEST')

    def test_xy_limit_and_contact_alignment_do_not_abort_search(self):
        args = self.Config().stage3_args(True, search_path='square')
        self.assertFalse(hasattr(args, 'max_xy_command'))
        self.assertFalse(hasattr(args, 'place_yaw_tolerance_deg'))
        c = self.multi.MultilayerStage3.__new__(self.multi.MultilayerStage3)
        c.return_motion = self.multi.ReturnMotion()
        c.yaw_search = self.multi.YawSearch()
        c.args = args
        c.search_offset_base = np.array([.03, 0.])
        c.rate_limit_xy(np.array([.04, 0.]), .02)
        self.assertGreater(c.search_offset_base[0], .03)
        c.flow = SimpleNamespace(both_force_since=0.)
        c.z_model = SimpleNamespace(offsets={a: 0. for a in self.poses})
        c.remember_search_lowest = Mock()
        c.start_moving_z_search = Mock()
        cycle = SimpleNamespace(now=1., both_at_target=True, error=dict(x=.03, y=0., z=.01, yaw=.2),
            tray_estimate=[0., 0., .5], poses=self.poses, support={a: 2. for a in self.poses},
            raw_wrenches={a: np.zeros(6) for a in self.poses})
        c.step_initial_descent(cycle)
        self.assertEqual(c.stage_name, 'XY_SEARCH')
        c.start_moving_z_search.assert_called_once()

    def test_multilayer_initial_contact_reuses_single_retreat_and_keeps_search_model(self):
        args = self.Config(state_path=str(self.path)).stage3_args(True)
        self.state.save_state(dict(stage2_complete=True,
            place_tray_target=[0., 0., .4, 0., 0., 0., 1.],
            tray_tcp_grasp={'left': [0., .2, 0., 0., 0., 0., 1.],
                            'right': [0., -.2, 0., 0., 0., 0., 1.]}), str(self.path))
        from tray_control.common.feedback import WrenchSample
        for held in self.poses:
            with self.subTest(first_contact=held):
                c = self.multi.MultilayerStage3(args)
                c.flow = self.multi.MultilayerFlow(stage_started=0.)
                c.anchor_poses = {a: p.copy() for a, p in self.poses.items()}
                c.check_robot_state_without_tcp = Mock()
                c.commanded_descent_pose = Mock(return_value=([0., 0., .5, 0., 0., 0., 1.], {}))
                c.tcp_pair_checked = Mock(return_value=self.poses)
                c.tray_error = Mock(return_value=dict(x=.02, y=0., z=.01, yaw=0.))
                c.baseline = {a: np.zeros(6) for a in self.poses}
                forces = {a: 0. for a in self.poses}
                c.read_wrench_sample = lambda a, dt: WrenchSample(np.zeros(6), forces[a])
                now = 0.
                def advance():
                    nonlocal now
                    now += .02
                    cycle, _ = c.read_cycle(now, .02)
                    c.step_initial_descent(cycle)
                    c.publish_cycle(cycle)
                    return cycle
                for _ in range(30):
                    advance()
                moving = next(a for a in self.poses if a != held)
                start = c.z_model.offsets.copy()
                forces[held] = args.target_force
                advance()
                forces[held] = 0.  # 上退卸载不解除接触锁存，也不暂停另一侧。
                for _ in range(60):
                    cycle = advance()
                self.assertTrue(cycle.reached[held])
                self.assertFalse(cycle.both_at_target)
                self.assertAlmostEqual(c.z_model.offsets[held], start[held] - .0005)
                self.assertGreater(c.z_model.offsets[moving], start[moving] + .003)
                forces[moving] = args.target_force
                cycle = advance()
                self.assertTrue(cycle.both_at_target)
                frozen = c.z_model.offsets.copy()
                forces[moving] = 0.
                c.enter_xy_search = Mock()
                for _ in range(17):
                    advance()
                self.assertEqual(c.z_model.offsets, frozen)
                c.enter_xy_search.assert_called()
                c.stage_name = 'XY_SEARCH'
                c.step_moving_z_search = Mock(return_value=frozen)
                c.publish_cycle(cycle)
                c.step_moving_z_search.assert_called_once_with(cycle.support, cycle.dt)
                c.cleanup()

    def test_contact_descent_uses_shared_distance_type_and_timing_interface(self):
        from tray_control.observability.timing import CycleTiming
        self.assertIs(self.stage3.CycleTiming, CycleTiming)
        self.assertIs(self.multi.ContactDescentState, self.stage3.ContactDescentState)
        distance = self.stage3.CONTACT_RETREAT_DISTANCE_M
        self.assertEqual(distance, .0005)
        self.assertEqual(self.multi.CONTACT_RETREAT_DISTANCE_M, distance)
        args = self.Config(state_path=str(self.path)).stage3_args(False)
        c = self.stage3.SingleStage3Place(args)
        with patch.object(self.stage3, 'input', side_effect=RuntimeError('stop before servo')), \
                patch.object(self.stage3, 'MovingZParameters', wraps=self.stage3.MovingZParameters) as params:
            with self.assertRaisesRegex(RuntimeError, 'stop before servo'):
                c.run_compliant_descent_and_release()
        self.assertEqual(params.call_args.kwargs['retreat_distance'], distance)
        message, *values = self.stage3.rospy.logwarn.call_args.args
        self.assertIn('上退0.5mm', message % tuple(values))
        c.cleanup()
        self.state.save_state(dict(stage2_complete=True,
            place_tray_target=[0.,0.,.4,0.,0.,0.,1.],
            tray_tcp_grasp={'left':[0.,.2,0.,0.,0.,0.,1.],
                            'right':[0.,-.2,0.,0.,0.,0.,1.]}), str(self.path))
        c = self.multi.MultilayerStage3(self.Config(state_path=str(self.path)).stage3_args(True))
        self.assertIsInstance(c.contact_descent, self.stage3.ContactDescentState)
        self.assertEqual(c.contact_retreat_params.retreat_distance, distance)
        message, *values = self.multi.rospy.logwarn.call_args.args
        self.assertIn('初始接触上退0.5mm', message % tuple(values))
        c.cleanup()

    def test_stage3_publishes_beyond_old_depth_and_side_difference_limits(self):
        args = self.Config(state_path=str(self.path)).stage3_args(True)
        for key in ('max_downward_distance', 'max_upward_retreat', 'max_z_difference'):
            self.assertFalse(hasattr(args, key))
        c = self.multi.MultilayerStage3.__new__(self.multi.MultilayerStage3)
        c.return_motion = self.multi.ReturnMotion()
        c.yaw_search = self.multi.YawSearch()
        c.args, c.stage_name, c.robot = args, 'Z_DESCENT', Mock()
        c.z_model = self.common.SwitchingZAdmittance(args)
        c.z_model.offsets = {ArmGroup.LEFT: .18, ArmGroup.RIGHT: .25}
        from tray_control.stage3 import ContactDescentState
        c.contact_descent = ContactDescentState(offsets=c.z_model.offsets)
        c.contact_retreat_params = self.multi.MovingZParameters(
            speed_limit=args.max_free_speed, acceleration_limit=args.max_acceleration,
            retreat_distance=self.stage3.CONTACT_RETREAT_DISTANCE_M, travel_limit=None, equilibrium_limit=None)
        c.anchor_poses = {arm: pose.copy() for arm, pose in self.poses.items()}
        c.search_offset_base, c.return_motion.commands = np.zeros(2), None
        cycle = SimpleNamespace(reached={a: False for a in self.poses},
                                support={a: 2.1 for a in self.poses}, dt=.02)
        c.publish_cycle(cycle)
        c.robot.servol.assert_called_once()
        target = c.robot.servol.call_args.args[0]
        self.assertAlmostEqual(target[2], self.poses[ArmGroup.LEFT][2] - .18)
        self.assertAlmostEqual(target[9], self.poses[ArmGroup.RIGHT][2] - .25)

    def test_all_three_search_paths_remain_available_with_fixed_z_strategy(self):
        self.state.save_state(dict(stage2_complete=True,
            place_tray_target=[0., 0., .4, 0., 0., 0., 1.],
            tray_tcp_grasp={'left': [0., .2, 0., 0., 0., 0., 1.],
                           'right': [0., -.2, 0., 0., 0., 0., 1.]}), str(self.path))
        removed = ('moving_z_search_enabled', 'search_z_mode', 'search_pause_xy_during_retreat',
                   'radius_yaw_search_enabled', 'xy_force_capture_enabled', 'seating_timeout')
        for route in ('square', 'circle', 'spiral'):
            with self.subTest(route=route):
                args = self.Config(state_path=str(self.path)).stage3_args(True, search_path=route)
                self.assertTrue(all(not hasattr(args, key) for key in removed))
                c = self.multi.MultilayerStage3(args)
                self.assertFalse(hasattr(c, 'confirmation'))
                target, done = c.xy_path.target(np.zeros(2), elapsed=.02, yaw=.3)
                self.assertEqual(target.shape, (2,))
                self.assertFalse(done)
                c.cleanup()

    def test_search_end_returns_xy_then_always_starts_yaw_and_releases(self):
        self.state.save_state(dict(stage2_complete=True,
            place_tray_target=[0., 0., .4, 0., 0., 0., 1.],
            tray_tcp_grasp={'left': [0., .2, 0., 0., 0., 0., 1.],
                           'right': [0., -.2, 0., 0., 0., 0., 1.]}), str(self.path))
        c = self.multi.MultilayerStage3(self.Config(state_path=str(self.path)).stage3_args(True))
        c.last_commands = {a: p.copy() for a, p in self.poses.items()}
        c.anchor_poses = {a: p.copy() for a, p in self.poses.items()}
        c.baseline = {a: np.zeros(6) for a in self.poses}
        c.history.poses = {a: p.copy() for a, p in self.poses.items()}
        for p in c.history.poses.values():
            p[2] -= .002  # Z must remain compliant; return checks only XY/orientation.
        c.history.drop = .002
        c.history.offset = np.array([.001, -.001])
        c.search_start_z = .8
        c.flow = self.multi.MultilayerFlow(stage_started=0.)
        c.start_return_to_lowest(.001)
        self.assertEqual(c.stage_name, 'RETURN_SEARCH_LOWEST')
        self.assertIsNotNone(c.return_motion.z_axis)
        c.advance_search_return(self.poses, 1., .02)
        self.assertEqual(c.stage_name, 'RETURN_SEARCH_LOWEST')
        c.advance_search_return(self.poses, 1.31, .02)
        self.assertEqual(c.stage_name, 'YAW_SEARCH_LOWEST')
        raw = {a: np.zeros(6) for a in self.poses}
        raw[ArmGroup.LEFT][1] = 2.51
        worker = Mock()
        with patch.object(self.multi, 'DualGripperWorker', return_value=worker):
            c.advance_lowest_yaw_search(self.poses, [0., 0., .7], .8, 1.4, .02, raw)
        self.assertEqual(c.stage_name, 'OPENING')
        worker.start.assert_called_once()
        c.cleanup()

    def test_multilayer_real_cycle_dispatches_capture_and_direct_seating(self):
        for direct in (False, True):
            with self.subTest(direct=direct):
                args = self.Config(state_path=str(self.path)).stage3_args(True)
                args.search_path = 'square'
                args.square_ring_step = .002
                grasp = {'left': [0., .2, 0., 0., 0., 0., 1.],
                         'right': [0., -.2, 0., 0., 0., 0., 1.]}
                self.state.save_state(dict(stage2_complete=True,
                    place_tray_target=[0., 0., .4, 0., 0., 0., 1.],
                    tray_tcp_grasp=grasp), str(self.path))
                c = self.multi.MultilayerStage3(args)
                c.robot = Mock()
                c.robot.set_servo_params.return_value = True
                c.tcp_pair_checked = Mock(return_value={a: p.copy() for a, p in self.poses.items()})
                c.baseline = {arm: np.zeros(6) for arm in self.poses}
                c.read_wrench_sample = Mock(return_value=SimpleNamespace(support=2.1, raw=np.zeros(6)))
                c.wrenches = Mock()
                c.wrenches.snapshot.return_value = (np.zeros(6), 1., 1)
                def estimate(poses):
                    z = .4 if direct else (.416 if c.stage_name == 'XY_SEARCH' else .42)
                    x = 0. if direct else .002
                    return [x, 0., z, 0., 0., 0., 1.], {}
                c.estimate_tray = Mock(side_effect=estimate)
                c.check_robot_state_without_tcp = Mock()
                c.start_force_log = Mock()
                samples, observations = [], []
                c.calculate_search_observation = Mock(wraps=c.calculate_search_observation)
                c.write_force_sample = lambda *a, **kw: (
                    samples.append(c.stage_name), observations.append(kw['observation']))
                worker = Mock()
                worker.status.return_value = (True, None)
                clock = SimpleNamespace(now=1.)
                rate = Mock()
                rate.sleep.side_effect = lambda: setattr(clock, 'now', clock.now + .02)
                with patch.object(self.multi, 'input', return_value=''), \
                        patch.object(self.multi, 'DualGripperWorker', return_value=worker), \
                        patch.object(self.multi.time, 'monotonic', side_effect=lambda: clock.now), \
                        patch.object(self.multi.rospy, 'Rate', return_value=rate), \
                        patch.object(self.multi.rospy, 'is_shutdown', return_value=False):
                    c.run_multilayer_descent()
                self.assertEqual(c.calculate_search_observation.call_count, len(observations))
                self.assertEqual(len(observations), len(samples))
                self.assertTrue(c.released)
                self.assertEqual(c.direct_seating_selected, direct)
                self.assertEqual(c.stage_name, 'OPENING')
                worker.start.assert_called_once()
                self.assertEqual(c.robot.servol.call_count, rate.sleep.call_count + 1)
                for call in c.robot.servol.call_args_list:
                    self.assertEqual(len(call.args[0]), 14)
                if direct:
                    self.assertIn('FINAL_Z_SEATING', samples)
                    self.assertIn('OPENING', samples)
                else:
                    self.assertIn('XY_SEARCH', samples)
                    self.assertNotIn("FINAL_Z_SEATING", samples)

    def test_record_top_large_xyz_yaw_difference_is_reported_and_saved(self):
        args=self.Config(state_path=str(self.path)).stage3_args(True)
        self.state.save_state({'layer_index':1,'target_layer_index':2},str(self.path))
        pose=[.8,.9,.8,0.,0.,2**-.5,2**-.5]
        expected=[.1,.2,.4,0.,0.,0.,1.]
        output=io.StringIO()
        with patch.object(self.multi,'capture_top_tray_pose',return_value=pose),contextlib.redirect_stdout(output):
            result=self.multi.record_top_pose(args,expected_pose=expected)
        saved=self.state.load_state(str(self.path))
        self.assertEqual(result,pose)
        self.assertEqual(saved['top_tray_pose'],pose)
        self.assertEqual(saved['layer_index'],2)
        self.assertTrue(saved['stage3_complete'])
        self.assertIn('+400.000mm',output.getvalue())
