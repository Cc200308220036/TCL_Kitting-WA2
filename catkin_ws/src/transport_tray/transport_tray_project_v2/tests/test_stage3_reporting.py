"""位姿输出、基线CSV和精确状态事件的离线检查。"""
import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def pure_module(name):
    # 不导入tray_control.__init__中的ROS接口。
    import importlib.util
    from types import ModuleType
    namespace = "_reporting_math"
    if namespace not in sys.modules:
        package = ModuleType(namespace)
        package.__path__ = [str(ROOT / "tray_control")]
        sys.modules[namespace] = package
    locations = {
        "force_recording": "observability.force_recording",
        "state": "common.state",
        "pose_reporting": "observability.pose_reporting",
    }
    name = locations.get(name, name)
    importlib.import_module(namespace + "." + name.rsplit(".", 1)[0])
    qualified = namespace + "." + name
    spec = importlib.util.spec_from_file_location(qualified, ROOT / "tray_control" / (name.replace(".", "/") + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


recording = pure_module("force_recording")
pure_module("state")
reporting = pure_module("pose_reporting")


class ForceRecordingTests(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.baseline = {"left": [1, 2, 3, .1, .2, .3], "right": [-1, -2, -3, -.1, -.2, -.3]}
        self.now = 100.
        self.recorder = recording.ForceCsvRecorder(self.temp.name, self.baseline, clock=lambda: self.now,
                                                  extra_columns=["left_search_relative_fx_n"])
        self.addCleanup(self.recorder.close)

    def rows(self):
        self.recorder.close()
        with open(self.recorder.path, newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_event_forwarding_uses_active_csv_and_keeps_timing_separate(self):
        recording.activate(self.recorder)
        try:
            with patch.object(recording, 'timing_state') as timing:
                self.now = 101.
                recording.state('Z_DESCENT')
                recording.mark('contact', 'left')
                recording.deactivate(object())  # 非当前记录器不能注销。
                recording.mark('contact', 'right')
                recording.deactivate(self.recorder)
                recording.state('AFTER_CLOSE')
                recording.mark('ignored', 'not recorded')
            self.assertEqual(timing.call_args.args, ('AFTER_CLOSE',))
            rows = self.rows()
            self.assertEqual([r['detail'] for r in rows if r['event'] == 'contact'], ['left', 'right'])
            self.assertFalse(any(r['event'] == 'ignored' for r in rows))
            self.assertEqual(self.recorder.stage, 'Z_DESCENT')
        finally:
            recording.deactivate(self.recorder)

    def test_baseline_and_six_axis_relative_wrench(self):
        self.recorder.set_state("Z_DESCENT")
        self.recorder.sample({"left": [2, 1, 5, .4, .1, .8], "right": [-2, -1, -5, -.4, -.1, -.8]},
                             {"left": 1.9, "right": -1.8})
        sample = next(row for row in self.rows() if row["row_type"] == "sample")
        self.assertEqual(float(sample["left_fx_baseline_n"]), 1.)
        self.assertEqual(float(sample["left_fx_delta_n"]), 1.)
        self.assertEqual(float(sample["left_fy_delta_n"]), -1.)
        self.assertEqual(float(sample["left_fz_delta_n"]), 2.)
        self.assertAlmostEqual(float(sample["left_tx_delta_nm"]), .3)
        self.assertAlmostEqual(float(sample["right_tz_delta_nm"]), -.5)
        self.assertEqual(float(sample["left_support_fz_n"]), 1.9)

    def test_state_transition_timestamp_and_event_rows_do_not_reuse_force(self):
        self.now = 101.
        self.recorder.set_state("Z_DESCENT")
        self.recorder.sample(self.baseline)
        self.now = 103.25
        self.recorder.set_state("XY_SEARCH")
        self.recorder.set_state("XY_SEARCH")
        rows = self.rows()
        transitions = [r for r in rows if r["row_type"] == "state_transition"]
        self.assertEqual(len(transitions), 2)
        event = transitions[-1]
        self.assertEqual((event["state_from"], event["state_to"]), ("Z_DESCENT", "XY_SEARCH"))
        self.assertEqual(float(event["elapsed_s"]), 3.25)
        self.assertEqual(event["left_fx_raw_n"], "")
        self.assertEqual(event["left_fx_delta_n"], "")
        self.assertEqual(float(event["left_fx_baseline_n"]), 1.)

    def test_diagnostics_and_control_clock_are_preserved(self):
        self.now = 102.
        self.recorder.writerow({"stage": "XY_SEARCH", "elapsed_s": .75,
                               "left_fx_raw_n": 4., "left_search_relative_fx_n": 2.6})
        sample = next(r for r in self.rows() if r["row_type"] == "sample")
        self.assertEqual(float(sample["elapsed_s"]), 2.)
        self.assertEqual(float(sample["control_elapsed_s"]), .75)
        self.assertEqual(float(sample["left_search_relative_fx_n"]), 2.6)
        self.assertEqual(float(sample["left_fx_delta_n"]), 3.)

    def test_console_transition_matches_csv_time_and_skips_duplicate_state(self):
        self.now = 104.
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(recording.time, "time", return_value=0.):
            self.recorder.set_state("Z_DESCENT")
            self.recorder.set_state("Z_DESCENT")
            self.now = 106.
            self.recorder.set_state("XY_SEARCH")
        self.assertEqual(output.getvalue().count("[Stage3状态切换]"), 2)
        self.assertIn("北京时间=1970-01-01 08:00:00.000", output.getvalue())
        self.assertIn("CSV t=6.000s", output.getvalue())
        self.assertIn("上一状态2.000s", output.getvalue())
        row = next(r for r in self.rows() if r["state_to"] == "XY_SEARCH")
        self.assertEqual(float(row["elapsed_s"]), 6.)

    def test_abort_and_partial_samples_are_saved_on_close(self):
        self.recorder.sample(self.baseline)
        self.recorder.mark("abort", "反馈超时")
        self.recorder.set_state("ABORTED")
        rows = self.rows()
        self.assertTrue(any(r["event"] == "abort" and r["detail"] == "反馈超时" for r in rows))
        self.assertTrue(any(r["state_to"] == "ABORTED" for r in rows))
        self.assertTrue(any(r["row_type"] == "sample" for r in rows))
        self.assertEqual(rows[-1]["event"], "log_closed")


class PoseReportingTests(TestCase):
    def test_pose6d_uses_meters_and_xyz_euler_degrees(self):
        q = Rotation.from_euler("xyz", [1., -2., 30.], degrees=True).as_quat()
        np.testing.assert_allclose(reporting.pose6d([.1, .2, .3, *q]), [.1, .2, .3, 1., -2., 30.], atol=1e-10)

    def test_unconfirmed_yaw_release_still_reports_signed_z_difference(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            delta = reporting.print_place_result([0., 0., .43, 0., 0., 0., 1.],
                        [0., 0., .437, 0., 0., 0., 1.], release_reason="Yaw搜索未观察到新增下沉")
        self.assertAlmostEqual(delta, 7.)
        self.assertIn("P_place目标6D", output.getvalue())
        self.assertIn("实际放置顶层tray", output.getvalue())
        self.assertIn("+7.000mm", output.getvalue())
        self.assertIn("未独立确认入槽", output.getvalue())
