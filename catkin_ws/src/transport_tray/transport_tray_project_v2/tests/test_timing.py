"""耗时累计的离线检查，不加载ROS，不执行动作。"""
import contextlib
import importlib.util
import io
from pathlib import Path
from unittest import TestCase, mock

spec = importlib.util.spec_from_file_location(
    "_timing", Path(__file__).resolve().parents[1] / "tray_control/observability/timing.py")
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


class TimingTests(TestCase):
    def setUp(self):
        self.now = 0.0
        self.timer = timing.StateTimer("Stage3多层", clock=lambda: self.now)

    def tearDown(self):
        timing._active = None

    def test_repeated_state_accumulates_without_overlapping_time(self):
        self.timer.switch(None, "XY_SEARCH")
        self.now = 2.0
        self.timer.switch(None, "XY_SEARCH")  # 同一周期重复通知不分段
        self.now = 3.0
        self.timer.switch(None, "YAW_SEARCH")
        self.now = 5.0
        self.timer.switch(None, "XY_SEARCH")
        self.now = 6.0
        self.timer.finish()
        self.assertEqual(self.timer.totals[("Stage3多层", "XY_SEARCH")], 4.0)
        self.assertEqual(self.timer.visits[("Stage3多层", "XY_SEARCH")], 2)
        self.assertEqual(sum(self.timer.totals.values()), 6.0)

    def test_confirmation_time_is_separate_and_previous_state_restored(self):
        timing._active = self.timer
        timing.state("Z_DESCENT")
        self.now = 2.0
        with timing.phase(None, "人工确认等待"):
            self.now = 7.0
        self.assertEqual(self.timer.current, ("Stage3多层", "Z_DESCENT"))
        self.now = 8.0
        self.timer.finish()
        self.assertEqual(self.timer.totals[("Stage3多层", "人工确认等待")], 5.0)
        self.assertEqual(self.timer.totals[("Stage3多层", "Z_DESCENT")], 3.0)

    def test_stage_boundary_does_not_double_count(self):
        timing._active = self.timer
        with timing.phase("Stage0", "视觉识别"):
            self.now = 2.0
            timing.state("预抓取到位")
            self.now = 5.0
        self.now = 6.0
        self.timer.finish()
        self.assertEqual(sum(v for (g, _), v in self.timer.totals.items() if g == "Stage0"), 5.0)
        self.assertEqual(sum(self.timer.totals.values()), 6.0)

    def test_exception_reports_partial_summary_and_does_not_swallow_error(self):
        def fail():
            timing.state("XY_SEARCH")
            self.now = 4.0
            raise KeyboardInterrupt()
        output = io.StringIO()
        with mock.patch.object(timing, "StateTimer", return_value=self.timer), contextlib.redirect_stdout(output):
            with self.assertRaises(KeyboardInterrupt):
                timing.run_with_summary(fail, "Stage3多层")
        self.assertIn("异常/中断", output.getvalue())
        self.assertIn("XY_SEARCH：4.000s", output.getvalue())
        self.assertIsNone(timing._active)

    def test_cycle_timing_keeps_feedback_servo_and_sleep_breakdown(self):
        now = [1.]
        with mock.patch.object(timing.time, 'monotonic', side_effect=lambda: now[0]):
            cycle = timing.CycleTiming(.02)
            self.assertIn('不可用', cycle.detail(.02))
            cycle.begin(now[0])
            with cycle.section('tcp'):
                now[0] += .003
            with cycle.section('servo'):
                now[0] += .002
            now[0] += .001
            rate = mock.Mock()
            rate.sleep.side_effect = lambda: now.__setitem__(0, now[0] + .014)
            cycle.finish(rate)
        rate.sleep.assert_called_once()
        self.assertAlmostEqual(cycle.previous['tcp'], .003)
        self.assertAlmostEqual(cycle.previous['servo'], .002)
        self.assertAlmostEqual(cycle.previous['compute'], .001)
        self.assertAlmostEqual(cycle.previous['sleep'], .014)
        self.assertIn('TCP=3.0', cycle.detail(.02))
