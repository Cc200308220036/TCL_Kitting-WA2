"""抓拍工具回归：真实 PNG 读写，ROS/相机订阅边界使用替身。"""

import importlib.util
import io
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stderr

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipIf(cv2 is None, "OpenCV is required for capture tests")
class CaptureTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("capture_one_pair", ROOT / "tools/capture_one_pair.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory(prefix="fdpose-capture-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "captures"
        self.color = np.full((2, 3, 3), [21, 73, 198], dtype=np.uint8)
        self.depth = np.array([[0, 1, 65535], [1000, 2000, 4000]], dtype=np.uint16)
        self.rgb_msg = SimpleNamespace(encoding="rgb8")
        self.depth_msg = SimpleNamespace(encoding="16UC1")

    def decode(self, depth=None, encoding="16UC1", color=None):
        bridge = Mock()
        bridge.imgmsg_to_cv2.side_effect = [self.color if color is None else color,
                                           self.depth if depth is None else depth]
        return self.module.decode_pair(bridge, self.rgb_msg, SimpleNamespace(encoding=encoding))

    def test_color_requests_bgr_and_depth_is_preserved(self):
        bridge = Mock()
        bridge.imgmsg_to_cv2.side_effect = [self.color, self.depth]
        color, depth = self.module.decode_pair(bridge, self.rgb_msg, self.depth_msg)
        self.assertEqual(bridge.imgmsg_to_cv2.call_args_list[0].kwargs, {"desired_encoding": "bgr8"})
        self.assertEqual(bridge.imgmsg_to_cv2.call_args_list[1].kwargs, {"desired_encoding": "passthrough"})
        np.testing.assert_array_equal(color, self.color)
        np.testing.assert_array_equal(depth, self.depth)
        np.testing.assert_array_equal(self.decode(encoding="mono16")[1], self.depth)

    def test_float_meters_round_to_mm_and_invalid_values_become_zero(self):
        meters = np.array([[1.2344, 2.0006, np.nan], [np.inf, -1, 0]], dtype=np.float32)
        before = meters.copy()
        _, result = self.decode(meters, "32FC1")
        np.testing.assert_array_equal(result, [[1234, 2001, 0], [0, 0, 0]])
        self.assertEqual(result.dtype, np.uint16)
        np.testing.assert_array_equal(meters, before)

    def test_out_of_range_depth_is_rejected_without_saturation(self):
        with self.assertRaisesRegex(ValueError, "65.535"):
            self.decode(np.full((2, 3), 66, dtype=np.float32), "32FC1")

    def test_wrong_shapes_and_depth_formats_are_rejected(self):
        for depth, encoding in [(np.zeros((1, 3), np.uint16), "16UC1"),
                                (np.zeros((2, 3, 1), np.uint16), "16UC1"),
                                (np.zeros((2, 3), np.uint8), "8UC1"),
                                (np.zeros((2, 3), np.float32), "16UC1")]:
            with self.subTest(encoding=encoding, shape=depth.shape), self.assertRaises(ValueError):
                self.decode(depth, encoding)
        with self.assertRaises(ValueError):
            self.decode(color=np.zeros((2, 3, 3), dtype=np.float32))

    def test_png_roundtrip_two_files_and_unique_sample_directories(self):
        one = self.module.save_pair(self.root, self.color, self.depth)
        two = self.module.save_pair(self.root, self.color, self.depth)
        self.assertNotEqual(one, two)
        self.assertEqual(sorted(p.relative_to(one).as_posix() for p in one.rglob("*") if p.is_file()),
                         ["depth/0.png", "rgb/0.png"])
        np.testing.assert_array_equal(cv2.imread(str(one / "rgb/0.png")), self.color)
        restored = cv2.imread(str(one / "depth/0.png"), cv2.IMREAD_UNCHANGED)
        self.assertEqual(restored.dtype, np.uint16)
        np.testing.assert_array_equal(restored, self.depth)
        self.assertEqual(stat.S_IMODE((one / "depth/0.png").stat().st_mode), 0o644)
        self.assertFalse(list(self.root.glob(".capture_*")))

    def test_second_write_failure_cleans_partial_sample_only(self):
        existing = self.module.save_pair(self.root, self.color, self.depth)
        real_write = cv2.imwrite
        def fail_depth(path, array, options):
            return False if "/depth/" in path else real_write(path, array, options)
        with patch.object(self.module.cv2, "imwrite", side_effect=fail_depth), self.assertRaises(OSError):
            self.module.save_pair(self.root, self.color, self.depth)
        self.assertEqual(list(self.root.iterdir()), [existing])

    def ros_boundary(self, frames=None, shutdown=False):
        subscribers = [Mock(), Mock()]
        factory = Mock(side_effect=subscribers)
        synchronizer = Mock()
        if frames:
            synchronizer.registerCallback.side_effect = lambda callback: [callback(*pair) for pair in frames]
        filters = SimpleNamespace(Subscriber=factory, TimeSynchronizer=Mock(return_value=synchronizer),
                                  ApproximateTimeSynchronizer=Mock(return_value=synchronizer))
        modules = {"rospy": SimpleNamespace(is_shutdown=Mock(return_value=shutdown)),
                   "message_filters": filters, "sensor_msgs.msg": SimpleNamespace(Image=object)}
        return subscribers, filters, modules

    def test_strict_sync_takes_first_pair_and_unsubscribes(self):
        first = (self.rgb_msg, self.depth_msg)
        subscribers, filters, modules = self.ros_boundary(frames=[first, (object(), object())])
        with patch.dict(sys.modules, modules):
            self.assertEqual(self.module.capture_pair("/rgb", "/depth", 1, 0), first)
        filters.TimeSynchronizer.assert_called_once_with(subscribers, 10)
        filters.ApproximateTimeSynchronizer.assert_not_called()
        for subscriber in subscribers:
            subscriber.unregister.assert_called_once()

    def test_approximate_sync_is_opt_in(self):
        subscribers, filters, modules = self.ros_boundary(frames=[(self.rgb_msg, self.depth_msg)])
        with patch.dict(sys.modules, modules):
            self.module.capture_pair("/rgb", "/depth", 1, .03)
        filters.ApproximateTimeSynchronizer.assert_called_once_with(subscribers, 10, .03, allow_headerless=False)
        filters.TimeSynchronizer.assert_not_called()

    def test_timeout_and_shutdown_unsubscribe_without_creating_output(self):
        for shutdown, error in ((False, TimeoutError), (True, RuntimeError)):
            subscribers, _, modules = self.ros_boundary(shutdown=shutdown)
            with patch.dict(sys.modules, modules), \
                 patch.object(self.module.time, "monotonic", side_effect=[0, 2]), self.assertRaises(error):
                self.module.capture_pair("/rgb", "/depth", 1, 0)
            for subscriber in subscribers:
                subscriber.unregister.assert_called_once()
        self.assertFalse(self.root.exists())

    def test_second_subscription_failure_cleans_first_subscription(self):
        subscribers, filters, modules = self.ros_boundary()
        filters.Subscriber.side_effect = [subscribers[0], RuntimeError("subscribe failed")]
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "subscribe failed"):
            self.module.capture_pair("/rgb", "/depth", 1, 0)
        subscribers[0].unregister.assert_called_once()

    def test_default_output_is_project_relative_not_cwd(self):
        args = self.module.build_parser().parse_args([])
        self.assertEqual(args.out_dir, ROOT / "data/samples/captures")
        self.assertEqual(args.slop, 0)

    def test_invalid_wait_and_slop_rejected_before_ros_import(self):
        for arguments in (("--wait", "0"), ("--wait", "nan"), ("--slop", "-1"), ("--slop", "inf")):
            with patch.object(sys, "argv", ["capture", *arguments]), redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as error:
                self.module.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
