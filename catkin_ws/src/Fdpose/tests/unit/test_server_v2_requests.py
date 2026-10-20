"""V2 请求边界和互斥回归：真实线程/锁，模拟 ROS、GPU 和文件保存。"""

from queue import Queue
from threading import Event, Lock, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from test_mesh_interfaces import load_subject


class ServiceError(Exception):
    pass


class ObservedLock:
    """记录线程确实到达锁入口，避免靠 sleep 猜测线程调度。"""

    def __init__(self):
        self.lock = Lock()
        self.attempts = Queue()

    def __enter__(self):
        self.attempts.put(True)
        self.lock.acquire()
        return self

    def __exit__(self, *exc):
        self.lock.release()


class ServerV2RequestTests(unittest.TestCase):
    def setUp(self):
        config = load_subject("requests_config_subject", "src/fdpose/ros/server_config.py")
        tools = load_subject("requests_tools_subject", "src/fdpose/ros/server_utils.py")
        self.server = load_subject("requests_server_subject", "src/fdpose/ros/server_v2.py", {
            "fdpose.ros.server_config": config, "fdpose.ros.server_utils": tools,
            "fdpose.ros.server": None,
        })
        self.server.rospy.ServiceException = ServiceError
        self.server.ObjPose = SimpleNamespace
        self.server.PoseEstResponse = lambda **kw: SimpleNamespace(**kw)
        self.server.FdposeLoadMeshResponse = lambda **kw: SimpleNamespace(**kw)
        self.server.cv2.cvtColor.side_effect = lambda image, code: image[..., ::-1].copy()
        self.node = self.server.PoseEstimatorNodeV2.__new__(self.server.PoseEstimatorNodeV2)
        self.node._request_lock = Lock()
        self.node.config = config.ServerConfig(min_mask_pixels=1, min_valid_depth_pixels=1,
                                               min_valid_depth_ratio=1.0, roi_margin=0)
        self.node.K = np.eye(3, dtype=np.float32)
        self.rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        self.depth = np.full((4, 6), 1000, dtype=np.uint16)
        self.mask = np.full((4, 6), 65535, dtype=np.uint16)
        self.images = {"rgb": self.rgb, "depth": self.depth, "mask": self.mask}

        def decode(key, encoding):
            image = self.images[key]
            if isinstance(image, Exception):
                raise image
            return image

        self.node.bridge = Mock()
        self.node.bridge.imgmsg_to_cv2.side_effect = decode
        self.estimator = Mock()
        self.estimator.register.return_value = np.eye(4, dtype=np.float32)
        self.node.objects = {"object": {"pose_estimator": self.estimator}}
        self.node.matrix_to_ros_pose = Mock(return_value="camera_pose")
        self.node.debug_writer = Mock(enabled=True)
        self.node.debug_writer.save_request.return_value = None
        self.node._read_mesh_config = Mock(return_value={})
        self.request = SimpleNamespace(color_image="rgb", depth_image="depth",
                                       items=[SimpleNamespace(label="object", mask="mask")])

    def assert_lock_released(self):
        acquired = self.node._request_lock.acquire(blocking=False)
        if acquired:
            self.node._request_lock.release()
        self.assertTrue(acquired, "callback left the lock held")

    def test_invalid_rgbd_shapes_fail_request_before_inference(self):
        cases = (
            ("rgb", np.zeros((4, 6)), "RGB must"),
            ("rgb", np.zeros((4, 6, 4)), "RGB must"),
            ("rgb", np.zeros((0, 6, 3)), "RGB must"),
            ("depth", np.zeros((4, 6, 1)), "Depth must"),
            ("depth", np.zeros((0, 6)), "Depth must"),
            ("depth", np.zeros((1, 6)), "size mismatch"),
            ("depth", np.zeros((4, 7)), "size mismatch"),
        )
        for key, image, message in cases:
            with self.subTest(key=key, shape=image.shape):
                self.images.update(rgb=self.rgb, depth=self.depth)
                self.images[key] = image
                with self.assertRaisesRegex(ServiceError, message):
                    self.node.handle_pose_estimation(self.request)
                self.estimator.register.assert_not_called()
                self.assert_lock_released()
        self.images.update(rgb=self.rgb, depth=self.depth)
        self.assertEqual(len(self.node.handle_pose_estimation(self.request).obj_poses), 1)

    def test_low_nonzero_background_does_not_change_target_depth_unit(self):
        self.depth.fill(1)
        self.depth[1:3, 2:4] = 1000
        self.mask.fill(0)
        self.mask[1:3, 2:4] = 65535
        response = self.node.handle_pose_estimation(self.request)
        self.assertEqual(len(response.obj_poses), 1)
        self.estimator.register.assert_called_once()
        np.testing.assert_allclose(self.estimator.register.call_args.kwargs["depth"], np.ones((2, 2)))
        self.assert_lock_released()

    def test_rgbd_decode_errors_have_service_error_and_preserve_cause(self):
        for key in ("rgb", "depth"):
            with self.subTest(key=key):
                self.images.update(rgb=self.rgb, depth=self.depth)
                error = ValueError("corrupt image buffer")
                self.images[key] = error
                with self.assertRaisesRegex(ServiceError, "Invalid RGB-D request") as caught:
                    self.node.handle_pose_estimation(self.request)
                self.assertIs(caught.exception.__cause__, error)
                self.assert_lock_released()
        self.estimator.register.assert_not_called()

    def test_invalid_masks_skip_only_bad_item_including_broadcast_shapes(self):
        for image in (np.ones((1, 6)), np.ones((4, 1)), np.ones((4, 6, 1)),
                      np.ones((0, 6)), np.ones((4, 7)), ValueError("corrupt mask")):
            with self.subTest(shape=getattr(image, "shape", "decode error")):
                self.estimator.register.reset_mock()
                self.images["bad"] = image
                self.request.items = [SimpleNamespace(label="object", mask="bad"),
                                      SimpleNamespace(label="object", mask="mask")]
                response = self.node.handle_pose_estimation(self.request)
                self.assertEqual(len(response.obj_poses), 1)
                self.estimator.register.assert_called_once()
                self.server.rospy.logerr.assert_not_called()
                self.assert_lock_released()

    def test_all_items_rejected_keeps_empty_result_contract(self):
        for mask, depth in ((np.zeros_like(self.mask), self.depth),
                            (self.mask, np.zeros_like(self.depth)),
                            (np.ones((1, 6)), self.depth)):
            with self.subTest(mask_shape=mask.shape):
                self.images.update(mask=mask, depth=depth)
                self.assertEqual(self.node.handle_pose_estimation(self.request).obj_poses, [])
                self.estimator.register.assert_not_called()
        self.server.rospy.logerr.assert_not_called()

    def test_roi_preparation_failure_is_not_logged_as_estimation_failure(self):
        self.node._crop_by_mask = Mock(side_effect=ValueError("invalid ROI"))
        self.assertEqual(self.node.handle_pose_estimation(self.request).obj_poses, [])
        self.estimator.register.assert_not_called()
        self.assertIn("input preparation failed", self.server.rospy.logwarn.call_args.args[0])
        self.server.rospy.logerr.assert_not_called()

    def test_estimation_failure_does_not_abort_next_item_or_hold_lock(self):
        self.request.items *= 2
        self.estimator.register.side_effect = [RuntimeError("GPU failure"), np.eye(4)]
        response = self.node.handle_pose_estimation(self.request)
        self.assertEqual(len(response.obj_poses), 1)
        self.assertEqual(self.estimator.register.call_count, 2)
        self.assertTrue(any("estimation failed" in str(call.args[0])
                            for call in self.server.rospy.logerr.call_args_list))
        self.assert_lock_released()

    def test_mesh_error_releases_lock(self):
        self.node._read_mesh_config.side_effect = RuntimeError("registry unavailable")
        request = SimpleNamespace(cancel=False, goods=["object"], mesh_paths=[])
        with self.assertRaisesRegex(RuntimeError, "registry unavailable"):
            self.node.load_meshes_from_yaml(request)
        self.assert_lock_released()
        self.assertEqual(len(self.node.handle_pose_estimation(self.request).obj_poses), 1)

    def run_serialized_pair(self, first, second, entered, release):
        self.node._request_lock = ObservedLock()
        outcomes = Queue()
        second_done = Event()

        def call(name, action):
            try:
                outcomes.put((name, action()))
            except Exception as exc:
                outcomes.put((name, exc))
            finally:
                if name == "second":
                    second_done.set()

        first_thread = Thread(target=call, args=("first", first), daemon=True)
        second_thread = Thread(target=call, args=("second", second), daemon=True)
        first_thread.start()
        second_started = False
        try:
            self.assertTrue(entered.wait(3), "first callback did not reach guarded work")
            self.node._request_lock.attempts.get(timeout=3)
            second_thread.start()
            second_started = True
            self.node._request_lock.attempts.get(timeout=3)
            self.assertFalse(second_done.is_set(), "second callback bypassed the shared lock")
        finally:
            release.set()
            first_thread.join(timeout=3)
            if second_started:
                second_thread.join(timeout=3)
        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        results = dict(outcomes.get(timeout=3) for _ in range(2))
        for result in results.values():
            if isinstance(result, Exception):
                raise result
        return results

    def test_two_pose_requests_are_serialized(self):
        entered, release = Event(), Event()

        def register(**kwargs):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test release timeout")
            return np.eye(4)

        self.estimator.register.side_effect = register
        pose = lambda: self.node.handle_pose_estimation(self.request)
        results = self.run_serialized_pair(pose, pose, entered, release)
        self.assertEqual([len(results[key].obj_poses) for key in ("first", "second")], [1, 1])
        self.assertEqual(self.estimator.register.call_count, 2)

    def test_unload_waits_until_pose_visualization_finishes(self):
        entered, release = Event(), Event()

        def save(*args):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test release timeout")
            self.assertIn("object", self.node.objects)

        self.node.debug_writer.save_request.side_effect = save
        request = SimpleNamespace(cancel=True, goods=["object"], mesh_paths=[])
        results = self.run_serialized_pair(
            lambda: self.node.handle_pose_estimation(self.request),
            lambda: self.node.load_meshes_from_yaml(request), entered, release)
        self.assertEqual(len(results["first"].obj_poses), 1)
        self.assertTrue(results["second"].success)
        self.assertEqual(self.node.objects, {})
        self.server.rospy.logwarn.assert_not_called()

    def test_pose_waits_for_mesh_loading(self):
        entered, release = Event(), Event()
        self.node.objects = {}

        def load(label, path):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test release timeout")
            self.node.objects[label] = {"pose_estimator": self.estimator}
            return True

        self.node._load_mesh_from_path = Mock(side_effect=load)
        request = SimpleNamespace(cancel=False, goods=["object"], mesh_paths=["mock.ply"])
        results = self.run_serialized_pair(
            lambda: self.node.load_meshes_from_yaml(request),
            lambda: self.node.handle_pose_estimation(self.request), entered, release)
        self.assertTrue(results["first"].success)
        self.assertEqual(len(results["second"].obj_poses), 1)

    def test_one_debug_record_per_request_with_duplicate_labels(self):
        self.request.items *= 2
        response = self.node.handle_pose_estimation(self.request)
        self.assertEqual(len(response.obj_poses), 2)
        self.node.debug_writer.save_request.assert_called_once()
        call = self.node.debug_writer.save_request.call_args.args
        self.assertIs(call[0], self.rgb)
        self.assertIs(call[2], self.depth)
        records = call[5]
        self.assertEqual([record["instance_id"] for record in records], [0, 1])
        self.assertEqual([record["status"] for record in records], ["success", "success"])

    def test_debug_disabled_does_not_save_or_retain_masks(self):
        self.node.debug_writer.enabled = False
        self.assertEqual(len(self.node.handle_pose_estimation(self.request).obj_poses), 1)
        self.node.debug_writer.save_request.assert_not_called()

    def test_no_pose_still_records_skipped_status(self):
        self.images["mask"] = np.zeros_like(self.mask)
        self.assertEqual(self.node.handle_pose_estimation(self.request).obj_poses, [])
        records = self.node.debug_writer.save_request.call_args.args[5]
        self.assertEqual(records[0]["reason"], "mask_too_small")


if __name__ == "__main__":
    unittest.main()
