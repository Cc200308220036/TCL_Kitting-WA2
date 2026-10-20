"""新版 YOLO 编排回归；真实数组/检测转换，ROS 和 GPU 边界使用替身。"""
import sys
import threading
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

with patch.dict(sys.modules, {"rospy": Mock()}):
    from fdpose.pipeline import FdposePipeline
from test_mesh_interfaces import load_subject


class FdposePipelineTest(unittest.TestCase):
    def setUp(self):
        self.client_module = load_subject("pipeline_fdpose_client", "src/fdpose/ros/client.py")
        modules = patch.dict(sys.modules, {"fdpose.ros.client": self.client_module})
        modules.start()
        self.addCleanup(modules.stop)
        self.color = np.full((2, 3, 3), [1, 2, 3], dtype=np.uint8)
        self.depth = np.full((2, 3), 1000, dtype=np.uint16)
        self.refined = np.full((2, 3), 1.2, dtype=np.float32)
        self.item = SimpleNamespace(label="wp_2", confidence=0.9, bbox=[0, 0, 3, 2],
                                    mask=np.ones((2, 3), dtype=np.uint16))
        self.response = SimpleNamespace(have_objs=True, items=[self.item])
        self.fdpose = Mock()
        self.fdpose.estimate.return_value = ["pose"]
        self.segmentation = Mock(spec=["segment"])
        self.segmentation.segment.return_value = self.response
        self.refiner = Mock(spec=["refine"])
        self.refiner.refine.return_value = self.refined
        self.pipeline = FdposePipeline(self.fdpose, self.segmentation, self.refiner)

    def process(self, **kwargs):
        inputs = dict(color=self.color, depth=self.depth, labels=["wp_2"])
        inputs.update(kwargs)
        return self.pipeline.process_rgbd(**inputs)

    def test_new_response_and_real_detection_conversion(self):
        self.assertEqual(self.process(labels=["wp_2", "wp_2"]), ["pose"])
        self.fdpose.load_meshes.assert_called_once_with(["wp_2"])
        request = self.segmentation.segment.call_args.kwargs
        self.assertEqual(set(request), {"color", "depth", "labels"})
        self.assertIs(request["color"], self.color)
        self.assertIs(request["depth"], self.depth)
        call = self.fdpose.estimate.call_args.kwargs
        self.assertIs(call["color_bgr"], self.color)
        self.assertIs(call["depth"], self.refined)
        detection = call["detections"][0]
        self.assertIsInstance(detection, self.client_module.DetectionInput)
        self.assertEqual(detection.label, "wp_2")
        self.assertEqual(detection.bbox, (0, 0, 3, 2))
        self.assertAlmostEqual(detection.confidence, 0.9)
        np.testing.assert_array_equal(detection.mask, self.item.mask)

    def test_ros_mask_decoding_uses_existing_client(self):
        mask = self.item.mask.copy()
        self.item.mask = object()
        bridge = self.client_module.CvBridge.return_value
        bridge.imgmsg_to_cv2.return_value = mask
        self.process()
        bridge.imgmsg_to_cv2.assert_called_once_with(
            self.item.mask, desired_encoding="passthrough")
        np.testing.assert_array_equal(
            self.fdpose.estimate.call_args.kwargs["detections"][0].mask, mask)

    def test_rgb_converted_once_and_input_unchanged(self):
        before = self.color.copy()
        self.process(color_order="rgb")
        bgr = self.segmentation.segment.call_args.kwargs["color"]
        np.testing.assert_array_equal(bgr, before[:, :, ::-1])
        self.assertTrue(bgr.flags.c_contiguous)
        self.assertIs(self.refiner.refine.call_args.args[0], bgr)
        self.assertIs(self.fdpose.estimate.call_args.kwargs["color_bgr"], bgr)
        np.testing.assert_array_equal(self.color, before)

    def test_raw_depth_units_external_mesh_and_tf(self):
        for depth in (self.depth, self.refined):
            transform = np.eye(4)
            self.process(depth=depth, run_depth_refine=False,
                         load_mesh=False, T_target_camera=transform)
            call = self.fdpose.estimate.call_args.kwargs
            self.assertIs(call["depth"], depth)
            self.assertIs(call["T_target_camera"], transform)
        self.refiner.refine.assert_not_called()
        self.fdpose.load_meshes.assert_not_called()

    def test_serial_order(self):
        calls = Mock()
        for name, method in (("mesh", self.fdpose.load_meshes),
                             ("seg", self.segmentation.segment),
                             ("refine", self.refiner.refine), ("pose", self.fdpose.estimate)):
            calls.attach_mock(method, name)
        self.process()
        self.assertEqual([call[0] for call in calls.mock_calls], ["mesh", "seg", "refine", "pose"])

    def test_parallel_waits_for_both_and_uses_same_frame(self):
        barrier = threading.Barrier(2)
        def segment(*, color, depth, labels):
            self.assertIs(color, self.color)
            self.assertIs(depth, self.depth)
            self.assertEqual(labels, ["wp_2"])
            barrier.wait(timeout=3)
            return self.response
        def refine(color, depth):
            self.assertIs(color, self.color)
            self.assertIs(depth, self.depth)
            barrier.wait(timeout=3)
            return self.refined
        self.segmentation.segment.side_effect = segment
        self.refiner.refine.side_effect = refine
        self.assertEqual(self.pipeline.process_rgbd_parallel(
            self.color, self.depth, ["wp_2"]), ["pose"])
        self.assertIs(self.fdpose.estimate.call_args.kwargs["depth"], self.refined)

    def test_parallel_without_refiner_uses_raw_depth(self):
        pipeline = FdposePipeline(self.fdpose, self.segmentation)
        self.assertEqual(pipeline.process_rgbd_parallel(
            self.color, self.depth, ["wp_2"]), ["pose"])
        self.assertIs(self.fdpose.estimate.call_args.kwargs["depth"], self.depth)

    def test_no_detection_skips_serial_refiner_and_pose(self):
        for response in (SimpleNamespace(have_objs=False, items=[]),
                         SimpleNamespace(have_objs=True, items=[])):
            self.segmentation.segment.return_value = response
            self.assertEqual(self.process(), [])
        self.refiner.refine.assert_not_called()
        self.fdpose.estimate.assert_not_called()

    def test_parallel_no_detection_still_joins_refiner(self):
        self.segmentation.segment.return_value = SimpleNamespace(have_objs=False, items=[])
        self.assertEqual(self.pipeline.process_rgbd_parallel(
            self.color, self.depth, ["wp_2"]), [])
        self.refiner.refine.assert_called_once()
        self.fdpose.estimate.assert_not_called()

    def test_service_errors_are_not_hidden(self):
        for process in (self.pipeline.process_rgbd, self.pipeline.process_rgbd_parallel):
            for method in (self.segmentation.segment, self.refiner.refine):
                method.side_effect = RuntimeError("service unavailable")
                with self.assertRaisesRegex(RuntimeError, "service unavailable"):
                    process(self.color, self.depth, ["wp_2"])
                method.side_effect = None
        self.fdpose.estimate.assert_not_called()

    def test_mesh_failure_stops_segmentation(self):
        self.fdpose.load_meshes.side_effect = RuntimeError("mesh load failed")
        with self.assertRaisesRegex(RuntimeError, "mesh load failed"):
            self.process()
        self.segmentation.segment.assert_not_called()

    def test_pose_empty_result_is_preserved(self):
        self.fdpose.estimate.return_value = []
        self.assertEqual(self.process(), [])

    def test_bad_input_fails_before_service_calls(self):
        cases = [dict(color=np.zeros((2, 3), dtype=np.uint8)),
                 dict(color=self.color.astype(np.float32)), dict(color_order="invalid"),
                 dict(depth=None), dict(depth=np.zeros((3, 2), dtype=np.uint16)),
                 dict(depth=self.depth.astype(np.float64)), dict(labels=[]),
                 dict(labels=None), dict(labels=["wp_2", " "]), dict(labels=[123])]
        for inputs in cases:
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                self.process(**inputs)
        self.fdpose.load_meshes.assert_not_called()
        self.segmentation.segment.assert_not_called()

    def test_label_suffix_is_not_interpreted(self):
        self.process(labels=" wp_2_unstable ")
        self.fdpose.load_meshes.assert_called_once_with(["wp_2_unstable"])
        self.assertEqual(self.segmentation.segment.call_args.kwargs["labels"], ["wp_2_unstable"])

    def test_missing_refiner_fails_before_mesh_load(self):
        self.pipeline.depth_refine = None
        with self.assertRaises(RuntimeError):
            self.process(run_depth_refine=True)
        self.fdpose.load_meshes.assert_not_called()

    def test_bad_mask_or_refined_depth_never_reaches_pose(self):
        self.item.mask = np.ones((3, 2), dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "mask"):
            self.process()
        self.item.mask = np.ones((2, 3), dtype=np.uint16)
        for depth in (np.ones((3, 2), dtype=np.float32), self.depth.astype(np.int32),
                      np.zeros((2, 3), dtype=np.uint16),
                      np.full((2, 3), np.nan, dtype=np.float32)):
            self.refiner.refine.return_value = depth
            for process in (self.pipeline.process_rgbd, self.pipeline.process_rgbd_parallel):
                with self.subTest(depth=depth), self.assertRaises(ValueError):
                    process(self.color, self.depth, ["wp_2"])
        self.fdpose.estimate.assert_not_called()

    def test_zero_raw_depth_requires_successful_refinement(self):
        zero = np.zeros_like(self.depth)
        with self.assertRaisesRegex(ValueError, "finite positive"):
            self.process(depth=zero, run_depth_refine=False)
        self.assertEqual(self.process(depth=zero), ["pose"])

    def test_camera_provider_called_once_and_reuses_pipeline(self):
        provider = Mock(return_value=(self.color, self.depth))
        self.assertEqual(self.pipeline.process_from_camera(
            provider, ["wp_2"], run_depth_refine=False, load_mesh=False), ["pose"])
        provider.assert_called_once_with()
        self.fdpose.load_meshes.assert_not_called()
        self.assertIs(self.fdpose.estimate.call_args.kwargs["depth"], self.depth)

    def test_bad_camera_provider(self):
        with self.assertRaises(TypeError):
            self.pipeline.process_from_camera(None, ["wp_2"])
        with self.assertRaises(ValueError):
            self.pipeline.process_from_camera(lambda: (self.color,), ["wp_2"])

    def test_factory_uses_new_yolo_signature_and_v2_defaults(self):
        factory = Mock(return_value=self.segmentation)
        module = SimpleNamespace(SegServiceClient=factory)
        with patch.dict(sys.modules, {"naviai_yolo.ros.seg_service_client": module}):
            with patch.object(self.client_module, "FdposeClient", return_value=self.fdpose) as fdpose:
                pipeline = FdposePipeline.from_ros_services(timeout=12)
        factory.assert_called_once_with(
            service_name="/instance_segmentation_service", wait_timeout_sec=12.0)
        fdpose.assert_called_once_with(mesh_service="/fdpose_v2/load_mesh",
                                      pose_service="/fdpose_v2/pose_estimation", timeout=12)
        self.assertIs(pipeline.segmentation, self.segmentation)

    def test_custom_factories_keep_depth_contract(self):
        seg, refine = Mock(), Mock()
        FdposePipeline.from_ros_services(
            segmentation_client_factory=seg, segmentation_service="/custom_seg",
            depth_refine_client_factory=refine, depth_refine_service="/depth",
            timeout=5, segmentation_client_kwargs={"custom": True})
        seg.assert_called_once_with(service_name="/custom_seg", wait_timeout_sec=5.0, custom=True)
        refine.assert_called_once_with(service_name="/depth", timeout_sec=5.0)

    def test_factory_rejects_ambiguous_parameters(self):
        for key in ("timeout_sec", "wait_timeout_sec", "service_name"):
            with self.assertRaises(ValueError):
                FdposePipeline.from_ros_services(
                    segmentation_client_factory=Mock(), segmentation_client_kwargs={key: 1})
        with self.assertRaises(ValueError):
            FdposePipeline.from_ros_services(depth_refine_service="/depth")

    def config_file(self, text):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "pipeline.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_yaml_values_and_explicit_override(self):
        path = self.config_file(
            "mesh_service: /mesh_custom\npose_service: /pose_custom\n"
            "segmentation_service: /yolo_custom\ntimeout: 8\n")
        seg = Mock()
        with patch.object(self.client_module, "FdposeClient") as fdpose:
            FdposePipeline.from_ros_services(config_path=path, segmentation_client_factory=seg)
            fdpose.assert_called_once_with(mesh_service="/mesh_custom", pose_service="/pose_custom", timeout=8.0)
            seg.assert_called_once_with(service_name="/yolo_custom", wait_timeout_sec=8.0)
            fdpose.reset_mock()
            seg.reset_mock()
            FdposePipeline.from_ros_services(
                config_path=path, segmentation_client_factory=seg,
                pose_service="/pose_override", segmentation_service="/seg_override", timeout=3)
            fdpose.assert_called_once_with(mesh_service="/mesh_custom", pose_service="/pose_override", timeout=3.0)
            seg.assert_called_once_with(service_name="/seg_override", wait_timeout_sec=3.0)

    def test_optional_config_and_missing_fields_use_defaults(self):
        for path in (None, self.config_file(""), self.config_file("timeout: 6\n")):
            seg = Mock()
            with patch.object(self.client_module, "FdposeClient") as fdpose:
                FdposePipeline.from_ros_services(config_path=path, segmentation_client_factory=seg)
                self.assertEqual(fdpose.call_args.kwargs["mesh_service"], "/fdpose_v2/load_mesh")
                self.assertEqual(fdpose.call_args.kwargs["pose_service"], "/fdpose_v2/pose_estimation")
                self.assertEqual(seg.call_args.kwargs["service_name"], "/instance_segmentation_service")

    def test_missing_file_is_not_silently_ignored(self):
        path = self.config_file("").with_name("absent.yaml")
        seg = Mock()
        with self.assertRaises(FileNotFoundError):
            FdposePipeline.from_ros_services(config_path=path, segmentation_client_factory=seg)
        seg.assert_not_called()

    def test_bad_config_fails_before_any_client_is_constructed(self):
        for contents in ("[", "[]", "timeout: 0", "timeout: true", "timeout: .nan",
                         "timeout: .inf", "timeout: '20'", "mesh_service: ''",
                         "pose_service: null", "segmentation_service: 5",
                         "services: {}", "client_file: /some/code.py"):
            with self.subTest(contents=contents):
                seg = Mock()
                with patch.object(self.client_module, "FdposeClient") as fdpose:
                    with self.assertRaises(ValueError):
                        FdposePipeline.from_ros_services(
                            config_path=self.config_file(contents), segmentation_client_factory=seg)
                    seg.assert_not_called()
                    fdpose.assert_not_called()

    def test_injection_never_reads_connection_yaml(self):
        loader = Mock()
        with patch.dict(FdposePipeline.__init__.__globals__, {"_load_connection_config": loader}):
            pipeline = FdposePipeline(self.fdpose, self.segmentation, self.refiner)
        loader.assert_not_called()
        self.assertIs(pipeline.fdpose, self.fdpose)

    def test_bad_explicit_timeout_is_not_ignored(self):
        for value in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FdposePipeline.from_ros_services(
                    config_path=None, timeout=value, segmentation_client_factory=Mock())


if __name__ == "__main__":
    unittest.main()
