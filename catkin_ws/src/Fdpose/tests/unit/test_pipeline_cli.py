"""新版本地 Pipeline 入口及历史 V2 入口回归；不连接实际服务。"""

import contextlib
import io
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_mesh_interfaces import load_subject


ROOT = Path(__file__).resolve().parents[2]


class PipelineCliTests(unittest.TestCase):
    def setUp(self):
        client = load_subject("cli_fdpose_client", "src/fdpose/ros/client.py")
        self.module = load_subject(
            "pipeline_cli_subject", "tools/clients/test_pipeline.py",
            {"fdpose.ros.client": client},
        )
        self.module.rospy.ROSException = type("ROSException", (Exception,), {})
        self.module.rospy.ServiceException = type("ServiceException", (Exception,), {})

    def args(self):
        return self.module.build_parser().parse_args(
            ["--rgb", "color.png", "--depth", "depth.png", "--labels", "wp_2"])

    def test_default_cli_config_is_project_file(self):
        args = self.args()
        self.assertEqual(Path(args.config), ROOT / "config/fdpose_pipeline.yaml")
        self.assertIsNone(args.timeout)

    def test_cli_delegates_to_public_pipeline(self):
        color, depth = np.zeros((2, 3, 3), dtype=np.uint8), np.ones((2, 3), dtype=np.uint16)
        self.module.cv2.imread.side_effect = [color, depth]
        args = self.args()
        with patch.object(self.module.FdposePipeline, "from_ros_services") as factory:
            factory.return_value.process_rgbd.return_value = []
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(self.module.run_local_test(args), 0)
        factory.assert_called_once_with(config_path=args.config, timeout=None)
        call = factory.return_value.process_rgbd.call_args.kwargs
        self.assertIs(call["color"], color)
        self.assertIs(call["depth"], depth)
        self.assertEqual(call["labels"], ["wp_2"])
        self.assertIn("returned poses: 0", output.getvalue())

    def test_bad_file_does_not_construct_clients(self):
        self.module.cv2.imread.return_value = None
        with patch.object(self.module.FdposePipeline, "from_ros_services") as factory:
            with self.assertRaises(ValueError):
                self.module.run_local_test(self.args())
        factory.assert_not_called()

    def test_main_initializes_ros_and_reports_failure(self):
        self.module.rospy.myargv.return_value = [
            "test_pipeline.py", "--rgb", "color.png", "--depth", "depth.png", "--labels", "wp_2"]
        with patch.object(self.module, "run_local_test", side_effect=ValueError("bad config")):
            self.assertEqual(self.module.main(), 1)
        self.module.rospy.init_node.assert_called_once_with("fdpose_pipeline_local_test", anonymous=True)
        self.module.rospy.logerr.assert_called_once()

    def test_historical_installed_entry_warns_and_forwards_to_v2(self):
        server = Mock()
        with patch.dict(sys.modules, {"fdpose.ros.server_v2": server}):
            with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
                with contextlib.redirect_stderr(io.StringIO()) as stderr:
                    runpy.run_path(str(ROOT / "scripts/pose_estimator_node.py"), run_name="__main__")
        server.main.assert_called_once_with()
        self.assertIn("not Legacy", stderr.getvalue())
        self.assertIn("/fdpose_v2/pose_estimation", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
