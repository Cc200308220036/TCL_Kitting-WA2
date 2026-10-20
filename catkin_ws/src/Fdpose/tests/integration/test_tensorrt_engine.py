# 运行：python -B -m unittest tests.integration.test_tensorrt_engine -v（目标 GPU）。
# 实际调用：native/tensorrt/build/pyTRTEng 扩展和 assets/engines/；无 CUDA 时跳过。
# 配置：不读取 ROS YAML；先运行 tools/build/prepare_tensorrt.py。
"""GPU 回归：需先在 Orin 执行 prepare_tensorrt.py；无 CUDA 时跳过。"""

from pathlib import Path
import sys
import unittest


class TensorRTEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        cls.torch = torch
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "native/tensorrt/build"))
        import pyTRTEng
        cls.engine_type = pyTRTEng.TensorRTEngine
        cls.engine_path = root / "assets/engines/test_refine_f16_b1.eng"

    def setUp(self):
        self.engine = self.engine_type(2, 2)
        self.assertTrue(self.engine.loadEngineModel(str(self.engine_path)))
        self.inputs = [self.torch.rand(1, 6, 160, 160, device="cuda") for _ in range(2)]
        self.outputs = [self.torch.empty(1, 3, device="cuda") for _ in range(2)]

    def test_rejects_wrong_count_shape_dtype_device_and_layout(self):
        bad_inputs = (
            self.inputs[:1],
            [self.inputs[0][:, :5].contiguous(), self.inputs[1]],
            [self.inputs[0].half(), self.inputs[1]],
            [self.inputs[0].cpu(), self.inputs[1]],
            [self.inputs[0].transpose(2, 3), self.inputs[1]],
        )
        for inputs in bad_inputs:
            with self.subTest(shape=tuple(inputs[0].shape)), self.assertRaises(RuntimeError):
                self.engine.inferEngineModel(inputs, self.outputs)
        with self.assertRaises(RuntimeError):
            self.engine.inferEngineModel(self.inputs, [self.outputs[0][:, :2], self.outputs[1]])

    def test_failed_reload_invalidates_context(self):
        with self.assertRaises(RuntimeError):
            self.engine.loadEngineModel(str(self.engine_path.parent / "missing-test-engine.eng"))
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            self.engine.inferEngineModel(self.inputs, self.outputs)
        self.assertTrue(self.engine.loadEngineModel(str(self.engine_path)))
        self.assertTrue(self.engine.inferEngineModel(self.inputs, self.outputs))

    def test_non_default_stream_produces_same_result(self):
        torch = self.torch
        self.assertTrue(self.engine.inferEngineModel(self.inputs, self.outputs))
        reference = [value.clone() for value in self.outputs]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # clone 和 fill 在当前流排队，进入扩展前不做全设备同步。
            inputs = [value.clone() for value in self.inputs]
            outputs = [torch.full_like(value, float("nan")) for value in self.outputs]
            self.assertTrue(self.engine.inferEngineModel(inputs, outputs))
            for actual, expected in zip(outputs, reference):
                self.assertTrue(torch.isfinite(actual).all().item())
                self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
