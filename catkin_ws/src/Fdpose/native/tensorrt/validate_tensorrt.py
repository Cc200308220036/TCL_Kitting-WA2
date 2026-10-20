#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 运行：python native/tensorrt/validate_tensorrt.py [--report /路径/report.json]（目标 GPU）。
# 实际执行：加载 pyTRTEng 扩展并与同权重 PyTorch 网络比较；也由 prepare_tensorrt.py 调用。
# 配置：命令行 Engine/扩展路径及 assets/checkpoints/ 模型配置；不读取 ROS YAML。
"""在目标 GPU 上校验三套 Engine 的实际推理，并与同权重 PyTorch FP32 对比。

固定随机输入用于检测部署/数值问题，不代表真实 RGB-D 的位姿精度。
只读取 Engine；默认输出 JSON 到终端，--report 可另外保存报告。
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SPECS = (
    ("score", "test_score_f16_b168.eng", 168, ((1, 168),)),
    ("refine_b168", "test_refine_f16_b168.eng", 168, ((168, 3), (168, 3))),
    ("refine_b1", "test_refine_f16_b1.eng", 1, ((1, 3), (1, 3))),
)


def validate(extension_dir, engine_paths, atol=0.02, rtol=0.02):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run in the fdpose environment on Orin")
    # 先载入待验证的扩展，避免模型导入时的 paths.py 把旧 build 放到搜索路径前面。
    sys.path.insert(0, str(Path(extension_dir).resolve()))
    import pyTRTEng
    if Path(pyTRTEng.__file__).resolve().parent != Path(extension_dir).resolve():
        raise RuntimeError(f"Wrong pyTRTEng loaded: {pyTRTEng.__file__}")
    from export_onnx import load_score_model, load_refine_model

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report = {"passed": True, "seed": 2026, "atol": atol, "rtol": rtol,
              "reference": "PyTorch FP32", "input": "fixed synthetic RGB+XYZ tensors",
              "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
              "models": {}}
    with torch.no_grad():
        for key, _, batch, shapes in SPECS:
            engine = pyTRTEng.TensorRTEngine(2, len(shapes))
            if not engine.loadEngineModel(str(engine_paths[key])):
                raise RuntimeError(f"Engine load returned false: {key}")
            model = load_score_model() if key == "score" else load_refine_model()
            generator = torch.Generator(device="cpu").manual_seed(2026)
            # RGB 为 [0,1]，归一化 XYZ 为 [-1,1]；跨后端使用完全相同的输入。
            host_inputs = [torch.rand((batch, 6, 160, 160), generator=generator) for _ in range(2)]
            for tensor in host_inputs:
                tensor[:, 3:] = tensor[:, 3:] * 2 - 1
            result = {"engine": str(Path(engine_paths[key]).resolve()), "streams": [], "outputs": []}
            # 默认流和非默认流都测，覆盖 PyTorch 输入生产与 TensorRT 执行的顺序。
            for stream in (torch.cuda.default_stream(), torch.cuda.Stream()):
                with torch.cuda.stream(stream):
                    inputs = [tensor.cuda().contiguous() for tensor in host_inputs]
                    reference = model(*inputs, L=batch) if key == "score" else model(*inputs)
                    names = ("score_logit",) if key == "score" else ("trans", "rot")
                    expected = [reference[name].float() for name in names]
                    outputs = [torch.full(shape, float("nan"), device="cuda") for shape in shapes]
                    torch.cuda.synchronize()
                    timings = []
                    for iteration in range(3):
                        for tensor in outputs:
                            tensor.fill_(float("nan"))
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        if not engine.inferEngineModel(inputs, outputs):
                            raise RuntimeError(f"Inference returned false: {key}")
                        torch.cuda.synchronize()
                        timings.append((time.perf_counter() - start) * 1000)
                        for name, actual, target in zip(names, outputs, expected):
                            if actual.shape != target.shape or not torch.isfinite(actual).all().item():
                                raise RuntimeError(f"Invalid output shape/values: {key}/{name}")
                            if not torch.isfinite(target).all().item():
                                raise RuntimeError(f"Non-finite PyTorch reference: {key}/{name}")
                            passed = bool(torch.allclose(actual, target, atol=atol, rtol=rtol))
                            report["passed"] &= passed
                            error = (actual - target).abs()
                            result["outputs"].append({"name": name, "shape": list(actual.shape),
                                                      "stream": "default" if not result["streams"] else "non_default",
                                                      "iteration": iteration,
                                                      "max_abs_error": error.max().item(),
                                                      "mean_abs_error": error.mean().item(),
                                                      "passed": passed})
                    result["streams"].append({"kind": "default" if len(result["streams"]) == 0 else "non_default",
                                              "first_ms": timings[0], "repeat_ms": timings[1:]})
            report["models"][key] = result
            del model, engine, inputs, outputs, expected, reference
            torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-dir", type=Path, default=ROOT / "native/tensorrt/build")
    for key, filename, _, _ in SPECS:
        parser.add_argument("--" + key.replace("_", "-"), type=Path, default=ROOT / "assets/engines" / filename)
    parser.add_argument("--atol", type=float, default=0.02, help="FP16 对 FP32 的逐元素绝对误差阈值")
    parser.add_argument("--rtol", type=float, default=0.02, help="FP16 对 FP32 的逐元素相对误差阈值")
    parser.add_argument("--report", type=Path, help="可选 JSON 报告路径")
    args = parser.parse_args()
    import math
    if not all(math.isfinite(v) and v >= 0 for v in (args.atol, args.rtol)):
        parser.error("atol and rtol must be finite and non-negative")
    try:
        report = validate(args.extension_dir, {key: getattr(args, key) for key, _, _, _ in SPECS}, args.atol, args.rtol)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        return 0 if report["passed"] else 1
    except Exception as exc:
        print(f"TensorRT validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
