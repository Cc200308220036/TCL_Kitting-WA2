#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 运行：python native/tensorrt/compare_pose_backends.py --sample-dir assets/warmup/wp_2。
# 实际执行：本文件分别启动两个后端子进程，调用 src/fdpose/core/estimator.py::register。
# 配置：样例 warmup.yaml、assets/meshes/registry.yaml 和模型资产；不读取 V2 服务 YAML。
"""用固定 RGB-D/Mask/Mesh 比较 PyTorch 与 TensorRT 的完整 register 流程。

两个后端分别在新进程中运行；使用样例内参、完整图像、相同随机种子。
这是核心推理回归工具，不启动 ROS，不等同于 V1/V2 Service 联调。
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]


def run_backend(args):
    sys.path.insert(0, str(ROOT / "src"))
    import cv2
    import numpy as np
    import torch
    import trimesh
    import yaml
    from fdpose.core.estimator import FoundationPose
    from fdpose.core.utils import set_seed, draw_posed_3d_box, draw_xyz_axis
    from fdpose.learning.training import predict_score, predict_pose_refine
    import nvdiffrast.torch as dr

    sample_dir = args.sample_dir.resolve()
    sample = yaml.safe_load((sample_dir / "warmup.yaml").read_text())
    rgb = cv2.imread(str(sample_dir / sample["rgb"]))
    depth = cv2.imread(str(sample_dir / sample["depth"]), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(sample_dir / sample["mask"]), cv2.IMREAD_GRAYSCALE)
    if rgb is None or depth is None or mask is None:
        raise ValueError("Cannot read sample RGB/depth/mask")
    if rgb.shape[:2] != depth.shape or depth.shape != mask.shape or not (mask > 0).any():
        raise ValueError("Sample dimensions do not match, or mask is empty")
    K = np.loadtxt(sample_dir / sample["camera_intrinsics"]).reshape(3, 3)
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid sample camera intrinsics")
    unit = sample.get("depth_unit", "mm")
    if unit not in ("mm", "m"):
        raise ValueError(f"Unsupported depth_unit: {unit}")
    depth = np.nan_to_num(depth.astype(np.float32), nan=0, posinf=0, neginf=0)
    depth *= 0.001 if unit == "mm" else 1.0
    mask = mask > 0
    if np.count_nonzero(depth[mask] >= 0.001) < 4:
        raise ValueError("No valid depth inside the sample mask")
    color = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    registry = yaml.safe_load((ROOT / "assets/meshes/registry.yaml").read_text())
    mesh_path = args.mesh or ROOT / "assets/meshes" / registry["meshes"][sample["label"]]["path"]
    mesh = trimesh.load(str(mesh_path), force="mesh")
    bbox = np.asarray(mesh.bounds).copy()

    set_seed(0)
    if args.backend == "torch":
        # 仅在这个测试子进程中禁用 TRT 导入，不修改服务配置或磁盘 Engine。
        predict_score.pyTRTEng = predict_pose_refine.pyTRTEng = None
        predict_score._PYTRT_IMPORT_ERROR = predict_pose_refine._PYTRT_IMPORT_ERROR = "disabled for regression"
    start = time.perf_counter()
    scorer = predict_score.ScorePredictor()
    refiner = predict_pose_refine.PoseRefinePredictor()
    if args.backend == "trt" and not (scorer.use_trt and refiner.use_trt):
        raise RuntimeError("TensorRT backend is not active; fallback is not accepted in this test")
    estimator = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                              mesh=mesh, scorer=scorer, refiner=refiner,
                              glctx=dr.RasterizeCudaContext(), debug=0,
                              debug_dir=str(args.output_dir / args.backend))
    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - start
    output_dir = args.output_dir / args.backend
    output_dir.mkdir(parents=True, exist_ok=True)
    timings, poses = [], []
    for index in range(args.repeats):
        set_seed(0)
        torch.cuda.synchronize()
        start = time.perf_counter()
        pose = estimator.register(K=K.copy(), rgb=color.copy(), depth=depth.copy(),
                                  ob_mask=mask.copy(), iteration=args.iterations)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        pose = np.asarray(pose).reshape(4, 4)
        if not np.isfinite(pose).all():
            raise RuntimeError("Non-finite estimated pose")
        poses.append(pose.tolist())
        np.save(output_dir / f"pose_{index}.npy", pose)
        vis = draw_posed_3d_box(K, img=color.copy(), ob_in_cam=pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=pose, scale=0.1, K=K,
                            thickness=3, transparency=0, is_input_rgb=True)
        if not cv2.imwrite(str(output_dir / f"vis_{index}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("Failed to save pose visualization")
    report = {"backend": args.backend, "sample": str(sample_dir), "mesh": str(mesh_path),
              "iterations": args.iterations, "seed": 0, "use_roi": False,
              "torch_amp": bool(scorer.amp), "initialization_seconds": init_seconds,
              "register_seconds": timings, "T_camera_object": poses}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=ROOT / "assets/warmup/wp_2")
    parser.add_argument("--mesh", type=Path, help="可选 Mesh 路径；默认按样例 Label 查注册表")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3, help="首轮 + 至少一轮后续 register")
    parser.add_argument("--output-dir", type=Path, help="新建输出目录，默认 assets/engines/validation/ 下独立目录")
    parser.add_argument("--backend", choices=("torch", "trt"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 2:
        parser.error("iterations must be >= 1 and repeats must be >= 2")
    if args.backend:
        run_backend(args)
        return
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    else:
        parent = ROOT / "assets/engines/validation"
        parent.mkdir(parents=True, exist_ok=True)
        args.output_dir = Path(tempfile.mkdtemp(prefix="pose_", dir=str(parent)))
    for backend in ("torch", "trt"):
        command = [sys.executable, str(Path(__file__).resolve()), "--backend", backend,
                   "--sample-dir", str(args.sample_dir.resolve()),
                   "--output-dir", str(args.output_dir.resolve()),
                   "--iterations", str(args.iterations), "--repeats", str(args.repeats)]
        if args.mesh:
            command.extend(["--mesh", str(args.mesh.resolve())])
        subprocess.run(command, check=True)
    import numpy as np
    reports = {key: json.loads((args.output_dir / key / "report.json").read_text()) for key in ("torch", "trt")}
    differences = []
    for a, b in zip(reports["torch"]["T_camera_object"], reports["trt"]["T_camera_object"]):
        a, b = np.asarray(a), np.asarray(b)
        angle = np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)))
        differences.append({"translation_difference_m": float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
                            "rotation_difference_deg": float(angle)})
    reports["differences"] = differences
    reports["note"] = "Differences are not ground-truth accuracy; rotation differences do not account for object symmetry."
    report_path = args.output_dir / "comparison.json"
    report_path.write_text(json.dumps(reports, indent=2) + "\n")
    print(f"Pose regression report: {report_path}")


if __name__ == "__main__":
    main()
