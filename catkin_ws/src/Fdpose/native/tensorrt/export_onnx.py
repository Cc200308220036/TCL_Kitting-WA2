#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 运行：python native/tensorrt/export_onnx.py（模型导出，不启动 ROS Service）。
# 实际调用：src/fdpose/learning/models/ 中的网络，将模型导出到 assets/onnx/。
# 配置：assets/checkpoints/ 下对应模型的 config.yml 和权重；之后运行 tools/build/prepare_tensorrt.py。
"""
Export FDPose score/refine PyTorch models to ONNX for TensorRT engine building.

Usage:
    conda activate fdpose
    python native/tensorrt/export_onnx.py

Output:
    score_b168.onnx
    refine_b168.onnx
    refine_b1.onnx

Then build TRT engines:
    /usr/src/tensorrt/bin/trtexec --onnx=score_b168.onnx    --saveEngine=test_score_f16_b168.eng    --fp16
    /usr/src/tensorrt/bin/trtexec --onnx=refine_b168.onnx  --saveEngine=test_refine_f16_b168.eng  --fp16
    /usr/src/tensorrt/bin/trtexec --onnx=refine_b1.onnx    --saveEngine=test_refine_f16_b1.eng    --fp16
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fdpose.learning.models.score_network import ScoreNetMultiPair
from fdpose.learning.models.refine_network import RefineNet
from fdpose.paths import ONNX_DIR, WEIGHTS_DIR


SCORE_WEIGHTS = WEIGHTS_DIR / "2024-01-11-20-02-45"
REFINE_WEIGHTS = WEIGHTS_DIR / "2023-10-28-18-33-37"

C_IN = 6
H, W = 160, 160


class ScoreNetWrapper(nn.Module):
    """Wrap ScoreNetMultiPair to fix L and only take A, B as inputs."""
    def __init__(self, model, L=168):
        super().__init__()
        self.model = model
        self.L = L

    def forward(self, A, B):
        out = self.model(A, B, self.L)
        return out["score_logit"]


class RefineNetWrapper(nn.Module):
    """Wrap RefineNet to return (trans, rot) tuple."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, A, B):
        out = self.model(A, B)
        return out["trans"], out["rot"]


def load_score_model():
    cfg = OmegaConf.load(str(SCORE_WEIGHTS / "config.yml"))
    model = ScoreNetMultiPair(cfg=cfg, c_in=C_IN)
    ckpt = torch.load(str(SCORE_WEIGHTS / "model_best.pth"), map_location="cuda")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.cuda().eval()
    return model


def load_refine_model():
    cfg = OmegaConf.load(str(REFINE_WEIGHTS / "config.yml"))
    model = RefineNet(cfg=cfg, c_in=C_IN, n_view=1)
    ckpt = torch.load(str(REFINE_WEIGHTS / "model_best.pth"), map_location="cuda")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.cuda().eval()
    return model


def export_score():
    print("=" * 60)
    print("Exporting score model (batch=168, L=168)...")
    print("=" * 60)
    model = load_score_model()
    wrapper = ScoreNetWrapper(model, L=168).cuda().eval()

    B, L = 1, 168
    dummy_A = torch.randn(B * L, C_IN, H, W, device="cuda")
    dummy_B = torch.randn(B * L, C_IN, H, W, device="cuda")

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = str(ONNX_DIR / "score_b168.onnx")
    torch.onnx.export(
        wrapper,
        (dummy_A, dummy_B),
        onnx_path,
        input_names=["A", "B"],
        output_names=["score_logit"],
        opset_version=17,
        dynamic_axes=None,
    )
    print(f"Saved: {onnx_path}")
    del model, wrapper
    torch.cuda.empty_cache()


def export_refine(batch_size, tag):
    print("=" * 60)
    print(f"Exporting refine model (batch={batch_size})...")
    print("=" * 60)
    model = load_refine_model()
    wrapper = RefineNetWrapper(model).cuda().eval()

    dummy_A = torch.randn(batch_size, C_IN, H, W, device="cuda")
    dummy_B = torch.randn(batch_size, C_IN, H, W, device="cuda")

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = str(ONNX_DIR / f"refine_{tag}.onnx")
    torch.onnx.export(
        wrapper,
        (dummy_A, dummy_B),
        onnx_path,
        input_names=["A", "B"],
        output_names=["trans", "rot"],
        opset_version=17,
        dynamic_axes=None,
    )
    print(f"Saved: {onnx_path}")
    del model, wrapper
    torch.cuda.empty_cache()


def main():
    export_score()
    export_refine(168, "b168")
    export_refine(1, "b1")

    print("\n" + "=" * 60)
    print("ONNX export done. Now build TRT engines:")
    print("=" * 60)
    print("""
cd /root/catkin_ws/src/fdpose

/usr/src/tensorrt/bin/trtexec --onnx=assets/onnx/score_b168.onnx   --saveEngine=assets/engines/test_score_f16_b168.eng   --fp16
/usr/src/tensorrt/bin/trtexec --onnx=assets/onnx/refine_b168.onnx --saveEngine=assets/engines/test_refine_f16_b168.eng --fp16
/usr/src/tensorrt/bin/trtexec --onnx=assets/onnx/refine_b1.onnx   --saveEngine=assets/engines/test_refine_f16_b1.eng   --fp16
""")


if __name__ == "__main__":
    main()
