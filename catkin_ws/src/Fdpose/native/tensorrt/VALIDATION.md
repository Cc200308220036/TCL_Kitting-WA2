# TensorRT 准备与回归验证

代码已提供以下检查，实际 GPU 验收需要在目标 Orin 上执行。不能根据本地单元测试
通过就认定 JetPack 5/6 或 TensorRT 8/10 均已兼容。

## 1. 准备与网络数值检查

在项目根目录、FDPose Conda 环境下，先停止使用本项目 Engine 的服务：

```bash
python tools/build/prepare_tensorrt.py
python tools/build/prepare_tensorrt.py --check
```

输入包括三份 ONNX、C++/CUDA 源码、当前 Python/PyTorch/CUDA/TensorRT 环境，
以及用于参考输出的原始 checkpoint 和配置。工具不自动安装依赖，不自动重新导出 ONNX。
权重变更时应先用 `native/tensorrt/export_onnx.py` 更新 ONNX。

三份 Engine 的对外张量都必须是 CUDA、连续布局、float32：

| Engine | A、B 的形状 | 输出 |
|---|---|---|
| score | `(168, 6, 160, 160)` | `score_logit: (1, 168)` |
| refine_b168 | `(168, 6, 160, 160)` | `trans, rot: (168, 3)` |
| refine_b1 | `(1, 6, 160, 160)` | `trans, rot: (1, 3)` |

FP16 是 Engine 内部计算精度，不改变这些 I/O 的 float32 约定。
当前封装仅支持上述固定形状模型，不支持任意 batch 或动态 shape。
Refine 的输出是网络原始更新量，不是最终位姿。

准备工具使用固定种子的 RGB+XYZ 合成输入，依次验证三个模型，在默认和非默认流上
各推理三次，与同权重 PyTorch FP32 输出比较形状、有限值和逐元素误差。
`manifest.json` 内的 `validation` 保存误差及耗时。阈值默认 `atol=0.02, rtol=0.02`，
属于初始部署检查标准，尚未在用户两套 Orin 环境标定。近似相等不能代替真实数据精度评估。

若比较失败，终端会输出误差报告，产物不安装。单独诊断可执行：

```bash
python native/tensorrt/validate_tensorrt.py --report assets/engines/validation/network.json
```

该命令支持 `--atol` / `--rtol` 用于误差调查；修改它们不会改变 prepare 的验收阈值。
不要仅为通过测试而放宽阈值。优先检查 ONNX、权重、模型配置及实际加载的扩展。

## 2. 固定真实样例的完整位姿对比

```bash
python native/tensorrt/compare_pose_backends.py --sample-dir assets/warmup/wp_2
```

样例目录需提供 `warmup.yaml` 引用的彩色图、深度图、Mask、相机内参文本。
Mesh 默认由样例 Label 在注册表查找，也可通过 `--mesh /绝对路径/model.ply` 指定。
工具不改变 ROS 配置，两个独立子进程分别运行 PyTorch 和 TensorRT。
使用完整图像和样例内参，不启用 Service ROI，因此这是核心 register 回归，
不是服务时延基准。PyTorch 沿用默认 AMP，与第一节的 FP32 网络参考不同。

输出默认在 `assets/engines/validation/pose_*/`，包含各轮相机系 Pose、
相机系包围框/坐标轴可视化、初始化和各轮耗时、后端间平移与旋转差异。
自定义 `--output-dir` 必须指定尚不存在的目录，以免覆盖旧实验。
首轮时间不是系统冷启动时间：模型初始化单独统计，操作系统缓存和 GPU 状态仍可能影响结果。
对称物体的旋转差异应结合模型对称性人工评估，脚本不自动判定精度合格。

## 3. 异常和流顺序回归

```bash
python -B -m unittest tests.integration.test_tensorrt_engine -v
```

检查错误张量数量/形状/类型/设备/布局会抛出异常，加载失败不会沿用旧 context，
并检查在非默认 PyTorch CUDA 流中产生的输入能得到正确结果。
无 PyTorch/CUDA 时测试会跳过；跳过不代表 GPU 验证通过。

封装使用 PyTorch 当前 CUDA 流，并检查绑定、提交和同步结果，保留原来的同步返回接口。
这遵循 [PyTorch 当前流语义](https://docs.pytorch.org/cppdocs/api/cuda/utilities.html)
和 [TensorRT enqueueV3 接口约定](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/classnvinfer1_1_1_i_execution_context.html)。

## 4. 安装与复用边界

准备脚本检测工具版本、相关动态库内容、CUDA 编译器版本及 GPU/PyTorch 环境，
并对 ONNX、模型参考权重、相关源码和生成产物计算校验值。
环境和输入匹配时不会重编译，但仍运行实际数值验证；`--check` 不安装文件。

构建在独立临时目录执行，安装前保留旧文件备份。可捕获的安装异常触发恢复；
恢复失败时保留 `.fdpose-trt-backup-*/restore.json` 及剩余备份，并打印位置。
不保证断电或强制终止后的整套文件原子性，因此应在服务停止时准备 Engine。
项目根目录 `.fdpose-tensorrt.lock` 是构建互斥锁文件；进程退出后锁自动释放，文件保留。

本地可运行的回归：

```bash
PYTHONPATH=src python -B -m unittest discover -s tests/unit -v
```
