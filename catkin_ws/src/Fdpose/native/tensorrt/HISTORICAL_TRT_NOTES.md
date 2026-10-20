# TensorRT 历史排障记录（非当前部署指南）

当前准备、数值验证和固定样例回归请先阅读 [VALIDATION.md](VALIDATION.md)。
下文保留历史排障记录；其中关于版本、文件头和性能的推断不能作为当前两套
Orin 环境的验收依据。当前扩展加载失败抛出异常，由预测器捕获后回退。

## 问题现象

FDPose 服务启动时，`fdpose_server.log` 出现 3 次错误：

```
loading trt engine
1: [runtime.cpp::parsePlan::314] Error Code 1: Serialization
   (Serialization assertion plan->header.magicTag == rt::kPLAN_MAGIC_TAG failed.)
```

每次启动 FDPose 服务都会出现，导致 3 个 TRT 引擎全部加载失败，自动回退到 PyTorch 推理（速度慢 3-5 倍）。

## 根本原因

### 环境对比

| 项目 | 引擎文件（旧） | 容器环境（当前） |
|------|---------------|-----------------|
| 编译时间 | 2024-11-12（别的机器） | — |
| TensorRT 版本 | 8.5 或更早（推测） | 8.6.2 |
| 文件格式 | `ptrt` 裸 plan 格式 | `ftrt` 容器格式 |
| GPU 架构 | sm_70 (Volta) | sm_87 (Orin) |
| 典型 GPU | V100, GTX 1180 | Jetson Orin |
| 平台 | 未知 | aarch64 (ARM64) |

### 文件格式差异（根因分析）

通过对比新旧引擎文件头部（hex dump），发现 **TRT 8.6 改变了引擎文件格式**：

**旧引擎（TRT 8.5 或更早）—— 裸 plan 格式：**
```
0x00: 70 74 72 74  → "ptrt" (直接是 plan，无外层容器)
0x08: e8 00 00 00  → plan 数据
0x10: 7e 45 ed 01  → 时间戳/哈希
0x18: 02 80 01 00  → plan 数据（无版本信息）
```

**新引擎（TRT 8.6.2）—— ftrt 容器格式：**
```
0x00: 66 74 72 74  → "ftrt" (文件容器魔数)
0x08: 01 00 00 00  → 容器版本 = 1
0x10: 4c fe f5 01  → 时间戳/哈希
0x18: 08 06 02 03  → TRT 版本 = 8.6.2.3  ← 版本号嵌在这里
0x20: 4e 47 4e 45  → "NGNE" 引擎段标记
0x30: 70 74 72 74  → "ptrt" (真正的 plan 在容器内部)
```

### 错误发生机制

```
TRT 8.6.2 的 deserializeCudaEngine 读取文件:
  1. 期望开头是 "ftrt" (容器魔数)
  2. 实际读到的是 "ptrt" (旧版裸 plan 魔数)
  3. 魔数不匹配 → magicTag == kPLAN_MAGIC_TAG 校验失败
  4. 反序列化失败 → loadEngineModel 返回 False
  5. 代码自动回退到 PyTorch 推理
```

### 为什么不兼容

TRT 8.6 引入了 `ftrt` 容器格式，在 plan 外面包了一层容器，里面嵌入了版本号（如 `08 06 02 03` = 8.6.2.3）。旧引擎是 TRT 8.6 之前编译的，直接用 `ptrt` 裸格式，没有外层容器。

```
旧机器 (TRT 8.5-)          当前容器 (TRT 8.6.2)
┌──────────────┐           ┌──────────────┐
│ ptrt (裸plan) │  ──→  ✗  │ 期望 ftrt 容器 │
│              │           │              │
│ 无版本信息     │           │ 有版本 8.6.2.3 │
└──────────────┘           └──────────────┘
     编译于 2024-11            运行于 2026-07
```

TensorRT 引擎文件**不向后兼容**，大版本（8.5→8.6）文件格式变了就必须重新编译。这也是 NVIDIA 的官方建议：**换 TRT 版本必须重建引擎**。

### GPU 架构也不匹配（双重不兼容）

除了 TRT 版本不同，**GPU 架构也不一样**。通过搜索引擎文件内的 CUDA 标记发现：

```bash
# 旧引擎文件中的 CUDA 算力标记
strings eng_backup_*/test_score_f16_b168.eng | grep 'cuda_sm'
# 输出: $__cuda_sm70  ← 编译目标是 Volta (sm_70)

# 新引擎文件中的 CUDA 算力标记
strings test_score_f16_b168.eng | grep 'target sm'
# 输出: .target sm_87   ← 编译目标是 Orin (sm_87)
```

| | 旧引擎 | 新引擎（当前） |
|---|---|---|
| CUDA 算力标记 | `$__cuda_sm70` | `.target sm_87` |
| GPU 架构 | sm_70 (Volta) | sm_87 (Orin) |
| 典型 GPU | V100, GTX 1180 | Jetson Orin |

所以旧引擎是在 **Volta GPU (sm_70)** 的机器上用 **TRT 8.5 或更早** 编译的，当前环境是 **Orin GPU (sm_87) + TRT 8.6.2**，存在**双重不兼容**：
1. TRT 版本不同（8.5 → 8.6.2），文件格式变了
2. GPU 架构不同（sm_70 → sm_87），编译的 CUDA kernel 不通用

### 原来的 TRT 版本能否查看

**不能精确查看**，原因：

| 方法 | 能否看到 | 说明 |
|------|---------|------|
| 旧 `.eng` 文件头部 | ❌ | `ptrt` 格式不嵌入版本号 |
| 旧 `.eng` 文件内字符串 | ❌ | 无版本号字符串 |
| 旧 `.eng` 文件 CUDA 标记 | ✅ 部分 | 只能看到 sm_70（GPU 架构），看不到 TRT 版本 |
| CMakeCache.txt | ❌ | 当前容器编译 pyTRTEng.so 的记录，不是原始引擎的 |
| 编译文档/日志 | ❌ | 容器里没有原始编译记录 |

只能推断：
- **TRT 版本**：8.5 或更早（因为没有 `ftrt` 容器格式，该格式是 8.6 才引入的）
- **GPU 架构**：sm_70 Volta（从引擎文件内的 `$__cuda_sm70` 标记得知）
- **编译时间**：2024-11-12（文件时间戳）

### 如何查看当前 TRT 版本

```bash
# 方法1: dpkg（最直接）
dpkg -l | grep libnvinfer-bin
# 输出: ii  libnvinfer-bin  8.6.2.3-1+cuda12.2  arm64  TensorRT binaries

# 方法2: 库文件版本
ls -la /usr/lib/aarch64-linux-gnu/libnvinfer.so*
# 输出: libnvinfer.so -> libnvinfer.so.8.6.2

# 方法3: trtexec 输出第一行（任意命令都带版本号）
/usr/src/tensorrt/bin/trtexec --help 2>&1 | head -1
# 输出: &&&& RUNNING TensorRT.trtexec [TensorRT v8602]
```

### 如何查看引擎文件版本

```bash
# 查看引擎文件头部魔数
xxd -l 4 xxx.eng
# "ftrt" = TRT 8.6+ 容器格式
# "ptrt" = TRT 8.5- 裸 plan 格式

# TRT 8.6+ 引擎在 offset 0x18 嵌入了版本号
xxd -s 0x18 -l 4 xxx.eng
# 08 06 02 03 = TRT 8.6.2.3
```

## 涉及的 3 个引擎

| 引擎文件 | 用途 | 输入 shape | 输出 |
|----------|------|-----------|------|
| `test_score_f16_b168.eng` | 评分网络 | A,B: (168, 6, 160, 160) | score_logit: (1, 168) |
| `test_refine_f16_b168.eng` | 位姿细化网络 (batch 168) | A,B: (168, 6, 160, 160) | trans: (168, 3), rot: (168, 3) |
| `test_refine_f16_b1.eng` | 位姿细化网络 (batch 1) | A,B: (1, 6, 160, 160) | trans: (1, 3), rot: (1, 3) |

- `f16` = FP16 半精度
- `b168` / `b1` = batch size 168 / 1
- `c_in = 6` = RGB(3) + XYZ(3)，不使用 normal

## 解决方案

在目标 Orin 当前环境中执行统一准备工具：

```bash
cd /root/catkin_ws/src/fdpose
conda activate fdpose
python tools/build/prepare_tensorrt.py
```

该工具将仓库已有的 `assets/onnx/*.onnx` 作为输入，并自动完成：

1. 检测当前 TensorRT、CUDA、GPU、Python 和 PyTorch；
2. 针对当前环境编译 `native/tensorrt/build/pyTRTEng*.so`；
3. 使用当前 `trtexec` 生成 `assets/engines/*.eng`；
4. 使用刚编译的扩展执行三个 Engine，并与同权重 PyTorch FP32 做数值比较；
5. 写入 `assets/engines/manifest.json`，供下次判断是否需要重建。

环境和模型未变化时命令直接返回，不会重复编译。`--check` 只检查，
`--force` 强制重建。兼容脚本 `rebuild_trt_engines.sh` 现在只是调用该工具的
`--force` 包装器。

只有模型权重或网络结构变化、需要更新 ONNX 时，才单独执行：

```bash
python native/tensorrt/export_onnx.py
python tools/build/prepare_tensorrt.py
```

## 验证方法

### 1. 检查 fdpose_server.log（stdout）

```bash
cat /root/catkin_ws/src/fdpose/runtime/logs/fdpose_server.log | grep -i "trt\|engine\|fallback"
```

| 输出 | 含义 |
|------|------|
| `loading trt engine` 后无 `magicTag failed` | ✅ 引擎加载成功 |
| 无 `fallback to PyTorch` | ✅ 未回退 |
| `output size must be multiple of 128` | ✅ TRT 推理运行中 |

### 2. 检查 ROS 日志（logging.info 输出）

```bash
cat /root/.ros/log/pose_estimator_track_node_*.log | grep -i "tensorrt\|engine"
```

成功时显示：
```
[root][INFO]: Using TensorRT score engine: .../test_score_f16_b168.eng
[root][INFO]: Using TensorRT refine engines
```

### 3. 日志通道说明

| 消息来源 | 输出通道 | 出现位置 |
|----------|----------|----------|
| C++ `std::cout`（如 `loading trt engine`） | stdout | `fdpose_server.log` |
| Python `logging.info`（如 `Using TensorRT`） | rospy 日志处理器 | ROS 日志文件 |
| Python `logging.warning`（如 `fallback`） | rospy 日志处理器 | ROS 日志文件 |

`fdpose_server.log` 里看不到 `Using TensorRT` 是正常的，因为 `logging.info` 不走 stdout。

## 代码容错机制

`predict_score.py` 和 `predict_pose_refine.py` 都有自动回退逻辑：

```python
self.use_trt = False
try:
    self.trt_model = pyTRTEng.TensorRTEngine(2, 1)
    if self.trt_model.loadEngineModel(eng_path):
        self.use_trt = True
        logging.info(f"Using TensorRT score engine: {eng_path}")
    else:
        logging.warning("TensorRT score engine load failed, fallback to PyTorch")
except Exception as exc:
    logging.warning(f"TensorRT score engine unavailable ({exc}), fallback to PyTorch")
```

推理时根据 `use_trt` 选择路径：

```python
if self.use_trt:
    output_ = torch.zeros((1, A.shape[0]), dtype=torch.float32).cuda()
    self.trt_model.inferEngineModel([A, B], [output_])
    output = {"score_logit": output_}
else:
    with torch.cuda.amp.autocast(enabled=self.amp):
        output = self.model(A, B, L=len(A))
```

所以即使 TRT 加载失败，FDPose 仍能正常运行（PyTorch 回退），只是速度慢。

## 性能对比

| 模式 | 典型推理时间 | 说明 |
|------|------------|------|
| PyTorch FP32 | ~100ms | 最慢 |
| PyTorch + AMP FP16 | ~50ms | 回退模式 |
| **TensorRT FP16** | **~10-20ms** | 最快（当前） |

## 何时需要重新编译

以下情况需要重新运行 `prepare_tensorrt.py`：

- 升级 TensorRT 版本（如 8.6 → 10.x）
- 更换 GPU 型号（如 Orin → 其他）
- 更换平台（如 aarch64 → x86）
- 更新了 PyTorch 模型权重

## 文件清单

```
native/tensorrt/
├── export_onnx.py              # 模型变化时重新导出 ONNX
├── rebuild_trt_engines.sh      # prepare_tensorrt.py --force 兼容包装
├── CMakeLists.txt              # CMake 构建文件
├── include/                    # C++ 头文件
├── src/                        # C++ 源文件（含 pybind）
├── plugins/                    # 自定义 TRT 插件（layerNorm）
└── build/                      # 当前环境编译的 pyTRTEng.so

assets/onnx/                    # 跨环境输入模型
assets/engines/                 # 当前设备 Engine 和 manifest.json
tools/build/prepare_tensorrt.py # 推荐的一键准备入口
```
