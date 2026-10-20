# FDPose project structure

本项目核心负责 Mesh 加载、物体位姿估计及其调用接口。任务配置、机器人动作、专用示教和关系持久化属于调用方。
独立使用见 [FDPose README](../README.md)；附加视觉编排见 [Pipeline 文档](PIPELINE.md)。

```text
fdpose/
├── config/              # Server、Pipeline、服务部署、相机内参及兼容外参配置
├── scripts/             # ROS/CLI 入口；manage_services_bash.sh 纯 Bash 启停，原 Python 管理入口保留
├── src/fdpose/
│   ├── core/            # FoundationPose 核心
│   ├── learning/        # 网络及输入处理
│   ├── ros/             # FDPose Server、公共 Client 和服务工具
│   ├── pipeline/        # 外部分割、深度补全、FDPose 编排及在线测试
│   │   ├── __init__.py  # 导出 FdposePipeline
│   │   └── pipeline.py  # 新版 YOLO Service + FDPose V2 编排、连接配置读取
│   ├── transforms.py    # 通用相对位姿计算，仅依赖 NumPy
│   └── bundlesdf/       # 上游组件；mycuda 仍被 core 引用
├── native/              # C++/CUDA/TensorRT 源码、验证工具和本机构建
├── assets/
│   ├── meshes/
│   │   ├── registry.yaml
│   │   ├── objects/     # 已注册 Mesh
│   │   └── unregistered/ # 候选资产，可按路径加载
│   ├── checkpoints/
│   ├── onnx/
│   ├── engines/
│   └── warmup/
├── tools/
│   ├── capture_one_pair.py # 独立 ROS RGB-D 抓拍工具，仅写入一对彩色/深度 PNG，不调用模型
│   ├── manage_services.py # 启停实现：环境激活、Docker 执行、PID 身份和日志管理
│   ├── build/
│   ├── clients/         # test_fdpose_client.py（已有 Mask）、test_pipeline.py（调用 YOLO）
│   └── demos/
├── tests/               # 回归测试及视觉样例
├── data/                # 本地 RGB-D、Mask 样例
├── runtime/             # 日志、PID、调试输出
├── CMakeLists.txt
└── package.xml
```

公共接口：

- `fdpose.ros.client.FdposeClient`：Mesh 管理、Pose 请求、可选坐标变换。
- `fdpose.pipeline.FdposePipeline`：组合新版 YOLO SegServiceClient、可选深度补全 Client 与 FDPose V2。
- `fdpose.transforms.compute_relative_pose`：计算物体到目标的相对位姿。
- `fdpose.transforms.apply_relative_pose`：将该关系应用到当前物体位姿。

Label 仅表示物体身份，不编码稳定性或对称性；不再定义或剥离特殊后缀。
Service 和 Client 使用完整标签匹配 Mesh，具体约定见 [README 的输入输出说明](../README.md#label-约定)。

新版本地入口 `tools/clients/test_pipeline.py` 只读取 RGB-D 并调用公共 `FdposePipeline`。
`config/fdpose_pipeline.yaml` 仅保存服务名及等待时间，默认由 `from_ros_services()` 读取；
外部直接注入 Client 时不读配置。旧在线 CLI、旧分割协议适配及旧一键启停脚本已删除。
相机和 Action 交由主程序及各模块管理。
后台服务进程可由 `scripts/manage_services.sh start/stop/status` 管理，
部署信息位于 `config/fdpose_services.yaml`，使用和输出路径见 [服务一键启停](SERVICE_MANAGER_PYTHON.md)。
同一容器固定服务可先使用独立的 `scripts/manage_services_bash.sh`，参数和边界见 [纯 Bash 版](SERVICE_MANAGER_BASH.md)。
该 Bash 入口默认读取 `config/pipeline.conf`，原内置默认值保留；不读取 Python 管理版的 YAML。
V2 不再继承或导入 V1 服务类，相关代码职责如下：

- `ros/server_config.py`：V2 的只读配置数据类、YAML/ROS 加载与严格类型、范围校验。
- `ros/server_utils.py`：V1/V2 共用的无状态工具，不管理网络或服务生命周期。
- `ros/server_v2.py`：V2 自己的初始化、预热、Mesh 管理、估计请求和调试文件保存。
- `ros/debug_writer.py`：V2 调试目录、可读权限/原子 JSON、单图/三联图绘制与写入耗时；Service 只收集结果并触发保存。
- `ros/server.py`：V1 服务；仅共用工具转接，原有初始化与请求流程保留。

V1 仍保留独立源码入口；Legacy 服务实现已删除，旧整套流程启动脚本已退役。
`CMakeLists.txt` 安装项暂不修改，因此 `scripts/pose_estimator_node.py` 保留为明确提示并转发到 V2 的薄入口，
不再保留 Legacy 行为。安装配置尚未完整迁移，当前按源码目录方式使用。

正式测试和可复用验证工具保留；一次性排查文件放项目外临时目录。

## 运行入口与配置对应关系

以下命令均在项目根目录、对应依赖环境中执行。脚本开头注明实际实现及配置来源，
部署/样例 YAML 和 Bash 配置开头注明读取方；服务端配置与 Client 连接配置不会自动同步。

| 运行入口 | 实际实现/调用对象 | 配置来源 |
|---|---|---|
| `python scripts/pose_estimator_track_node_roi_v2.py` | `src/fdpose/ros/server_v2.py::main` | `config/fdpose_server_v2.yaml`，ROS 私有参数优先 |
| `python scripts/pose_estimator_node.py` | 同上，兼容旧安装名称 | 同上 |
| `python scripts/pose_estimator_track_node_roi.py` | `src/fdpose/ros/server.py::main`，V1 | ROS 私有参数；不读取 V2 YAML |
| `python tools/clients/test_fdpose_client.py` | 公共 `FdposeClient` → 已启动的 V2 | 命令行参数；Mesh 注册表由服务端读取 |
| `python tools/clients/test_pipeline.py --rgb ... --depth ... --labels ...` | 公共 `FdposePipeline` → YOLO/FDPose 服务 | `config/fdpose_pipeline.yaml` |
| `bash scripts/manage_services_bash.sh start` | Bash 管理器启动配置的服务脚本 | `config/pipeline.conf` |
| `bash scripts/manage_services.sh start` | `tools/manage_services.py` 启动配置的服务脚本 | `config/fdpose_services.yaml` |
| `python tools/capture_one_pair.py` | ROS RGB-D 同步抓拍，不调用模型 | 命令行话题参数，无 YAML |
| `python tools/build/prepare_tensorrt.py` | 构建 TensorRT 扩展/Engine 并运行验证 | ONNX、模型资产和当前环境；不读 ROS YAML |

补充说明：

- `assets/meshes/registry.yaml`：服务端 Label → Mesh 路径表；坏条目记录日志并跳过。
- `assets/warmup/wp_2/warmup.yaml`：V2 预热及 `compare_pose_backends.py` 的参考样例描述。
- `config/camera/intrinsic_data.npz`：服务端加载的相机内参；二进制文件不能直接添加头部注释。
- `assets/checkpoints/*/config.yml`：底层 Score/Refine 模型资产配置，不是部署参数，本次保持原文件不变。
- V2 深度消息固定为毫米，转米不再猜测数值范围；V1 保留旧兼容转换。
- V2 配置严格校验类型/范围；布尔与数字不要加引号，未知 YAML 键会告警。
- `tools/demos/` 仅保留历史数据集评测与离屏渲染工具，不作为当前 Service 联调入口。
- `tests/unit/` 为无 ROS/GPU 的回归；`tests/integration/test_tensorrt_engine.py` 需目标 GPU 验证。

单元回归命令（需 NumPy、OpenCV、PyYAML）：

```bash
PYTHONPATH=src python -B -m unittest discover -s tests/unit -v
```

### 本地样例与输出

`data/samples/current` 保留一组默认测试样例；旧历史目录 `runs/` 与旧对比脚本 `depth.py` 已清理。
新增抓拍工具默认写入
`data/samples/captures/<时间戳>_<标识>/rgb/0.png` 与 `depth/0.png`。
服务推理过程的原始数据和可视化仍写入 `runtime/debug`，不会自动归档到 samples。
