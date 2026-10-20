# Pipeline 使用说明（可选）

[返回 FDPose README](../README.md)

当前公共 [FdposePipeline](../src/fdpose/pipeline/pipeline.py) 对接新版
`naviai_yolo.ros.seg_service_client.SegServiceClient`（本次核对版本：`naviai_yolo_0909`）。
它只编排分割、可选深度补全、FDPose 位姿估计，不管理相机、机器人、任务或 Mesh 资产复制。

**已有 RGB-D 和 Mask 时直接使用 FdposeClient，不需要 Pipeline。**
旧在线 CLI、旧分割适配和一键启停脚本已清理；新版只保留公共编排器和一个薄的本地测试入口。
后台服务部署另由 [新的统一启停入口](SERVICE_MANAGER_PYTHON.md) 管理，不属于 Pipeline 类的职责。
同一容器内可先使用 [纯 Bash 版本](SERVICE_MANAGER_BASH.md)；原 Python 管理版保留供后续对比测试。

## 1. 运行前提与启动

主程序所在 Python 环境需要能够导入 `fdpose`、`naviai_yolo`、`rospy`、`cv_bridge`、
`naviai_manip_srvs` 和新版 `navi_types/InstSeg`。先构建并 source 对应 Catkin 工作空间；
安装环境及 FDPose 依赖见 [README](../README.md)。
若通过源码导入，则将 FDPose 与新版 YOLO 各自的 `src` 目录加入 PYTHONPATH，
不要再从旧 YOLO 的 `scripts/` 动态导入 Client。

先启动 ROS master，再在各自配置好依赖的终端启动服务：

也可以先填写 `config/fdpose_services.yaml`，再执行 `bash scripts/manage_services.sh start`，
按配置在当前环境或已有 Docker 容器中后台启动；停止用同一脚本的 `stop`，详见 [部署说明](SERVICE_MANAGER_PYTHON.md)。
以下是手动前台启动方式，两种方式选择一种即可：

```bash
# 新版 YOLO 所在工作空间已构建并 source
rosrun naviai_manip_seg seg_service_server.py

# 另一个终端，在 FDPose 项目根目录
python scripts/pose_estimator_track_node_roi_v2.py
```

默认服务：

| 模块 | Service |
|---|---|
| 新版 YOLO 分割 | `/instance_segmentation_service` |
| FDPose Mesh | `/fdpose_v2/load_mesh` |
| FDPose Pose | `/fdpose_v2/pose_estimation` |

两个服务必须在同一 ROS 图中可见。YOLO 的模型配置由 YOLO 服务管理，
FDPose 的内参和推理配置由 FDPose 服务管理；Pipeline 不加载它们。
`config/fdpose_pipeline.yaml` 已改为新版连接配置，仅由 `from_ros_services()` 读取。

## 2. 主程序注入 Client（推荐）

适用条件：主程序已经创建 Client，需要复用实例、分别设置各模块超时，
或希望在 Pipeline 外直接调用 FDPose 的 Mesh 管理接口。测试时也可以注入同接口替身。
构造函数只保存传入实例，不负责连接服务；传入的 Client 仍必须符合本文的固定调用协议。

主程序初始化 ROS 一次，创建各 Client，再传入 Pipeline：

```python
import rospy
from fdpose.ros.client import FdposeClient
from fdpose.pipeline import FdposePipeline
from naviai_yolo.ros.seg_service_client import SegServiceClient

rospy.init_node("vision_app")  # 已有节点时不要重复初始化

fdpose = FdposeClient()  # 默认 V2
segmentation = SegServiceClient(
    service_name="/instance_segmentation_service",
    wait_timeout_sec=10.0,
)
pipeline = FdposePipeline(
    fdpose_client=fdpose,
    segmentation_client=segmentation,
    depth_refine_client=None,  # 需要时传入外部模块的深度 Client 实例
)

# color_bgr 和 depth 由主程序读取文件或获取相机帧
poses = pipeline.process_rgbd(
    color=color_bgr,
    depth=depth,
    labels=["wp_2"],
)
for result in poses:
    print(result.label, result.T_object)  # 默认 T_camera_object
```

Pipeline 不会主动复制 Mesh 或写注册表；默认只按 labels 请求服务加载已注册 Mesh。
无检测项或 FDPose 返回空结果时得到 `[]`；输入错误、Mesh 加载失败及外部服务异常向上传播，
由主程序决定重试还是停止，不把调用失败伪装为“未检出”。

### 本地 RGB-D 测试

没有本地输入图时，可先运行 `python tools/capture_one_pair.py` 从相机抓拍一对 RGB-D，
默认保存到 `data/samples/captures/`；话题、同步和深度单位说明见 [README 第 2.4 节](../README.md)。
该工具只采样，不需要提前启动 YOLO/FDPose，也不会生成 Mask 或修改标定。

服务启动后，上述主程序中的图像可这样获取（无需相机，也无需预先提供 Mask）：

```python
import cv2
color_bgr = cv2.imread("data/samples/current/rgb/0.png", cv2.IMREAD_COLOR)
depth = cv2.imread("data/samples/current/depth/0.png", cv2.IMREAD_UNCHANGED)
if color_bgr is None or depth is None:
    raise ValueError("无法读取测试 RGB-D")
poses = pipeline.process_rgbd(color_bgr, depth, labels=["wp_2"])
```

测试图片中的 label 应与 YOLO 类别及 Mesh 注册项一致。如果已经有本地 Mask 且不启动 YOLO，
请使用 README 中的 `tools/clients/test_fdpose_client.py`，不经过 Pipeline。

## 3. 输入、结果和坐标约定

| 输入 | 约定 |
|---|---|
| color | 非空 H×W×3 uint8；默认 BGR，RGB 必须显式指定 `color_order="rgb"` |
| depth | H×W，uint16 表示毫米，float32 表示米；不根据数值猜单位 |
| labels | 非空标签列表，也支持单个字符串；去除首尾空格和重复项，不剥离任何后缀 |
| 分割 Mask | 必须与原图 H×W 一致，使用现有 `DetectionInput.from_yolo_item()` 转换 |
| 补全后的 depth | 与原图同尺寸，仍遵循 uint16 毫米 / float32 米的约定 |

RGB-D 必须已对齐，FDPose 服务的相机内参必须与输入图像的实际分辨率和成像几何对应。
Pipeline 不缩放图像、不改内参、不做深度配准；检查尺寸相同并不能证明两张图已经对齐。

新版 YOLO 的调用和返回为：

```python
response = segmentation.segment(color=color_bgr, depth=depth, labels=labels)
# response.have_objs / response.items；不包含输入 RGB-D
```

Pipeline 始终保留本次原图与原深度，只在启用补全时将 Pose 输入深度替换为补全结果。
YOLO 可省略深度，但 Pipeline 需要传入深度数组，不能将 YOLO 的零深度占位当作真实深度。
允许将有空洞的深度交给补全 Client；进入 Pose 前，最终深度必须至少含有有限正值。
物体 Mask 内深度是否足够，仍由 FDPose Service 的门禁判断。

返回 `PoseResult` 列表，矩阵字段为 `T_object`。
默认在相机坐标系；传入 `T_target_camera` 时由 FDPose Client 计算
`T_target_object = T_target_camera @ T_camera_object`。
这只改变返回矩阵，不改变 Service 在相机图像上的可视化。

## 4. 串行、并行、外部相机

- `process_rgbd(...)`：Mesh 加载 → 分割 → 可选深度补全 → Pose。
- `process_rgbd_parallel(...)`：Mesh 加载 → 分割与深度补全并行 → 等待两个结果 → Pose。
- `process_from_camera(frame_provider, labels, **process_kwargs)`：调用外部函数获取一帧，再走串行入口。

```python
poses = pipeline.process_rgbd_parallel(color_bgr, depth, ["wp_2"])

poses = pipeline.process_from_camera(
    frame_provider=camera.get_rgbd,  # 返回 (color, depth)，相机由主程序创建
    labels=["wp_2"],
)
```

没有深度 Client 时，并行入口自动使用原深度；串行入口默认根据是否注入深度 Client 决定是否补全，
也可传 `run_depth_refine=False` 跳过。显式开启但未注入 Client 会在服务调用前报错。

深度 Client 的接口为 `refine(color_bgr, depth)`，返回深度数组。
两个外部 Client 不能原地修改输入帧；同一 Pipeline 实例的多次请求应由主程序串行调用。
并行只针对这两个独立 Client，不意味着多个 FDPose 推理并行，也不保证共享 GPU 时更快。

串行无检测项时不调用深度补全；并行时补全可能已经执行，即使未检出也会等待其结束。
外部调用异常不会被吞掉。线程池不能强行终止已经发出的 ROS Service 请求，
服务就绪等待超时也不是推理请求超时；当前没有整条流程的硬超时或取消功能。

相机取帧后若需要并行，主程序直接取帧并调用 `process_rgbd_parallel()` 即可。
新版 YOLO Action 自带相机获取/等待语义，本轮没有接入此单帧流程。

## 5. Mesh 与可选坐标变换

默认 `load_mesh=True` 按标签查服务端注册表。外部路径不在注册表时应显式预加载：

```python
fdpose.load_mesh("my_part", mesh_path="/path/to/model.ply")
poses = pipeline.process_rgbd(
    color_bgr, depth, labels=["my_part"],
    load_mesh=False,
    T_target_camera=T_base_camera,  # 可选；由主程序获取
)
```

服务重启或 Mesh 卸载后需重新加载。Pipeline 不保留临时路径、不自动清空或覆盖已有 Mesh。
使用 `load_mesh=False` 时，主程序负责确保所需标签都已加载。

## 6. 按服务名创建 Client

适用条件：尚未创建 Client，希望只指定服务名即可快速组装流程，
并且接受统一的服务就绪等待时间。需要复用已有实例或外部构造接口不同时，使用第 2 节的实例注入。

调用前主程序仍需初始化 ROS。此方法创建新的 Client，可能等待服务就绪，
但不会启动服务或创建相机。新版 YOLO 仅在调用此工厂时导入：

```python
# 默认读取 FDPose 项目 config/fdpose_pipeline.yaml，与启动时的工作目录无关
pipeline = FdposePipeline.from_ros_services()

# 指定自己的连接配置；显式参数可覆盖 YAML 中的值
pipeline = FdposePipeline.from_ros_services(
    config_path="/path/to/pipeline.yaml",
    timeout=10.0,
)

# 完全不读取文件，按显式参数和内置默认值创建
pipeline = FdposePipeline.from_ros_services(
    # FDPose
    mesh_service="/fdpose_v2/load_mesh",
    pose_service="/fdpose_v2/pose_estimation",
    # YOLO
    segmentation_service="/instance_segmentation_service",
    # 深度补全默认不启用，省略对应参数组
    # 通用等待参数
    config_path=None,
    timeout=20.0,
)
```

参数按与构造函数一致的顺序分组：

| 分组 | 参数顺序 |
|---|---|
| FDPose | `mesh_service`、`pose_service` |
| YOLO | `segmentation_service`、`segmentation_client_factory`、`segmentation_client_kwargs` |
| 可选深度补全 | `depth_refine_service`、`depth_refine_client_factory`、`depth_refine_client_kwargs` |
| 通用参数 | `config_path`、`timeout` |

所有参数均为关键字参数。服务名与 timeout 省略或传 None 时使用配置值，
**优先级为显式参数 > YAML > 内置默认值**；`timeout=None` 不表示无限等待。
两种方式最终得到相同的 Pipeline，处理能力、串并行逻辑和结果格式完全相同，区别仅在于谁创建 Client。
内部创建的 FDPose Client 仍可通过 `pipeline.fdpose` 访问，用于显式加载外部 Mesh 等操作。

- 分割工厂默认是新版 `SegServiceClient`，构造参数为 `service_name / wait_timeout_sec`。
- 可用 `segmentation_client_factory` 替换，但必须满足上述构造和 `segment()` 接口。
- 可选深度工厂 `depth_refine_client_factory` 与 `depth_refine_service` 必须一起传；
  深度工厂沿用 `service_name / timeout_sec` 参数约定。
- 两者的额外构造参数可通过对应 `*_client_kwargs` 提供，不可覆盖服务名和等待超时。
- `timeout` 是服务就绪等待时间，不是整次推理时限。外部构造接口不同，直接注入实例更简单。

### YAML 保存什么

[默认配置](../config/fdpose_pipeline.yaml) 中各参数已有中文注释：

```yaml
mesh_service: /fdpose_v2/load_mesh
pose_service: /fdpose_v2/pose_estimation
segmentation_service: /instance_segmentation_service
timeout: 20.0
```

默认从固定模块导入 `fdpose.ros.client.FdposeClient` 和
`naviai_yolo.ros.seg_service_client.SegServiceClient`，不在 YAML 中配置 Python 文件或类路径。
深度补全的独立 Client 导入路径尚未确定，继续通过实例注入或显式工厂及服务名接入；
本轮不在配置中假设其实现，也不将旧的深度服务协议搬入公共 Pipeline。

labels、RGB-D、TF 和是否加载 Mesh 是每次处理的输入，不写入连接 YAML；
相机、模型、调试输出配置仍由各自模块管理。实例注入方式不读取此文件。

旧 YAML 的 `services/seg/paths/debug` 等结构不再支持。未知字段、错误格式、
空服务名和无效 timeout 会报错；文件不存在也会报错，不静默回退到其他服务。
配置在创建 Pipeline 时读取一次；修改配置后需重新创建实例。要完全跳过文件，请传 `config_path=None`。

## 7. 本地测试与清理边界

启动第 1 节的两个服务后，在 FDPose 根目录运行：

```bash
python tools/clients/test_pipeline.py \
  --rgb data/samples/current/rgb/0.png \
  --depth data/samples/current/depth/0.png \
  --labels wp_2
```

可用 `--config /path/to/pipeline.yaml` 或 `--timeout 10` 指定连接配置。
此入口只负责读取本地彩色/深度图、创建 Pipeline 和打印相机坐标系矩阵，
不另外实现 Mesh/Pose 请求或分割协议，也不保存客户端中间文件。
它使用注册表 Mesh，不启用深度补全；外部 Mesh 或深度补全场景使用前文的主程序集成方式。
不需要 YOLO、已有 Mask 时，仍使用 `tools/clients/test_fdpose_client.py`。

正式回归测试：[Pipeline 测试](../tests/unit/test_pipeline.py) 和
[CLI 测试](../tests/unit/test_pipeline_cli.py)。

```bash
PYTHONPATH=src python -B -m unittest discover -s tests/unit -p 'test_pipeline*.py' -v
```

测试使用 NumPy 数组、实际 FDPose 检测项转换和模拟 ROS 边界，覆盖新版响应、
串并行调用、深度单位、颜色顺序、Mask 尺寸、异常、工厂参数及 YAML 优先级/校验；
需要 NumPy 和 PyYAML，ROS 边界使用替身；
不能代替 Orin 上真实 YOLO + FDPose 服务联调。

`pipeline/` 现仅保留 `__init__.py` 和 `pipeline.py`。
旧 `pipeline_client.py`、`segmentation_client.py` 及其三个兼容入口已删除，
旧一键启动/停止脚本也已退役。旧命令不再可用，请改用上述本地入口或主程序集成。
相机订阅和 YOLO Action 调用交由主程序/各模块负责，不在此处维护第二套流程。
服务进程可使用独立的 `scripts/manage_services.sh` 管理，见 [服务一键启停](SERVICE_MANAGER_PYTHON.md)。
清理文件不会停止已经运行的旧服务；切换前请自行确认并停止旧部署进程。
