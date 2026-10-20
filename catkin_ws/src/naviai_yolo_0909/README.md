# naviai_yolo：实例分割使用与开发说明

本项目将 YOLO 实例分割封装为 ROS Service 和 Action，输出类别、置信度、bbox 和实例 Mask。
RGB-D 接口中的深度不参与 YOLO 推理；项目不负责相机标定、深度配准、三维重建或位姿估计。

| 使用场景 | 入口 | 依赖 |
|---|---|---|
| 已有图像数组或 RGB-D 文件，做一次分割 | `SegServiceClient.segment()` / `segment_files()` | ROS；模型在 Server |
| 从相机持续尝试，直到检出目标或任务结束 | `SegActionClient.send_goal()` | ROS、相机、Server 模型 |
| 已有 Mask，只读取区域和 bbox | `load_mask_file()` | NumPy / OpenCV，不需要 ROS 或模型 |
| 只抓拍一对同步 RGB-D | `capture_one_pair.py` | ROS、相机，不需要模型 |
| 自己的 Python 程序直接推理 | `NaviaiSeg.predict()` + 后处理 | 模型，不需要 ROS |

Service 只提供 YOLO 推理。`segment_files()` 是读取 RGB-D 后请求推理，**不是读取已有 Mask**。

## 阅读导航

- [业务集成快速开始：启动 Server、导入和调用 Client](#client-quickstart)
- [1. 文件架构与职责](#architecture)
- [2. 环境与启动准备](#environment)
- [3. 配置文件与参数](#configuration)
- [4. 图像、尺寸与结果约定](#image-contract)
- [5. Service 使用与方法](#service)
- [6. Action 使用与方法](#action)
- [7. 本地方法与其他工具](#tools)
- [8. Debug 数据保存与回放](#debug)
- [9. 测试、验收与常见问题](#verification)

逐段审核代码请阅读 [Service code review 路线](docs/SERVICE_CODE_REVIEW.md)。本文描述当前实现，不把待优化项当作已提供的能力。

<a id="client-quickstart"></a>

## 业务集成快速开始

**Server 负责配置和启动，Client 在调用方自己的业务代码中实例化并复用。** 不需要修改本项目的 Client 实现，也不从 `scripts/` 导入类。
ROS 包名为 `naviai_manip_seg`（用于 rosrun），Python 包名为 `naviai_yolo`（用于 import）。
以下示例需先完成 [环境准备](#environment)，每个终端 source ROS / catkin 环境，并确保 ROS Master 可用；路径和目标类别需换成实际值。

### 参数分别在哪里设置？

| 参数层级 | 参数 | 设置位置 |
|---|---|---|
| Server 模型与保存 | `seg_model_path`、`conf`、`imgsz`、`debug` | `config/model.yaml`；启动时读取，修改后重启 Server |
| Action 相机 | RGB / 深度话题 | `config/camera.yaml`；Service 不读取它 |
| Service Client 初始化 | `service_name`、`wait_timeout_sec` | `SegServiceClient(...)`；不加载模型 |
| Service 每次调用 | `color` / `color_path`、可选深度、`labels` | `segment(...)` / `segment_files(...)` |
| Action Client 初始化与调用 | `timeout_sec`；目标 `labels` | `SegActionClient(...)`；`send_goal(...)` |

`labels` 是每次请求的类别筛选，不是模型配置。Service 省略或传 `[]` 表示全部类别；Action 必须提供非空类别列表。
Client 不读取 model.yaml，也不能在一次调用中覆盖 Server 的 `conf` / `imgsz`。

### Service：一次分割，接入自己的业务代码

终端 A 启动 Server；默认读取包内 model.yaml，也可以指定自己的完整配置：

```bash
rosrun naviai_manip_seg seg_service_server.py
# 自定义配置时改用下面这条，不要同时启动两个同名 Server：
# rosrun naviai_manip_seg seg_service_server.py _model_config:=/absolute/path/model.yaml
```

下面的代码放在调用方自己的 Python 文件中。初始化节点和创建 Client 只做一次，不要放在逐帧循环里：

```python
import cv2
import rospy
from cv_bridge import CvBridge
from naviai_yolo.ros.seg_service_client import SegServiceClient

rospy.init_node('my_seg_business')  # 已初始化的业务节点不要重复调用。
client = SegServiceClient(
    service_name='instance_segmentation_service',
    wait_timeout_sec=10.0,  # 等待服务上线，不是单次推理请求超时。
)
bridge = CvBridge()

color_bgr = cv2.imread('/absolute/path/color.png', cv2.IMREAD_COLOR)
if color_bgr is None:
    raise ValueError('无法读取彩色图')
response = client.segment(color=color_bgr, depth=None, labels=['wp_2'])

if not response.have_objs:
    rospy.loginfo('请求成功，但未检出指定类别')
for item in response.items:
    foreground = bridge.imgmsg_to_cv2(item.mask, 'mono16') > 0
    print(item.label, item.confidence, list(item.bbox), foreground.shape)
    # 在这里使用 foreground 接自己的业务逻辑；color_bgr 仍由调用方持有。
```

已有相机 SDK 提供的 BGR 数组时，直接替换读取彩色图的部分，无需先保存文件。
`color` 要求非空 H×W×3、BGR uint8；可选 `depth` 要求同 H×W、uint16 毫米或 float32 米。没有深度传 None，Client 补零；无需预先缩放到 640×640。
如果输入本来就是文件，把上面的 `segment(...)` 调用替换为：

```python
response = client.segment_files(
    color_path='/absolute/path/color.png',
    depth_path='/absolute/path/depth.npy',  # 也可以是原值深度 PNG，省略则补零。
    labels=[],                           # 全部类别；也可指定 ['wp_2']。
)
```

两种方法都返回 `InstSegResponse`，只有 `have_objs` 和 `items`；每个 item 含 `label`、`confidence`、`bbox`、`mask`，不返回输入 RGB-D。
输入或通信错误会抛异常，不等同于正常“未检出”。包含异常处理、可直接运行的完整模板见 [examples/seg_service_call.py](examples/seg_service_call.py)，[模板启动参数见 5.2](#service-call-template)，[详细 API 见第 5 节](#service)。

### Action：相机找目标，成功后读取结果

Server 需要模型配置和可用的同步 RGB-D 相机话题：

```bash
rosrun naviai_manip_seg seg_action_server.py
# 自定义配置时追加 _model_config:=/absolute/path/model.yaml
# 以及 _camera_config:=/absolute/path/camera.yaml
```

在另一业务程序中调用；不需要从 Client 传入图像：

```python
import rospy
from cv_bridge import CvBridge
from naviai_yolo.ros.seg_action_client import SegActionClient

rospy.init_node('my_seg_action_business')  # 已初始化则省略。
client = SegActionClient(timeout_sec=15.0)
if client.send_goal(labels=['wp_2']):
    bridge = CvBridge()
    color_bgr = bridge.imgmsg_to_cv2(client.last_color_image, 'bgr8')
    for item in client.last_items:
        foreground = bridge.imgmsg_to_cv2(item.mask, 'mono16') > 0
        print(item.label, item.confidence, list(item.bbox), foreground.shape)
else:
    rospy.logwarn('任务未成功，不把缓存作为成功结果使用')
```

`send_goal()` 阻塞等待并返回 bool，检测数据从 `last_items` / `last_color_image` / `last_depth_image` 读取。多个目标类别中任意一类检出即可成功；同一个 Client 串行复用。
`timeout_sec` 只控制 Client 等待结果的时间，不修改 Server 当前固定的 10 秒任务期限；连接失败等异常也可能传播。
完整模板见 [examples/seg_action_call.py](examples/seg_action_call.py)，具体超时和异常边界见 [第 6 节](#action)。

<a id="architecture"></a>

## 1. 文件架构与职责

仓库目录为 `naviai_yolo_0909`，ROS 包名为 `naviai_manip_seg`，Python 导入包名为 `naviai_yolo`。

```text
naviai_yolo_0909/
├── README.md                       # 完整使用说明（本文）
├── config/
│   ├── model.yaml                  # 模型、推理参数、Server 保存开关
│   └── camera.yaml                 # Action / 抓拍使用的相机话题
├── models/                         # 权重，默认 best_wrc.pt
├── data/                           # 样例、历史数据与源码默认输出
├── src/naviai_yolo/
│   ├── configuration.py            # 配置读取、校验、资源/输出路径
│   ├── segmentation.py             # NaviaiSeg：模型加载和预测适配
│   ├── postprocess.py              # 类别过滤、Mask 恢复/二值化、SegDetection
│   ├── mask_io.py                  # 已有 Mask 文件的本地读取
│   ├── debug_writer.py             # 分析记录、JSON 和可视化写盘
│   ├── ros/
│   │   ├── seg_service_server.py   # 一次请求的一次 YOLO 分割
│   │   ├── seg_service_client.py   # 数组 / 文件两种输入方法
│   │   ├── seg_action_server.py    # 取帧、推理、反馈、取消和任务终态
│   │   ├── seg_action_client.py    # 发送目标、等待结果、缓存反馈
│   │   ├── camera_sub.py           # 严格同步 RGB-D，缓存最新帧
│   │   ├── image_io.py             # Service / 抓拍的解码与校验
│   │   ├── debug_recording.py      # ROS 来源、处理状态与保存对接
│   │   └── capture_one_pair.py     # 抓拍工具实现
│   └── yolo11_seg/                 # 内置第三方 Ultralytics YOLO 运行时
├── scripts/
│   ├── seg_service/
│   │   ├── seg_service_server.py   # Server 启动入口
│   │   └── seg_service_client.py   # 文件调用命令行入口
│   └── seg_action/
│       ├── seg_action_server.py    # Server 启动入口
│       └── seg_action_client.py    # Action 调用命令行入口
├── tools/capture_one_pair.py       # 抓拍命令行入口
├── examples/
│   ├── seg_service_call.py         # Service 实例化、调用与 Mask 读取模板
│   └── seg_action_call.py          # 最小 Action Python 模板
├── tests/                          # 不依赖真实 ROS / 权重的回归测试
├── docs/
│   ├── README.md                   # 文档导航
│   └── SERVICE_CODE_REVIEW.md      # Service 审核清单
├── setup.py                        # Python 库和第三方运行时打包
├── CMakeLists.txt                  # catkin 构建与入口/资源安装
├── package.xml                     # ROS 包名与依赖
└── requirements.txt               # 历史 Python 环境清单
```

`scripts/` 和 `tools/` 是薄启动入口，业务在 `src/naviai_yolo/`。库调用应导入 `naviai_yolo.*`，不从脚本动态加载实现。
`examples/` 演示调用方如何使用这些库接口；接入自己的项目时参考示例，不需要复制或修改 `src/` 中的 Client 实现。

```text
Service Client：segment_files() → segment() → ROS 请求 ─┐
                                                     ├→ NaviaiSeg.predict()
Action Server：CameraSubscriber → 任务内取帧 ───────────┘       ↓
                                                   postprocess_segmentation()
                                                            ↓
                                            DetItem → Service 响应 / Action 反馈

Server 帧处理由 DebugRecorder 包围，退出时调用 DebugWriter 保存。

已有 Mask：load_mask_file() → postprocess_mask_file() → SegDetection 列表
           （本地独立调用，不经过 ROS / 模型 / 自动保存链路）
```

`yolo11_seg/` 是模型运行时，不是另一个 Service 或 Action；后处理和 ROS 封装在它的外层。
Service 与 Action 在不同进程启动时，各自加载模型和保存组件，不共享模型实例或显存分配。

<a id="environment"></a>
## 2. 环境与启动准备

### 2.1 运行依赖

代码面向项目既有的 ROS Noetic / Python 3.8 环境，需要：

- ROS：`rospy`、`actionlib`、`actionlib_msgs`、`sensor_msgs`、`cv_bridge`、`message_filters`、`rospkg`。
- 自定义消息包：`navi_types`，包含 `InstSeg`、`InstSegAction`、`DetItem` 等生成类型。
- 推理：与目标设备匹配的 PyTorch、torchvision，以及内置 YOLO 所需依赖。
- 图像与配置：NumPy、OpenCV、PyYAML；构建还需要 `catkin_pkg` 和 `setuptools`。

`requirements.txt` 是历史环境列表，包含设备相关版本，不是已验证可跨设备直接安装的最小依赖锁文件。
尤其要匹配 Python、CUDA/设备环境、PyTorch、torchvision 和 cv_bridge；本机单元测试不能代替目标设备环境验收。

自定义 `.srv/.msg/.action` 定义不在本仓库，字段类型以部署环境为准。环境激活后可检查：

```bash
python -c "import rospy, cv_bridge, torch, cv2, numpy, yaml; from navi_types.srv import InstSeg; from navi_types.msg import InstSegAction, DetItem; print('imports OK')"
rossrv show navi_types/InstSeg
rosmsg show navi_types/DetItem
```

### 2.2 catkin 构建与运行

以下命令以 Bash 为例；`/absolute/path/...`、`/path/to/...` 都是占位路径，需替换。
先将项目和 `navi_types` 放入或链接到 catkin 工作空间的 `src/`。

```bash
source /opt/ros/noetic/setup.bash
cd /path/to/catkin_ws
catkin_make -DPYTHON_EXECUTABLE=/path/to/seg/bin/python
source devel/setup.bash
```

需要安装空间时：

```bash
catkin_make install
source install/setup.bash
```

每个运行终端都要激活运行环境并 source 对应 ROS / catkin 环境。
调用前需要可访问 ROS Master；若尚未启动，在独立终端执行 `roscore`。

catkin 安装包含 Python 库、五个可执行入口、配置、`.pt` 权重、文档和示例。ROS 命令使用包名：

```bash
rosrun naviai_manip_seg seg_service_server.py
```

### 2.3 源码直接运行

后文的 `python scripts/...`、`python tools/...` 默认在仓库根目录执行；薄入口也支持绝对脚本路径。
直接在自己的程序中导入本包，或运行 `examples/`，先 source catkin 环境；仅用源码时也可设置：

```bash
export PYTHONPATH="/absolute/path/naviai_yolo_0909/src${PYTHONPATH:+:$PYTHONPATH}"
```

本地 Mask 读取只需能导入本包和 NumPy / OpenCV，不需要 ROS、模型配置或权重。

<a id="configuration"></a>
## 3. 配置文件与参数

### 3.1 model.yaml：模型与 Server 保存配置

当前默认有效配置如下；修改后重启 Server，不支持热更新：

```yaml
seg_model_path: models/best_wrc.pt
imgsz: 640
conf: 0.6

debug:
  output_dir: data/seg
  save_records: true
  save_visualizations: true
```

| 字段 | 含义与校验 | 谁读取 |
|---|---|---|
| `seg_model_path` | 非空权重路径；文件必须存在，需使用实例分割权重 | Service / Action Server |
| `conf` | 有限数值，范围 `[0, 1]`，检测置信度阈值 | Service / Action Server |
| `imgsz` | 正整数，或两个正整数组成的 `[height, width]` | Service / Action Server |
| `debug.output_dir` | 自动保存根目录；未设置时为 `data/seg` | Server 保存组件 |
| `debug.save_records` | 保存彩色图、原值深度、Mask、检测列表和元数据 | Server 保存组件 |
| `debug.save_visualizations` | 每帧保存一张 RGB + 全部返回实例的叠加图 | Server 保存组件 |

`conf/imgsz` 不接受字符串数字、布尔值或 NaN/Inf。保存开关必须是 YAML 布尔值 `true/false`，不能写成字符串 `"false"`。
省略 `debug` 或某个保存开关时，对应保存默认关闭；仓库配置显式开启两项。两项都关闭，不创建调试目录，但 Service 的输入校验仍执行。

`imgsz` 不是客户端输入尺寸限制，也可以配置为 `[480, 640]`；预处理保持比例、填边并满足模型 stride 要求。
当前没有暴露 `device`、FP16、batch、IoU、任务时长等 YAML 配置项，添加这些键不会自动生效。

### 3.2 camera.yaml：Action / 抓拍的图像来源

当前生效的话题配置为：

```yaml
camera_rgb_topic: /zj_humanoid/sensor/realsense_head/color/image_raw
camera_depth_topic: /zj_humanoid/sensor/realsense_head/aligned_depth_to_color/image_raw
```

根据实际相机修改。两个话题必须发布 `sensor_msgs/Image`；Action 和抓拍使用 `TimeSynchronizer`，队列长度为 10，要求消息时间戳严格一致。两个话题都有数据，不代表一定能配成一对。

`chest_camera_param` 当前未被分割/抓拍代码使用，修改它不会改变 YOLO 或完成 RGB-D 配准。
Service Server / Client 不读取相机配置，采集和选帧由调用方负责。

### 3.3 ROS 私有参数与命令行参数

| 节点 | ROS 私有参数 | 默认值 / 作用 |
|---|---|---|
| Service Server | `~model_config` | 空值使用包内 `config/model.yaml` |
| Action Server | `~model_config`、`~camera_config` | 空值分别使用包内 model.yaml / camera.yaml |
| Action Client | `~labels` | `['wp_2']`，要检出的类别 |
| Action Client | `~timeout_sec` | `60.0`，Client 等待结果的秒数，不改变 Server 的 10 秒期限 |
| 抓拍工具 | `~camera_config` | 空值使用包内 camera.yaml |

使用 `_参数名:=值` 传入 ROS 私有参数：

```bash
python scripts/seg_service/seg_service_server.py _model_config:=/absolute/path/model.yaml
python scripts/seg_action/seg_action_server.py \
  _model_config:=/absolute/path/model.yaml \
  _camera_config:=/absolute/path/camera.yaml
```

自定义 YAML 是完整替代文件，不与默认配置合并。model.yaml 必须含模型路径、conf 和 imgsz；camera.yaml 必须含两个话题键。
Service Client 的 `--color/--depth/--labels/--service` 和抓拍工具的 `--out-dir/--prefix/--wait` 是普通命令行参数，详见各自章节。
非空旧参数 `~mask_file/~mask_label` 会导致 Service Server 启动报错，需改用本地 Mask 方法。

### 3.4 路径相对哪里

| 路径 | 解析基准 |
|---|---|
| 相对 `~model_config/~camera_config` | 包资源根目录 |
| YAML 中相对 `seg_model_path` | 包资源根目录，不是自定义 YAML 所在目录 |
| 相对 `debug.output_dir`，源码运行 | 仓库根目录 |
| 相对 `debug.output_dir`，安装空间运行 | `${ROS_HOME:-~/.ros}/naviai_manip_seg/` |
| Service Client 的 RGB-D 文件路径 | Client 工作目录；文件在 Client 机器读取 |
| `load_mask_file()` 的相对路径 | 调用方工作目录；支持绝对路径与 `~` |
| 抓拍显式传入的相对 `--out-dir` | 调用方工作目录；省略时使用包的默认输出路径规则 |

自动 debug 保存发生在 Server 机器上，Client 不需要挂载 Server 的目录。需要明确输出位置时可填写绝对路径。

<a id="image-contract"></a>
## 4. 图像、尺寸与结果约定

### 4.1 输入图像

| 输入 | 约定 |
|---|---|
| 彩色数组 | 非空 `H×W×3`、`uint8`、BGR 顺序；RGB 数组需由调用方转换 |
| 深度数组 | `H×W`，与彩色图同尺寸；`uint16` 按毫米、`float32` 按米 |
| Service 深度 ROS 编码 | `16UC1` / `32FC1`，Server 实际解码并校验尺寸与 dtype |
| 未提供深度 | 仅 Client 数组/文件方法自动补同尺寸 uint16 零图，是协议占位，不是有效测距数据 |
| 文件深度 | 原值图像（如 uint16 PNG）或 `.npy`；禁止 pickle，不接受对象数组和伪装 NPZ |

深度 NaN/Inf 不会被自动修复。尺寸、类型符合要求，不代表深度有效或已经与 RGB 几何对齐。
不要用彩色化深度图作为真实深度，也不要为了适配 imgsz 单独缩放 RGB，导致它与深度失配。

### 4.2 不同尺寸如何推理

客户端无需预先把图片变成 640×640。对于当前图像调用路径：

1. YOLO 按 imgsz 和模型 stride 预处理，保持宽高比并进行 letterbox 填边。
2. 内置 `.pt` 推理允许矩形输入，实际网络输入不一定是 640×640。
3. 公共后处理去掉填边，并把 Mask 恢复成输入原图的 H×W；横图、竖图和奇数尺寸均有回归测试。
4. 浮点 Mask 按 `> 0.5` 二值化，整数 Mask 按 `> 0`；二值化后全空的实例被剔除。

较大的 imgsz 通常增加计算和内存开销；实际精度和耗时需要在自己的权重、设备和目标尺度上验收。

### 4.3 输出与类别筛选

| 输出形式 | Mask 数据 | 其他字段 |
|---|---|---|
| ROS `DetItem` | `sensor_msgs/Image`，`mono16`，背景 0、前景 65535 | label、confidence、bbox |
| 本地 `SegDetection` | NumPy `uint8`，背景 0、前景 1 | source_index、label、confidence、bbox |
| Debug Mask PNG | `uint8`，背景 0、前景 255 | 对应关系在 detections.json |

bbox 顺序为 `[x1, y1, x2, y2]`，使用原图坐标。YOLO bbox 来自模型输出并转为整数，不按最终 Mask 重算，也未额外裁剪到图像边界；本地文件 Mask 的 bbox 由前景计算，右下界为最大坐标加 1。

| labels 输入 | Service | Action Client |
|---|---|---|
| `None` / `[]` | 发送 `[]`，返回全部检出类别 | 不发送 Goal，返回 False |
| `['wp_2']` | 只返回这一类别 | 检出该类别非空 Mask 后成功 |
| `['wp_2', 'screw']` | 保留两类全部检出实例 | 任意一类检出即可成功，不要求两类同时出现 |
| 不存在的类别 | 正常返回空结果 | 持续尝试，直到超时或取消 |
| `'wp_2'` 单个字符串 | 非法，不能代替列表 | 非法，抛 ValueError |

类别名必须与权重中的名称一致；`wp_2`、`screw` 并不保证存在于所有模型中。
返回 Mask 当前不主动继承 color 的 header。Service 数组 Client 重新生成 ROS Image，也不会继承采集时间戳；需要原始 header 时应自行构造 ROS 请求并管理来源关联。

<a id="service"></a>
## 5. Service 使用与方法

### 5.1 协议与执行逻辑

服务名为 `instance_segmentation_service`，类型为 `navi_types/InstSeg`；在默认 ROS 命名空间中可通过 `/instance_segmentation_service` 查询。

| 方向 | 字段 | 作用 |
|---|---|---|
| 请求 | `labels` | 字符串列表；空列表不限制类别 |
| 请求 | `color_image` | 待分割的 ROS Image |
| 请求 | `depth_image` | 对齐深度 ROS Image；不参与模型推理，但必须符合输入约定 |
| 响应 | `have_objs` | 筛选并剔除空 Mask 后是否还有实例 |
| 响应 | `items` | `DetItem` 列表，包含 label、confidence、bbox、mask |

响应**不含输入 RGB-D**；调用方保留原图或消息。服务一次请求只处理一张图，不订阅相机、不等待某类目标出现，也不提供 Action 的取消机制。

Server 初始化时读取并校验配置、加载模型、尝试预热、初始化保存组件，最后注册服务。请求进入后依次解码/校验 RGB-D、YOLO 推理、服务端类别过滤、Mask 后处理、转换为 DetItem；保存开启时写入对应分析记录和可视化。

### 5.2 启动与命令行调用

终端 A 启动 Server：

```bash
python scripts/seg_service/seg_service_server.py
# catkin 环境中的等价入口：
# rosrun naviai_manip_seg seg_service_server.py
```

终端 B 核对协议并调用：

```bash
rosservice type /instance_segmentation_service

# 指定两个类别；深度可以是 PNG 或原值 NPY。
python scripts/seg_service/seg_service_client.py \
  --color /absolute/path/color.png \
  --depth /absolute/path/depth.npy \
  --labels wp_2 screw

# 不限制类别，也不提供深度：Client 补全零深度占位图。
python scripts/seg_service/seg_service_client.py --color /absolute/path/color.png
```

| CLI 参数 | 必填 / 默认值 | 含义 |
|---|---|---|
| `--color` | 必填 | 彩色图像文件路径，OpenCV 读取为 BGR |
| `--depth` | 可省略 | 对齐深度图像或 `.npy`；省略则补零 |
| `--labels` | 可省略 | 一个或多个类别；省略表示全部，不能只写参数名而不传值 |
| `--service` | `instance_segmentation_service` | 调用的服务名，协议仍固定为 InstSeg |

`--service` 不是消息类型切换开关。自定义服务名时需同时配置 Server 的 ROS 名称重映射。
CLI 成功完成请求时退出码为 0，正常“未检出”也是 0；捕获到输入/通信/服务异常时为 1，命令行参数解析错误为 2。

<a id="service-call-template"></a>

#### 业务集成模板：seg_service_call.py

[examples/seg_service_call.py](examples/seg_service_call.py) 展示 import、节点初始化、Client 实例化、文件调用、Mask 解码及异常处理。
运行模板前先按环境章节配置导入路径，并启动 Server。从仓库根执行：

```bash
python examples/seg_service_call.py \
  _color_path:=/absolute/path/color.png \
  _depth_path:=/absolute/path/depth.npy \
  '_labels:=[wp_2]'

# 只提供彩色图，补零深度，保留全部类别：
python examples/seg_service_call.py _color_path:=/absolute/path/color.png
```

| 模板 ROS 私有参数 | 默认值 / 是否必填 | 传给哪里 |
|---|---|---|
| `~color_path` | 必填 | `segment_files(color_path=...)` |
| `~depth_path` | None；省略则补零 | `segment_files(depth_path=...)` |
| `~labels` | `[]`；全部类别 | `segment_files(labels=...)` |
| `~service_name` | `instance_segmentation_service` | `SegServiceClient(service_name=...)` |
| `~wait_timeout_sec` | `10.0`；有限正数 | `SegServiceClient(wait_timeout_sec=...)`，只等待服务上线 |

这些私有参数属于**示例文件**，不是 `scripts/` 中 CLI 的 `--color/--depth/--labels/--service` 参数，也不是 Client 类会自动读取的配置。
业务项目直接调用类时，按下面 5.3 节传 Python 参数即可，无需照搬示例的 get_param()。已有数组时改用 `segment(...)`，模板内也标出了替换位置。
模板请求成功（含正常未检出）退出 0；缺少参数、输入/连接/服务调用或 Mask 解码错误退出 1。它不加载模型，不自动保存数据。
模板随包安装到 `examples/` 资源目录，不新增 `rosrun` 可执行入口。

### 5.3 Python Client 提供的方法

导入位置：`naviai_yolo.ros.seg_service_client.SegServiceClient`。

| API | 参数 | 返回 / 行为 |
|---|---|---|
| `SegServiceClient(service_name='instance_segmentation_service', wait_timeout_sec=10.0)` | 服务名、有限正数等待秒数 | 等待服务出现并创建 InstSeg 代理，不加载模型/相机配置 |
| `segment(color, depth=None, labels=None)` | BGR uint8 数组、可选深度数组、可选类别列表 | 校验并编码 ROS Image，调用服务，原样返回 InstSegResponse |
| `segment_files(color_path, depth_path=None, labels=None)` | RGB-D 文件路径、可选类别列表 | 读取文件后调用同一个 segment()，返回相同响应类型 |

业务方法只有这两个，不存在独立 camera / file / rgbd 三种 Client。两个方法不修改调用方数组，不自动保存图片。
`wait_timeout_sec` 仅限制等待服务出现的时间，**不是一次 RPC 的处理超时**。

文件方法完整示例：

```python
import rospy
from cv_bridge import CvBridge, CvBridgeError
from naviai_yolo.ros.seg_service_client import SegServiceClient

rospy.init_node('my_seg_service_client')
bridge = CvBridge()
try:
    client = SegServiceClient(wait_timeout_sec=10.0)
    response = client.segment_files(
        '/absolute/path/color.png',
        '/absolute/path/depth.npy',
        labels=['wp_2'],
    )
    if not response.have_objs:
        rospy.loginfo('请求成功，但没有符合条件的目标')
    for item in response.items:
        mask16 = bridge.imgmsg_to_cv2(item.mask, 'mono16')
        foreground = mask16 > 0  # bool H×W，可用于索引原图或深度。
        rospy.loginfo(f'{item.label}: conf={item.confidence:.3f}, bbox={list(item.bbox)}')
except (ValueError, CvBridgeError, rospy.ROSException, rospy.ServiceException) as exc:
    rospy.logerr(f'分割调用失败: {exc}')
```

数组方法示例；来自相机 SDK 的图像也用同一个方法，不需要先保存成文件：

```python
import cv2
import numpy as np
import rospy
from naviai_yolo.ros.seg_service_client import SegServiceClient

rospy.init_node('array_seg_client')
color_bgr = cv2.imread('/absolute/path/color.png', cv2.IMREAD_COLOR)
if color_bgr is None:
    raise ValueError('无法读取彩色图')
depth = np.load('/absolute/path/depth.npy', allow_pickle=False)
client = SegServiceClient()
response = client.segment(color_bgr, depth, labels=[])
# color_bgr / depth 仍由调用方持有；空 labels 保留全部检出类别。
```

### 5.4 已有 ROS Image 时直接调用

如果上层已经持有带原始 header 的 ROS 消息，可直接使用 InstSeg 代理，不必转成数组再编码。
以下片段假设 `color_msg`、`depth_msg` 是调用方采集到的同帧消息：

```python
import rospy
from navi_types.srv import InstSeg

rospy.init_node('raw_instseg_client')
rospy.wait_for_service('instance_segmentation_service', timeout=10.0)
proxy = rospy.ServiceProxy('instance_segmentation_service', InstSeg)
response = proxy(labels=['wp_2'], color_image=color_msg, depth_image=depth_msg)
```

直接请求同样受 Server 的尺寸/编码校验；不能省略 depth 消息。没有真实深度时，应自行生成合法的同尺寸零图。

### 5.5 空结果与错误的区别

| 情况 | 对外行为 |
|---|---|
| 正常推理未检出、labels 无匹配、最终 Mask 全空 | `have_objs=False, items=[]` |
| Client 文件不存在或图像尺寸/dtype 不合法 | 通常抛 ValueError，不发出分割 RPC |
| 服务不可达或通信失败 | ROS 异常传播给调用方 |
| Server 解码、校验、推理或后处理失败 | 抛 `rospy.ServiceException`，不伪装成空结果 |
| Debug 写盘失败 | 告警，不因写盘故障把正常分割变成空结果 |

<a id="action"></a>
## 6. Action 使用与方法

### 6.1 协议与任务语义

Action 名称为 `naviai_manip_inst_seg_action`，类型为 `navi_types/InstSegAction`。

| 消息 | 字段 | 用途 |
|---|---|---|
| Goal | `labels` | 非空的目标类别列表 |
| Feedback | `status`、`have_objs`、`items` | 当前帧的处理状态和检测实例；正常反馈 status 为 200 |
| Feedback | `color_image`、`depth_image` | 与本帧检测对应的 RGB-D ROS 消息 |
| Result | `success` | 仅表示任务结果，不携带检测图像或 items |

Server 订阅相机，收到 Goal 后清空旧缓存，在任务循环中消费最新同步帧；未检出会继续尝试，任意一个目标类别有非空 Mask 即可成功。
它不是持续视频分割发布器，也不是“等所有目标类别都齐全”的任务。无帧、无匹配或全空 Mask 都不会触发成功。

Server 在处理阶段之间及提交终态前检查取消/超时，取消优先于超时和成功：取消为 preempted，超时/处理失败为 aborted，成功为 succeeded。
单帧 CvBridge 解码错误会记录并继续尝试下一帧；其他执行异常通过统一收尾提交失败结果。

Action 的深度目前用于反馈和保存，没有套用 Service 的推理前深度尺寸/编码校验。因此消费 Feedback 时仍需检查深度是否可用和对齐。

### 6.2 启动、调用与现成模板

终端 A：

```bash
python scripts/seg_action/seg_action_server.py
# 自定义相机时加：_camera_config:=/absolute/path/camera.yaml
```

终端 B：

```bash
python scripts/seg_action/seg_action_client.py '_labels:=[wp_2, screw]' _timeout_sec:=15

# catkin 入口等价用法：
# rosrun naviai_manip_seg seg_action_client.py '_labels:=[wp_2]' _timeout_sec:=15
```

完整模板见 [examples/seg_action_call.py](examples/seg_action_call.py)，已完成节点初始化、发送目标、等待与读取缓存：

```bash
python examples/seg_action_call.py '_labels:=[wp_2]'
```

模板固定构造 `timeout_sec=15.0`，只读取 `~labels`，不要假定它也接受 `~timeout_sec`。
模板安装在包的 `examples/` 资源目录，不是 `rosrun` 可执行入口。

### 6.3 Python Client 提供的方法和缓存

导入位置：`naviai_yolo.ros.seg_action_client.SegActionClient`。

| API | 说明 |
|---|---|
| `SegActionClient(timeout_sec=60.0, save_vis=False)` | 连接 Action Server；timeout_sec 必须是有限正数 |
| `send_goal(labels)` | 阻塞发送并等待一次任务，返回 bool；超时自动取消 |
| `feedback_callback(feedback)` | actionlib 自动调用，缓存有效检测；业务代码通常不直接调用 |
| `last_items` | 最后一次有效反馈中的非空 DetItem 列表 |
| `last_color_image` / `last_depth_image` | 对应有效反馈的 RGB-D ROS Image |
| `last_status` | 最近收到反馈的 status；不是 Action 最终状态 |

每次 send_goal() 开始先清空缓存，包括拒绝空 labels 的情况。只有以下条件同时成立，才返回 True：

1. 最终状态为 SUCCEEDED。
2. Result.success 为 True。
3. 已缓存非空的检测 items。

Result 成功但没收到有效反馈，仍返回 False。返回 False 时，即使缓存里有检测，也不能当作成功任务的结果使用。
同一 Client 应串行调用 send_goal，不提供多 Goal 并发调用保证。非法 labels 抛 ValueError；连接失败等异常也可能传播，不能把所有失败都理解为 bool False。

```python
import rospy
from cv_bridge import CvBridge
from naviai_yolo.ros.seg_action_client import SegActionClient

rospy.init_node('my_seg_action_client')
client = SegActionClient(timeout_sec=15.0)
if client.send_goal(['wp_2']):
    bridge = CvBridge()
    color = bridge.imgmsg_to_cv2(client.last_color_image, 'bgr8')
    depth = bridge.imgmsg_to_cv2(client.last_depth_image, 'passthrough')
    for item in client.last_items:
        mask = bridge.imgmsg_to_cv2(item.mask, 'mono16') > 0
        rospy.loginfo(f'{item.label}: bbox={list(item.bbox)}')
else:
    rospy.logwarn('任务未成功，不消费缓存作为成功结果')
```

`save_vis` 是兼容旧调用的弃用参数，传 True 只告警，不保存图片；CLI 不再读取 `~save_vis`。自动保存统一配置在 Server 的 model.yaml。

### 6.4 超时和取消的实际边界

| 阶段 | 当前限制 | 使用的时钟 |
|---|---|---|
| Client 构造时连接 Server | 固定 10 秒等待参数 | actionlib 原有 ROS 时间等待 |
| Server 执行一个 Goal | 固定 10 秒，包含取帧、推理和 debug 保存 | 单调墙钟 |
| Client 等待任务结果 | timeout_sec，默认 60 秒，示例 15 秒 | 单调墙钟 + Event |
| Client 超时后的取消确认 | 最多再等待 2 秒，然后返回 False | 单调墙钟 |

Server 的 10 秒是代码常量，不受 Client 的 timeout_sec 影响。同步模型推理和写盘无法立即中断，实际返回可能超过期限；返回后会再次检查，不把迟到结果提交为成功。
取消确认超时也不保证 Server 已停止。收到检测 Feedback 后无需主动取消，Server 会自行成功结束；过早取消反而会让最终状态变为失败。

使用仿真时间时，启动连接阶段仍可能受暂停/缺失的 `/clock` 影响，不能把结果等待的墙钟保证推广到连接阶段。
封装没有独立公共 `cancel()` 方法。需要外部手动取消时，可由控制线程调用底层 `client.client.cancel_goal()`；不要在阻塞的 send_goal() 后才取消已结束任务，多任务控制需自行使用 actionlib 管理。
Action CLI 的正常进程退出码不代表任务成功与否，应检查日志中的 success；程序集成应检查 send_goal() 返回值。

<a id="tools"></a>
## 7. 本地方法与其他工具

### 7.1 已有 Mask：load_mask_file()

位置：`naviai_yolo.mask_io`。完整签名为 `load_mask_file(mask_path, image_shape, label)`。

| 参数 / 返回 | 说明 |
|---|---|
| `mask_path` | 本地文件路径，str 或 Path；每次调用重新读取，不缓存 |
| `image_shape` | 对应 RGB 的 `(height, width)` 或 `color.shape`，用来校验尺寸 |
| `label` | 调用方赋予该 Mask 的类别名 |
| 返回 | `List[SegDetection]`：全黑为 []，有前景时为单个实例 |

```python
from naviai_yolo.mask_io import load_mask_file

detections = load_mask_file(
    '/absolute/path/mask.png',
    image_shape=(720, 1280),
    label='wp_2',
)
for detection in detections:
    print(detection.label, detection.confidence, detection.bbox)
    mask01 = detection.mask  # uint8，原图尺寸，0/1。
```

支持单通道 uint8 / uint16 PNG 等图像；非零区域全部作为同一个实例，不拆连通域，也不解释像素类别编号。
置信度固定为 1.0，只是已有 Mask 的约定，不是模型推理得分。文件不存在、无法解码、通道数或尺寸错误会抛 ValueError。
它不加载模型、ROS、model.yaml，不自动保存 debug，不提供远程文件读取或图像配准；RGB、深度与 Mask 是否来自同一帧由调用方保证。

### 7.2 抓拍：capture_one_pair.py

用途是采集一对同步 RGB-D 供后续分析，**不会调用 YOLO**，也不受 model.yaml 的 debug 开关控制。

```bash
python tools/capture_one_pair.py \
  --out-dir /absolute/path/captures \
  --prefix sample \
  --wait 5 \
  _camera_config:=/absolute/path/camera.yaml

# catkin 中的等价入口：
# rosrun naviai_manip_seg capture_one_pair.py --wait 5
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--out-dir` | 由包输出规则解析的 `data/rgbd_captures` | 保存目录，具体路径规则见配置章节 |
| `--prefix` | `cap` | 文件名前缀，使用简单名称，不作为目录路径 |
| `--wait` | `5.0` | 等待同步帧的墙钟秒数，必须为有限正数 |
| `~camera_config` | 包内 camera.yaml | RGB-D 话题配置 |

保存文件使用本机保存时的毫秒时间戳，不是相机采集时间戳：

```text
captures/
├── sample_<timestamp>_color.png   # BGR uint8 图像编码为 PNG
└── sample_<timestamp>_depth.png   # 相机为 16UC1：uint16，毫米
    或 sample_<timestamp>_depth.npy # 相机为 32FC1：float32，米
```

深度不会强行转换成 uint16，不做单位换算。工具会校验 RGB-D 尺寸/编码并检查写入结果：成功退出码 0；等待失败、无帧、解码或写入失败为 1；非法 CLI 参数为 2。
写入失败可能保留已经写出的部分文件，不自动删除。该工具不生成 session.json / meta.json，也不承诺整对文件原子提交。

其实现类也可复用：

| `PairCapture` API | 行为 |
|---|---|
| `PairCapture(out_dir, prefix)` | 读取 camera.yaml，并立即注册 RGB-D 同步订阅；调用前需初始化 rospy |
| `save_latest()` | 保存最新的一对消息，返回 bool；没有帧时返回 False，不负责等待 |
| `cb(rgb_msg, depth_msg)` | 同步订阅回调，更新 latest；通常不手工调用 |

只需一次抓拍时优先使用上面的 CLI，它已经包含等待逻辑。

### 7.3 相机组件：CameraSubscriber

位置：`naviai_yolo.ros.camera_sub.CameraSubscriber`，供 Action 或上层采集流程复用，不是单独运行模式。

| 方法 | 行为 |
|---|---|
| `CameraSubscriber()` | 读取 camera.yaml 并初始化缓存，尚未订阅 |
| `start_subscribing()` | 启动严格同步 RGB-D 订阅 |
| `get_camera_data()` | 非阻塞返回 `{'rgb': rgb_msg, 'depth': depth_msg}`，无帧为 None；取出即消费 |
| `clear_camera_data()` | 清空尚未消费的缓存帧 |
| `stop_subscribing()` | 解除订阅并清空缓存 |

只保留最新同步帧，新帧覆盖未消费的旧帧；不是帧历史队列，也没有最大帧龄检查。
上层需要自己限制等待时间。以下示例仅在一秒内尝试取得一帧，不调用模型：

```python
import time
import rospy
from naviai_yolo.ros.camera_sub import CameraSubscriber

rospy.init_node('camera_component_example')
camera = CameraSubscriber()
camera.start_subscribing()
try:
    deadline = time.monotonic() + 1.0
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        frame = camera.get_camera_data()
        if frame is not None:
            rospy.loginfo(f"RGB: {frame['rgb'].width}x{frame['rgb'].height}")
            break
        time.sleep(0.01)
finally:
    camera.stop_subscribing()
```

### 7.4 直接调用模型与公共后处理

自己的非 ROS 程序需要推理时，可以复用 `NaviaiSeg`，不必启动 Server：

```python
import cv2
from naviai_yolo.segmentation import NaviaiSeg
from naviai_yolo.postprocess import postprocess_segmentation

color = cv2.imread('/absolute/path/color.png', cv2.IMREAD_COLOR)
if color is None:
    raise ValueError('无法读取彩色图')
model = NaviaiSeg('/absolute/path/best_wrc.pt')
boxes, masks, names = model.predict(color, conf=0.6, imgsz=640)
detections = postprocess_segmentation(
    boxes, masks, names, color.shape, target_labels=['wp_2'],
)
```

`NaviaiSeg(model_path)` 直接使用传入路径，不自动读取 model.yaml；相对模型路径相对于调用方工作目录。重复推理时复用同一个实例，不要每帧加载模型。
`predict(bgr_image, conf, imgsz)` 返回 NumPy boxes、masks 和类别映射 names；无框时返回 `(None, None, None)`，后处理会得到 []。
模型有框但没有分割 Mask 时会报错，通常表示使用了检测权重而不是分割权重。

| 公共方法 | 用途 |
|---|---|
| `postprocess_segmentation(boxes, masks, names, image_shape, target_labels=None, mask_threshold=0.5)` | 类别筛选、去填边、恢复尺寸、二值化、剔除空 Mask，返回 SegDetection 列表 |
| `postprocess_mask_file(mask, image_shape, label)` | 处理已经读取的 Mask 数组；不负责读文件 |
| `mask_to_mono16(mask_binary)` | 转换为背景 0、前景 65535 的 uint16 数组 |
| `mask_to_uint8(mask_binary)` | 转换为背景 0、前景 255 的 uint8 数组 |
| `render_overlay(color, detections)`，位于 `naviai_yolo.debug_writer` | 返回叠加 Mask、bbox、label 的新图，不修改原图，也不自动写盘 |

这些本地方法不创建 Server 的分析记录。`SegDetection.source_index` 是原始预测中的实例序号，不是跟踪 ID。

<a id="debug"></a>
## 8. Debug 数据保存与回放

### 8.1 谁保存、保存什么

只有 Service / Action Server 根据 model.yaml 自动保存；两个 Client 均不自动保存。
`save_records` 和 `save_visualizations` 独立，关闭两者即可关闭自动 debug 输出。
预热、Action 无帧等待和接收 Goal 本身不会产生样本；Action 每个实际处理帧会产生记录，包括未检出和处理失败的帧。

```text
data/seg/<run_id>/
├── session.json
├── records/
│   ├── 000001/
│   │   ├── color.png           # 未叠加标记的彩色图
│   │   ├── depth.npy           # 保留 dtype / 数值的深度
│   │   ├── meta.json           # 来源、处理状态、文件清单
│   │   ├── detections.json     # 检测列表及 Mask 路径
│   │   └── masks/
│   │       ├── 000.png         # 单帧实例 0 的二值 Mask
│   │       └── 001.png         # 单帧实例 1 的二值 Mask
│   └── 000002/...
└── visualizations/
    ├── 000001.jpg              # 同一 sample_id 的 RGB + 全部实例叠加
    └── 000002.jpg
```

上图展示两项都开启且输入可保存的情况；没有检测时不创建 masks 目录。只开可视化不会创建 records；只开 records 不会创建 visualizations。
color.png 是解码后的 BGR 像素编码成 PNG，不是原始 ROS 消息字节；depth.npy 不归一化、不换算单位，使用 `np.load(path, allow_pickle=False)` 读取。
可视化只混合前景区域，再画框和标签；无检测时标注状态。JPEG 有损，不应拿它代替原图进行数值分析。

### 8.2 run_id、sample_id 和 request_id

| 标识 | 含义 |
|---|---|
| `run_id` | 每次 Server 启动独立生成：UTC 时间（含微秒）+ Server 类型 + 随机后缀 |
| `sample_id` | 当前 run 中从 `000001` 递增，线程安全分配；用于配对 records 与可视化 |
| `request_id` | Action 同一 Goal 的帧共享 UUID；Service 为 null |
| `instance_id` | detections.json 中的单帧局部序号，对应 masks/000.png 等 |

样本标识是 `(run_id, sample_id)`，不能只看 `000001`，也不能把实例编号当跨帧 ID。
并发时编号分配顺序不等于处理完成顺序。Service 协议不传 Client 文件路径或输入方法，Server 无法从 sample_id 推导原始文件名，也无法区分 segment() 和 segment_files()。

### 8.3 session.json 与 meta.json

session.json 保存本次运行的 schema_version、run_id、Server 类型、接口名、模式、模型路径、conf/imgsz、Mask 阈值和 debug 配置。
它不复制权重、不计算模型哈希；严格复现实验还需自行保留对应代码和模型版本。

| meta.json 字段 | 含义 |
|---|---|
| `schema_version` | 当前格式版本 1 |
| `run_id/sample_id/request_id` | 运行、帧和 Action Goal 的关联标识 |
| `received_at/processed_at/saved_at` | 开始处理、完成处理、开始写记录的 UTC 时间；不是相机采集时间 |
| `processing_ms` | 处理上下文内耗时，不含 debug 写盘；Service 包含深度校验，Action 保存阶段的深度解码不计入 |
| `color/depth` | ROS 编码、frame_id、stamp、保存数组的 shape/dtype；深度还包含 unit |
| `target_labels` | 此次服务端筛选条件，Service 的 [] 表示全部类别 |
| `status/error` | 帧处理状态 success、no_detections 或 error，以及异常信息 |
| `detection_count` | 本帧保存的实例数量 |
| `files` | color、depth、detections 相对文件名；不可用输入为 null |
| `record_complete` | 本次分析记录是否按计划写完；不是推理成功标志 |

示意节选如下（实际文件还包含其余时间与来源字段）：

```json
{
  "schema_version": 1,
  "run_id": "example_run",
  "sample_id": "000001",
  "request_id": null,
  "target_labels": ["wp_2"],
  "status": "success",
  "error": null,
  "detection_count": 1,
  "record_complete": true,
  "files": {
    "color": "color.png",
    "depth": "depth.npy",
    "detections": "detections.json"
  }
}
```

ROS stamp 以 `{secs, nsecs}` 保存，缺失或全零为 null，不用保存时间冒充采集时间。
深度 `16UC1` 记为毫米、`32FC1` 记为米，未知编码单位为 null；Action 的 debug 深度解码失败会记录 `depth.error`，不改变 RGB 推理结果。
Service 的无效深度则在推理前报错；诊断记录可能保存仍能解码的错误输入，回放前必须检查 status。

detections.json 保存 `instance_id/source_index/label/confidence/bbox/mask`。正常帧保存的是按请求类别筛选后的结果，Service 只有 labels=[] 才保留全部检出类别。
Action 的 meta.status 是帧状态，不是 Goal 最终状态：一帧可能已检出并保存，但随后取消、超时或反馈失败，整个任务仍失败；保存结果也不等于它已成功发送给 Client。

### 8.4 保存失败与数据完整性

保存目前是同步执行，会增加请求耗时和磁盘占用；没有自动抽样、配额、轮转、清理或异步队列。
创建运行目录失败会告警并关闭该次运行的保存；单帧写入失败只告警，不覆盖正常分割结果。

JSON 使用同目录临时文件写完再原子替换，避免最终更新失败时截断已有 JSON；这不保证整个样本目录原子提交或断电持久性。
`record_complete=False` 表示记录未写完；即使为 True，也可能按计划跳过不可用深度，必须同时检查 status、files、depth.error 和文件是否存在。
历史 debug_images、旧 masks 等数据不自动迁移或删除。

### 8.5 回放分析样本

回放是把保存的 RGB-D 再送入当前 YOLO，不是直接加载旧检测结果；模型或配置不同，结果可能不同。

```python
import json
from pathlib import Path
import rospy
from naviai_yolo.ros.seg_service_client import SegServiceClient

sample = Path('/absolute/path/data/seg/RUN_ID/records/000001')
meta = json.loads((sample / 'meta.json').read_text(encoding='utf-8'))
if not meta['record_complete'] or meta['status'] not in ('success', 'no_detections'):
    raise ValueError('示例仅回放完整且处理正常的样本')
files = meta['files']
if not files.get('color') or not files.get('depth') or meta.get('depth', {}).get('error'):
    raise ValueError('样本缺少有效 RGB-D')

rospy.init_node('replay_seg_sample')
client = SegServiceClient()
response = client.segment_files(
    sample / files['color'], sample / files['depth'], labels=meta['target_labels'],
)
```

若只想加载旧 Mask，使用本地 load_mask_file() 读取 `masks/000.png`；类别与 bbox 等原结果参考 detections.json，不需要启动 Server。

<a id="verification"></a>
## 9. 测试、验收与常见问题

### 9.1 本地回归测试

在具备 NumPy、OpenCV、PyYAML 的 Python 环境中，从仓库根目录运行：

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v

# 只验证本地 Mask 方法：
python -B -m unittest discover -s tests -p 'test_mask_io.py' -v

# 只验证 Service 保存后的 RGB-D 回放链路：
PYTHONPATH=src:tests python -B -m unittest \
  test_ros_interfaces.ROSInterfaceTests.test_service_client_replays_server_debug_record -v
```

当前测试集含 103 项测试，覆盖包名与安装资源定位、后处理、文件读取、配置校验、Service/Action 适配、Client 调用模板、类别过滤、超时取消、相机缓存和 debug 保存。
ROS 通信、CvBridge 和模型使用替身；图像处理、文件写入和内置 LetterBox 的尺寸往返使用真实代码。
测试不依赖真实权重或 GPU，也不证明真实 ROS 消息序列化、CUDA 环境或模型并发安全。

### 9.2 目标设备验收顺序

1. 按环境章节验证 Python/ROS 导入和 navi_types 字段，确认 ROS Master、模型路径和实际类别名。
2. 启动 Service；用同一 RGB-D 分别检查不传 labels、指定存在类别、指定不存在类别，核对 have_objs/items。
3. 检查返回 Mask 的 mono16 编码、原图尺寸和前景位置，使用横竖图、奇数尺寸图做对齐验证。
4. 用错误尺寸深度验证 ServiceException，与正常未检出的空结果区分。
5. 开启 debug，核对原图/深度/Mask/叠加图的编号及文件完整性，再回放一个有效样本。
6. 检查相机两个话题及其时间戳；启动 Action，验证检出、无目标超时、取消和最终 success。
7. 用抓拍工具分别验证 16UC1 PNG 或 32FC1 NPY 的类型与数值，确认输出目录可写。
8. 在实际设备上再测长期运行、并发请求、GPU 占用和开启保存后的端到端延迟。

### 9.3 常见问题

| 现象 | 优先检查 |
|---|---|
| `ModuleNotFoundError: naviai_yolo` | 是否 source catkin；源码库调用是否设置正确 PYTHONPATH |
| 无法导入 navi_types / cv_bridge | 消息包是否构建并 source，当前 Python 是否与 ROS 扩展兼容 |
| Service Client 等不到服务 | Master、命名空间/服务名、Server 是否因配置或权重错误尚未注册 |
| Client 与 Server 类型不匹配 | `rosservice type` 应为 navi_types/InstSeg，检查是否误启动旧脚本 |
| 明明有目标却返回空 | 类别名、conf、模型权重及二值化后 Mask 是否为空；不要把 ServiceException 当空结果 |
| Action 相机话题有数据却总是无帧 | 两个 topic 名、消息类型和 header.stamp 是否严格一致 |
| Client 把 timeout 设为 60，Action 仍约 10 秒结束 | Server 任务期限固定 10 秒；Client timeout 只是自己的等待上限 |
| 仿真时间下 Client 连接一直等 | 连接阶段仍依赖 actionlib ROS 时间；检查 `/use_sim_time` 与 `/clock` |
| 没有生成 debug 文件 | Server 的配置文件、两个保存开关、Debug output 日志及目录权限 |
| 安装后找不到源码 data/seg 的新文件 | 相对输出默认落在 ROS_HOME 下的包目录，见路径规则 |
| 深度尺寸/类型错误 | 使用原值单通道深度；uint16 毫米或 float32 米，不使用彩色化深度 |
| 抓拍失败但留下 color.png | 第二个文件可能写入失败；工具不保证成对原子提交，检查返回值/退出码 |

排查默认相机时可分别查看两个话题的频率和时间戳，例如：

```bash
rostopic hz /zj_humanoid/sensor/realsense_head/color/image_raw
# 另一个终端检查 depth 频率：
rostopic hz /zj_humanoid/sensor/realsense_head/aligned_depth_to_color/image_raw
```



