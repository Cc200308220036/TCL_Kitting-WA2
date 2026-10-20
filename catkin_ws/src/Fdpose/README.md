# FDPose

FDPose 的核心是：**加载 Mesh，使用 RGB-D 和目标 Mask 估计物体位姿**。
项目提供 ROS Service、可直接导入的 Python Client，以及独立的复制注册、坐标变换和相对位姿计算工具。

已有 RGB-D 和 Mask 即可使用，不需要启动 YOLO 或深度补全。
相机采集、分割、任务管理、TCP 计算和机械臂执行由调用方负责。
本文以公共 Client 默认连接的 **V2 服务**为例；旧入口说明放在后面。

## 1. 提供的功能与输入输出

- 加载 Mesh：按磁盘注册表中的 label 加载，或直接提供外部文件路径。
- 位姿估计：支持单目标和多目标，返回物体在相机坐标系下的 `4×4` 位姿矩阵。
- Mesh 卸载/重载：按 label 卸载、卸载全部，或显式重新加载缓存。
- 复制注册：由用户单独调用，将单个 Mesh 复制到资产目录并写入注册表；重名报错。
- 可选坐标变换：Client 接受调用方提供的外参，将相机系位姿转换到目标坐标系。
- 相对位姿工具：只做矩阵计算，不管理示教文件、任务或机器人。

输入约定：

| 输入 | 要求 |
|---|---|
| `color_bgr` | `uint8`、`H×W×3`，OpenCV BGR 顺序 |
| `depth` | `H×W`；整数按毫米，浮点数按米，由 Client 转换 |
| `mask` | 与 RGB-D 等大的二维数组；大于零表示目标前景 |
| `label` | 与加载的 Mesh 标签一致；不通过特殊后缀表达状态 |
| 相机内参 | 由服务端配置，必须与输入图像的分辨率及标定匹配 |

输出 `PoseResult` 包含 `label` 和 `T_object`。
不提供外参时，`T_object = T_camera_object`，平移单位为米。
本地 Mask 不需要 bbox；Client 会自动从 Mask 计算。

### Label 约定

label 只表示物体身份，加载、卸载、分割输入和 Pose 输出使用同一完整标签。
没有稳定性或对称性后缀协议，也不再自动剥离 `_unstable`。
建议统一使用小写字母开头、数字和下划线分段的标签，例如 `wp_2`、`my_part`；
单 Mesh 加载和复制注册会按此规则校验名称。

## 2. 快速使用：启动 V2 并测试本地数据

### 2.1 运行前提

- 使用已配置好的 ROS、CUDA 和 `fdpose` Python/Conda 环境。
- ROS master 可用，Catkin 消息包 `naviai_manip_srvs` 已编译并加载环境。
- 服务端可访问 Mesh、模型权重、相机内参及本机所需的 CUDA 扩展。
- 外部路径加载需要 `FdposeLoadMesh.srv` 中已有 `string[] mesh_paths` 字段。
- 核心 FDPose 不要求安装分割模块、创建 `seg` 环境或连接相机。

项目尚无完整通用依赖安装脚本，请使用部署环境已有配置。
以下命令均在 FDPose 项目根目录运行；每个终端均需加载 ROS/Catkin 和 Python 环境。
尚未准备本机 TensorRT 产物时，参考后面的加速准备说明。

### 2.2 启动服务

终端一：启动 ROS master，已有 master 时无需重复启动。

```bash
roscore
```

终端二：只启动 FDPose V2。

```bash
conda activate fdpose
python scripts/pose_estimator_track_node_roi_v2.py
```

默认服务：

| 服务 | 名称 |
|---|---|
| Mesh 加载/卸载 | `/fdpose_v2/load_mesh` |
| 位姿估计 | `/fdpose_v2/pose_estimation` |

可用 `rosservice list` 检查服务是否出现。前台启动的节点用 `Ctrl+C` 停止。

#### 可选：使用 pipeline.conf + Bash 后台启停

需要后台运行、记录 PID/日志时，先修改 [config/pipeline.conf](config/pipeline.conf) 中的
容器、项目路径、Conda 环境、ROS setup 和网络地址，再执行：

```bash
# 只启动 FDPose，不要求安装 YOLO；这是前台启动方式的替代，不要重复启动。
bash scripts/manage_services_bash.sh start fdpose
bash scripts/manage_services_bash.sh status fdpose
bash scripts/manage_services_bash.sh stop fdpose

# 需要整套服务时：默认 YOLO + FDPose，深度补全由配置开关控制。
bash scripts/manage_services_bash.sh start

# 使用另一份配置，--config 放在动作之前；stop 时保持相同目标配置。
bash scripts/manage_services_bash.sh --config /absolute/path/pipeline.conf start fdpose
```

`CONTAINER=""` 表示直接在当前 Linux 环境运行；从宿主机进入已有容器时，
填写容器名和容器内的 `FDPOSE_DIR`。未指定配置路径时，脚本按自身所在项目位置查找默认文件，
不受终端当前目录影响；环境变量优先于 .conf，未配置项仍有内置默认值。

完整配置示例、终端环境继承、日志查看及故障排查见 **[pipeline.conf + Bash 使用手册](docs/SERVICE_MANAGER_BASH.md)**。
start/status 仅检查进程存活，不表示模型预热完成。原 [Python 管理版](docs/SERVICE_MANAGER_PYTHON.md)
保留供后续测试，两套工具不要同时启动同一批服务。

### 2.3 不启动 YOLO，直接测试本地 RGB-D 和 Mask

终端三：

```bash
python tools/clients/test_fdpose_client.py
```

默认测试 `wp_2`，读取：

```text
data/samples/current/rgb/0.png
data/samples/current/depth/0.png
data/samples/current/masks/0.png
```

指定自己的数据：

```bash
python tools/clients/test_fdpose_client.py \
  --label wp_2 \
  --rgb /path/to/rgb.png \
  --depth /path/to/depth.png \
  --mask /path/to/mask.png
```

这个 CLI 按注册表加载 Mesh，`--label` 必须已注册；
需要直接加载外部 Mesh 路径时，使用下面的 Python Client 接口。
本地样例仍需要匹配服务端内参，不能只换图片却继续使用不匹配的标定。

公共 Client 与本地测试 CLI 均默认连接 V2，无需再指定服务名。
这个文件是手动集成测试入口，主程序集成请导入公共 `FdposeClient`，不要导入测试脚本。

本地 Service/Client 联调统一使用此入口；旧独立 Demo 和旧 V1 手动测试已删除。
它验证 Mesh 加载与位姿返回是否连通，不自动判断位姿精度；精度验证需要真实标注或人工检查。
配置文件与运行入口的对应关系见 [项目结构与入口说明](docs/PROJECT_STRUCTURE.md#运行入口与配置对应关系)。

### 2.4 从相机话题保存一对本地 RGB-D

[tools/capture_one_pair.py](tools/capture_one_pair.py) 是独立的样例采集工具，
不调用 FDPose、YOLO 或深度补全服务，也不改变 Pipeline 的相机接口。
运行环境需要 ROS、`sensor_msgs`、`message_filters`、`cv_bridge`、NumPy 和 OpenCV；
先启动相机并配置 ROS 网络，在 FDPose 项目根目录执行：

```bash
python tools/capture_one_pair.py
```

默认订阅 `/zj_humanoid/sensor/realsense_head/color/image_raw` 和
`/zj_humanoid/sensor/realsense_head/aligned_depth_to_color/image_raw`，
等待最多 10 秒，收到第一对时间戳完全相同的图像后保存并退出，释放订阅。
使用其他相机时指定话题；时间戳不完全一致时，可显式允许近似同步：

```bash
python tools/capture_one_pair.py \
  --rgb-topic /camera/color/image_raw \
  --depth-topic /camera/aligned_depth_to_color/image_raw \
  --wait 10 \
  --slop 0.03
```

`--slop 0.03` 表示允许约 30 ms 时间差，移动场景需评估时间错位影响；默认 0 为严格同步。
深度必须已由相机驱动对齐到彩色图，工具会校验尺寸，但**不会执行空间对齐**。

默认输出基于脚本所属项目位置，不受终端当前目录影响。每次独立生成样例，不覆盖 current 或旧样例：

```text
data/samples/captures/<UTC时间戳>_<随机标识>/
├── rgb/0.png       # 彩色 PNG，OpenCV 读取为 uint8 BGR
└── depth/0.png     # 原始数值深度 PNG，uint16，单位毫米；不是深度伪彩图
```

可用 `--out-dir /absolute/path/captures` 指定输出根目录，仍会在其下创建独立样例目录；
自定义相对路径以终端当前目录为基准。启动日志会打印保存的完整路径。
`16UC1/mono16` 按毫米原值保存；`32FC1` 按米乘 1000 后舍入为毫米，最多约 0.5 mm 舍入误差，
非有限值和非正值保存为 0。超过 65.535 m 的有效浮点深度会报错，不静默截断。
读取深度务必使用 `cv2.IMREAD_UNCHANGED`；普通看图软件显示偏黑不代表深度值丢失。

工具仅保存这两张图，不生成 Mask、不保存/替换相机标定。
有 YOLO 的完整流程可用采样结果测试（先准备相应服务和注册 Mesh）：

```bash
python tools/clients/test_pipeline.py \
  --rgb /path/to/sample/rgb/0.png \
  --depth /path/to/sample/depth/0.png \
  --labels wp_2
```

直接调用 `test_fdpose_client.py` 时仍需另外提供对应 Mask。
FDPose 服务端内参必须与采集相机、彩色图分辨率一致，不能仅替换图片而沿用不匹配的标定。
抓拍超时、图像转换或保存失败时返回非零退出码；两张图写完才发布样例目录，失败时清理本次临时图像。

## 3. 在主程序中使用 FdposeClient

代码位于 [src/fdpose/ros/client.py](src/fdpose/ros/client.py)。
主程序负责初始化 ROS 节点；Client 不启动服务，不读取相机，不调用分割。

从项目根目录运行自己的程序时，可通过 `PYTHONPATH=src python /path/to/your_app.py`
让 Python 找到 `fdpose` 包。内置启动脚本已处理源码导入路径。

### 3.1 单目标估计

```python
import cv2
import rospy
from fdpose.ros.client import FdposeClient

rospy.init_node("fdpose_app", anonymous=True)  # 主程序已有节点时不要重复初始化
client = FdposeClient()
client.wait_until_ready()  # 可省略，首次服务调用也会等待

client.load_mesh("wp_2")

color = cv2.imread("data/samples/current/rgb/0.png", cv2.IMREAD_COLOR)
depth = cv2.imread("data/samples/current/depth/0.png", cv2.IMREAD_UNCHANGED)
mask = cv2.imread("data/samples/current/masks/0.png", cv2.IMREAD_GRAYSCALE)
if any(image is None for image in (color, depth, mask)):
    raise RuntimeError("图片读取失败，请检查路径")

result = client.estimate_one(
    label="wp_2",
    color_bgr=color,
    depth=depth,
    mask=mask,
)
print(result.label)
print(result.T_object)  # T_camera_object
```

同一 Mesh 通常只需加载一次，后续可以对不同帧重复调用估计接口。

### 3.2 Mesh 加载与独立复制注册

下面各例表示不同使用方式，不需要连续执行：

```python
# 已注册资产：按 label 加载。
client.load_mesh("part")

# 外部路径：直接加载，不复制、不写注册表、不在 Client 记住路径。
client.load_mesh("test_part", mesh_path="/path/to/model.ply")

# 需要长期保存时，用户显式调用复制注册，再单独加载。
saved_path = client.register_mesh("saved_part", "/path/to/model.ply")
client.load_mesh("saved_part")

# 替换同名服务端缓存，不修改磁盘注册表。
client.load_mesh("test_part", mesh_path="/path/to/new.ply", reload=True)

# 批量加载；空路径表示查服务端注册表。
client.load_meshes(
    ["test_part", "saved_part"],
    mesh_paths=["/path/to/new.ply", None],
)

# 只卸载指定标签；不删除文件或注册表。
client.clear_meshes(["test_part"])
# client.clear_meshes()  # 明确需要时才卸载全部
```

- `register_mesh(label, mesh_path) -> str` 只复制单个文件到 `assets/meshes/objects/<label>/<原文件名>` 并写入 [注册表](assets/meshes/registry.yaml)，返回保存路径，不调用服务；无需初始化 ROS 节点。
- 同名 label 已注册就抛出 `ValueError`，即使来源相同也报错；目标目录已存在则抛出 `FileExistsError`。不提供覆盖，不删除旧资产或源文件。
- 临时路径仅用于这次加载。加载成功后，位姿估计只需 label；卸载或服务重启后，未注册的 Mesh 必须再次提供路径。
- V2 对同 label、同来源复用缓存；不同来源返回失败并保留旧缓存。更新时显式传 `reload=True`；同路径内容更新也需要重载，因为没有内容哈希检查。
- `reload=True` 先卸载本次标签再加载；批量接口的 `clear_existing=True` 会先卸载全部。两次请求不是原子事务，应在暂停相关推理时执行；失败不自动恢复旧缓存。
- `clear_meshes()`、`clear_meshes(None)`、`clear_meshes([])` 都表示卸载全部。空白/非字符串标签会报错，不会被过滤成全部卸载。
- 客户端和服务端均需能访问路径；不上传文件。客户端相对路径以当前工作目录为基准，发送前转换为绝对路径。
- 复制不自动处理外部 MTL/纹理依赖，有此类依赖时先用完整源目录下的路径直接加载。注册使用文件锁和原子替换，失败只清理本次未提交的副本；YAML 会重新序列化，不保留注释/排版。
- 不同 Client 连接同一服务端时共享服务端标签空间。临时加载了已有注册标签的其他文件后，可通过 `load_mesh(label, reload=True)` 恢复磁盘注册版本。

旧接口迁移：已移除临时关联、`unregister_mesh()`、`copy_to_registry` 和 `overwrite`。
旧的临时 `register_mesh(label, path)` 应改为 `load_mesh(label, mesh_path=path)`；
现在 `register_mesh()` 明确表示实际复制并写表，不会在加载过程中自动调用。

### 3.3 多目标与可选坐标变换

以下代码复用已创建的 `client`，图像、各目标 Mask 和外参由主程序提供：

```python
from fdpose.ros.client import DetectionInput

results = client.estimate(
    color_bgr=color,
    depth=depth,
    detections=[
        DetectionInput(label="part_a", mask=mask_a),
        DetectionInput(label="part_b", mask=mask_b),
    ],
    # T_target_camera=T_base_camera,  # 可选；两个 Mesh 需先加载
)
for result in results:
    print(result.label, result.T_object)
```

`DetectionInput` 包含 `label`、`mask`、`confidence=1.0` 和可选 `bbox`。
已有分割响应时，可调用 `DetectionInput.from_yolo_item(item)` 转换；
这只是格式适配，不会调用 YOLO。

坐标约定：

```text
T_A_B：将 B 坐标系中的点变换到 A 坐标系
T_target_object = T_target_camera @ T_camera_object
```

`estimate()` 和 `estimate_one()` 都接受 `T_target_camera=None`。
传入 `T_base_camera` 时输出 base 系位姿；不传时输出 camera 系位姿。
Client 不自动查询 TF，服务端可视化始终使用相机坐标系。

### 3.4 公开接口与异常

| 接口 | 功能/返回 |
|---|---|
| `FdposeClient(mesh_service, pose_service, timeout=20.0)` | 创建服务代理封装，默认连接 V2 |
| `wait_until_ready(timeout=None)` | 等待两个服务就绪 |
| `register_mesh(label, mesh_path)` | 显式复制写表，返回绝对路径 |
| `load_mesh(label, mesh_path=None, *, reload=False)` | 加载一个 Mesh |
| `load_meshes(labels, clear_existing=False, mesh_paths=None, *, reload=False)` | 批量加载 |
| `clear_meshes(labels=None)` | 卸载指定或全部缓存 |
| `estimate(color_bgr, depth, detections, T_target_camera=None)` | 返回 `list[PoseResult]` |
| `estimate_one(label, color_bgr, depth, mask, confidence=1.0, bbox=None, T_target_camera=None)` | 返回一个 `PoseResult` |

`timeout` 用于等待服务出现，不是推理执行超时。
本地输入校验可能抛出 `ValueError` / `TypeError` / 文件相关异常。
服务不可用、加载失败或调用失败使用 `FdposeClientError`；
ROS 原始异常保留在异常链中。

`estimate()` 只返回成功目标，全部跳过时可以返回空列表；
`estimate_one()` 没得到对应 label 时抛出异常。Client 在发请求前会整体校验输入，
单个非法 Mask 可能阻止整批发送；服务端收到请求后才执行逐目标容错。

## 4. V2 配置、预热与输出文件

默认读取 [config/fdpose_server_v2.yaml](config/fdpose_server_v2.yaml)，文件内已有中文参数注释。
优先级为：ROS 私有参数 > YAML > 代码默认值。修改配置后需要重启节点，不支持热更新。

主要配置：

| 参数 | 作用 |
|---|---|
| `est_refine_iter` | 位姿细化迭代次数 |
| `use_roi` / `roi_margin` | Mask ROI 裁剪与边缘扩展 |
| `min_mask_pixels` | 最少前景像素 |
| `min_depth_m` / `max_depth_m` | Mask 内有效深度范围，单位米 |
| `min_valid_depth_pixels` / `min_valid_depth_ratio` | 最少有效深度数量及比例 |
| `camera_info_dir` | 标定目录，包含带 `3×3 K` 的 `intrinsic_data.npz` |
| `save_debug_files` / `save_axis_vis` / `save_debug_montage` | raw、单张位姿图、三联图的独立保存开关 |
| `warmup_enabled` / `warmup_dir` | 固定样例前向预热 |
| `mesh_service_name` / `pose_service_name` | 服务名，Client 需与之匹配 |

配置中的相机和预热相对路径以项目根目录为基准。
布尔配置使用 YAML 的 `true/false`，不要写字符串 `"false"`。
缺失配置文件会告警并使用默认值；已有文件的解析错误会中止启动。

临时覆盖配置：

```bash
python scripts/pose_estimator_track_node_roi_v2.py \
  _save_axis_vis:=true \
  _save_debug_files:=false \
  _warmup_enabled:=true
```

预热默认关闭。开启后会在注册 Service 前使用 `assets/warmup/wp_2/` 做一次前向，
随后移除临时 Mesh、保留共享网络及上下文；启动会更久。
预热失败会记录错误并继续启动，不代表预热已经成功。

调试开关相互独立；`save_debug_files` 和 `save_axis_vis` 默认开启，三联图默认关闭：

```yaml
save_debug_files: true
save_axis_vis: true
save_debug_montage: false
```

每次启动创建一个 UTC 时间戳加随机短标识的运行目录，不再使用 `v2/current_request`：

```text
runtime/debug/<启动时间戳>_<短标识>/
├── session.json                  # 版本、配置、完整图内参 K
├── raw/
│   └── 000001/                   # 第一次有效解码的 RGB-D 请求
│       ├── color.png             # 完整图，仅保存一次
│       ├── depth.npy             # 服务收到的 mono16 深度原值（毫米）
│       ├── masks/000.png         # 请求内实例编号，不是 label
│       └── results.json          # 各实例 label、状态、原因、相机系 Pose
└── vis/
    ├── 000001_000_pose_wp_2.png       # 请求 1、实例 0、标签 wp_2，原分辨率位姿图
    └── 000001_000_montage_wp_2.png    # 同一实例的三联图，开启对应开关后生成
```

同名目标通过实例编号区分，不再相互覆盖。原始文件与可视化通过请求编号对应。
可视化文件名为 `<请求编号>_<实例编号>_<pose或montage>_<label>.png`。
label 后缀保留字母、数字、下划线和短横线，其他字符替换为下划线，最长 48 个字符；
这只影响文件名，JSON 和服务中的完整 label 不变。历史图片不会自动重命名。
Pose 矩阵保存在 `results.json` 的各项 `pose` 字段中，不再逐 label 写 `pose.npy`。
成功实例生成可视化；被跳过/失败的实例记录状态，能够正常解码的 Mask 随 raw 保存。
整帧 RGB-D 无法解码或尺寸不匹配时仍返回服务异常，不保存该帧。

三联图为“位姿图 + 原 Mesh 表面点投影 + 输入深度伪彩色图”，统一使用完整图内参和
`T_camera_object`，不转到 base；黄色点是模型投影，不是实测点云，也不做遮挡推断。
深度图展示本次用于估计的深度，固定色条范围沿用 `min_depth_m/max_depth_m`，无效值为黑色。
服务收到的是补全深度时，这里展示补全后的深度，不提供补全前后的对比。
每栏最大宽度 640，保持比例；单张位姿图仍保留原分辨率。

仅开启三联图也可工作；两个可视化都开启时复用一次位姿绘图。关闭三联图时不做采样/投影/着色。
最多 2000 个模型表面采样点在首次三联图时生成并随 Mesh 缓存，卸载/重载后失效；
采样使用局部 RNG，不改变推理的随机状态。无需开启底层 estimator 的 debug。

新建文件权限为 `0644`、目录为 `0755`，JSON 先写临时文件再原子替换，避免读到半写内容；
没有逐文件 fsync，不承诺断电后的持久化。已有父目录权限不自动修改，其他用户读取还需拥有父目录的遍历权限。
raw 的 `record_complete` 表示原始记录是否写完；可视化错误另记在 `errors`，不影响 Pose 返回。
三个开关全关时不创建新的调试目录；关闭开关不会删除历史数据，也没有自动保留数量/磁盘清理策略。

保存同步执行、占用请求锁，磁盘慢时会延长响应时间。PNG 使用低压缩级别以降低编码耗时。
每个请求的 `FDPose DEBUG` 日志分别给出 `raw`（原始数据及 JSON 写入）、`render`（可视化绘制）、
`vis_write`（可视化编码和写入）及 `total`（整个保存过程）毫秒数；首次三联图的采样时间包含在 render 中。
JSON 中 `timing_before_final_json_ms` 是最终 JSON 写入前的计时快照，以日志 total 为完整保存耗时。
若保存开销不可接受，可以独立关闭 raw/单图/三联图；真实 Orin 延迟需实机测量。

V2 对整帧 RGB-D 解码或尺寸错误返回服务异常；单个目标 Mask、深度门限、
ROI 准备或推理失败会记录日志并跳过该目标。调试保存失败不撤销成功的位姿。
Pose 和 Mesh 回调共用互斥锁，保护共享网络、缓存和调试读写；接口仍同步阻塞。

日志中的 `register time` 是调用返回 CPU 位姿的耗时，不是纯 GPU 时间；
`lock_wait` 是等待锁的时间，`total_time` 包含等待、预处理、推理和保存。
正式请求使用 `perf_counter()`，未额外增加 CUDA 全设备同步。

## 5. 通用相对位姿计算

该工具只依赖 NumPy，不启动 ROS，不读取相机或机器人状态，也不管理关系文件：

```python
from fdpose.transforms import compute_relative_pose, apply_relative_pose

# 两个位姿必须处于同一参考坐标系，且对应同一采集时刻。
T_object_target = compute_relative_pose(
    T_reference_object_at_teach,
    T_reference_target_at_teach,
)

# 后续物体移动后，应用同一相对关系。
T_reference_target_now = apply_relative_pose(
    T_reference_object_now,
    T_object_target,
)
```

`target` 可以是另一物体、工具或任意目标坐标系。多个关系由调用方用列表或字典管理，
命名、存储和执行顺序也由调用方决定。原 `fdpose.robot` 导入路径已移除；
原有纯矩阵计算可改用上述两个函数，矩阵乘法约定不变。

## 6. TensorRT 加速准备与验证

仓库保存通用 ONNX 和 TensorRT 扩展源码，不要求 Ubuntu 20.04 与 22.04
共用同一份已编译 Engine。第一次部署，以及 JetPack、TensorRT、CUDA、
Python、PyTorch、ONNX 或插件源码变化后，在目标 Orin 上执行：

```bash
conda activate fdpose
python tools/build/prepare_tensorrt.py
```

该命令自动完成：

1. 检测当前 TensorRT、CUDA、GPU、Python 和 PyTorch 环境；
2. 编译 `native/tensorrt/build/pyTRTEng*.so`；
3. 从 `assets/onnx/` 生成 `assets/engines/` 下的三份 Engine；
4. 使用刚编译的扩展执行三份 Engine，在默认及非默认 CUDA 流上与同权重 PyTorch FP32 对比；
5. 验证通过后安装产物，把环境、文件校验值和数值验证报告写入 `assets/engines/manifest.json`。

环境和输入未变化时会直接复用现有产物。只检查而不修改文件：

```bash
python tools/build/prepare_tensorrt.py --check
```

强制重新生成：

```bash
python tools/build/prepare_tensorrt.py --force
```

准备前请停止使用这些 Engine 的 FDPose 服务。构建或数值验证失败不会安装
staging 产物；安装时遇到可捕获异常会恢复旧文件。此恢复机制不覆盖断电或
`kill -9`，恢复失败会打印保留下来的备份目录。工具会拒绝同时运行第二个构建进程。

验证还需要当前环境能导入 FDPose 模型依赖，并具备原始 checkpoint 和模型配置。
固定合成输入的默认阈值为 `atol=0.02, rtol=0.02`，逐元素检查
`abs(TRT - PyTorch) <= atol + rtol * abs(PyTorch)`。这是 FP16 部署检查阈值，
不代表位姿精度保证；不通过时应检查 ONNX/权重是否一致及误差报告。

单独复查现有 Engine（不重建；可保存 JSON 报告）：

```bash
python native/tensorrt/validate_tensorrt.py \
  --report assets/engines/validation/network.json
```

用固定 RGB-D、Mask 和 Mesh 对比完整位姿推理，无需启动 ROS 或 YOLO：

```bash
python native/tensorrt/compare_pose_backends.py \
  --sample-dir assets/warmup/wp_2 \
  --iterations 3 --repeats 3
```

输出在 `assets/engines/validation/pose_<随机后缀>/`：两个后端各自的
`pose_*.npy`、`vis_*.png`、`report.json`，以及汇总 `comparison.json`。
使用样例内参、完整图像和相同种子，输出始终为 `T_camera_object`。
PyTorch 组沿用预测器默认 AMP，TensorRT 组必须真正启用 TensorRT。
报告区分模型初始化、首轮 register 和后续 register 耗时；可视化保存不计入推理耗时。
这是核心算法的后端对比，不是 V1/V2 Service 测试，也没有真值精度判定；
旋转差异没有消除物体对称性。

Orin 上检查异常输入、重载失败和非默认 CUDA 流：

```bash
python -B -m unittest tests.integration.test_tensorrt_engine -v
```

FDPose 在 TensorRT 扩展或 Engine **加载失败**时仍回退到 PyTorch；
TensorRT **运行中失败**会明确报错，由 Service 记录目标失败，不继续使用无效输出。
详细边界与检查说明见 [TensorRT 验证说明](native/tensorrt/VALIDATION.md)。

## 7. 旧入口与兼容说明

公共 Client 和本地测试 CLI 默认使用 V2；旧整套 Pipeline 启停脚本已清理，Catkin 安装行为未改变。
当前纯 Bash 后台启停入口为 `scripts/manage_services_bash.sh`，读取 `config/pipeline.conf`，
见 [Bash 使用手册](docs/SERVICE_MANAGER_BASH.md)。另一套 `scripts/manage_services.sh` 是保留的
Python 管理入口，使用不同的 YAML 配置，见 [Python 启停文档](docs/SERVICE_MANAGER_PYTHON.md)。
需要 YOLO 的新版流程见 [Pipeline 文档](docs/PIPELINE.md)，本地入口为 `tools/clients/test_pipeline.py`。

| 版本 | 源码启动入口 | Mesh / Pose 服务默认名 |
|---|---|---|
| V2 | `scripts/pose_estimator_track_node_roi_v2.py` | `/fdpose_v2/load_mesh` / `/fdpose_v2/pose_estimation` |
| V1 ROI | `scripts/pose_estimator_track_node_roi.py` | `/fdpose_load_mesh` / `/pose_estimation_service` |

V2 独立于 V1 服务类，共享每个节点内的 Score/Refine 网络和渲染上下文，
增加 YAML 配置、预热、输入门限及 Mesh 来源检查；两者核心估计器仍为
[src/fdpose/core/estimator.py](src/fdpose/core/estimator.py)。

连接已经启动的 V1 时，显式配置公共 Client：

```python
client = FdposeClient(
    mesh_service="/fdpose_load_mesh",
    pose_service="/pose_estimation_service",
)
```

本地测试 CLI 如需连接 V1，也要显式指定旧服务名：

```bash
python tools/clients/test_fdpose_client.py \
  --load-mesh-service /fdpose_load_mesh \
  --pose-service /pose_estimation_service
```

历史 Legacy 服务实现已删除。`CMakeLists.txt` 的安装项暂不修改，因此保留
`scripts/pose_estimator_node.py` 这个薄入口，但它现在会明确提示并转发到 V2，
不再提供 Legacy 行为或旧服务名。建议直接使用 `pose_estimator_track_node_roi_v2.py`。
这不代表 Catkin 安装规则已完整迁移，本文仍按源码目录方式启动。

## 8. 测试

在包含 NumPy 和 PyYAML 的 Python 环境中运行完整单元测试；ROS/GPU 边界使用模拟依赖，
测试数据创建于系统临时目录并自动清理：

```bash
PYTHONPATH=src python -B -m unittest discover -s tests/unit -v
```

`test_client_registration.py` 使用真实 YAML、文件复制和文件锁，验证独立复制注册、
同名拒绝、失败清理、并发注册及无状态路径加载；缺少 PyYAML 时该组测试会跳过。
`test_mesh_cache_safety.py` 串联 Client 与 V2 Mesh 回调，验证空白标签拒绝、同名来源冲突、
显式重载和恢复磁盘版本；只模拟 ROS 消息、几何解析和 GPU，不连接真实服务。

涉及真实 ROS、CUDA 和相机的集成测试需在部署环境中显式运行，
TensorRT 验证说明见 [VALIDATION.md](native/tensorrt/VALIDATION.md)。

## 9. 目录与附加文档

- `src/fdpose/ros/`：Service、公共 Client 和兼容入口。
- `src/fdpose/core/`、`src/fdpose/learning/`：FoundationPose 核心及网络。
- `config/`：配置；`assets/`：Mesh、权重及加速资产；`runtime/`：运行输出。
- [完整目录与代码职责](docs/PROJECT_STRUCTURE.md)。
- [Pipeline 使用说明](docs/PIPELINE.md)：可选的分割、深度补全和流程编排，独立 FDPose 使用无需阅读。
- [pipeline.conf + Bash 使用手册](docs/SERVICE_MANAGER_BASH.md)：配置文件示例、容器内/宿主机运行、启停命令、环境继承、日志和故障排查。
- [Python 管理版启停说明](docs/SERVICE_MANAGER_PYTHON.md)：保留的另一套实现，使用 `config/fdpose_services.yaml`。
- [RGB-D 抓拍工具](tools/capture_one_pair.py)：订阅相机并保存一对本地测试图，使用说明见第 2.4 节。
- [TensorRT 验证说明](native/tensorrt/VALIDATION.md)。
