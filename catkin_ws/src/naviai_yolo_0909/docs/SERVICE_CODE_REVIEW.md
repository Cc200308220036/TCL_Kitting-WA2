# Service code review 路线

审核对象是当前 YOLO 项目唯一的 Service：`instance_segmentation_service`。
Client 与 Server 均使用 `navi_types/InstSeg`，不再区分带 labels / 不带 labels 两套协议。
不涉及 Fdpose 修改。
本文描述现状；“待确认”项不是已经实现的保证。
完整运行命令、Client 方法和配置说明见 [项目 README 的 Service 章节](../README.md#service)。

## 1. 先确定协议，不要从相机或第三方 YOLO 开始读

| 项目 | 统一约定 |
|---|---|
| 服务名 / 类型 | instance_segmentation_service / navi_types/InstSeg |
| Client / Server | seg_service_client.py / seg_service_server.py |
| 请求 | labels、color_image、depth_image |
| 响应 | have_objs、items；不包含输入图像 |
| 类别过滤 | Server 处理，Client 不重复过滤 |
| 空 labels | Client 的 None / [] 均发送 []，Server 保留全部检出类别 |
| 推理方式 | 仅 YOLO；已有 Mask 使用独立本地方法，不经过 Service |

统一 `SegServiceClient` 的协议固定为 InstSeg；原 RGB-D 由调用方保留，不依赖响应回传。
`segment()` 和 `segment_files()` 是同一个 Client 的两个输入方法，不是两个服务模式。
Service 一次请求处理一张图，不订阅相机，不等待目标出现，不提供 Action 的取消机制。
深度不参与分割推理；项目也没有在这条链路中提供位姿估计或 RGB-D 标定/配准。

自定义 `.srv` / `.msg` 定义不在本项目内。最终字段、类型和顺序以部署环境生成的消息为准。

## 2. 按一次请求的顺序阅读

以下文件均相对项目根目录。建议先完整读 Server 的 `inst_seg_callback()`，再按下表逐层展开。
`scripts/` 只是启动包装，业务实现都在 `src/naviai_yolo/`；不必先逐行读第三方运行时。

```text
segment_files() ──读取文件──> segment()
                               │ 校验数组、编码为 ROS Image
                               ▼
                InstSeg 请求（labels + color/depth）
                               ▼
                 SegServiceServer.inst_seg_callback()
                 ├─ 解码并校验 RGB-D（失败抛 ServiceException）
                 └─ NaviaiSeg.predict() → postprocess_segmentation(labels)
                               ▼
                     转换为 ROS DetItem 列表
                               ▼
               InstSegResponse：have_objs + 筛选后的 items
                               ▼
                      Client 直接返回调用方

Server 帧处理由 DebugRecorder.record() 包围，退出时调用 DebugWriter：
分析数据 → records/<sample_id>/；可视化 → visualizations/<sample_id>.jpg
```

| 顺序 | 文件 / 函数 | 这一段必须核对什么 |
|---|---|---|
| ① 调用边界 | `src/naviai_yolo/ros/seg_service_client.py`：`segment_files`、`segment` | 两条入口是否最终走同一校验/RPC；错误输入是否在发请求前失败；labels 是否作为请求字段发送 |
| ② 初始化 | `src/naviai_yolo/ros/seg_service_server.py`：`__init__`；`src/naviai_yolo/configuration.py` | conf/imgsz 是否在模型加载前校验；服务注册前模型/保存组件是否就绪；旧 Mask 参数是否明确拒绝；Client 与 Server 是否同名同类型 |
| ③ 请求处理 | `src/naviai_yolo/ros/seg_service_server.py`：`inst_seg_callback`；`src/naviai_yolo/ros/image_io.py` | BGR/深度解码校验、服务端 labels 筛选、ServiceException 与合法空结果是否区分 |
| ④ 模型边界 | `src/naviai_yolo/segmentation.py`：`NaviaiSeg.predict` | 只输出 NumPy boxes/masks/names；禁止 YOLO 自带保存；有框无 Mask 必须报错 |
| ⑤ 后处理 | `src/naviai_yolo/postprocess.py` | 类别映射、letterbox 去边、恢复原图大小、二值化、空 Mask 剔除、bbox 与实例对应 |
| ⑥ ROS 输出 | Server 回调中的 DetItem 转换与 InstSegResponse 构造 | label/confidence/bbox 与后处理一一对应；Mask 是原图大小的 mono16，前景 65535；没有假定响应带 RGB-D |
| ⑦ 保存旁路 | `src/naviai_yolo/ros/debug_recording.py`、`src/naviai_yolo/debug_writer.py` | 保存只在 Server；关闭时无额外深度解码；写盘失败不改变分割返回；分析数据能否配对回放 |

### 输入与输出的明确约定

- color：非空 H×W×3、uint8、BGR 顺序。来自其他 SDK 的 RGB 数组需由调用方转换。
- depth：与 color 同 H×W，uint16 按毫米、float32 按米；代码不换算单位，也不验证几何对齐。
- Client 的 depth 省略时补全零 uint16 图，只是协议占位，不能当作有效测距结果。直接 RPC 必须传合法 depth。
- Server 在推理前解码并验证 RGB-D，关闭 debug 也执行；接受 16UC1/32FC1，不因深度含 NaN/Inf 而改变数值。
- 文件深度支持原值图像或 `.npy`；禁止 pickle，对象数组、损坏文件、伪装 NPZ 会被拒绝。
- Client 把数组重新编码为 ROS Image，没有继承采集时的 header/stamp；若需要相机原始时间戳，当前数组 API 不提供该参数。
- YOLO bbox 使用模型输出的原图坐标并转为整数，不根据最终 Mask 重算，也未额外裁剪到图像边界。
- 内部 Mask 是 uint8 的 0/1；ROS Mask 是 mono16 的 0/65535；分析 PNG 是 uint8 的 0/255。

本地读取不属于上述 Service 协议：`src/naviai_yolo/mask_io.py` 的 `load_mask_file()`
返回 SegDetection 列表；单通道非零区域作为一个实例，全黑返回 []，错误抛 ValueError。
它不初始化 ROS/模型、不写 debug，也不会替代 `segment_files()` 的 RGB-D 推理调用。

### 保存逻辑要单独审核

`run_id + sample_id` 唯一定位一个样本；同一 Server 启动周期内 sample_id 递增。
Service 的 request_id 为 null，不能凭客户端的文件名推导 sample_id。
源图保存的是解码后的 BGR PNG，而不是原始 ROS 消息字节；深度 `.npy` 保留解码后的 dtype/数值。
Server 保存的检测结果就是按请求 labels 筛选后的结果；labels=[] 才保存全部检出类别。

`status` 表示分割处理状态；`record_complete` 表示本次分析记录是否按计划写完，两者不同。
即使 record_complete=true，深度也可能因解码失败而缺失，必须同时检查 `files` 和 `depth.error`。
JSON 本次已改为同目录临时文件 + 原子替换；文件序列化/写入失败不会截断已有 JSON。
这不包含 PNG/NPY 的事务保证，也不代表断电后必然保留全部文件。

## 3. 逐项检查失败行为

| 场景 | 当前实际行为 | Review 判断 |
|---|---|---|
| Client 文件不存在、数组类型/尺寸不合法 | 抛 ValueError，不发出分割 RPC | 上层是否捕获并给出清晰错误 |
| 服务不可达、传输失败 | ROS 异常传播给调用方 | 不应伪装成未检出 |
| 正常推理未检出、二值化后全空 | have_objs=false，items=[] | 合法业务结果 |
| 服务端解码/推理/后处理异常 | 记录日志，抛 rospy.ServiceException；保存开启时记录 error 样本 | Client 必须捕获异常，不作为合法空结果 |
| 绕开 Client，直接发送异常 depth | 推理前拒绝：尺寸/编码/dtype 不合法或无法解码均调用失败 | 校验格式不等于验证几何对齐或测距有效性 |
| 错误样本的 debug 深度解码失败 | metadata.depth.error 记录失败，不覆盖原始处理异常 | 回放方必须同时检查 status/files/depth.error |
| debug 写盘失败 | 告警；分割结果不因写盘失败而变成空结果 | 已有 meta 不应被最终更新写坏；失败样本不要直接用于分析 |

## 4. 对照测试，不只看实现

在具备 NumPy、OpenCV、PyYAML 的 Python 环境中，从项目根运行：

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

当前共 103 项自动测试。ROS 通信、CvBridge、模型推理使用替身，图像处理和文件读写使用真实库。
InstSeg 请求/响应替身限制了字段，响应仅允许 have_objs/items，避免把旧 RGB-D 响应当作新协议。
它们验证接口适配逻辑，不证明真实 ROS 消息序列化、真实权重输出或 GPU 并发安全。

| 审核主题 | 建议优先看的测试 |
|---|---|
| 两个 Client 方法同一行为 | `test_service_client_files_and_arrays_return_same_mask_result` |
| Service 集成模板的参数、结果和异常处理 | `test_service_example_passes_parameters_and_decodes_masks`、`test_service_example_defaults_and_no_detections_are_successful`、`test_service_example_reports_missing_parameters_and_call_errors` |
| Client/Server 协议配对 | `test_service_client_uses_same_name_and_protocol_as_server` |
| labels 的真实位置 | `test_service_client_files_send_labels_and_return_server_response`、`test_service_client_empty_labels_keep_all_and_no_match_clears_have_objs` |
| 输入错误不发 RPC | `test_service_client_invalid_inputs_are_rejected_before_rpc`、`test_service_client_npy_rejects_invalid_arrays_before_rpc` |
| 服务注册前完成初始化 | `test_yolo_model_and_recorder_are_ready_before_service_registration` |
| 不再提供 Mask 服务模式 | `test_removed_mask_service_parameters_fail_before_model_loading` |
| 本地 Mask 读取 | `tests/test_mask_io.py`（无 ROS/模型导入、有效/空/无效文件、尺寸与 bbox） |
| 原图尺寸和空 Mask | `tests/test_postprocess.py`、`test_services_drop_empty_masks_and_record_no_detections` |
| 服务端输入校验 | `test_service_rejects_invalid_rgbd_before_inference_with_debug_off` |
| debug 异常隔离/复用深度 | `test_invalid_depth_raises_but_debug_disk_errors_do_not_change_response`、`test_service_recording_reuses_validated_depth` |
| 奇数尺寸去边 | `test_real_letterbox_roundtrip_preserves_odd_size_edges`（`tests/test_postprocess.py`，直接使用内置 LetterBox 类） |
| 非法配置启动即失败 | `test_invalid_model_config_fails_before_loading_or_registering` |
| ROS 包名与安装路径 | `test_ros_package_names_are_consistent`、`test_installed_resource_lookup_uses_new_ros_package_name`（`tests/test_runtime.py`） |
| JSON 最终更新失败 | `test_final_metadata_write_failure_keeps_incomplete_marker_and_visualization`（`tests/test_debug_writer.py`） |
| 保存数据可回放 | `test_service_client_replays_server_debug_record` |
| Service 过滤和异常 | `test_plain_service_filters_labels_and_handles_prediction_error` |
| 空 labels 返回全部、指定类别和无匹配 | `test_plain_service_empty_labels_keep_all_and_nonempty_labels_filter` |

除表中注明独立文件者，测试均在 `tests/test_ros_interfaces.py`。
例如单独检查回放链路：

```bash
PYTHONPATH=src:tests python -B -m unittest test_ros_interfaces.ROSInterfaceTests.test_service_client_replays_server_debug_record -v
```

## 5. 最后进行真实 ROS 验收

先 source 正确的 ROS / catkin 环境，确认 model.yaml 指向真实分割权重。
以下 `/absolute/path/...` 需替换成自己准备的对齐数据；不要同时启动两个同名服务端。

```bash
# 终端 A：YOLO 推理服务
python scripts/seg_service/seg_service_server.py

# 终端 B：核对真实协议，然后调用
rosservice type /instance_segmentation_service
rossrv show navi_types/InstSeg
rosmsg show navi_types/DetItem
python scripts/seg_service/seg_service_client.py --color /absolute/path/color.png --depth /absolute/path/depth.npy --labels wp_2
```

复用同一 Client 命令，分别检查保留全部类别、命中筛选、未命中筛选，以及 debug 数据回放。
配置修改需重启 Server。用日志中的 Debug output 找到本次运行目录，不猜 sample_id 对应哪个请求。
省略 --labels 时应返回全部，指定不存在的类别时应返回 have_objs=false、items=[]。
用错误尺寸深度和模拟推理异常验证客户端能收到 ServiceException；
正常未检出仍须返回合法空结果。此项需要真实 ROS 验收，替身测试不覆盖 TCPROS 序列化。

### 本轮 Review 的待确认项（没有顺手修改）

- [ ] 上层是否已捕获 ServiceException？内部处理失败现在通过 ROS 异常传播，不能再依赖空结果。
- [ ] Mask header 是否必须继承 color header？目前未提供；输入的格式校验也不保证几何对齐。
- [ ] 是否需要单次 RPC 超时？Client 的 wait_timeout_sec 仅等待服务出现，不限制一次请求处理时长。
- [ ] 真实模型的首次调用、并发请求、长时间运行是否通过设备测试？第三方推理锁不能替代整条服务链路的并发验收。
- [ ] 同步写盘带来的延迟和磁盘增长是否可接受？目前没有配额、自动清理或异步保存。

本次仅保留带 labels 的 InstSeg，移除无 labels 的 ManipInstSeg 实现、入口及依赖，
Mask 文件读取已独立为本地 load_mask_file() 方法，不再提供 Service 分支。
Client 的数组/文件两个方法保持不变，均返回 InstSegResponse。
Fdpose 暂未迁移：旧 Service 类型、服务名、启动路径和依赖响应 RGB-D 的代码都需要分别检查。
已有 catkin 安装空间可能保留旧脚本，升级时需清理对应安装残留；本次没有操作外部部署环境。
本轮另外完善了 YOLO 的配置校验、Action Client 结果/取消等待的墙钟超时和参数校验，
以及抓拍写入检查与浮点深度保存。Action 启动连接仍依赖 actionlib 的 ROS 时间等待。
Fdpose 文件未修改，留待 fdpose_pipeline 阶段迁移。
