# transport_tray_project_v2 使用说明

本项目控制 WA2 机器人完成托盘的视觉定位、双臂抓取、抬升转腰和逐层堆叠。`main.py` 是连续搬运入口，支持 1～3 层；四个 `test_stage*.py` 用于分阶段调试。以下说明以当前代码和默认配置为准。

## 1. 快速运行

先加载现场已验证的 ROS、机器人和视觉运行环境，再进入本目录：

```bash
cd catkin_ws/src/transport_tray/transport_tray_project_v2

# 只打印计划，不执行运动
python3 main.py

# 连续搬运三层：共同Z寻槽、方形路线；抓取后分别抬升1、1、7cm
python3 main.py --execute --layers 3 --search-path square --lift-cm 1 1 7
```

运行过程中按终端提示确认动作和视觉采集。对 `main.py` 和 V2 四个独立入口，`--execute` 才会进入实机流程；无该参数或使用 `--help` 时不导入 ROS、不发送运动。程序从第一层开始执行，`--layers 2` 表示搬运第一、二层，不表示从第二层恢复。

| 常用参数 | 用法与默认值 |
|---|---|
| `--layers` | 总层数：`1`、`2`、`3`；默认 `3`。 |
| `--lift-cm L1 L2 L3` | 三层抓取后的抬升高度，单位 cm；默认 `1 1 7`，即使只运行一层也要提供三个值。 |
| `--search-path` | `square` 方形回字路线、`circle` 同心圆、`spiral` 螺旋；主程序默认 `square`。 |
| `--xy-offset X Y` | 到 P_pre 后双臂共同 XY 偏置，单位 m；默认 `0 0`。 |
| `--config` | 配置 YAML 路径；默认 `tray_control/config/transport_tray.yaml`。 |
| `--state-file` | 共享状态 JSON 路径；默认 `tray_control/runtime/tray_state.json`。 |

多层 XY 寻槽已固定使用共同 Z，不需要也不支持 `--search-z-mode`。更换路线只需更改 `--search-path`。

运行依赖包括 ROS 的 `rospy`、`geometry_msgs`、`rosservice`、TF/RGB-D 数据，`naviai_controller`、Jodell 夹爪服务，以及外部 `tray_pose/scripts`、`Fdpose` 工程与模型。数值依赖见 [requirements.txt](requirements.txt)。本目录不包含完整机器人驱动和视觉部署环境。

视觉适配器默认从所在工作空间的 `catkin_ws/src` 查找视觉工程；移动项目后可显式设置：

```bash
export TRAY_CATKIN_SRC=/实际工作空间/catkin_ws/src
```

更换机器人或相机时需核对夹爪服务、抓取标定、相机外参和 `main.py` 中返回取料的关节目标。

## 2. Stage0 抓取标定

标定保存“参考托盘位姿”和“人工对准预抓取点时的左右 TCP 位姿”。Stage0 据此计算新托盘位置对应的左右预抓取目标，跟随托盘平移和 Yaw 变化。它与 Stage1 的腕力零偏采集是两回事，也不会重新计算相机外参。

标定入口为 [tray_control/stage0_calibrate.py](tray_control/stage0_calibrate.py)，由原 `test_stage0.py` 迁移，使用 V2 的视觉、配置和标定保存接口，不再依赖上级搬运脚本。该工具不发送运动。

### 标定步骤与命令

1. 加载 ROS、机器人和视觉环境，将托盘放在稳定支撑面上；采图时保持头部、腰部和托盘静止。
2. 在 V2 目录执行以下命令，程序先识别参考托盘，再提示人工对准夹爪。该命令只读取位姿并保存标定，不发送运动，**不要添加 `--execute`**。

```bash
cd catkin_ws/src/transport_tray/transport_tray_project_v2
python3 tray_control/stage0_calibrate.py --calibrate
```

3. 看到参考托盘识别结果后，保持托盘不动，用现场示教方式将左右**实际夹爪末端中心**分别对准托盘两侧中心的预抓取点，并调整好夹爪姿态。无需把左右电机 TCP 坐标调成相同高度或对称值。
4. 在提示处按 Enter，程序读取双 TCP 并保存 JSON；输入其他内容或 Ctrl+C 取消。指定文件会被覆盖，需要保留旧标定时先自行备份。

| 保存内容 | 含义 |
|---|---|
| `reference_tray_xyzquat` | 视觉识别的参考托盘 BASE 位姿。 |
| `reference_tcp_xyzquats.left/right` | 人工对准后的左右电机 TCP BASE 位姿。 |
| `vision_backend` / `vision_label` | 采集时的视觉后端和托盘类别，需与 YAML 一致。 |

不加参数仅打印计划，`--help` 查看选项，均不导入 ROS。默认读取 V2 YAML 的视觉参数、相机外参和标定保存路径；可用 `--config`、`--calibration-file` 等显式覆盖。覆盖类别或外参后，实际搬运的 YAML 也应同步；脚本不会自动修改 YAML。

位姿格式均为 `[x, y, z, qx, qy, qz, qw]`，位置单位 m。YAML 的 `common.paths.calibration_file` 应指向上述文件；`--tray-pose-camera-to-head` 应与 `common.vision.tray_pose_camera_to_head` 使用同一份相机外参。

### 标定后核对

先只计算并打印预抓取目标，不执行 MoveL；此预览会读取视觉，并默认写入独立的 `tray_control/runtime/stage0_preview.json`（可通过 `--state-file` 覆盖），不改变主流程状态：

```bash
python3 tray_control/stage0_calibrate.py --preview
```

核对左右目标位置和姿态后，再按第 1 节启动 V2 主流程，或使用 `test_stage0_stage1_pose_record.py --execute` 验证抓取；后者还会闭爪、抬升，不能当作只做标定的命令。更换夹爪安装、预抓取姿态、托盘类型或相机外参后，应重新核对并按需重标定。

## 3. 当前搬运逻辑

```text
每层：Stage0 视觉预抓取 → Stage1 导纳靠拢并闭爪 → 同步抬升 → Stage2 三步转腰
      ├─ 第一层：预下降5cm → 接触放置 → 张爪撤离 → 视觉记录首层
      └─ 第二/三层：计算并到达P_pre → 接触找平 → 直接落座或XY寻槽
                    → 必要时回最低点并Yaw搜索 → 张爪撤离 → 视觉记录新顶层
层间：固定图像和相机变换 → 顶面计算与返回取料并行 → 等待记录成功 → 下一层
```

| 阶段 | 主要行为 |
|---|---|
| Stage0 预抓取 | 识别托盘，读取抓取标定，对齐托盘的 180° 对称姿态，计算并移动到左右 TCP 预抓取目标。 |
| Stage1 抓取 | 沿预抓取双 TCP 的水平连线向内导纳靠拢；目标力满足稳定条件后并行闭爪，闭爪期间继续导纳；保存抓取变换。 |
| 抓取后抬升 | 双臂同步沿 BASE +Z 执行相对 MoveL，不记录力。主程序显式传入每层抬升高度。 |
| Stage2 转腰 | 同步 MoveJ 依次修改 `q4=0.2 → q3=1.5 → q4=0.5 rad`，速度 `0.5`、加速度 `0.1`。 |
| 首层放置 | 主程序先同步下降 5cm，再运行接触放置；独立单层测试没有这段 5cm 预下降。 |
| 多层预放置 | 默认用保存的抓取变换和当前双 TCP 反算持物托盘；根据上一层顶面规划 P_place/P_pre，先 XY、后 Z 同步 MoveL，再加共同 XY 偏置。当前托盘低于 P_pre 时只跳过 Z 段。 |

### Stage3 的接触、寻槽和释放

- **单层和多层初始接触共用 `update_contact_descent()`**：双侧共同导纳下降，任一侧达到目标力即锁存；先侧上退 **0.5mm** 后保持，另一侧继续下降；两侧都达到过目标力后冻结。先侧卸载不会解除锁存。
- **单层**：双侧锁存后先退出伺服，再自动张爪。
- **多层**：双侧锁存后保持原 0.3s 等待；托盘中心距 P_place 小于 1mm 且左右下降差不超过 5mm 时进入直接落座，否则进入 XY 寻槽。
- **XY 寻槽**：保持入口左右高度差，施加相同 Z 增量；Z 使用持物基线，不二次归零。任一侧连续 3 周期超出 1.3N 时暂停 XY、共同上退 0.7mm，再恢复下降；较大侧降到 1.1N 及以下后允许下一轮退让。
- **捕获条件**：相对寻槽起点下沉达到 3mm，或任一侧相对寻槽起点的 `|ΔFx|`/`|ΔFy|` 超阈值即释放；默认第二层 2N、第三层 3.5N。路线走完或搜索超时则回历史最低下沉点的 XY/姿态，再进行 Yaw 搜索。
- **Yaw 搜索**：共同 Z 导纳下依次扫描 `0 → +3° → -3° → 0`。任一侧相对持物基线的 `|ΔFy|>2.5N`，或满足新增下沉及去抖条件时释放；Yaw 扫描耗尽或超时仍按“未确认入槽”自动释放。
- **多层张爪时继续保持伺服指令**，张爪稳定后退出伺服。单层、多层共用撤离顺序：下探 1.3mm → 沿双 TCP 水平连线各外撤 20mm → 同步上抬 30mm。

初始接触的 0.5mm 与 XY 寻槽的 0.7mm 是两种独立退让。原恒力找平实现 `SwitchingZAdmittance` 仍保留，但当前初始接触没有使用它；暂未提供配置切换开关。释放或视觉记录成功表示程序完成对应步骤，不单独证明机械入槽。

## 4. 主要调用接口

业务入口在 [tray_control/api.py](tray_control/api.py)。`SingleLayerTray`、`MultiLayerTray` 创建时传入 `confirm` 回调，可选传入 `before_vision`、配置和状态路径；调用前由调用方初始化一次 ROS 节点。API 会执行动作，不提供 CLI 的计划模式。

| 对外接口 | 功能与调用关系 |
|---|---|
| `tray.prep(speed=..., acceleration=...)` | 调用 Stage0 规划并到达预抓取位。 |
| `tray.grip(free_speed=..., contact_speed=..., acceleration=...)` | 调用 Stage1 完成靠拢闭爪并保存抓取状态；需要预抓取完成。 |
| `tray.lift(height_m=..., speed=..., acceleration=..., robot=...)` | 调用 `stage2.lift_tray()` 同步抬升；需要抓取成功。 |
| `stage2.turn_waist(robot)` | 三步转腰；`main.py` 在抬升后直接调用。 |
| `SingleLayerTray.place(robot=..., initial_lower_distance=...)` | 单层放置、撤离、首层记录；主程序额外传 `initial_lower_distance=0.05`。 |
| `MultiLayerTray.move_to_pre(robot=..., xy_offset=...)` | 持物定位、规划并执行 P_pre；需要有效上一层顶面和抓取变换。 |
| `MultiLayerTray.place(search_path="square", ...)` | 运行多层 Stage3、撤离并记录新顶层；需要 `move_to_pre()` 成功。**直接调用 API 时路线默认是 `spiral`，需显式指定 `square`。** |

API 返回 `StageResult(step, completed, state_path, measurements, targets)`；执行失败会抛出异常并记录失败状态。状态文件传递抓取变换、上一层顶面和阶段完成标记，不能代替机器人实际起点。

| 阶段内部关键接口 | 职责 |
|---|---|
| `stage0.save_calibration()` / `plan_pregrasp()` / `move_dual_to_pregrasp()` | 标定保存 / 预抓取规划 / 执行。 |
| `TrayApproach.prepare_grippers() → wait_ready() → calibrate_bias() → run_approach()` | 夹爪准备、反馈就绪、无接触力基线、导纳抓取。 |
| `stage2.capture_held_tray() → plan_preplacement() → move_to_pre()` | 获取持物状态、生成 P_place/P_pre 目标、执行平移。 |
| `update_contact_descent()` | 共用初始接触锁存、先侧定距上退和另一侧下降。 |
| `SingleStage3Place.run()` | 单层基线、接触放置、张爪和撤离。 |
| `MultilayerStage3.run_experiment()` | 多层准备、控制循环、撤离及结果保存。 |
| `read_cycle() → step_stage() → publish_cycle()` | 多层每周期读取一次反馈、判断阶段、发布一次双臂指令；CSV 在阶段判断前记录。 |
| `step_initial_descent()` / `step_xy_search()` / `step_yaw_search()` | 初始接触交接、XY 捕获与路径推进、Yaw 捕获与扫描。 |
| `finish_search_and_release()` / `Stage3Place.retreat()` | 统一多层搜索结果及张爪入口 / 公共三段撤离。 |

## 5. 文件职责

```text
tray_control/
├── api.py / config.py              # 业务接口（含StageResult）、配置
├── stage*.py                      # 阶段控制，仍保留在包根目录
├── common/                        # 运行支撑、反馈、夹爪、状态、视觉
├── algorithms/                    # 导纳模型、寻槽路线
├── observability/                 # CSV、事件、计时、终端结果
└── config/ / calibration/ / assets/ / runtime/  # 配置、资源与运行状态
```

三个子包的 `__init__.py` 仅作包标识，不加载额外控制逻辑。配置、资源、状态和日志目录保持原位置；主程序及四个独立脚本仍在项目根目录。

### 入口与业务控制

| 文件 | 一句话说明 |
|---|---|
| [main.py](main.py) | 层数循环、阶段顺序、层间返回和顶面记录并行调度。 |
| [tray_control/stage0_calibrate.py](tray_control/stage0_calibrate.py) | 采集Stage0参考托盘和双TCP，或预览预抓取目标；不发送运动。 |
| [terminal.py](terminal.py) | 人工动作确认和视觉准备确认。 |
| [tray_control/__init__.py](tray_control/__init__.py) | 导出单层、多层业务类及执行结果类型。 |
| [tray_control/api.py](tray_control/api.py) | 业务接口、StageResult结果类型、阶段前置条件、状态更新及控制器生命周期。 |
| [tray_control/config.py](tray_control/config.py) | 读取和校验 YAML，将配置及接口覆盖值转换为各阶段参数。 |
| [tray_control/stage0.py](tray_control/stage0.py) | 视觉抓取规划、标定对齐和预抓取运动。 |
| [tray_control/stage1.py](tray_control/stage1.py) | 沿 TCP 连线的导纳靠拢、接触判断和并行闭爪。 |
| [tray_control/stage2.py](tray_control/stage2.py) | 同步抬升、三步转腰、持物定位及 P_pre 规划执行。 |
| [tray_control/stage3.py](tray_control/stage3.py) | 共用接触找平函数和单层放置控制循环。 |
| [tray_control/stage3_multi.py](tray_control/stage3_multi.py) | 多层接触、XY/Yaw 寻槽、释放状态机及顶面记录。 |
| [tray_control/stage3_common.py](tray_control/stage3_common.py) | Stage3 公共反馈、原恒力模型、伺服、到位等待、撤离和清理。 |

### 公共支撑、算法与观测

| 文件 | 一句话说明 |
|---|---|
| [tray_control/common/runtime.py](tray_control/common/runtime.py) | 操作互斥、确认回调、控制器退出和腕力订阅清理。 |
| [tray_control/common/feedback.py](tray_control/common/feedback.py) | 腕力缓存、滤波、阈值锁存、反馈等待及基线采样。 |
| [tray_control/common/gripper.py](tray_control/common/gripper.py) | 双夹爪 ROS 服务、并行任务和完成确认。 |
| [tray_control/common/state.py](tray_control/common/state.py) | 位姿校验、抓取变换、双 TCP 反算融合、目标计算和 JSON 状态读写。 |
| [tray_control/common/vision.py](tray_control/common/vision.py) | YOLO 托盘位姿适配、识别重试、固定帧后台处理及首层记录。 |
| [tray_control/algorithms/moving_z_admittance.py](tray_control/algorithms/moving_z_admittance.py) | 移动平衡点 Z 导纳和定距退让数学模型。 |
| [tray_control/algorithms/search_path.py](tray_control/algorithms/search_path.py) | 方形、同心圆、螺旋路线生成和搜索时间预算。 |
| [tray_control/observability/stage3_logging.py](tray_control/observability/stage3_logging.py) | 多层 CSV 字段及采样组装。 |
| [tray_control/observability/force_recording.py](tray_control/observability/force_recording.py) | CSV缓冲写入、记录器激活/注销及阶段事件转发。 |
| [tray_control/observability/timing.py](tray_control/observability/timing.py) | 控制周期诊断、阶段计时、按层汇总及人工等待区分。 |
| [tray_control/observability/pose_reporting.py](tray_control/observability/pose_reporting.py) | BASE 下 XYZ、欧拉角和放置高度差的终端输出。 |

### 独立调试入口

四个脚本均可先不加 `--execute` 查看计划，实机执行时需处于该阶段的实际起点；不能直接将四条命令当成自动连续流程。

| 文件 / 执行命令 | 功能及与主程序的区别 |
|---|---|
| `python3 test_stage0_stage1_pose_record.py --execute` | 预抓取、抓取、抬升 1cm 并记录视觉/TCP 对照；不转腰或张爪，默认使用本轮独立状态文件。 |
| `python3 test_stage2_waist_three_steps.py --execute` | 仅执行三步转腰，速度/加速度由脚本顶部常量传入。 |
| `python3 test_stage3_single_z_place.py --execute` | 从当前悬空夹持位置直接做单层放置并记录首层，不执行主流程的 5cm 预下降。 |
| `python3 test_stage3_multi_z_search_place.py --execute --search-path square` | 到 P_pre 后执行多层放置；**当前脚本显式使用视觉定位持物托盘**，主流程则默认按 YAML 使用双 TCP 反算。 |

抓取测试与后续阶段衔接时，显式传入相同 `--state-file`；多层还必须已有有效上一层顶面记录。

### 离线测试文件（`tests/`）

| 文件 | 验证内容 |
|---|---|
| `test_v2_runtime.py` | 模拟机器人和视觉边界，验证业务接口、状态及阶段调用链。 |
| `test_single_contact_descent.py` | 初始接触锁存、定距上退及下降速度。 |
| `test_common_search_z.py` | 共同 Z 导纳、力基线和超力退让。 |
| `test_stage3_simplified.py` | 多层阶段分支和状态切换。 |
| `test_yaw_fy_capture.py` | Yaw 力触发及释放判据。 |
| `test_search_path.py` | 三种寻槽路线和时间预算。 |
| `test_stage3_multi_test_options.py` | 多层独立入口的参数传递。 |
| `test_main_record_top.py` | 采图、后台顶面记录和返回取料的衔接。 |
| `test_stage3_reporting.py` | 位姿打印、CSV 字段和事件。 |
| `test_timing.py` | 状态计时及汇总。 |
| `stage3_probe.py` | 供测试提取并运行生产控制方法，不连接硬件。 |

## 6. 参数在哪里改

配置文件为 [tray_control/config/transport_tray.yaml](tray_control/config/transport_tray.yaml)。距离/速度/加速度默认使用 m、m/s、m/s²，角度按字段注明的 rad 或 deg；夹爪开度使用 mm。YAML 中的相对资源路径以 `tray_control/` 为基准。

| 内容 | 修改位置 / 当前值 |
|---|---|
| 视觉、外参、夹爪服务、力参数 | YAML 的 `common.vision`、`common.stage1`、`common.stage3`。 |
| 主流程抬升高度 | CLI `--lift-cm`，默认 `1 1 7`cm；优先于 YAML 的 `lift_heights_by_layer`。 |
| 初始放置下降速度/加速度 | `common.stage3.descent_motion`；单层 6mm/s、多层 3mm/s，均为 20mm/s²。当前共用接触函数使用 `speed`，多层 `contact_speed` 不参与该初始找平计算。 |
| 初始接触退让距离 | `stage3.py` 的 `CONTACT_RETREAT_DISTANCE_M=0.0005`，单层、多层的运动、参数和日志共同引用，改这一处即可。 |
| 堆叠高度和预放置间隙 | `multi_layer.preplace`；`stack_pitch=33mm`、`preplace_clearance=20mm`，P_place.Z=上一层顶面.Z+层高，P_pre.Z=P_place.Z+间隙。 |
| 持物定位方式 | `held_tray_pose_from_tcp: true` 为双 TCP 反算，`false` 为视觉；独立多层测试显式覆盖为视觉。 |
| 路线尺寸、捕获力、共同 Z 参数 | `multi_layer.search`；方形最大半径 10mm、方环间距 2mm，第二/三层 XY 力阈值为 2/3.5N。 |
| XY 速度 | `config.py` 的 `MOTION_DEFAULTS["place_multi"]["xy_speed"]`，默认 2mm/s；API 可传 `xy_speed` 覆盖。 |
| 撤离距离/速度/加速度 | `stage3_common.py` 的 `Stage3Place.retreat()` 内 `steps`，三段参数集中列出。 |
| 转腰 / 返回取料目标 | `stage2.py` 的 `WAIST_*` / `main.py` 的 `restore_initial_position()`。 |

## 7. 状态、记录与补记

- **状态**：`tray_control/runtime/tray_state.json` 保存顶层位姿、层号、抓取关系和阶段结果。重新抓取保留上一层顶面；首层重新记录会重建状态并清除旧多层任务，不生成备份。
- **日志**：多层力与事件 CSV 位于 `logs/stage3/`，路径也写入状态文件；单层只统计时间，不生成力 CSV。多层分析曲线用 `row_type=sample`、横轴 `elapsed_s`，状态事件使用同一时间轴。Z 支撑力和 Yaw 相对力使用持物基线，XY 捕获力使用寻槽起点基线。
- **计时**：主程序退出时按层汇总耗时，人工确认时间单列；CSV 保留真实时间轴。非末层先固定 RGB-D 与相机变换，再并行计算顶面、返回取料；记录成功后才开始下一层。
- **目录**：`pose_record_runs/` 保存抓取对照实验；`analysis/`、`results/` 为分析和结果资料；`tray_control/calibration/`、`assets/` 保存抓取标定和相机外参。`EXTRACTION_MANIFEST.json` 是早期提取快照，不代表当前代码校验值。

多层已撤离但视觉失败时，只补记顶面，不重新运动：

```bash
python3 main.py --execute --record-top-after-place
# 使用自定义状态文件时添加相同的 --state-file 路径
```

该操作要求状态中已标记多层释放撤离完成，沿用原目标层号。若现场只有已放好的第一层且双臂已撤离，可重新建立首层记录：

```bash
python3 test_stage3_single_z_place.py --execute --record-top-only
```

## 8. 离线检查

```bash
python3 -m unittest discover -s tests -q
python3 -m compileall -q tray_control main.py
```

测试模拟硬件和视觉接口，不连接机器人；通过离线测试不能代替实机验证。
