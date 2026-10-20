# Python 服务进程管理器

[返回 README](../README.md) · [Pipeline 使用](PIPELINE.md)

同一容器内固定服务现在可先使用 [独立纯 Bash 版](SERVICE_MANAGER_BASH.md)。
本文保留 Python 版配置及用法供后续验证；两套入口不要同时启动同一批服务。

入口为 [scripts/manage_services.sh](../scripts/manage_services.sh)，支持 `start`、`stop`、`status`。
它只管理后台服务进程，不调用 Pipeline、不读取相机，也不改变模型或位姿估计逻辑。
具体进程管理由 [tools/manage_services.py](../tools/manage_services.py) 实现。

## 1. 使用

先按实际机器修改 [config/fdpose_services.yaml](../config/fdpose_services.yaml) 中的容器、
环境、工作目录和服务脚本路径，然后在 FDPose 根目录执行：

```bash
bash scripts/manage_services.sh start
bash scripts/manage_services.sh status
bash scripts/manage_services.sh stop

# 也可以为 start 指定另一份部署配置
bash scripts/manage_services.sh start --config /absolute/path/services.yaml
```

默认示例启用 YOLO 和 FDPose V2；深度补全暂时禁用，填写实际模块信息后再启用。
只需要 FDPose 时，将 `yolo.enabled` 设为 `false`。
这些环境和路径是示例，不代表已匹配部署机器。

执行上述命令的控制端需要 Bash、Python 3.8+；`start` 另外需要 PyYAML。
默认使用控制端 `python3`，可通过 `SERVICE_MANAGER_PYTHON` 指定解释器路径。
执行服务的目标必须是 Linux；Docker 模式还需要控制端能够访问对应 Docker daemon，
容器中有 Bash 和 Python 3.8+。容器侧进程管理不依赖 PyYAML、ROS 或模型库。
实际服务仍需要自己的 Conda/venv、ROS 和推理依赖。

ROS master、容器和必要的网络配置需提前准备；脚本不会自动创建/启动容器或启动 roscore。

## 2. 配置参数

每个 `services` 子项是一个独立服务，同一容器内也可以使用不同虚拟环境。

| 参数 | 含义 |
|---|---|
| 子项名称，如 `fdpose` | 进程管理名称，用于 PID 和日志文件名；不是 ROS Service 名称 |
| `enabled` | 是否在 start 时启动 |
| `container` | 空字符串表示当前 Linux 环境；否则是已经运行的 Docker 容器名称或 ID |
| `manager_python` | 容器内管理进程用的 Python，默认 `python3`，无需激活模型环境 |
| `work_dir` | 服务启动时的工作目录 |
| `runtime_dir` | 目标环境中 PID、日志的存放目录 |
| `conda_sh` / `conda_env` | Conda 初始化脚本和环境名称（也可为环境路径） |
| `venv` | 使用普通 venv 时指定环境目录；不可同时指定 `conda_env` |
| `setup_files` | 激活虚拟环境后，按顺序 source 的 ROS/工作空间脚本 |
| `pythonpath` | 追加到已有 PYTHONPATH 前端的目录列表 |
| `python` | 服务解释器，默认 `python`，通常来自激活后的虚拟环境 |
| `script` | 服务入口 Python 文件 |
| `args` | 脚本参数列表，每个参数单独一个字符串，包括需要覆盖的 ROS 参数 |
| `env` | 环境变量，名称和值均为字符串，例如 ROS_MASTER_URI |

当前环境模式：相对 `work_dir`、`runtime_dir` 都以 **FDPose 项目根目录**为基准，
不以终端或 YAML 所在目录为基准。Docker 模式：这两个字段必须填写**容器内绝对路径**。
`script`、`setup_files`、`pythonpath`、`conda_sh`、`venv` 的相对路径以 `work_dir` 为基准。
路径不进行 `~` 或 `$变量` 展开，请使用明确路径。

例如只在已有容器中启动 FDPose：

```yaml
services:
  fdpose:
    enabled: true
    container: pose_container
    manager_python: /usr/bin/python3
    work_dir: /root/catkin_ws/src/Fdpose
    runtime_dir: /root/catkin_ws/src/Fdpose/runtime
    conda_sh: /opt/conda/etc/profile.d/conda.sh
    conda_env: fdpose
    setup_files:
      - /opt/ros/noetic/setup.bash
      - /root/catkin_ws/devel/setup.bash
    pythonpath: [src]
    python: python
    script: scripts/pose_estimator_track_node_roi_v2.py
    args: []
    env: {}
```

若使用 venv，移除 `conda_sh`、`conda_env`，增加 `venv: /absolute/path/to/venv`。
若直接使用已配置好的解释器，可都不填，并通过 `python` 指定解释器绝对路径。

部署配置只负责“在哪里、用哪个环境、启动哪个脚本”。
`fdpose_server_v2.yaml` 仍管理模型/预热/保存参数；`fdpose_pipeline.yaml` 仍管理 Client 连接。
它们不会自动相互同步；覆盖 ROS 服务名称时，Client 连接配置也需要对应修改。

## 3. PID 与日志

控制端固定在本项目中记录目标信息；实际服务 PID 和日志位于目标环境的 `runtime_dir`：

```text
控制端 FDPose/runtime/pids/services/
  fdpose.target.json       # 实际容器 ID、目标 runtime 路径等，用于后续定位
  manager.lock            # 防止多个控制命令同时操作

目标环境 runtime_dir/
  logs/
    fdpose_<UTC时间戳>.log  # stdout + stderr，每次启动独立文件
  pids/services/
    fdpose.pid.json        # 真正的服务进程 PID、启动 tick、开机标识和日志路径
    fdpose.pid.lock        # 目标侧操作锁
```

当前环境使用默认配置时，两部分就在同一个 `runtime` 下。
Docker 模式下 PID 属于容器内进程，不是宿主机 `docker exec` 的 PID。
日志保存在容器内；如需宿主机直接查看，请提前将目标 `runtime_dir` 挂载到宿主机。
脚本不会自动复制或转发容器日志。启动和 status 输出会显示具体日志路径。

默认本地日志可用 `tail -f runtime/logs/fdpose_<实际时间戳>.log` 查看。
PID/目标记录默认仅创建用户可读写，请使用同一部署用户和同一控制端项目进行管理。
日志不轮转、不自动清理；`stop` 保留日志，成功停止后移除对应 PID/目标记录，锁文件可以保留。

## 4. 启停边界

- `start` 按配置顺序提交后台启动；启动后的短暂存活检查**不代表模型预热或 ROS Service 已就绪**。
  `status` 也仅检查进程身份和存活；是否可调用请查看日志，并用 Client 等待服务或检查 ROS 服务列表。
- 重复 `start` 不重复创建已经记录且仍存活的进程；它不会扫描或接管手动启动的同名服务。
  切换到此脚本前，先自行停止原有部署，避免重复节点或 GPU 占用。
- 修改环境/容器/参数不会自动重启已有进程。请先 `stop` 再 `start`。
- `stop`、`status` 使用已保存的目标，不读取新 YAML；即使后来改了路径、禁用了服务或删除配置，
  仍按原记录管理。`--config` 仅影响 start，不限定 stop/status 的服务范围。
- `stop` 只处理这个控制端记录的服务，先校验 PID 身份，再向其进程组发送 SIGTERM；
  等待最多 10 秒后，只有身份仍匹配才发送 SIGKILL，并再等待最多 3 秒。
  PID 被复用、残留进程组无法确认、容器不可达等情况会报错并保留记录，不按名称批量杀进程。
- 服务应以前台 Python 进程方式启动，不应自行再次 daemonize/创建脱离原进程组的后台服务。
  脚本不管理这类脱离进程，也不提供异常退出后的自动拉起。
- 单个服务启动失败会继续尝试其余服务，最终返回非零退出码；已成功启动的服务不自动回滚，
  可以查看日志修正配置后重试，或执行 `stop` 停止本次已记录的服务。

## 5. 验证

```bash
python -B -m unittest discover -s tests/unit -p test_service_manager.py -v
```

测试使用系统临时目录验证配置、Shell 参数/环境激活、重复启动及记录管理；
Docker、Linux 进程身份和信号边界使用替身。实际 Orin/Docker/ROS 服务启停仍需在部署环境联调。
