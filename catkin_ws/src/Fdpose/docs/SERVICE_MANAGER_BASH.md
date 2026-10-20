# Bash 服务进程管理器（pipeline.conf）

[返回 README](../README.md) · [保留的 Python 版](SERVICE_MANAGER_PYTHON.md)

[manage_services_bash.sh](../scripts/manage_services_bash.sh) 独立实现 start、stop、status，
不调用 Python 管理器、不读 YAML；使用 [config/pipeline.conf](../config/pipeline.conf) 配置部署。
Python 仅用于运行模型 Service。
固定管理 YOLO、可选深度补全、FDPose V2，均在同一个 Linux 环境内执行。
默认启动顺序为 YOLO → 可选深度补全 → FDPose；先自行准备 ROS master 和已运行的容器。

日常使用只需修改 `config/pipeline.conf`，然后执行同一个 Bash 入口，不需要改启停代码。
本文中的路径和容器名称都是部署示例，需替换为实际值。

几个配置文件的职责不同，不会互相自动同步：

- `config/pipeline.conf`：本 Bash 脚本的容器、路径、虚拟环境、服务名称和日志目录。
- `config/fdpose_server_v2.yaml`：FDPose V2 的推理、预热、相机标定、图片保存等参数。
- `config/fdpose_pipeline.yaml`：`FdposePipeline.from_ros_services()` 创建 Client 时使用的连接配置。
- `config/fdpose_services.yaml`：另一套 Python 进程管理器的配置，本 Bash 脚本不读取它。

## 1. 在容器内使用

先修改 [config/pipeline.conf](../config/pipeline.conf) 中的环境、路径和服务名，
无需修改脚本的 `configure()`。该函数保留原有兜底默认值。
已经进入目标容器时，配置示例为：

```bash
# 以下内容写在 config/pipeline.conf 中；等号两侧不能有空格。
CONTAINER=""                         # 留空：就在当前环境中运行，不调用 Docker
# FDPOSE_DIR 不填，自动使用脚本所在项目的根目录。
YOLO_DIR="/root/catkin_ws/src/naviai_yolo"
CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
FDPOSE_ENV="fdpose"
YOLO_ENV="seg"
ROS_SETUP="/opt/ros/noetic/setup.bash"
WS_SETUP="/root/catkin_ws/devel/setup.bash"
ENABLE_DEPTH_REFINE=0                 # 暂不启动深度补全
```

其余 Service 入口、服务名等沿用文件中的配置。运行前也要核对文件中启用的
`ROS_IP`、`ROS_HOSTNAME`、`ROS_MASTER_URI`，不要把其他机器的地址直接用于本机。
只用 FDPose 时不必准备 YOLO 环境和目录，下面显式选择 `fdpose` 即可。

在容器终端执行：

```bash
cd /root/catkin_ws/src/Fdpose

# 全部服务：默认 YOLO + FDPose，深度补全由配置开关控制。
bash scripts/manage_services_bash.sh start
bash scripts/manage_services_bash.sh status
bash scripts/manage_services_bash.sh stop

# 或者只操作一个服务（不要与其他方式重复启动同一服务）。
bash scripts/manage_services_bash.sh start fdpose
bash scripts/manage_services_bash.sh status fdpose
bash scripts/manage_services_bash.sh stop fdpose

# 单服务启动支持追加参数，每个参数独立传入，不使用 eval。
bash scripts/manage_services_bash.sh start fdpose \
  _config_path:=/root/catkin_ws/src/Fdpose/config/fdpose_server_v2.yaml
```

不填服务选择时等于 `all`。深度补全默认禁用，`ENABLE_DEPTH_REFINE=1` 才参与 `start all`；
也可显式 `start depth`。`stop all` 始终检查三个服务的 PID，不受启用开关影响。
附加参数只支持 `start` 单个服务，不能传给 `start all` 或 stop/status。

无需先手动 `source pipeline.conf`，脚本会自动读取它。需要临时覆盖某项时：

```bash
FDPOSE_ENV=fdpose_test bash scripts/manage_services_bash.sh start fdpose
```

已经运行的进程不会因为修改配置或再次 start 而改变环境，需要先 stop 再 start。

### 配置查找与优先级

- 不提供配置路径：根据 **Bash 脚本所在目录的上一级（FDPose 根目录）**查找 `config/pipeline.conf`，
  与终端当前目录无关。默认文件不存在时告警，并沿用环境变量及内置默认值。
- 提供 `--config`：读取指定文件，路径相对执行命令时的当前目录；找不到或语法错误就报错，不静默回退。
  `--config` 放在 start/stop/status **之前**。
- 优先级为：环境变量 > 配置文件 > 内置默认值。允许留空的参数（如 FDPOSE_ENV、ROS_SETUP）
  显式传入空字符串仍可覆盖配置；必填参数沿用原来的非空默认规则。
- 项目路径不填时继续按原规则计算：当前环境的 FDPOSE_DIR 取脚本上一级；
  YOLO_DIR、DEPTH_DIR、WS_SETUP、RUNTIME_DIR 基于最终 FDPOSE_DIR 派生，不基于 .conf 所在目录。
- 文件是可信的 Bash 赋值文件，`source` 会执行内容；只写分组注释和变量赋值，不加载不可信文件。

未配置的路径沿用以下默认值；`FDPOSE_DIR` 在此指最终生效的项目根目录：

| 配置项 | 默认路径 |
|---|---|
| `FDPOSE_DIR` | 当前 Bash 脚本所在目录的上一级；Docker 模式必须明确填写容器内路径 |
| `YOLO_DIR` | `<FDPOSE_DIR>/../naviai_yolo_0909` |
| `DEPTH_DIR` | `<FDPOSE_DIR>/../../lingbot-depth-rqh` |
| `WS_SETUP` | `<FDPOSE_DIR>/../../devel/setup.bash` |
| `RUNTIME_DIR` | `<FDPOSE_DIR>/runtime` |

`*_SCRIPT` 的相对路径以对应的项目目录为基准。项目目录和 RUNTIME_DIR 必须是绝对路径；
setup 脚本也建议填写绝对路径。配置文件中的 `#` 表示注释，取消路径项的注释后赋值才会生效。

```bash
# 从任意目录运行，自动找脚本所属项目的默认配置。
bash /root/catkin_ws/src/Fdpose/scripts/manage_services_bash.sh start

# 使用另一份部署配置；停止时仍指定同一份，确保容器和输出路径一致。
bash scripts/manage_services_bash.sh --config /absolute/path/pipeline.conf start
bash scripts/manage_services_bash.sh --config /absolute/path/pipeline.conf stop
```

## 2. 从宿主机进入一个已有容器

在**宿主机项目里的** `config/pipeline.conf` 填写：

```bash
CONTAINER="pose_container"                  # 已经运行的实际容器名或 ID
FDPOSE_DIR="/root/catkin_ws/src/Fdpose"       # 容器内项目路径，不是宿主机路径
YOLO_DIR="/root/catkin_ws/src/naviai_yolo"    # 同一个容器内的 YOLO 路径
# CONDA_SH、*_ENV、*_SCRIPT、ROS_SETUP 等也填写容器内对应的值。
```

然后在宿主机的 FDPose 项目根目录执行：

```bash
bash scripts/manage_services_bash.sh start
bash scripts/manage_services_bash.sh status
bash scripts/manage_services_bash.sh stop
```

三次命令保持同一个容器及目标输出路径。也可以通过同名环境变量临时覆盖配置。
指定容器时必须显式填写容器内绝对路径 `FDPOSE_DIR`，不能使用宿主机路径；
因为脚本内容通过 stdin 传入容器，无法从宿主机脚本位置推断容器内的项目位置。
脚本通过 `docker exec` 把自身内容交给容器内 Bash 执行；不要求容器中有 Python 管理器或 PyYAML。
配置在控制端加载一次，然后转发配置及环境覆盖后的部署变量，不复制宿主机 PATH、Conda 环境或模型依赖。
容器端不会重新加载 .conf，也不要求能访问宿主机的配置路径。
未设置的变量仍使用容器内已有值或脚本默认值，派生路径留到容器内计算。

脚本不创建/启动容器，也不自动寻找容器。若在宿主机运行却把 CONTAINER 留空，
它会尝试在宿主机本地运行服务，并不会自动进入某个容器。

### 是否继承当前终端的环境

- 容器内运行且 CONTAINER 为空：服务继承当前终端已导出的环境变量，再应用配置覆盖、
  激活对应 Conda 环境，并 source ROS/工作空间脚本。它仍在同一个容器内，但成为独立后台进程。
- 从宿主机指定 CONTAINER：不继承宿主机完整终端环境；服务基于容器环境运行，
  接收脚本列出的部署变量，再激活容器内的 Conda/ROS 环境。
- 普通未 export 的终端变量、alias 不会自动传入新 Bash。已导出的部署变量优先于 .conf，
  但后续 source 的 ROS_SETUP / WS_SETUP / EXTRA_SETUP 仍可能再次修改服务实际运行环境。

若已在当前终端准备好 FDPose 环境、不希望再次激活或加载 setup，可仅对单服务这样运行：

```bash
FDPOSE_ENV="" ROS_SETUP="" WS_SETUP="" EXTRA_SETUP="" \
  bash scripts/manage_services_bash.sh start fdpose
```

此时仍会读取 .conf 的其他参数、切换到工作目录并设置 PYTHONPATH；
服务输出写入日志，不占用当前终端前台。启动命令返回后可以继续输入命令。

## 3. 配置项

| 环境变量 | 用途 |
|---|---|
| `CONTAINER` | 留空表示当前 Linux 环境，否则进入指定的已有容器 |
| `FDPOSE_DIR / YOLO_DIR / DEPTH_DIR` | 对应项目工作目录，使用目标环境的绝对路径 |
| `FDPOSE_SCRIPT / YOLO_SCRIPT / DEPTH_SCRIPT` | Python Service 入口，绝对路径或相对对应项目目录 |
| `FDPOSE_ENV / YOLO_ENV / DEPTH_ENV` | 对应 Conda 环境，默认 fdpose / seg / fdpose；显式空字符串表示不激活 |
| `CONDA_SH` | Conda 初始化脚本，默认 `/opt/conda/etc/profile.d/conda.sh` |
| `PYTHON_BIN` | 激活环境后运行的解释器，默认 python；也可指定绝对路径 |
| `ROS_SETUP / WS_SETUP` | ROS、Catkin 环境脚本，可显式设为空以跳过 |
| `EXTRA_SETUP` | 可选的额外环境脚本，Conda 和 ROS 初始化完成后 source |
| `EXTRA_LIBRARY_PATH` | 加到已有 LD_LIBRARY_PATH 前端的目录，使用冒号分隔 |
| `YOLO_SERVICE` | YOLO 服务名，通过 ROS remap 设置 |
| `FDPOSE_POSE_SERVICE / FDPOSE_MESH_SERVICE` | V2 位姿估计、Mesh 服务名，通过对应 ROS 私有参数设置 |
| `DEPTH_SERVICE / CAMERA_INFO` | 深度补全的服务名和 CameraInfo 话题，通过原模块 CLI 参数传入 |
| `ENABLE_DEPTH_REFINE` | 0 / 1，是否在 start all 时启用深度补全 |
| `RUNTIME_DIR` | 目标环境的绝对输出路径，默认 FDPose 项目下 runtime |

不会写死 ROS IP。若部署需要，显式设置 `ROS_IP`、`ROS_HOSTNAME`、`ROS_MASTER_URI`；
这里指 Bash 内置默认值不指定 IP；当前仓库的 .conf 已启用网络地址赋值，运行前须按实际网络核对。
服务名称改变后，Client 的连接名称也要对应修改。
如需沿用旧脚本中的 Orin 动态库搜索路径，可设置：

```bash
export EXTRA_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu
```

Conda 激活、ROS setup、额外 setup 按顺序执行，失败则不继续运行服务。
工作目录下的 `src` 会加入已有 PYTHONPATH 前端；模型配置和模型依赖仍由各模块管理。

## 4. 后台、PID 和日志

使用 `nohup + setsid` 后台启动，关闭标准输入，输出重定向到日志；Shell 最后 `exec python`，
记录独立进程组内的服务 PID，同时记录启动 tick 和开机标识用于身份检查。
目标 Linux 需要 Bash、`flock`、`setsid`、`nohup` 等常用工具。

```text
runtime/
  logs/fdpose_<UTC时间戳>_<随机标识>.log
  pids/bash/fdpose.pid
  pids/bash/manager.lock
```

日志每次启动独立保存，不自动轮转或清理。PID 文件包含 PID/身份标识以及日志路径，
不是旧版只含一个数字的 PID 文件，不要交给旧脚本处理。
Docker 模式下，日志和 PID 均在容器内；如需宿主机直接查看，提前挂载 runtime 目录。

查看日志时先运行 status，复制输出中的完整 `log=...` 路径，再执行：

```bash
# 在实际运行服务的环境中，替换为 status 输出的日志路径。
tail -f /实际/runtime/logs/fdpose_实际时间戳_实际标识.log

# 或在宿主机读取容器内日志，容器名和路径都替换为实际值。
docker exec pose_container tail -f /实际/runtime/logs/fdpose_实际时间戳_实际标识.log
```

`status` 显示 running 或 start 显示 started，只代表进程在检查时存活，不代表 GPU 预热完成。
实际可调用状态还需结合日志、ROS 服务可见性与 Client 等待结果判断。
这些日志目录与 Service 自己保存的 raw/vis 图片目录用途不同，RUNTIME_DIR 不改变图片保存参数。

- 重复 start 会跳过已记录且仍运行的服务，不自动重启。改变参数后先 stop 再 start。
- stop 校验身份后向进程组发 SIGTERM；最多等 10 秒，只有身份仍匹配才发 SIGKILL，再等最多 3 秒。
  无法确认的进程组保留记录并报错，不按进程名称批量结束。只处理此脚本留下的 PID。
- 成功停止后删除对应 PID 文件，保留日志。服务/残留进程组无法退出时保留记录供排查。
- 为简化单容器部署，**不在宿主机另存容器映射**。stop/status 必须使用原来的容器和 RUNTIME_DIR；
  更换这两项之前先停止旧部署。
- 启动仅做短暂进程存活检查，不等待模型预热或 ROS Service 就绪，也不自动重启崩溃服务。
  多个模型初始化可能重叠，准备好后再让 Client 调用；部分启动失败不会自动停止其余已启动服务。
- 与 Python 版使用不同 PID 目录，不互相接管；**不能同时用两套工具启动同一批服务**。
  切换管理方式前，先用原工具 stop。两版都不改变推理、位姿计算或 Service 内部保存逻辑。

## 5. 常见问题

### 修改了 .conf，但没有生效

先检查当前终端是否已 export 同名变量，例如用 `printenv FDPOSE_ENV` 查看，环境变量优先级更高。
确认 `--config` 是否指向另一份文件，并用原目标配置 stop 后再 start；重复 start 不会重启已有进程。

### 找不到 Conda、ROS setup 或服务文件

检查目标环境内的 `CONDA_SH`、`*_DIR`、`*_SCRIPT`、`ROS_SETUP` 和 `WS_SETUP`。
指定 Docker 容器时这些都是容器内路径。检查日志中的首个错误，尤其是环境激活或 source 失败；
启动工具不会安装 Conda、创建虚拟环境、编译消息包或下载模型。

### start 成功后，Client 仍然连不上

先确认进程是否随后退出或仍在预热，再核对 ROS master 是否可用、网络变量是否正确、
Client 名称是否与配置中的服务名称一致。工具不会启动 roscore，也不会自动同步 Client 的 YAML。

### stop 找不到原来的服务

使用启动时同一容器、同一 RUNTIME_DIR，以及有权限访问记录和进程的用户。
如果修改了配置，先用原容器/路径覆盖后停止旧部署。不能用 Bash 版接管 Python 版或手动启动的服务。
身份校验失败时不要直接根据 PID 数字强制结束进程，应先确认该进程是否仍是原来的服务。

### 容器内部还需要启动另一个容器吗

不需要。CONTAINER 为空就在当前环境执行；指定名称则从控制端进入已有容器一次，
容器内继续启动 Python Service，不会递归执行 Docker 或创建新容器。

## 6. 测试边界

正式测试位于 [test_service_manager_bash.py](../tests/unit/test_service_manager_bash.py)，
可运行 `python -B -m unittest discover -s tests/unit -p test_service_manager_bash.py -v`。
临时文件只创建于系统临时目录。
测试实际 Bash 参数传递和环境激活，Docker 与进程信号使用替身；不代表已经完成 Orin/ROS/GPU 联调。
