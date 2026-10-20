#!/usr/bin/env bash
# 运行：bash scripts/manage_services_bash.sh [--config /路径/pipeline.conf] start|stop|status。
# 实际执行：本文件管理进程，按 FDPOSE_SCRIPT/YOLO_SCRIPT/DEPTH_SCRIPT 启动各服务入口。
# 默认 FDPose 入口：scripts/pose_estimator_track_node_roi_v2.py；配置：config/pipeline.conf。
# 纯 Bash 版：固定服务在同一 Linux 容器中运行，不调用 manage_services.py。
# 部署配置放在项目根目录的 config/pipeline.conf，不依赖终端当前工作目录。
# 配置优先级：命令前赋值 / export 的环境变量 > pipeline.conf > configure() 默认值。
# 例如 FDPOSE_ENV=pose_test bash scripts/manage_services_bash.sh start fdpose。
# 本脚本不读取 fdpose_services.yaml；该 YAML 只用于另一个 Python 管理版本。
# CONTAINER 在加载配置后、进入目标环境前读取，可在 pipeline.conf 或环境变量中设置：
#   已在目标容器内：不设置 CONTAINER（或 export CONTAINER=""）。
#   从宿主机执行：export CONTAINER=容器名；export FDPOSE_DIR=/容器内/Fdpose绝对路径。
#   stop/status 也必须使用启动时的容器和 RUNTIME_DIR；更换目标前先停止旧部署。
set -eo pipefail

# 仅这些部署变量参与环境覆盖保护和 Docker 转发；不传递宿主机 PATH 等环境。
FDPOSE_CONFIG_KEYS=(CONTAINER FDPOSE_DIR YOLO_DIR DEPTH_DIR FDPOSE_SCRIPT YOLO_SCRIPT DEPTH_SCRIPT
  FDPOSE_ENV YOLO_ENV DEPTH_ENV CONDA_SH PYTHON_BIN ROS_SETUP WS_SETUP EXTRA_SETUP
  EXTRA_LIBRARY_PATH ROS_IP ROS_HOSTNAME ROS_MASTER_URI YOLO_SERVICE FDPOSE_POSE_SERVICE
  FDPOSE_MESH_SERVICE DEPTH_SERVICE CAMERA_INFO ENABLE_DEPTH_REFINE RUNTIME_DIR)

usage() {
  echo '用法：bash scripts/manage_services_bash.sh [--config 文件路径] start|stop|status [all|yolo|depth|fdpose] [服务参数...]'
  echo '示例：FDPOSE_ENV=fdpose bash scripts/manage_services_bash.sh start fdpose'
  echo '容器：CONTAINER=名称 FDPOSE_DIR=/容器内/Fdpose bash scripts/manage_services_bash.sh start'
  echo '附加参数仅支持单个服务；stop/status 请使用启动时相同的容器和 RUNTIME_DIR。'
}

load_config() {
  local config_file=${1:-} key index
  local -a override_names override_values
  if [[ -z $config_file ]]; then
    config_file=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/config/pipeline.conf
    # 未提供路径且默认配置不存在时，保留原来的环境变量 / 内置默认值行为。
    [[ -e $config_file ]] || { echo "默认配置不存在，使用内置默认值：$config_file" >&2; return 0; }
  fi
  [[ -f $config_file && -r $config_file ]] || { echo "配置文件不可读：$config_file" >&2; return 1; }
  # source 对不含 / 的文件名会查 PATH；转换成绝对路径，保证读取指定的文件。
  config_file=$(cd -- "$(dirname -- "$config_file")" && pwd)/$(basename -- "$config_file")
  for key in "${FDPOSE_CONFIG_KEYS[@]}"; do
    if [[ ${!key+x} ]]; then override_names+=("$key"); override_values+=("${!key}"); fi
  done
  bash -n "$config_file" || return 1
  # .conf 为可信的 Bash 赋值文件；source 会执行其内容，不加载不可信文件。
  source "$config_file" || return 1
  for ((index=0; index<${#override_names[@]}; index++)); do
    printf -v "${override_names[$index]}" '%s' "${override_values[$index]}"
  done
  # 包括配置中新设的 ROS 网络变量，必须导出后才能被后台服务继承。
  for key in "${FDPOSE_CONFIG_KEYS[@]}"; do
    if [[ ${!key+x} ]]; then export "$key"; fi
  done
}

configure() {
  # 下方保留原有兜底值；日常部署请修改 config/pipeline.conf，无需修改启停代码。
  # ==================== 1. 项目目录与服务入口 ====================
  # 以下目录均指实际运行服务的 Linux/容器环境，不是宿主机挂载前的路径。
  # FDPose 根目录，必须是绝对路径；在容器内直接执行时，默认取本脚本的上一级目录。
  # 从宿主机指定 CONTAINER 时，必须在配置文件或环境变量中提供容器内 FDPOSE_DIR。
  FDPOSE_DIR=${FDPOSE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
  # 同一容器的项目路径；*_SCRIPT 为绝对路径或相对对应 *_DIR 的路径。
  # YOLO 项目根目录（不是 scripts 目录）；默认兄弟目录名是示例，部署时按实际名称修改。
  YOLO_DIR=${YOLO_DIR:-$FDPOSE_DIR/../naviai_yolo_0909}
  # 深度补全项目根目录；不启动 depth 时无需准备该目录。
  DEPTH_DIR=${DEPTH_DIR:-$FDPOSE_DIR/../../lingbot-depth-rqh}
  # FDPose 当前使用 V2 入口；不需要为启停功能修改 Service 本体。
  FDPOSE_SCRIPT=${FDPOSE_SCRIPT:-scripts/pose_estimator_track_node_roi_v2.py}
  # 新版 YOLO 的 Service 入口，路径相对 YOLO_DIR。
  YOLO_SCRIPT=${YOLO_SCRIPT:-scripts/seg_service/seg_service_server.py}
  # 深度补全入口，需支持下方 --service-name / --camera-info-topic 参数。
  DEPTH_SCRIPT=${DEPTH_SCRIPT:-inference_depth_refine_service_v1.py}

  # ==================== 2. Python / Conda 环境 ====================
  # *_ENV 是对应服务的 Conda 环境名称或环境绝对路径，不是容器名称。
  # 可显式设为空：不激活 Conda，直接使用 PYTHON_BIN 指定的解释器。
  # ${变量-默认值}：只在变量未设置时使用默认值，允许用空字符串关闭环境激活。
  # ${变量:-默认值}：变量未设置或为空时都使用默认值，例如下面的 PYTHON_BIN。
  # FDPose 的推理环境，需要包含其 CUDA/PyTorch/TensorRT 等实际依赖。
  FDPOSE_ENV=${FDPOSE_ENV-fdpose}
  # YOLO 的推理环境，可以与 FDPose 不同，但两者仍在同一个容器中。
  YOLO_ENV=${YOLO_ENV-seg}
  # 深度补全环境，默认名为 fdpose；如果模块使用独立环境，在这里修改。
  DEPTH_ENV=${DEPTH_ENV-fdpose}
  # 同一容器共用的 Conda 初始化脚本，建议绝对路径；不是 conda 可执行文件。
  # 只要启用的服务需要激活 Conda，该文件就必须存在，不能留空。
  CONDA_SH=${CONDA_SH-/opt/conda/etc/profile.d/conda.sh}
  # 各服务共用的解释器命令；默认 python 会从各自激活后的环境中查找。
  # 也可填解释器绝对路径（例如 venv/bin/python），但该覆盖会作用于所有本次启动的服务。
  PYTHON_BIN=${PYTHON_BIN:-python}

  # ==================== 3. ROS 与动态库环境 ====================
  # 激活 Conda 后，按 ROS_SETUP → WS_SETUP → EXTRA_SETUP 的顺序 source。
  # 建议都使用目标环境的绝对路径；三个 setup 参数均允许显式设为空以跳过。
  # ROS 发行版环境脚本；默认 Noetic，需匹配实际部署，不会自动启动 roscore。
  ROS_SETUP=${ROS_SETUP-/opt/ros/noetic/setup.bash}
  # 已编译 Catkin 工作空间的环境脚本，用于找到自定义消息和服务类型。
  # 默认假设 FDPose 位于 <catkin_ws>/src/Fdpose，非此布局需修改。
  WS_SETUP=${WS_SETUP-$FDPOSE_DIR/../../devel/setup.bash}
  EXTRA_SETUP=${EXTRA_SETUP-}  # 可选：最后 source 的环境脚本，可设置 ROS_IP、库路径等。
  # 多个库目录用冒号分隔；为空则不追加。这里填路径值，不填 export 命令。
  # Orin 示例：/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu
  EXTRA_LIBRARY_PATH=${EXTRA_LIBRARY_PATH-}  # 加到已有 LD_LIBRARY_PATH 前面，不覆盖原值。
  # 不设置固定 IP；沿用目标环境中的 ROS_IP、ROS_HOSTNAME、ROS_MASTER_URI。
  # 需要覆盖网络配置时，在执行脚本前 export 对应变量，或在 EXTRA_SETUP 中设置。

  # ==================== 4. ROS 服务名称与模块开关 ====================
  # 以下是 ROS 通信名称，不是进程管理名称；修改后 Client 连接配置也要同步修改。
  # 命令中的 yolo/depth/fdpose 是固定管理名称，不随下面的 ROS 名称改变。
  # YOLO 分割服务：通过 ROS remap 将原 instance_segmentation_service 映射到此名称。
  YOLO_SERVICE=${YOLO_SERVICE:-/instance_segmentation_service}
  # FDPose 位姿估计服务，对应 V2 的 ~pose_service_name 参数。
  FDPOSE_POSE_SERVICE=${FDPOSE_POSE_SERVICE:-/fdpose_v2/pose_estimation}
  # FDPose Mesh 加载/清除服务，对应 V2 的 ~mesh_service_name 参数。
  FDPOSE_MESH_SERVICE=${FDPOSE_MESH_SERVICE:-/fdpose_v2/load_mesh}
  # 深度补全服务名，传给深度模块的 --service-name；不启用 depth 时不使用。
  DEPTH_SERVICE=${DEPTH_SERVICE:-/lingbot/depth_refine}
  # 深度模块读取内参的 CameraInfo 话题，不是 RGB/Depth 图像话题；按相机实际名称修改。
  CAMERA_INFO=${CAMERA_INFO:-/zj_humanoid/sensor/realsense_head/color/camera_info}
  # 仅接受 0 / 1：控制 start all 是否加入深度补全，默认 0。
  # 显式 start depth 不受此开关限制；stop all 仍会检查之前启动的 depth。
  ENABLE_DEPTH_REFINE=${ENABLE_DEPTH_REFINE:-0}

  # ==================== 5. 启停日志与 PID 输出 ====================
  # 必须是目标环境的绝对路径；start/stop/status 使用同一路径才能找到相同的记录。
  # Docker 模式下在容器内写入；如需宿主机查看，请预先挂载此目录。
  # 仅控制启停日志与 PID，不覆盖 FDPose Service 自己的图片/调试保存配置。
  RUNTIME_DIR=${RUNTIME_DIR:-$FDPOSE_DIR/runtime}
  [[ $FDPOSE_DIR == /* && $RUNTIME_DIR == /* ]] || { echo 'FDPOSE_DIR 和 RUNTIME_DIR 必须为绝对路径' >&2; return 1; }
  [[ $ENABLE_DEPTH_REFINE == 0 || $ENABLE_DEPTH_REFINE == 1 ]] || return 1
  # 以下两个目录由 RUNTIME_DIR 派生，不需要单独配置；Bash 与 Python 的 PID 记录隔离。
  PID_DIR=$RUNTIME_DIR/pids/bash
  # 每次启动独立日志；stop 保留日志，只删除成功停止的 PID 记录。
  LOG_DIR=$RUNTIME_DIR/logs
}

select_service() {
  case "$1" in
    yolo)
      work_dir=$YOLO_DIR; model_env=$YOLO_ENV; script=$YOLO_SCRIPT
      service_args=("instance_segmentation_service:=$YOLO_SERVICE") ;;
    depth)
      work_dir=$DEPTH_DIR; model_env=$DEPTH_ENV; script=$DEPTH_SCRIPT
      service_args=(--service-name "$DEPTH_SERVICE" --camera-info-topic "$CAMERA_INFO") ;;
    fdpose)
      work_dir=$FDPOSE_DIR; model_env=$FDPOSE_ENV; script=$FDPOSE_SCRIPT
      service_args=("_pose_service_name:=$FDPOSE_POSE_SERVICE" "_mesh_service_name:=$FDPOSE_MESH_SERVICE") ;;
  esac
}

run_service() {
  # 子 Shell 单独启用失败即退出；不启用 nounset，兼容 Conda/ROS 的环境脚本。
  set -e
  local directory=$1 environment=$2 entry=$3
  shift 3
  cd -- "$directory"
  if [[ -n $environment ]]; then
    source "$CONDA_SH"
    conda activate "$environment"
  fi
  [[ -z $ROS_SETUP ]] || source "$ROS_SETUP"
  [[ -z $WS_SETUP ]] || source "$WS_SETUP"
  [[ -z $EXTRA_SETUP ]] || source "$EXTRA_SETUP"
  export PYTHONPATH="$directory/src${PYTHONPATH:+:$PYTHONPATH}"
  if [[ -n $EXTRA_LIBRARY_PATH ]]; then
    export LD_LIBRARY_PATH="$EXTRA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  exec "$PYTHON_BIN" -u "$entry" "$@"
}

read_identity() {
  local stat_line
  local -a fields
  [[ $1 =~ ^[0-9]+$ && $1 -gt 1 ]] || return 1
  [[ -r /proc/$1/stat ]] || return 1
  { IFS= read -r stat_line < "/proc/$1/stat"; } 2>/dev/null || return 1
  read -r -a fields <<< "${stat_line##*) }"
  [[ ${fields[0]} != Z && ${#fields[@]} -ge 20 ]] || return 1
  proc_parent=${fields[1]}; proc_group=${fields[2]}; proc_ticks=${fields[19]}
}

read_record() {
  { read -r saved_pid saved_ticks saved_boot; IFS= read -r saved_log; } < "$pid_file" || return 1
  [[ $saved_pid =~ ^[0-9]+$ && $saved_pid -gt 1 && $saved_ticks =~ ^[0-9]+$ && -n $saved_boot ]] || {
    echo "无效 PID 记录，拒绝操作：$pid_file" >&2; return 1;
  }
}

is_owned() {
  [[ $saved_boot == "$current_boot" ]] && read_identity "$saved_pid" &&
    [[ $proc_group == "$saved_pid" && $proc_ticks == "$saved_ticks" ]]
}

group_exists() {
  local candidate
  kill -0 -- "-$saved_pid" 2>/dev/null || return 1
  if read_identity "$saved_pid" && [[ $proc_group == "$saved_pid" ]]; then return 0; fi
  # 容器 PID 1 可能尚未回收僵尸；只把仍在运行的组成员视为残留服务。
  for candidate in /proc/[0-9]*/stat; do
    candidate=${candidate#/proc/}; candidate=${candidate%/stat}
    if read_identity "$candidate" && [[ $proc_group == "$saved_pid" ]]; then return 0; fi
  done
  return 1
}

start_service() {
  local name=$1 pid stamp log temporary
  if [[ -f $pid_file ]]; then
    read_record || return 1
    if is_owned; then
      echo "$name: already running pid=$saved_pid log=$saved_log"; return 0
    fi
    if [[ $saved_boot == "$current_boot" ]] && group_exists; then
      echo "$name: 旧进程组身份无法确认，保留记录，拒绝启动" >&2; return 1
    fi
  fi
  select_service "$name"
  [[ $script == /* ]] || script=$work_dir/$script
  [[ $work_dir == /* && -d $work_dir && -f $script ]] || {
    echo "$name: 工作目录必须为绝对路径，且目录和脚本必须存在：$work_dir / $script" >&2; return 1;
  }
  # 创建日志后再启动。nohup 处理挂断信号；setsid 建立独立进程组。
  stamp=$(date -u +%Y%m%dT%H%M%S)
  log=$(mktemp "$LOG_DIR/${name}_${stamp}_XXXXXX.log") || return 1
  temporary=$(mktemp "$PID_DIR/.${name}.XXXXXX") || return 1
  export CONDA_SH ROS_SETUP WS_SETUP EXTRA_SETUP EXTRA_LIBRARY_PATH PYTHON_BIN
  export -f run_service
  set +m  # 禁用作业控制，避免 setsid 因子进程已经是组长而再次 fork。
  nohup setsid bash -c 'run_service "$@"' -- "$work_dir" "$model_env" "$script" \
    "${service_args[@]}" "${extra_args[@]}" </dev/null >"$log" 2>&1 9>&- &
  pid=$!
  # 等待 setsid 建立进程组，而不是等待模型就绪。
  local attempt
  for ((attempt=0; attempt<20; attempt++)); do
    if read_identity "$pid" && [[ $proc_group == "$pid" ]]; then break; fi
    sleep .05
  done
  if ! read_identity "$pid" || [[ $proc_group != "$pid" ]]; then
    # 未建立独立进程组时，仅对仍属于当前管理 Shell 的子进程请求退出。
    if read_identity "$pid" && [[ $proc_parent == "$$" ]]; then kill "$pid" 2>/dev/null || true; fi
    rm -f -- "$temporary"
    echo "$name: 启动失败，无法确认服务 PID；日志：$log" >&2; return 1
  fi
  if ! printf '%s %s %s\n%s\n' "$pid" "$proc_ticks" "$current_boot" "$log" > "$temporary" ||
     ! mv -f -- "$temporary" "$pid_file"; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    rm -f -- "$temporary"
    echo "$name: PID 保存失败，已请求本次进程退出；日志：$log" >&2; return 1
  fi
  sleep .3
  read_record || return 1
  if ! is_owned; then echo "$name: 启动后退出；日志：$log" >&2; return 1; fi
  echo "$name: started pid=$pid log=${log}（仅表示进程存活）"
}

stop_service() {
  local name=$1 attempt
  if is_owned; then
    kill -TERM -- "-$saved_pid" || return 1
    for ((attempt=0; attempt<100; attempt++)); do
      group_exists || break
      sleep .1
    done
    if group_exists && is_owned; then
      kill -KILL -- "-$saved_pid" || return 1
      for ((attempt=0; attempt<30; attempt++)); do
        group_exists || break
        sleep .1
      done
    fi
  fi
  if [[ $saved_boot == "$current_boot" ]] && group_exists; then
    echo "$name: 进程组仍存在或身份不匹配，未清除记录，请核查" >&2; return 1
  fi
  rm -- "$pid_file"
  echo "$name: stopped（日志保留：${saved_log}）"
}

main() {
  local config_file='' config_loaded=0
  if [[ ${1:-} == --config ]]; then
    [[ $# -ge 2 && -n $2 ]] || { echo '--config 需要文件路径' >&2; return 1; }
    config_file=$2; shift 2
  elif [[ ${1:-} == --config-loaded ]]; then
    # 内部 Docker 转发：只使用已经传入的值，不重新读取容器里的另一份配置。
    config_loaded=1; shift
  fi
  local action=${1:-} selected=${2:-all} key name failed=0
  local -a forwarded services extra_args
  case "$action" in -h|--help) usage; return 0 ;; start|stop|status) ;; *) usage; return 1 ;; esac
  case "$selected" in all|yolo|depth|fdpose) ;; *) usage; return 1 ;; esac
  if [[ $# -ge 2 ]]; then shift 2; else shift; fi
  extra_args=("$@")
  if [[ ${#extra_args[@]} -gt 0 && ( $action != start || $selected == all ) ]]; then
    echo '附加参数只允许用于 start 单个服务' >&2; return 1
  fi
  if [[ $config_loaded == 0 ]]; then load_config "$config_file" || return 1; fi
  if [[ -n ${CONTAINER:-} ]]; then
    [[ ${FDPOSE_DIR:-} == /* ]] || { echo '指定 CONTAINER 时，必须提供容器内绝对路径 FDPOSE_DIR' >&2; return 1; }
    # 转发配置和环境覆盖后的值；未设的派生路径留给容器内 configure() 计算。
    for key in "${FDPOSE_CONFIG_KEYS[@]}"; do
      if [[ $key != CONTAINER && ${!key+x} ]]; then forwarded+=(-e "$key=${!key}"); fi
    done
    docker exec -i "${forwarded[@]}" -e CONTAINER= "$CONTAINER" bash -s -- --config-loaded "$action" "$selected" \
      "${extra_args[@]}" < "${BASH_SOURCE[0]}"
    return $?
  fi
  [[ $(uname -s) == Linux ]] || { echo '服务执行目标必须是 Linux；可在宿主机指定 CONTAINER' >&2; return 1; }
  configure || return 1
  for key in flock setsid nohup; do command -v "$key" >/dev/null || { echo "缺少命令：$key" >&2; return 1; }; done
  IFS= read -r current_boot < /proc/sys/kernel/random/boot_id
  if [[ $action != start && ! -d $PID_DIR ]]; then echo '没有 Bash 版管理的进程'; return 0; fi
  mkdir -p -- "$PID_DIR" "$LOG_DIR"
  # 锁只由管理脚本持有，后台服务关闭 FD 9，避免 stop 永久等待。
  exec 9>"$PID_DIR/manager.lock"
  flock -n 9 || { echo '另一个 Bash 启停命令正在操作，请稍后重试' >&2; return 1; }
  if [[ $selected != all ]]; then
    services=("$selected")
  elif [[ $action == start ]]; then
    services=(yolo)
    [[ $ENABLE_DEPTH_REFINE != 1 ]] || services+=(depth)
    services+=(fdpose)
  else
    services=(fdpose depth yolo)  # stop 不受当前启用开关影响，仍检查原来的 PID 文件。
  fi
  for name in "${services[@]}"; do
    pid_file=$PID_DIR/$name.pid
    if [[ $action == start ]]; then
      start_service "$name" || failed=1
    elif [[ ! -f $pid_file ]]; then
      echo "$name: not managed"
    elif ! read_record; then
      failed=1
    elif [[ $action == stop ]]; then
      stop_service "$name" || failed=1
    elif is_owned; then
      echo "$name: running pid=$saved_pid log=$saved_log"
    else
      echo "$name: stopped or stale pid=$saved_pid log=$saved_log"
    fi
  done
  return "$failed"
}

# source 时仅加载函数，便于正式测试；直接执行或 docker 内 bash -s 时运行入口。
if [[ ${BASH_SOURCE[0]:-} == "$0" || -z ${BASH_SOURCE[0]:-} ]]; then main "$@"; fi
