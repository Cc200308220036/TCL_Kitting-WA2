#!/usr/bin/env python3
# 运行：python tools/manage_services.py start|stop|status；也可经 scripts/manage_services.sh 调用。
# 实际执行：按部署配置中的 script 启动服务进程，不执行视觉 Pipeline。
# 配置：config/fdpose_services.yaml（start --config 可覆盖）；stop/status 使用保存的状态。
"""Linux/已有 Docker 容器内的服务启停；控制端不需要 ROS、模型或 GPU 依赖。"""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone


def write_state(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def process_identity(pid):
    """Linux 内核启动 tick + 本次开机标识，避免仅凭 kill(pid, 0) 判断身份。"""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    fields = text[text.rfind(")") + 2:].split()
    if fields[0] == "Z":
        return None
    return {"pgid": int(fields[2]), "start_ticks": fields[19]}


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def matches(state):
    pid = state["pid"]
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        raise ValueError("Invalid managed PID record; refusing to signal")
    return (state["boot_id"] == boot_id()
            and process_identity(pid) == {"pgid": pid, "start_ticks": state["start_ticks"]})


def group_members(pgid):
    members = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            identity = process_identity(int(path.name))
            if identity and identity["pgid"] == pgid:
                members.append(int(path.name))
    return members


def launch_command(spec):
    """用户路径和参数全部 shell quote；不接受任意 command 字符串。"""
    directory = Path(spec["work_dir"])
    script = directory / spec["script"]
    if not directory.is_dir() or not script.is_file():
        raise ValueError(f"Service work_dir/script not found: {directory}, {script}")
    conda_env, venv = spec.get("conda_env"), spec.get("venv")
    if conda_env and venv:
        raise ValueError("Choose conda_env or venv, not both")
    setup = [str(directory / path) for path in spec.get("setup_files", [])]
    conda_sh = str(directory / spec.get("conda_sh", ""))
    venv_activate = str(directory / venv / "bin/activate") if venv else None
    if conda_env:
        setup.insert(0, conda_sh)
    if venv:
        setup.insert(0, venv_activate)
    for path in setup:
        if not Path(path).is_file():
            raise ValueError(f"Environment setup file not found: {path}")
    commands = ["set -e", "cd " + shlex.quote(str(directory))]
    if conda_env:
        commands += ["source " + shlex.quote(conda_sh), "conda activate " + shlex.quote(conda_env)]
    elif venv:
        commands.append("source " + shlex.quote(venv_activate))
    commands += ["source " + shlex.quote(str(directory / path)) for path in spec.get("setup_files", [])]
    for key, value in spec.get("env", {}).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid environment variable name: {key}")
        commands.append(f"export {key}={shlex.quote(value)}")
    if spec.get("pythonpath"):
        prefix = ":".join(str(directory / path) for path in spec["pythonpath"])
        commands.append("export PYTHONPATH=" + shlex.quote(prefix) + '${PYTHONPATH:+:"$PYTHONPATH"}')
    argv = [spec.get("python", "python"), "-u", str(script), *spec.get("args", [])]
    commands.append("exec " + " ".join(shlex.quote(arg) for arg in argv))
    return ["/bin/bash", "-c", "\n".join(commands)]


def agent(action, spec):
    """在目标环境执行；PID 和 log 均位于该环境的 runtime_dir。"""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Service execution target must be Linux (Orin or Linux Docker container)")
    name = spec["name"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Invalid service name")
    root = Path(spec["runtime_dir"])
    state_path = root / "pids/services" / f"{name}.pid.json"
    if action != "start" and not state_path.exists():
        return {"status": "stopped"}
    with locked(state_path.with_suffix(".lock")):
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        live = state is not None and matches(state)
        if action == "status":
            return dict(state or {}, status="running" if live else "stopped_or_stale")
        if action == "start" and live:
            return dict(state, status="already_running")
        # 身份不匹配时绝不发信号；残留进程组需要人工核查，不用新进程覆盖记录。
        if state and not live and state["boot_id"] == boot_id() and group_members(state["pid"]):
            raise RuntimeError("PID identity changed or process group remains; refusing to start/stop it")
        if action == "stop":
            if live:
                os.killpg(state["pid"], signal.SIGTERM)
                deadline = time.monotonic() + 10
                while matches(state) and time.monotonic() < deadline:
                    time.sleep(.1)
                if matches(state):
                    os.killpg(state["pid"], signal.SIGKILL)
                    deadline = time.monotonic() + 3
                    while matches(state) and time.monotonic() < deadline:
                        time.sleep(.1)
                if matches(state) or group_members(state["pid"]):
                    raise RuntimeError("Managed process/group has not exited; PID record retained")
            state_path.unlink(missing_ok=True)
            return dict(state or {}, status="stopped")
        command = launch_command(spec)
        log_dir = root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        log = log_dir / f"{name}_{stamp}.log"
        with log.open("ab", buffering=0) as output:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        try:
            identity = process_identity(process.pid)
            if not identity or identity["pgid"] != process.pid:
                raise RuntimeError(f"Service exited before PID registration; see {log}")
            state = dict(identity, pid=process.pid, boot_id=boot_id(), name=name,
                         log=str(log), started_at=stamp)
            write_state(state_path, state)
        except Exception:
            # 只清理此次 Popen 产生且尚未退出的进程，不留下未记录的服务。
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
            raise
        time.sleep(.3)
        if process.poll() is not None:
            raise RuntimeError(f"Service exited during startup; see {log}")
        return dict(state, status="started")


def read_config(path, project_root):
    import yaml  # 仅控制端 start 需要 PyYAML；stop/status 和容器侧仅需标准库。
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or set(data) != {"services"} or not isinstance(data["services"], dict):
        raise ValueError("Config must contain a services mapping")
    allowed = {"enabled", "container", "manager_python", "runtime_dir", "work_dir", "script",
               "python", "conda_sh", "conda_env", "venv", "setup_files", "pythonpath", "env", "args"}
    result = {}
    for name, original in data["services"].items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Invalid service name: {name!r}")
        if not isinstance(original, dict) or original.keys() - allowed:
            raise ValueError(f"Invalid fields for service {name}")
        enabled = original.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a YAML boolean")
        if not enabled:
            continue
        spec = dict(original, name=name)
        for key in ("container", "manager_python", "runtime_dir", "work_dir", "script", "python",
                    "conda_sh", "conda_env", "venv"):
            if key in spec and not isinstance(spec[key], str):
                raise ValueError(f"{name}.{key} must be a string")
            if key in ("manager_python", "runtime_dir", "python") and key in spec and not spec[key].strip():
                raise ValueError(f"{name}.{key} must not be empty")
        for key in ("setup_files", "pythonpath", "args"):
            if not isinstance(spec.get(key, []), list) or any(not isinstance(v, str) for v in spec.get(key, [])):
                raise ValueError(f"{name}.{key} must be a list of strings")
        env = spec.get("env", {})
        if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                          or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)
                                          for k, v in env.items()):
            raise ValueError(f"{name}.env must map environment names to strings")
        if not spec.get("work_dir") or not spec.get("script"):
            raise ValueError(f"{name} needs work_dir and script")
        if spec.get("conda_env") and (not spec.get("conda_sh") or spec.get("venv")):
            raise ValueError(f"{name}: conda requires conda_sh and cannot be combined with venv")
        for key in ("work_dir", "runtime_dir"):
            value = spec.get(key, str(project_root / "runtime"))
            if spec.get("container") and (key not in spec or not Path(value).is_absolute()):
                raise ValueError(f"{name}.{key} must be an explicit absolute path inside the container")
            spec[key] = str((project_root / value).resolve()) if not spec.get("container") else value
        result[name] = spec
    return result


def container_id(name):
    result = subprocess.run(["docker", "inspect", "--type", "container", "--format",
                             "{{.Id}} {{.State.Running}}", name], check=True, capture_output=True, text=True)
    identifier, running = result.stdout.strip().split()
    if running != "true":
        raise RuntimeError(f"Container is not running: {name}; start it separately")
    return identifier


def run_agent(action, target, spec=None):
    payload = dict(spec or {}, name=target["name"], runtime_dir=target["runtime_dir"])
    if target.get("container_id"):
        command = ["docker", "exec", "-i", target["container_id"], target["manager_python"],
                   "-c", Path(__file__).read_text(), "--worker"]
    else:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    result = subprocess.run(command, input=json.dumps({"action": action, "spec": payload}),
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Service worker failed")
    return json.loads(result.stdout)


def manage(action, config_path, project_root):
    state_dir = project_root / "runtime/pids/services"
    if action != "start" and not state_dir.exists():
        print("No managed services.")
        return 0
    failed = False
    with locked(state_dir / "manager.lock"):
        if action == "start":
            services = read_config(config_path, project_root)
            if not services:
                print("No enabled services.")
            targets = services
        else:
            services = {}
            targets = {p.name[:-len(".target.json")]: None for p in state_dir.glob("*.target.json")}
        for name in targets:
            path = state_dir / f"{name}.target.json"
            try:
                if path.exists():
                    target = json.loads(path.read_text())
                    if action == "start":
                        prior = run_agent("status", target)
                        if prior["status"] == "running":
                            print(f"{name}: already running pid={prior['pid']} log={prior['log']} (saved target)")
                            continue
                        # 在原目标上确认并清理旧记录后，才允许按新配置换目标。
                        run_agent("stop", target)
                elif action != "start":
                    continue
                if action == "start":
                    spec = services[name]
                    target = {"name": name, "runtime_dir": spec["runtime_dir"],
                              "container_id": container_id(spec["container"]) if spec.get("container") else None,
                              "manager_python": spec.get("manager_python", "python3")}
                    # 先记录目标，启动响应丢失时 stop/重试仍能找到真实 PID 文件。
                    write_state(path, target)
                result = run_agent(action, target, services.get(name))
                if action == "stop":
                    path.unlink()
                elif action == "start":
                    write_state(path, dict(target, pid=result.get("pid"), log=result.get("log")))
                location = target.get("container_id") or "current environment"
                print(f"{name}: {result['status']} pid={result.get('pid', '-')} target={location} log={result.get('log', '-')}")
            except Exception as exc:
                failed = True
                print(f"{name}: ERROR: {exc}", file=sys.stderr)
    return int(failed)


def main():
    if sys.argv[1:] == ["--worker"]:
        payload = json.load(sys.stdin)
        print(json.dumps(agent(payload["action"], payload["spec"])))
        return 0
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "status"))
    parser.add_argument("--config", type=Path, default=root / "config/fdpose_services.yaml",
                        help="start 使用的部署配置；stop/status 使用已记录的目标，不读取新配置")
    args = parser.parse_args()
    return manage(args.action, args.config, root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Service manager: {exc}", file=sys.stderr)
        raise SystemExit(1)
