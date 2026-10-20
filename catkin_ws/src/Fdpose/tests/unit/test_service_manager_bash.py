"""纯 Bash 管理器：真实 Shell/环境/参数，Docker 和信号边界使用替身。"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/manage_services_bash.sh"


class BashServiceManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fdpose-bash-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.work = self.root / "work with spaces"
        self.work.mkdir()
        self.env = dict(os.environ, FDPOSE_DIR=str(self.work), CONTAINER="",
                        RUNTIME_DIR=str(self.root / "runtime"), ROS_SETUP="", WS_SETUP="",
                        EXTRA_SETUP="", EXTRA_LIBRARY_PATH="", FDPOSE_ENV="", YOLO_ENV="",
                        DEPTH_ENV="", PYTHON_BIN=sys.executable, ENABLE_DEPTH_REFINE="0")
        self.entry = self.work / "fixture service.py"
        self.entry.write_text(
            "import json, os, sys\n"
            "print(json.dumps({'args': sys.argv[1:], 'trace': os.environ.get('TRACE'), "
            "'pythonpath': os.environ.get('PYTHONPATH'), 'libs': os.environ.get('LD_LIBRARY_PATH')}))\n"
        )
        self.env["FDPOSE_SCRIPT"] = self.entry.name

    def shell(self, body, *args):
        return subprocess.run(["bash", "-c", 'source "$1"; shift\n' + body, "test", str(SCRIPT), *args],
                              env=self.env, capture_output=True, text=True, timeout=5)

    def test_syntax_and_help_need_no_python_manager(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        result = subprocess.run(["bash", str(SCRIPT), "--help"], env=self.env,
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("start|stop|status", result.stdout)

    def test_configuration_and_service_names(self):
        self.env.update(FDPOSE_POSE_SERVICE="/custom/pose", FDPOSE_MESH_SERVICE="/custom/mesh")
        result = self.shell('configure; select_service fdpose; printf "%s\\n" "$work_dir" "$model_env" "${service_args[@]}"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [str(self.work), "", "_pose_service_name:=/custom/pose", "_mesh_service_name:=/custom/mesh"])

    def test_real_shell_conda_ros_setup_and_quoted_arguments(self):
        conda = self.root / "conda init.sh"
        conda.write_text('conda() { export TRACE="conda:$2"; }\n')
        ros = self.root / "ros setup.sh"
        ros.write_text('export TRACE="$TRACE:ros"\nexport PYTHONPATH=existing\n')
        extra = self.root / "extra setup.sh"
        extra.write_text('export TRACE="$TRACE:extra"\nexport LD_LIBRARY_PATH=oldlibs\n')
        self.env.update(CONDA_SH=str(conda), FDPOSE_ENV="pose env", ROS_SETUP=str(ros),
                        EXTRA_SETUP=str(extra), EXTRA_LIBRARY_PATH="/cuda/lib:/nvidia/lib")
        marker = self.root / "must_not_exist"
        args = ["two words", "x'y", "$(touch " + str(marker) + ")"]
        result = self.shell('configure; run_service "$FDPOSE_DIR" "$FDPOSE_ENV" "$FDPOSE_SCRIPT" "$@"', *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value["args"], args)
        self.assertEqual(value["trace"], "conda:pose env:ros:extra")
        self.assertEqual(value["pythonpath"], str(self.work / "src") + ":existing")
        self.assertEqual(value["libs"], "/cuda/lib:/nvidia/lib:oldlibs")
        self.assertFalse(marker.exists())

    def test_failed_activation_does_not_execute_service(self):
        conda = self.root / "bad conda.sh"
        conda.write_text('conda() { return 7; }\n')
        self.env.update(CONDA_SH=str(conda), FDPOSE_ENV="missing")
        result = self.shell('configure; run_service "$FDPOSE_DIR" "$FDPOSE_ENV" "$FDPOSE_SCRIPT"')
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, "")

    def test_extra_args_require_single_start(self):
        for args in [("start", "all", "--test"), ("stop", "fdpose", "--test")]:
            result = subprocess.run(["bash", str(SCRIPT), *args], env=self.env,
                                    capture_output=True, text=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("附加参数只允许", result.stderr)

    def test_docker_forwards_only_explicit_values_and_script(self):
        binary = self.root / "bin"
        binary.mkdir()
        docker = binary / "docker"
        docker.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@"\n'
                          'content=$(</dev/stdin)\n[[ $content == *"run_service()"* ]]\n')
        docker.chmod(0o755)
        self.env.update(PATH=str(binary) + os.pathsep + self.env["PATH"], CONTAINER="one container")
        result = subprocess.run(["bash", str(SCRIPT), "start", "fdpose", "_test:=two words"],
                                env=self.env, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.rstrip("\0").split("\0")
        self.assertEqual(args[:2], ["exec", "-i"])
        self.assertIn("FDPOSE_ENV=", args)
        self.assertIn("CONTAINER=", args)
        self.assertFalse(any(value.startswith("PATH=") for value in args))
        self.assertEqual(args[-8:], ["one container", "bash", "-s", "--", "--config-loaded", "start", "fdpose", "_test:=two words"])

        # 配置只存在于控制端，容器、路径及环境可全部从配置读取。
        config = self.root / "pipeline with spaces.conf"
        config.write_text('CONTAINER="configured box"\nFDPOSE_DIR="/target/project"\nFDPOSE_ENV="configured_env"\n')
        for key in ("CONTAINER", "FDPOSE_DIR", "FDPOSE_ENV"):
            self.env.pop(key, None)
        result = subprocess.run(["bash", str(SCRIPT), "--config", config.name, "start", "fdpose"],
                                env=self.env, cwd=self.root, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.rstrip("\0").split("\0")
        self.assertIn("FDPOSE_DIR=/target/project", args)
        self.assertIn("FDPOSE_ENV=configured_env", args)
        self.assertIn("configured box", args)
        self.assertIn("--config-loaded", args)
        self.assertNotIn(str(config), args)

    def test_docker_requires_explicit_target_project_path(self):
        self.env.update(CONTAINER="box", FDPOSE_DIR="relative")
        result = subprocess.run(["bash", str(SCRIPT), "start"], env=self.env,
                                capture_output=True, text=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("容器内绝对路径", result.stderr)

    def test_duplicate_start_does_not_spawn(self):
        record = self.root / "test.pid"
        record.write_text("12345 456 boot\nlog with spaces.log\n")
        result = self.shell('configure; pid_file=$1; is_owned() { return 0; }; '
                            'nohup() { echo unexpected; return 1; }; start_service fdpose', str(record))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already running pid=12345", result.stdout)
        self.assertNotIn("unexpected", result.stdout)

    def test_stop_mismatched_identity_keeps_record_and_does_not_signal(self):
        record = self.root / "test.pid"
        record.write_text("12345 456 boot\nlog.log\n")
        result = self.shell('pid_file=$1; current_boot=boot; read_record; '
                            'is_owned() { return 1; }; group_exists() { return 0; }; '
                            'kill() { echo unexpected; }; stop_service fdpose', str(record))
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(record.exists())
        self.assertNotIn("unexpected", result.stdout)

    def test_stop_signals_verified_group_and_keeps_log(self):
        record = self.root / "test.pid"
        log = self.root / "test.log"
        log.write_text("keep")
        record.write_text("12345 456 boot\n" + str(log) + "\n")
        result = self.shell('pid_file=$1; current_boot=boot; read_record; '
                            'is_owned() { return 0; }; group_exists() { return 1; }; '
                            'kill() { printf "signal:%s\\n" "$*"; }; stop_service fdpose', str(record))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("signal:-TERM -- -12345", result.stdout)
        self.assertFalse(record.exists())
        self.assertTrue(log.exists())

    def test_reused_pid_and_reboot_fail_identity_check(self):
        result = self.shell('saved_pid=12345; saved_ticks=456; saved_boot=boot; current_boot=boot; '
                            'read_identity() { proc_group=12345; proc_ticks=999; }; '
                            'if is_owned; then exit 4; fi; current_boot=newboot; '
                            'if is_owned; then exit 5; fi')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_group_with_no_live_members_is_not_treated_as_running(self):
        result = self.shell('saved_pid=12345; kill() { return 0; }; '
                            'read_identity() { return 1; }; '
                            'if group_exists; then exit 4; fi')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_config_preserves_environment_overrides_including_empty(self):
        config = self.root / "pipeline.conf"
        config.write_text('FDPOSE_DIR="/from/config"\nFDPOSE_ENV="config_env"\n'
                          'YOLO_ENV="config_yolo"\nROS_SETUP="/config/ros.sh"\n')
        self.env.pop("YOLO_ENV", None)
        self.env.pop("YOLO_DIR", None)
        result = self.shell('load_config "$1"; configure; '
                            'printf "%s\\n" "$FDPOSE_DIR" "$FDPOSE_ENV" "$YOLO_ENV" "$ROS_SETUP" "$YOLO_DIR"', str(config))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [str(self.work), "", "config_yolo", "",
                                                      str(self.work) + "/../naviai_yolo_0909"])

    def test_config_ros_network_values_are_exported_to_children(self):
        config = self.root / "pipeline.conf"
        config.write_text('ROS_IP="192.0.2.10"\nROS_MASTER_URI="http://192.0.2.1:11311"\n')
        self.env.pop("ROS_IP", None)
        self.env.pop("ROS_MASTER_URI", None)
        result = self.shell('load_config "$1"; bash -c \'printf "%s\\n" "$ROS_IP" "$ROS_MASTER_URI"\'', str(config))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["192.0.2.10", "http://192.0.2.1:11311"])

    def test_default_config_and_root_resolve_from_script_not_cwd(self):
        project = self.root / "project"
        (project / "scripts").mkdir(parents=True)
        (project / "config").mkdir()
        script = project / "scripts/manage_services_bash.sh"
        script.write_text(SCRIPT.read_text())
        (project / "config/pipeline.conf").write_text('YOLO_ENV="from_default_file"\n')
        self.env.pop("FDPOSE_DIR", None)
        self.env.pop("YOLO_ENV", None)
        self.env.pop("YOLO_DIR", None)
        self.env.pop("RUNTIME_DIR", None)
        body = 'source "$1"; load_config ""; configure; printf "%s\\n" "$FDPOSE_DIR" "$YOLO_ENV" "$RUNTIME_DIR"'
        result = subprocess.run(["bash", "-c", body, "test", str(script)], cwd=self.work,
                                env=self.env, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [str(project), "from_default_file", str(project / "runtime")])

        # 默认文件缺失时保留旧逻辑，而不是要求用户必须提供配置。
        (project / "config/pipeline.conf").unlink()
        result = subprocess.run(["bash", "-c", body, "test", str(script)], cwd=self.work,
                                env=self.env, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("默认配置不存在", result.stderr)
        self.assertEqual(result.stdout.splitlines(), [str(project), "seg", str(project / "runtime")])

    def test_missing_explicit_and_invalid_config_fail_before_start(self):
        invalid = self.root / "invalid.conf"
        invalid.write_text('FDPOSE_ENV="unterminated\n')
        for config in (self.root / "missing.conf", invalid):
            result = subprocess.run(["bash", str(SCRIPT), "--config", str(config), "start"],
                                    env=self.env, capture_output=True, text=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("服务执行目标必须是 Linux", result.stderr)
            self.assertFalse((self.root / "runtime").exists())


if __name__ == "__main__":
    unittest.main()
