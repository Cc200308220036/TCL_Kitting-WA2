"""部署管理回归：真实配置/临时文件，Docker 和进程信号边界使用替身。"""

import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("service_manager", ROOT / "tools/manage_services.py")
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


class ServiceManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.work = self.root / "work with spaces"
        self.work.mkdir()
        self.script = self.work / "example service.py"
        self.script.write_text("print('fixture')\n")
        self.config = self.root / "services.yaml"
        self.spec = {"name": "fdpose", "work_dir": str(self.work), "script": self.script.name,
                     "runtime_dir": str(self.root / "target_runtime"), "python": sys.executable}
        self.state_path = self.root / "target_runtime/pids/services/fdpose.pid.json"
        self.state = {"pid": 12345, "pgid": 12345, "start_ticks": "123", "boot_id": "boot",
                      "log": "saved.log"}

    def write_state(self):
        self.state_path.parent.mkdir(parents=True)
        self.state_path.write_text(json.dumps(self.state))

    def config_text(self, text):
        self.config.write_text(text)
        return manager.read_config(self.config, self.root)

    def test_config_local_resolution_and_disabled_placeholder(self):
        cfg = self.config_text("services:\n  fdpose:\n    work_dir: .\n    script: server.py\n"
                               "  depth:\n    enabled: false\n")
        self.assertEqual(list(cfg), ["fdpose"])
        self.assertEqual(cfg["fdpose"]["work_dir"], str(self.root))
        self.assertEqual(cfg["fdpose"]["runtime_dir"], str(self.root / "runtime"))

    def test_invalid_config_is_rejected_before_execution(self):
        cases = ["services: []", "services:\n  ../bad: {}",
                 "services:\n  fdpose: {enabled: 'false'}",
                 "services:\n  fdpose: {work_dir: '.', script: x, unknown: x}",
                 "services:\n  fdpose: {work_dir: '.', script: x, args: '--x'}",
                 "services:\n  fdpose: {container: box, work_dir: '.', script: x}",
                 "services:\n  fdpose: {work_dir: '.', script: x, conda_env: pose, venv: /env}"]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.config_text(text)

    def test_shell_quoting_preserves_args_without_executing_them(self):
        marker = self.root / "should_not_exist"
        self.script.write_text("import json, os, sys\nprint(json.dumps([sys.argv[1:], os.environ['DEMO']]))\n")
        self.spec.update(args=["two words", "$(touch " + str(marker) + ")", "x'y"],
                         env={"DEMO": "$(false); quoted value"})
        output = subprocess.check_output(manager.launch_command(self.spec), text=True)
        self.assertEqual(json.loads(output), [self.spec["args"], self.spec["env"]["DEMO"]])
        self.assertFalse(marker.exists())

    def test_activation_ros_setup_and_pythonpath_order(self):
        conda = self.work / "conda init.sh"
        conda.write_text('conda() { export TRACE="conda:$2"; }\n')
        setup = self.work / "ros setup.sh"
        setup.write_text('export TRACE="$TRACE:ros"\nexport PYTHONPATH="old/path"\n')
        self.script.write_text("import json, os\nprint(json.dumps([os.environ['TRACE'], os.environ['PYTHONPATH']]))\n")
        self.spec.update(conda_sh=str(conda), conda_env="pose env", setup_files=[str(setup)], pythonpath=["src"])
        result = json.loads(subprocess.check_output(manager.launch_command(self.spec), text=True))
        self.assertEqual(result, ["conda:pose env:ros", str(self.work / "src") + ":old/path"])

    def test_pid_reuse_and_reboot_do_not_match(self):
        with patch.object(manager, "boot_id", return_value="boot"), \
             patch.object(manager, "process_identity", return_value={"pgid": 12345, "start_ticks": "999"}):
            self.assertFalse(manager.matches(self.state))
        with patch.object(manager, "boot_id", return_value="other-boot"):
            self.assertFalse(manager.matches(self.state))
        with self.assertRaises(ValueError):
            manager.matches(dict(self.state, pid=-1))

    def test_venv_activation_without_conda(self):
        activate = self.work / "venv/bin/activate"
        activate.parent.mkdir(parents=True)
        activate.write_text("export VENV_FIXTURE=yes\n")
        self.script.write_text("import os\nprint(os.environ['VENV_FIXTURE'])\n")
        self.spec["venv"] = "venv"
        self.assertEqual(subprocess.check_output(manager.launch_command(self.spec), text=True).strip(), "yes")

    def test_linux_proc_identity_handles_spaces_and_parentheses_in_comm(self):
        fields = ["S", "1", "12345"] + ["0"] * 16 + ["456"]
        with patch.object(Path, "read_text", return_value="12345 (python (worker)) " + " ".join(fields)):
            self.assertEqual(manager.process_identity(12345), {"pgid": 12345, "start_ticks": "456"})

    def test_duplicate_start_does_not_spawn(self):
        self.write_state()
        with patch.object(manager.sys, "platform", "linux"), \
             patch.object(manager, "matches", return_value=True), \
             patch.object(manager.subprocess, "Popen") as spawn:
            self.assertEqual(manager.agent("start", self.spec)["status"], "already_running")
        spawn.assert_not_called()

    def test_start_records_real_target_pid_and_detaches_stdio(self):
        process = Mock(pid=12345)
        process.poll.return_value = None
        with patch.object(manager.sys, "platform", "linux"), \
             patch.object(manager, "boot_id", return_value="boot"), \
             patch.object(manager, "process_identity", return_value={"pgid": 12345, "start_ticks": "123"}), \
             patch.object(manager.subprocess, "Popen", return_value=process) as spawn, \
             patch.object(manager.time, "sleep"):
            result = manager.agent("start", self.spec)
        self.assertEqual(result["status"], "started")
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        self.assertEqual(spawn.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(json.loads(self.state_path.read_text())["pid"], 12345)
        self.assertTrue(Path(result["log"]).exists())

    def test_stop_verified_process_and_cleanup_pid_only(self):
        self.write_state()
        with patch.object(manager.sys, "platform", "linux"), \
             patch.object(manager, "matches", side_effect=[True, False, False, False]), \
             patch.object(manager, "group_members", return_value=[]), \
             patch.object(manager.os, "killpg") as kill:
            self.assertEqual(manager.agent("stop", self.spec)["status"], "stopped")
        kill.assert_called_once_with(12345, signal.SIGTERM)
        self.assertFalse(self.state_path.exists())

    def test_stop_refuses_identity_mismatch_with_existing_group(self):
        self.write_state()
        with patch.object(manager.sys, "platform", "linux"), \
             patch.object(manager, "matches", return_value=False), \
             patch.object(manager, "boot_id", return_value="boot"), \
             patch.object(manager, "group_members", return_value=[12345]), \
             patch.object(manager.os, "killpg") as kill:
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                manager.agent("stop", self.spec)
        kill.assert_not_called()
        self.assertTrue(self.state_path.exists())

    def test_stop_escalates_only_while_identity_still_matches(self):
        self.write_state()
        with patch.object(manager.sys, "platform", "linux"), \
             patch.object(manager, "matches", side_effect=[True, True, True, False, False]), \
             patch.object(manager, "group_members", return_value=[]), \
             patch.object(manager.time, "monotonic", side_effect=[0, 11, 12]), \
             patch.object(manager.os, "killpg") as kill:
            self.assertEqual(manager.agent("stop", self.spec)["status"], "stopped")
        self.assertEqual([call.args for call in kill.call_args_list],
                         [(12345, signal.SIGTERM), (12345, signal.SIGKILL)])

    def test_missing_pid_stop_is_idempotent_without_creating_target_dirs(self):
        with patch.object(manager.sys, "platform", "linux"):
            self.assertEqual(manager.agent("stop", self.spec), {"status": "stopped"})
        self.assertFalse(self.state_path.parent.exists())

    def test_docker_id_is_resolved_and_stopped_container_rejected(self):
        with patch.object(manager.subprocess, "run", return_value=Mock(stdout="id123 true\n")) as run:
            self.assertEqual(manager.container_id("pose box"), "id123")
        self.assertEqual(run.call_args.args[0][-1], "pose box")
        with patch.object(manager.subprocess, "run", return_value=Mock(stdout="id123 false\n")):
            with self.assertRaises(RuntimeError):
                manager.container_id("pose box")

    def test_docker_exec_uses_pinned_id_and_stdin_payload(self):
        target = {"name": "fdpose", "runtime_dir": "/output", "container_id": "id123", "manager_python": "python3"}
        with patch.object(manager.subprocess, "run", return_value=Mock(returncode=0, stdout='{"status":"running"}')) as run:
            self.assertEqual(manager.run_agent("status", target)["status"], "running")
        self.assertEqual(run.call_args.args[0][:5], ["docker", "exec", "-i", "id123", "python3"])
        self.assertEqual(json.loads(run.call_args.kwargs["input"])["spec"]["runtime_dir"], "/output")

    def test_stop_uses_saved_target_even_if_config_deleted(self):
        directory = self.root / "runtime/pids/services"
        directory.mkdir(parents=True)
        path = directory / "fdpose.target.json"
        saved = {"name": "fdpose", "runtime_dir": "/old", "container_id": "old-id", "manager_python": "python3"}
        path.write_text(json.dumps(saved))
        with patch.object(manager, "run_agent", return_value={"status": "stopped"}) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(manager.manage("stop", self.config, self.root), 0)
        run.assert_called_once_with("stop", saved, None)
        self.assertFalse(path.exists())

    def test_failed_start_preserves_target_for_recovery_and_continues_other_services(self):
        configs = {"one": dict(self.spec, name="one"), "two": dict(self.spec, name="two")}
        def run(action, target, spec=None):
            if target["name"] == "one":
                raise RuntimeError("lost startup response")
            return {"status": "started", "pid": 456, "log": "two.log"}
        with patch.object(manager, "read_config", return_value=configs), \
             patch.object(manager, "run_agent", side_effect=run), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(manager.manage("start", self.config, self.root), 1)
        self.assertTrue((self.root / "runtime/pids/services/one.target.json").exists())
        self.assertEqual(json.loads((self.root / "runtime/pids/services/two.target.json").read_text())["pid"], 456)


if __name__ == "__main__":
    unittest.main()
