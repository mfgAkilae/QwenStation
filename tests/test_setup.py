import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_station import config as config_module  # noqa: E402
from qwen_station import installer, setup  # noqa: E402
from qwen_station.config import PRESETS, Config, load_config  # noqa: E402
from qwen_station.supervisor import LogBuffer, ManagedLauncher  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
ENGINE = PROJECT / "qwen_station" / "engine"


class FakeSetup(setup.Setup):
    """Setup without WSL: the streamed commands are recorded and answered from a script."""

    def __init__(self, config, lines=(), code=0, exists=False, wsl=("ok",)):
        super().__init__(config, LogBuffer())
        self.commands = []
        self.lines = list(lines)
        self.code = code
        self.exists = exists
        self.wsl_states = list(wsl)  # one answer per question, the last one repeats
        self.install_fails = False

    def distro_exists(self):
        return self.exists

    def wsl_state(self):
        state = self.wsl_states.pop(0) if len(self.wsl_states) > 1 else self.wsl_states[0]
        return state, "2.7.3" if state == "ok" else ""

    def prerequisites(self):
        return [{"id": "wsl", "label": "WSL", "ok": True, "blocking": False, "detail": ""}]

    def _stream(self, command, stdin=None, markers=False):
        self.commands.append(command)
        if command[:2] == ["wsl.exe", "--install"]:
            if self.install_fails:
                return 1
            self.exists = True
        if markers:
            for line in self.lines:
                self.handle_line(line, markers=True)
            return self.code
        return 0


class ParsingTests(unittest.TestCase):
    def test_markers(self):
        self.assertEqual(setup.parse_marker("::step vllm done"), ("step", "vllm", ["done"]))
        self.assertEqual(setup.parse_marker("::progress target 12 99"), ("progress", "target", ["12", "99"]))
        self.assertIsNone(setup.parse_marker("Resolved 212 packages"))
        self.assertIsNone(setup.parse_marker("::step vllm"))

    def test_lines_are_cut_at_newlines_and_progress_redraws(self):
        lines, rest = setup.split_lines(b"one\r\ntwo 10%\rtwo 20%\rthree\nhal")
        self.assertEqual(lines, [("one", False), ("two 10%", True), ("two 20%", True), ("three", False)])
        self.assertEqual(rest, b"hal")
        self.assertEqual(setup.split_lines(rest + b"f", final=True), ([("half", False)], b""))

    def test_a_trailing_carriage_return_waits_for_its_newline(self):
        lines, rest = setup.split_lines(b"one\r")
        self.assertEqual((lines, rest), ([], b"one\r"))
        self.assertEqual(setup.split_lines(rest + b"\ntwo\n")[0], [("one", False), ("two", False)])


@unittest.skipUnless(os.name == "nt", "the WSL setup with its Windows paths")
class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        patcher = mock.patch.dict(os.environ, {"LOCALAPPDATA": self.temp.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def config(self):
        return Config(mode="managed", distro="QwenStation")

    def test_a_full_run_creates_the_distribution_and_saves_the_settings(self):
        done = []
        runner = FakeSetup(self.config(), ["::step vllm start", "Resolved 212 packages", "::step vllm done",
                                           "::step target start", "::progress target 500 1000", "::step target done"])
        runner.on_done = lambda: done.append(True)
        runner.state = "running"
        runner._run(r"D:\QwenStation")
        self.assertEqual(runner.state, "done")
        self.assertTrue(runner.fresh and done)
        self.assertEqual(runner.commands[0][:5], ["wsl.exe", "--install", "Ubuntu-26.04", "--name", "QwenStation"])
        self.assertIn(r"D:\QwenStation\wsl", runner.commands[0])
        self.assertEqual(runner.commands[-1][-3:], ["bash", "/opt/qwen-station/engine/provision.sh", "all"])
        self.assertEqual(runner._steps["vllm"]["status"], "done")
        self.assertEqual(runner._steps["target"]["progress"], [1000, 1000])
        self.assertEqual(runner._steps["app"]["status"], "skipped")  # not a frozen exe
        saved = json.loads((Path(self.temp.name) / "QwenStation" / "qwen-station.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, {"mode": "managed", "distro": "QwenStation", "prefix": "/opt/qwen-station",
                                 "engine_location": r"D:\QwenStation", "setup_pending": False})
        self.assertNotIn("Resolved 212 packages", [line["text"] for line in runner.log.since(0) if line["level"] == "step"])

    def test_a_failing_step_is_reported_and_an_existing_distribution_is_kept(self):
        runner = FakeSetup(self.config(), ["::step vllm start", "::step vllm fail"], code=1, exists=True)
        runner.state = "running"
        runner._run(r"D:\QwenStation")
        snapshot = runner.snapshot()
        self.assertEqual(snapshot["state"], "failed")
        self.assertIn("vLLM", snapshot["error"])
        self.assertFalse(any(command[:2] == ["wsl.exe", "--install"] for command in runner.commands))
        self.assertEqual({step["id"]: step["status"] for step in snapshot["steps"]}["vllm"], "failed")
        self.assertFalse((Path(self.temp.name) / "QwenStation" / "qwen-station.json").exists())

    def saved(self):
        return json.loads((Path(self.temp.name) / "QwenStation" / "qwen-station.json").read_text(encoding="utf-8"))

    def test_a_missing_wsl_is_installed_and_the_setup_goes_on(self):
        runner = FakeSetup(self.config(), wsl=("missing", "ok"))
        runner.state = "running"
        with mock.patch.object(installer, "run_elevated", return_value=0) as elevated:
            runner._run(r"D:\QwenStation")
        elevated.assert_called_once_with("wsl.exe", "--install", "--no-distribution")
        self.assertEqual(runner.state, "done")

    def test_a_fresh_wsl_that_needs_a_restart_is_continued_afterwards(self):
        runner = FakeSetup(self.config(), wsl=("missing",))
        runner.state = "running"
        with mock.patch.object(installer, "run_elevated", return_value=0), \
                mock.patch.object(installer, "run_after_restart") as autostart:
            runner._run(r"D:\QwenStation")
        self.assertEqual(runner.state, "restart")
        autostart.assert_called_once()
        self.assertEqual((self.saved()["setup_pending"], self.saved()["engine_location"]), (True, r"D:\QwenStation"))
        self.assertFalse(runner.commands)  # nothing was tried inside WSL

        with mock.patch.object(config_module, "app_dir", return_value=Path(self.temp.name) / "elsewhere"):
            after = FakeSetup(load_config(), wsl=("ok",))  # the app starts again after the restart
        self.assertTrue(after.config.setup_pending)
        after.state = "needed"
        with mock.patch.object(setup, "is_local_drive", return_value=True), \
                mock.patch.object(setup.shutil, "disk_usage", return_value=mock.Mock(free=500 * 2 ** 30)), \
                mock.patch.object(setup.threading, "Thread") as thread:
            self.assertTrue(after.resume_after_restart())
        self.assertEqual(thread.call_args.kwargs["args"], (r"D:\QwenStation",))
        self.assertFalse(self.saved()["setup_pending"])
        after.state = "needed"
        self.assertFalse(after.resume_after_restart())  # once only

    def test_wsl_installed_in_this_run_but_not_usable_yet_asks_for_the_restart(self):
        runner = FakeSetup(self.config(), wsl=("missing", "ok"))
        runner.install_fails = True
        runner.state = "running"
        with mock.patch.object(installer, "run_elevated", return_value=0), mock.patch.object(installer, "run_after_restart"):
            runner._run(r"D:\QwenStation")
        self.assertEqual(runner.state, "restart")

    def test_declined_administrator_rights_and_an_old_wsl(self):
        runner = FakeSetup(self.config(), wsl=("missing",))
        runner.state = "running"
        with mock.patch.object(installer, "run_elevated", return_value=1223):
            runner._run(r"D:\QwenStation")
        self.assertEqual(runner.state, "failed")
        self.assertIn("administrator rights", runner.error)

        runner = FakeSetup(self.config(), wsl=("outdated", "ok"))
        runner.state = "running"
        runner._run(r"D:\QwenStation")
        self.assertEqual(runner.state, "done")
        self.assertEqual(runner.commands[0], ["wsl.exe", "--update"])

    def test_a_failing_distribution_names_disabled_virtualization(self):
        runner = FakeSetup(self.config())
        runner.install_fails = True
        runner.state = "running"
        original = runner._stream

        def stream(command, stdin=None, markers=False):
            if command[:2] == ["wsl.exe", "--install"]:
                runner.handle_line("Error code: Wsl/Service/CreateInstance/CreateVm/HCS/0x80370102")
            return original(command, stdin, markers)

        runner._stream = stream
        runner._run(r"D:\QwenStation")
        self.assertEqual(runner.state, "failed")
        self.assertIn("Virtualization is not enabled", runner.error)

    def test_blocking_prerequisites_stop_the_start(self):
        runner = FakeSetup(self.config())
        blocked = [{"id": "gpu", "label": "Graphics card", "ok": False, "blocking": True, "detail": "No NVIDIA graphics card found."}]
        with mock.patch.object(runner, "prerequisites", return_value=blocked):
            self.assertIn("No NVIDIA graphics card", runner.start(r"D:\QwenStation"))
        self.assertEqual(runner.state, "off")

    def test_prerequisites_report_gpu_and_wsl(self):
        runner = setup.Setup(self.config(), LogBuffer())
        card = {"name": "NVIDIA GeForce RTX 5090", "mem_total": 32607.0}
        with mock.patch.object(setup.gpu, "query_once", return_value=[card]), \
                mock.patch.object(runner, "wsl_state", return_value=("missing", "")):
            checks = {check["id"]: check for check in runner.prerequisites()}
        self.assertEqual((checks["gpu"]["ok"], checks["gpu"]["detail"]), (True, "RTX 5090, 32 GiB"))
        self.assertEqual((checks["wsl"]["ok"], checks["wsl"]["blocking"], checks["wsl"]["state"]), (True, False, "missing"))
        with mock.patch.object(setup.gpu, "query_once", side_effect=OSError("nvidia-smi")), \
                mock.patch.object(runner, "wsl_state", return_value=("ok", "2.7.3")):
            checks = {check["id"]: check for check in runner.prerequisites()}
        self.assertEqual((checks["gpu"]["ok"], checks["gpu"]["blocking"]), (False, True))
        small = {"name": "NVIDIA GeForce RTX 5080", "mem_total": 16303.0}
        with mock.patch.object(setup.gpu, "query_once", return_value=[small]), \
                mock.patch.object(runner, "wsl_state", return_value=("ok", "2.7.3")):
            self.assertFalse({check["id"]: check for check in runner.prerequisites()}["gpu"]["ok"])

    def test_start_rejects_bad_folders_and_full_drives(self):
        runner = FakeSetup(self.config())
        self.assertIn("full Windows path", runner.start("QwenStation"))
        with mock.patch.object(setup, "is_local_drive", return_value=False):
            self.assertIn("not a built-in drive", runner.start(r"P:\QwenStation"))
        with mock.patch.object(setup, "is_local_drive", return_value=True), \
                mock.patch.object(setup.shutil, "disk_usage", return_value=mock.Mock(free=10 * 2 ** 30)):
            self.assertIn("Only 10 GB free", runner.start(r"D:\QwenStation"))
        self.assertEqual(runner.state, "off")
        self.assertIn("not managed by the app", FakeSetup(Config(mode="legacy")).start(r"D:\QwenStation"))

    def test_check_compares_the_installed_engine_with_the_bundled_one(self):
        runner = FakeSetup(self.config(), exists=True)
        with mock.patch.object(runner, "installed", return_value={"engine": setup.bundled_engine_version()}):
            self.assertEqual(runner.check(), "done")
        with mock.patch.object(runner, "installed", return_value={"engine": "0.0.1"}):
            self.assertEqual((runner.check(), runner.reason), ("needed", "update"))
        runner.exists = False
        self.assertEqual((runner.check(), runner.reason), ("needed", "install"))
        self.assertEqual(FakeSetup(Config(mode="managed", mock=True)).check(), "off")


class ModeTests(unittest.TestCase):
    """How the app picks its mode on Windows. The Linux side is in test_linux.py."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        for patcher in (mock.patch.dict(os.environ, {"LOCALAPPDATA": self.temp.name}),
                        mock.patch.object(config_module, "IS_WINDOWS", True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_an_app_without_start_scripts_is_managed(self):
        with mock.patch.object(config_module, "app_dir", return_value=Path(self.temp.name)):
            config = load_config()
        self.assertEqual((config.mode, config.distro, config.prefix), ("managed", "QwenStation", "/opt/qwen-station"))

    def test_the_folder_with_the_start_scripts_stays_legacy(self):
        (Path(self.temp.name) / "run-variant-logged.sh").write_text("", encoding="utf-8")
        with mock.patch.object(config_module, "app_dir", return_value=Path(self.temp.name)):
            config = load_config()
        self.assertEqual((config.mode, config.distro), ("legacy", "Ubuntu"))

    def test_root_on_the_command_line_ignores_the_installed_settings(self):
        config_module.save_settings({"mode": "managed", "distro": "QwenStation"})
        with mock.patch.object(config_module, "app_dir", return_value=Path(self.temp.name) / "elsewhere"):
            self.assertEqual(load_config().mode, "managed")
            config = load_config(overrides={"root": r"D:\qwen-engine"})
        self.assertEqual((config.mode, config.distro), ("legacy", "Ubuntu"))

    def test_managed_launcher_runs_the_engine_as_root(self):
        launcher = ManagedLauncher(Config(mode="managed", distro="QwenStation"))
        command = launcher.command(PRESETS["262k"])
        self.assertEqual(command[:6], ["wsl.exe", "-d", "QwenStation", "-u", "root", "--exec"])
        self.assertEqual(command[-2:], ["bash", "/opt/qwen-station/engine/serve.sh"])
        self.assertIn("CTX=262144", command)
        self.assertIn("/opt/qwen-station/logs/status.txt", launcher.stop_script())


class BundleTests(unittest.TestCase):
    def test_bundled_patches_are_complete(self):
        patches = sorted((ENGINE / "patches").glob("*.py"))
        self.assertEqual([patch.name for patch in patches],
                         ["patch-vllm-autotune-seed.py", "patch-vllm-dflash-quant.py", "patch-vllm-eagle-group.py",
                          "patch-vllm-fp8-group-size.py", "patch-vllm-swblock-divisor.py"])
        for patch in patches:
            compile(patch.read_text(encoding="utf-8"), str(patch), "exec")

    def test_engine_pins_are_complete(self):
        pins = dict(line.split("=", 1) for line in (ENGINE / "engine.env").read_text(encoding="utf-8").splitlines()
                    if "=" in line and not line.startswith("#"))
        for key in ("ENGINE_VERSION", "VLLM_VERSION", "TARGET_REPO", "TARGET_REV", "DRAFTER_REPO", "DRAFTER_REV"):
            self.assertTrue(pins.get(key), key)
        self.assertEqual(pins["TARGET_DIR"], config_module.TARGET_DIR)
        self.assertEqual(pins["DRAFTER_DIR"], config_module.DRAFTER_DIR)
        self.assertTrue(setup.bundled_engine_version())
        golden = f"flashinfer-0.7.0.post1-sm120-{config_module.MODEL_ID}.json"
        self.assertTrue((ENGINE / "autotune" / golden).is_file())

    def test_every_setup_step_exists_in_the_provisioning_script(self):
        script = (ENGINE / "provision.sh").read_text(encoding="utf-8")
        for key, _, _ in setup.STEPS[4:]:  # the first four happen on the Windows side
            self.assertIn(f"step_{key}()", script, key)


class InstallerTests(unittest.TestCase):
    def test_a_source_checkout_is_never_installed(self):
        self.assertFalse(installer.install_needed())

    @unittest.skipUnless(os.name == "nt", "Windows paths")
    def test_install_location_is_per_user(self):
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": r"C:\Users\x\AppData\Local"}):
            self.assertEqual(str(installer.installed_exe()), r"C:\Users\x\AppData\Local\Programs\Qwen Station\Qwen-Station.exe")


if __name__ == "__main__":
    unittest.main()
