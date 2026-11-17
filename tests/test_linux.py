"""The native Linux mode and its release file. These tests run on Windows and Linux alike."""

import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_station import config as config_module  # noqa: E402
from qwen_station import installer, setup  # noqa: E402
from qwen_station.config import PRESETS, Config, load_config  # noqa: E402
from qwen_station.supervisor import LogBuffer, NativeLauncher, Supervisor  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent


class FakeNative(setup.NativeSetup):
    """The native setup with the provisioning script replaced by scripted output."""

    def __init__(self, config, lines=(), code=0):
        super().__init__(config, LogBuffer())
        self.commands = []
        self.lines = list(lines)
        self.code = code

    def prerequisites(self):
        return [{"id": "gpu", "label": "Graphics card", "ok": True, "blocking": False, "detail": ""}]

    def _stream(self, command, stdin=None, markers=False):
        self.commands.append(command)
        for line in self.lines:
            self.handle_line(line, markers=True)
        return self.code


class TempHome(unittest.TestCase):
    """Settings and default folders point into a temp folder, on either platform."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        for patcher in (
            mock.patch.dict(os.environ, {"LOCALAPPDATA": str(self.home / "local"), "XDG_CONFIG_HOME": str(self.home / "config"),
                                         "XDG_DATA_HOME": str(self.home / "data")}),
            mock.patch.object(config_module, "IS_WINDOWS", False),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def config(self):
        return Config(mode="native", prefix=config_module.native_prefix())

    def saved(self):
        return json.loads(config_module.settings_path().read_text(encoding="utf-8"))


class NativeSetupTests(TempHome):
    def test_a_full_run_copies_the_engine_and_provisions_below_the_folder(self):
        location = str(self.home / "disk" / "qwen-station")
        done = []
        runner = FakeNative(self.config(), ["::step vllm start", "::step vllm done",
                                            "::step target start", "::progress target 500 1000", "::step target done"])
        runner.on_done = lambda: done.append(True)
        runner.state = "running"
        runner._run(location)
        self.assertEqual((runner.state, runner.error), ("done", ""))
        self.assertTrue(done and runner.fresh)
        self.assertEqual(runner.commands, [["env", f"QWEN_STATION_PREFIX={location}", "bash",
                                            f"{location}/engine/provision.sh", "all"]])
        engine = Path(location) / "engine"
        self.assertTrue((engine / "serve.sh").is_file() and (engine / "patches" / "patch-vllm-autotune-seed.py").is_file())
        self.assertNotIn(b"\r\n", (engine / "provision.sh").read_bytes())
        self.assertEqual(runner.config.prefix, location)
        self.assertEqual(self.saved(), {"mode": "native", "prefix": location, "engine_location": location,
                                        "setup_pending": False})
        self.assertEqual(config_module.settings_path(), self.home / "config" / "qwen-station" / "qwen-station.json")
        steps = [step[0] for step in runner.steps]
        self.assertEqual(steps[0], "files")
        self.assertFalse({"app", "wsl", "distro"} & set(steps))

    def test_a_hint_from_the_provisioning_script_becomes_the_error(self):
        hint = "A compiler is missing. Run this once in a terminal, then start the setup again: sudo apt-get install -y build-essential"
        runner = FakeNative(self.config(), ["::step system start", f"::hint system {hint}", "::step system fail"], code=1)
        runner.state = "running"
        runner._run(str(self.home / "engine-root"))
        snapshot = runner.snapshot()
        self.assertEqual((snapshot["state"], snapshot["error"]), ("failed", hint))
        self.assertEqual({step["id"]: step["status"] for step in snapshot["steps"]}["system"], "failed")
        self.assertFalse(config_module.settings_path().exists())

    def test_the_folder_must_be_absolute_writable_and_large_enough(self):
        runner = FakeNative(self.config())
        self.assertIn("full path", runner.start("qwen-station"))
        with mock.patch.object(setup.shutil, "disk_usage", return_value=mock.Mock(free=10 * 2 ** 30)):
            self.assertIn("Only 10 GB free", runner.start(str(self.home / "small")))
        with mock.patch.object(setup.os, "access", return_value=False):
            self.assertIn("No write access", runner.start(str(self.home / "locked")))
        self.assertEqual(runner.state, "off")

    def test_an_existing_installation_stays_where_it_is(self):
        prefix = self.home / "engine-root"
        prefix.mkdir()
        (prefix / "installed.json").write_text(json.dumps({"engine": "0.0.1"}), encoding="utf-8")
        runner = FakeNative(Config(mode="native", prefix=str(prefix)))
        self.assertEqual((runner.check(), runner.reason), ("needed", "update"))
        self.assertEqual(runner.resolve_location(str(self.home / "elsewhere")), (str(prefix), ""))
        (prefix / "installed.json").write_text(json.dumps({"engine": setup.bundled_engine_version()}), encoding="utf-8")
        self.assertEqual(runner.check(), "done")
        (prefix / "installed.json").unlink()
        self.assertEqual((runner.check(), runner.reason), ("needed", "install"))

    def test_requirements_name_the_system_and_the_graphics_card(self):
        runner = setup.NativeSetup(self.config(), LogBuffer())
        card = {"name": "NVIDIA GeForce RTX 5090", "mem_total": 32607.0}
        with mock.patch.object(setup.gpu, "query_once", return_value=[card]), \
                mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup.platform, "machine", return_value="aarch64"):
            checks = {check["id"]: check for check in runner.prerequisites()}
        self.assertEqual(sorted(checks), ["gpu", "system"])
        self.assertEqual((checks["system"]["ok"], checks["system"]["blocking"]), (False, True))
        self.assertTrue(checks["gpu"]["ok"])
        self.assertEqual(runner.snapshot()["platform"], "linux")


class NativeModeTests(TempHome):
    def test_linux_defaults_to_the_native_mode_in_the_users_data_folder(self):
        with mock.patch.object(config_module, "app_dir", return_value=self.home / "app"):
            config = load_config()
        self.assertEqual((config.mode, config.prefix), ("native", str(self.home / "data" / "qwen-station")))
        self.assertIsInstance(setup.make_setup(config, LogBuffer()), setup.NativeSetup)
        self.assertIsInstance(Supervisor(config).launcher, NativeLauncher)

    def test_the_prefix_the_setup_saved_is_used(self):
        config_module.save_settings({"mode": "native", "prefix": "/data/qwen-station"})
        with mock.patch.object(config_module, "app_dir", return_value=self.home / "app"):
            self.assertEqual(load_config().prefix, "/data/qwen-station")

    def test_the_launcher_starts_serve_sh_directly_on_the_matched_card(self):
        launcher = NativeLauncher(Config(mode="native", prefix="/data/qwen.station", lan=True))
        command = launcher.command(PRESETS["262k"])
        self.assertEqual((command[0], command[-2:]), ("env", ["bash", "/data/qwen.station/engine/serve.sh"]))
        self.assertIn("VLLM_HOST=0.0.0.0", command)
        self.assertFalse(any(part.startswith("GPU_INDEX=") for part in command))
        launcher.gpu_index = 1
        self.assertEqual(launcher.command(PRESETS["262k"])[1], "GPU_INDEX=1")
        self.assertIn(r"pat='vllm serve /data/qwen\.station/models/'", launcher.stop_script())
        self.assertIn("/data/qwen.station/logs/status.txt", launcher.stop_script())
        self.assertTrue(launcher.distro_check()[0])

    def test_uninstall_removes_what_the_installer_script_created(self):
        paths = installer.linux_paths()
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "app":
                (path / "qwen_station").mkdir(parents=True)
            else:
                path.write_text("x", encoding="utf-8")
        with mock.patch.object(installer, "IS_WINDOWS", False):
            installer.uninstall_app()
        self.assertFalse(any(path.exists() for path in paths.values()))


class ReleaseFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("build_linux", PROJECT / "build-linux.py")
        cls.builder = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(PROJECT))
        spec.loader.exec_module(cls.builder)

    def test_the_payload_holds_the_app_with_unix_line_endings(self):
        with tarfile.open(fileobj=io.BytesIO(self.builder.payload()), mode="r:gz") as archive:
            members = {member.name: member for member in archive.getmembers()}
            serve = archive.extractfile("qwen_station/engine/serve.sh").read()
            icon = archive.extractfile("qwen-station.png").read()
        for name in ("run-station.py", "qwen_station/app.py", "qwen_station/dashboard.html",
                     "qwen_station/engine/provision.sh", "qwen_station/engine/engine.env",
                     "qwen_station/engine/patches/patch-vllm-autotune-seed.py"):
            self.assertIn(name, members)
        self.assertFalse([name for name in members if "__pycache__" in name or name.endswith((".pyc", ".ps1"))])
        self.assertNotIn(b"\r", serve)
        self.assertEqual(members["qwen_station/engine/serve.sh"].mode, 0o755)
        self.assertTrue(icon.startswith(b"\x89PNG"))

    def test_the_installer_script_and_the_uninstall_agree_on_the_paths(self):
        stub = (PROJECT / "packaging" / "linux-setup.sh").read_text(encoding="utf-8")
        self.assertTrue(stub.endswith("\n__PAYLOAD__\n"))
        for fragment in ('APP="$DATA/qwen-station/app"', 'BIN="$HOME/.local/bin"',
                         'DESKTOP="$DATA/applications/qwen-station.desktop"',
                         'ICON="$DATA/icons/hicolor/256x256/apps/qwen-station.png"'):
            self.assertIn(fragment, stub)
        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": "/x/data"}):
            paths = {name: path.as_posix() for name, path in installer.linux_paths().items()}
        self.assertEqual(paths["app"], "/x/data/qwen-station/app")
        self.assertEqual(paths["desktop"], "/x/data/applications/qwen-station.desktop")
        self.assertEqual(paths["icon"], "/x/data/icons/hicolor/256x256/apps/qwen-station.png")
        self.assertTrue(paths["launcher"].endswith("/.local/bin/qwen-station"))


if __name__ == "__main__":
    unittest.main()
