"""The native Linux mode and its release file. These tests run on Windows and Linux alike."""

import tests.isolation  # noqa: F401  first of all: no test may reach the real agent configs
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import qwen_station  # noqa: E402
from qwen_station import app as app_module  # noqa: E402
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
            mock.patch.object(Path, "home", return_value=self.home),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def config(self):
        return Config(mode="native", prefix=config_module.native_prefix())

    def saved(self):
        return json.loads(config_module.settings_path().read_text(encoding="utf-8"))


class NativeSetupTests(TempHome):
    def test_interrupted_setup_retains_its_finish_time_across_polls(self):
        runner = FakeNative(self.config())
        runner.started_at = 100
        with mock.patch.object(setup.time, "time", return_value=120):
            runner._fail("vllm", "Cancelled")
        with mock.patch.object(setup.time, "time", return_value=160):
            self.assertEqual(runner.snapshot()["finished_at"], 120)

    def test_a_fresh_process_reopens_the_custom_folder_after_a_failed_setup(self):
        location = self.home / "disk" / "engine"
        runner = FakeNative(self.config(), ["::step vllm start", "::step vllm fail"], code=1)
        runner._run(str(location))
        self.assertEqual(runner.state, "failed")
        with mock.patch.object(config_module, "app_dir", return_value=self.home / "app"):
            retry = FakeNative(load_config())
        self.assertEqual(retry.default_location(), str(location))

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
        self.assertEqual(self.saved(), {"mode": "native", "prefix": str(self.home / "engine-root")})
        self.assertIsNone(runner.installed())

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

    def test_vllm_is_asked_for_no_more_of_the_card_than_is_free(self):
        def utilization(launcher):
            parts = [part for part in launcher.command(PRESETS["262k"]) if part.startswith("GPU_UTIL=")]
            return parts[0].split("=", 1)[1] if parts else "0.92"

        launcher = NativeLauncher(Config(mode="native", prefix="/p"))
        self.assertEqual(utilization(launcher), "0.92")  # nothing measured: vLLM's own default
        # Reported on NixOS: 26.4 GiB free beside a desktop, vLLM found 25.85 of 31.41 GiB and wanted 0.92 of it.
        launcher.card = {"index": 0, "mem_free": 27034, "mem_total": 32607}
        self.assertEqual(utilization(launcher), "0.79")
        self.assertLess(0.79 * 31.41, 25.85)
        launcher.card = {"index": 0, "mem_free": 31500, "mem_total": 32607}  # nothing else on the card
        self.assertEqual(utilization(launcher), "0.92")
        launcher.gpu_index = 1
        launcher.card = {"index": 1, "mem_free": 27034, "mem_total": 32607}
        command = launcher.command(PRESETS["262k"])
        self.assertEqual(command[1:3], ["GPU_INDEX=1", "GPU_UTIL=0.79"])
        custom = NativeLauncher(Config(mode="native", prefix="/p", gpu_memory_utilization=0.7))
        custom.card = launcher.card
        self.assertEqual(utilization(custom), "0.7")  # a smaller setting of the user's own stays

    def test_uninstall_removes_what_the_installation_created(self):
        paths = installer.linux_paths()
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "app":
                (path / "station" / "qwen_station").mkdir(parents=True)
            else:
                path.write_text("x", encoding="utf-8")
        with mock.patch.object(installer, "IS_WINDOWS", False):
            installer.uninstall_app()
        self.assertFalse(any(path.exists() for path in paths.values()))


class ImageInstallTests(TempHome):
    """The AppImage copies its content into the user's data folder and continues from there."""

    def setUp(self):
        super().setUp()
        self.image = self.home / "mount"
        (self.image / "station" / "qwen_station").mkdir(parents=True)
        for name, text in ((installer.START_FILE, "#!/bin/sh\n"), (installer.BUILD_FILE, "1.3.0 abc\n"),
                           ("qwen-station.png", "png"), ("station/qwen_station/installer.py", "")):
            (self.image / name).write_text(text, encoding="utf-8")
        self.module_file = str(self.image / "station" / "qwen_station" / "installer.py")
        for patcher in (mock.patch.object(installer, "IS_WINDOWS", False),
                        mock.patch.dict(os.environ, {"APPDIR": str(self.image), "APPIMAGE": "/x/Qwen.AppImage"})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def handover(self, *arguments):
        with mock.patch.object(installer, "__file__", self.module_file), \
                mock.patch.object(installer.os, "execve") as execve, mock.patch("builtins.print"):
            problem = installer.linux_handover(list(arguments))
        return problem, execve

    def test_only_a_process_inside_the_image_counts_as_the_image(self):
        self.assertIsNone(installer.image_dir())  # this test runs from the checkout
        with mock.patch.object(installer, "__file__", self.module_file):
            self.assertEqual(installer.image_dir(), self.image.resolve())
            with mock.patch.dict(os.environ, {"APPDIR": ""}):
                self.assertIsNone(installer.image_dir())

    def test_the_first_start_installs_the_app_and_continues_as_the_installed_copy(self):
        problem, execve = self.handover("--setup")
        paths = installer.linux_paths()
        start = paths["app"] / installer.START_FILE
        self.assertIsNone(problem)
        self.assertTrue((paths["app"] / "station" / "qwen_station" / "installer.py").is_file())
        self.assertIn(f"exec '{start}' \"$@\"", paths["launcher"].read_text(encoding="utf-8"))
        entry = paths["desktop"].read_text(encoding="utf-8")
        self.assertIn("Exec=\"" + str(paths["launcher"]).replace("\\", "\\\\") + "\"\n", entry)
        self.assertEqual(paths["icon"].read_text(encoding="utf-8"), "png")
        program, argv, environment = execve.call_args.args
        self.assertEqual((program, argv), (str(start), [str(start), "--setup"]))
        self.assertFalse(set(installer.IMAGE_VARIABLES) & set(environment))

    def test_the_same_build_is_not_copied_again_and_another_build_replaces_it(self):
        self.handover()
        marker = installer.linux_paths()["app"] / "kept"
        marker.write_text("x", encoding="utf-8")
        self.handover()
        self.assertTrue(marker.is_file())
        (self.image / installer.BUILD_FILE).write_text("1.3.1 def\n", encoding="utf-8")
        _, execve = self.handover()
        self.assertFalse(marker.exists())
        self.assertEqual(execve.call_count, 1)

    def test_a_failed_installation_reports_its_reason_and_starts_nothing(self):
        with mock.patch.object(installer.shutil, "copytree", side_effect=OSError("disk full")):
            problem, execve = self.handover()
        self.assertEqual(problem, "disk full")
        execve.assert_not_called()

    def test_an_unpacked_image_in_the_temp_folder_is_removed_and_a_mounted_one_is_left_alone(self):
        self.handover()
        self.assertTrue(self.image.is_dir())
        unpacked = self.home / "appimage_extracted_0123"
        self.image.rename(unpacked)
        self.image = unpacked
        self.module_file = str(unpacked / "station" / "qwen_station" / "installer.py")
        start = installer.linux_paths()["app"] / installer.START_FILE
        with mock.patch.dict(os.environ, {"APPDIR": str(unpacked)}),                 mock.patch.object(installer.os, "access", side_effect=lambda path, mode: Path(path) == start):
            problem, execve = self.handover()
        self.assertIsNone(problem)
        self.assertFalse(unpacked.exists())
        self.assertEqual(execve.call_count, 1)

    def test_the_menu_entry_quotes_its_command(self):
        entry = installer.desktop_entry('/home/a b/"x"$y/qwen-station')
        self.assertIn('Exec="/home/a b/\\"x\\"\\$y/qwen-station"\n', entry)
        self.assertIn("Icon=qwen-station\n", entry)


@unittest.skipIf(os.name == "nt", "the window always opens on Windows")
class WindowTests(unittest.TestCase):
    """A display alone is not enough for a window: WSLg and ssh -X set one without any browser."""

    def can_show(self, display, programs=()):
        environment = {"DISPLAY": display, "WAYLAND_DISPLAY": ""}
        with mock.patch.dict(os.environ, environment), mock.patch.object(app_module, "_app_browsers", return_value=[]),                 mock.patch.object(app_module.shutil, "which", side_effect=lambda name: name if name in programs else None):
            return app_module.can_show_window()

    def test_no_display_means_no_window(self):
        self.assertFalse(self.can_show("", programs=("firefox",)))

    def test_a_display_without_any_browser_means_no_window(self):
        self.assertFalse(self.can_show(":0"))

    def test_a_display_with_a_browser_gets_the_window(self):
        self.assertTrue(self.can_show(":0", programs=("firefox",)))


class ReleaseFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("build_linux", PROJECT / "build-linux.py")
        cls.builder = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(PROJECT))
        spec.loader.exec_module(cls.builder)
        cls.files = {name: (data, mode) for name, data, mode in cls.builder.app_files()}

    def test_the_image_holds_the_app_with_unix_line_endings(self):
        for name in ("station/run-station.py", "station/qwen_station/app.py", "station/qwen_station/dashboard.html",
                     "station/qwen_station/engine/provision.sh", "station/qwen_station/engine/engine.env",
                     "station/qwen_station/engine/patches/patch-vllm-autotune-seed.py"):
            self.assertIn(name, self.files)
        self.assertFalse([name for name in self.files if "__pycache__" in name or name.endswith((".pyc", ".ps1"))])
        serve, mode = self.files["station/qwen_station/engine/serve.sh"]
        self.assertNotIn(b"\r", serve)
        self.assertEqual(mode, 0o755)
        self.assertTrue(self.files["qwen-station.png"][0].startswith(b"\x89PNG"))

    def test_the_top_of_the_image_has_what_the_format_and_the_installation_need(self):
        start, mode = self.files[installer.START_FILE]
        self.assertTrue(start.startswith(b"#!/bin/sh\n"))
        self.assertNotIn(b"\r", start)
        self.assertEqual(mode, 0o755)
        self.assertIn(b"station/run-station.py", start)
        version, build = self.files[installer.BUILD_FILE][0].decode().split()
        self.assertEqual(version, qwen_station.__version__)
        self.assertEqual(len(build), 12)
        self.assertIn(b"Exec=\"AppRun\"\n", self.files["qwen-station.desktop"][0])

    def test_the_paths_of_the_installation_follow_the_xdg_folders(self):
        with mock.patch.dict(os.environ, {"XDG_DATA_HOME": "/x/data"}):
            paths = {name: path.as_posix() for name, path in installer.linux_paths().items()}
        self.assertEqual(paths["app"], "/x/data/qwen-station/app")
        self.assertEqual(paths["desktop"], "/x/data/applications/qwen-station.desktop")
        self.assertEqual(paths["icon"], "/x/data/icons/hicolor/256x256/apps/qwen-station.png")
        self.assertTrue(paths["launcher"].endswith("/.local/bin/qwen-station"))


if __name__ == "__main__":
    unittest.main()
