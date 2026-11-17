"""The settings menu: validation, when a change takes effect, and the address the API listens on."""

import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_station import config as config_module  # noqa: E402
from qwen_station import clients, settings  # noqa: E402
from qwen_station import supervisor as supervisor_module  # noqa: E402
from qwen_station.app import Station  # noqa: E402
from qwen_station.config import PRESETS, Config, load_config  # noqa: E402
from qwen_station.relay import Relay  # noqa: E402
from qwen_station.supervisor import LAN_STATE, ManagedLauncher, MockLauncher, NativeLauncher, Supervisor, WslLauncher  # noqa: E402
from tests.test_station import free_port, request  # noqa: E402

ADDRESSES = [{"address": "192.168.1.20", "name": "Ethernet"}, {"address": "100.64.0.7", "name": "VPN"}]


class ValidationTests(unittest.TestCase):
    def check(self, **changes):
        config = Config()
        return settings.validate(config, settings.values(config), changes, ADDRESSES)

    def test_the_three_kinds_of_address_are_accepted(self):
        for host in ("127.0.0.1", "0.0.0.0", "192.168.1.20", "100.64.0.7"):
            merged, error, _ = self.check(api_host=host)
            self.assertEqual((merged.get("api_host"), error), (host, ""))

    def test_a_wrong_address_names_its_field(self):
        for host in ("10.9.9", "localhost", "::1", "", "127.0.0.2", "192.168.1.99"):
            merged, error, field = self.check(api_host=host)
            self.assertEqual((merged, field), ({}, "api_host"), host)
            self.assertTrue(error)
        self.assertIn("no network address 192.168.1.99", self.check(api_host="192.168.1.99")[1])

    def test_an_address_that_is_already_set_is_not_checked_against_the_machine_again(self):
        config = Config(api_host="10.1.2.3")  # the interface is down right now: other settings can still be saved
        merged, error, _ = settings.validate(config, settings.values(config), {"draft_tokens": 9}, ADDRESSES)
        self.assertEqual((merged["api_host"], merged["draft_tokens"], error), ("10.1.2.3", 9, ""))

    def test_ports_draft_tokens_and_the_environment_have_limits(self):
        self.assertEqual(self.check(api_port=80)[2], "api_port")
        self.assertEqual(self.check(station_port=70000)[2], "station_port")
        self.assertEqual(self.check(api_port=8010)[2], "api_port")  # the dashboard's port
        self.assertEqual(self.check(draft_tokens=6)[2], "draft_tokens")
        self.assertEqual(self.check(draft_tokens=16)[2], "draft_tokens")
        self.assertEqual(self.check(close_action="hide")[2], "close_action")
        self.assertEqual(self.check(extra_env="KV_DTYPE=nvfp4 broken")[2], "extra_env")
        self.assertEqual(self.check(extra_env="KV_DTYPE=nvfp4 XQA_DECODE=0")[1], "")
        self.assertEqual(self.check(extra_args="--a\n--b")[2], "extra_args")

    def test_unknown_settings_and_wrong_types_are_refused(self):
        self.assertEqual(self.check(root="C:\\x")[2], "root")
        self.assertEqual(self.check(api_port="8000")[2], "api_port")
        self.assertEqual(self.check(api_port=True)[2], "api_port")
        self.assertEqual(self.check(autostart="yes")[2], "autostart")


class AddressTests(unittest.TestCase):
    def test_the_older_lan_switch_means_all_interfaces_and_an_address_wins_over_it(self):
        self.assertEqual(Config().bind_host, "127.0.0.1")
        self.assertFalse(Config().shared)
        self.assertEqual(Config(lan=True).bind_host, "0.0.0.0")
        self.assertEqual(Config(lan=True, api_host="127.0.0.1").bind_host, "127.0.0.1")
        self.assertTrue(Config(api_host="192.168.1.20").shared)
        with self.assertRaises(ValueError):
            Config(api_host="nas.local").validate()

    def test_this_machine_reaches_a_single_address_directly_on_linux_and_through_127_0_0_1_in_wsl(self):
        self.assertEqual(Config(mode="native", api_host="192.168.1.20").api_base_url, "http://192.168.1.20:8000/v1")
        self.assertEqual(Config(mode="native", api_host="0.0.0.0").api_base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(Config(mode="managed", api_host="192.168.1.20").api_base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(Config(mode="native", api_host="192.168.1.20", mock=True).local_host, "127.0.0.1")

    def test_the_server_binds_the_address_on_linux_and_stays_private_inside_wsl(self):
        def host(launcher):
            return next(part for part in launcher.command(PRESETS["262k"]) if part.startswith("VLLM_HOST="))

        self.assertEqual(host(WslLauncher(Config(root=r"D:\engine"))), "VLLM_HOST=127.0.0.1")
        self.assertEqual(host(ManagedLauncher(Config(mode="managed", api_host="192.168.1.20"))), "VLLM_HOST=127.0.0.1")
        self.assertEqual(host(ManagedLauncher(Config(mode="managed", api_host="0.0.0.0"))), "VLLM_HOST=127.0.0.1")
        self.assertEqual(host(NativeLauncher(Config(mode="native", prefix="/p", api_host="192.168.1.20"))),
                         "VLLM_HOST=192.168.1.20")
        self.assertEqual(host(NativeLauncher(Config(mode="native", prefix="/p", lan=True))), "VLLM_HOST=0.0.0.0")

    def test_other_devices_get_the_main_address_for_all_interfaces_and_the_chosen_one_otherwise(self):
        with mock.patch.object(settings, "primary_address", return_value="192.168.1.20"):
            self.assertEqual(settings.network_url(Config()), "")
            self.assertEqual(settings.network_url(Config(api_host="0.0.0.0", api_port=9000)), "http://192.168.1.20:9000/v1")
            self.assertEqual(settings.network_url(Config(api_host="100.64.0.7")), "http://100.64.0.7:8000/v1")

    def test_the_settings_file_and_the_command_line_set_the_address(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "qwen-station.json"
            path.write_text(json.dumps({"api_host": "0.0.0.0"}), encoding="utf-8")
            self.assertEqual(load_config(str(path)).bind_host, "0.0.0.0")
            self.assertEqual(load_config(str(path), {"api_host": "127.0.0.1"}).bind_host, "127.0.0.1")


class TempData(unittest.TestCase):
    """The app's data folder points into a temp folder, on either platform."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        patcher = mock.patch.dict(os.environ, {"LOCALAPPDATA": str(self.home), "XDG_CONFIG_HOME": str(self.home)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def station(self, **overrides):
        config = Config(mode="managed", autostart=False, open_window=False, register_clients=False, **overrides)
        return Station(config, Supervisor(config, MockLauncher(config)))

    def saved(self):
        return json.loads(config_module.settings_path().read_text(encoding="utf-8"))


class SavingTests(TempData):
    def test_with_the_model_stopped_a_change_applies_at_once_and_is_written(self):
        station = self.station()
        result = station.save_settings({"api_host": "0.0.0.0", "draft_tokens": 11, "close_action": "keep"})
        self.assertEqual(result, {"ok": True, "changed": ["api_host", "close_action", "draft_tokens"], "pending": [],
                                  "after_restart": []})
        self.assertEqual((station.config.api_host, station.config.draft_tokens, station.config.close_action),
                         ("0.0.0.0", 11, "keep"))
        self.assertEqual(self.saved(), {"api_host": "0.0.0.0", "draft_tokens": 11, "close_action": "keep"})

    def test_while_the_model_runs_its_settings_wait_for_the_next_start(self):
        station = self.station()
        station.supervisor.state = "ready"
        result = station.save_settings({"api_host": "0.0.0.0", "api_port": 8003, "autostart": False})
        self.assertEqual(result["pending"], ["api_host", "api_port"])
        self.assertEqual((station.config.bind_host, station.config.api_port), ("127.0.0.1", 8000))  # still the running ones
        self.assertFalse(station.config.autostart)  # not a model setting: at once
        self.assertEqual(self.saved()["api_port"], 8003)
        self.assertEqual(station.settings_json()["values"]["api_port"], 8003)  # the menu shows what was saved
        self.assertEqual(station.state_json()["settings_pending"], ["api_host", "api_port"])
        station.supervisor.before_start()
        self.assertEqual((station.config.bind_host, station.config.api_port, station.pending), ("0.0.0.0", 8003, {}))

    def test_a_change_taken_back_before_the_restart_leaves_nothing_waiting(self):
        station = self.station()
        station.supervisor.state = "ready"
        station.save_settings({"draft_tokens": 11})
        self.assertEqual(station.save_settings({"draft_tokens": 7})["pending"], [])

    def test_the_dashboard_port_waits_for_the_next_start_of_the_app(self):
        station = self.station()
        result = station.save_settings({"station_port": 8020})
        self.assertEqual(result["after_restart"], ["station_port"])
        self.assertEqual(station.config.station_port, 8010)
        self.assertEqual(self.saved(), {"station_port": 8020})

    def test_an_invalid_value_changes_nothing(self):
        station = self.station()
        result = station.save_settings({"api_host": "10.9.9", "draft_tokens": 11})
        self.assertEqual((result["ok"], result["field"]), (False, "api_host"))
        self.assertEqual(station.config.draft_tokens, 7)
        self.assertFalse(config_module.settings_path().exists())

    def test_the_mock_writes_no_file(self):
        station = self.station(mock=True, api_port=free_port(), station_port=free_port())
        self.assertTrue(station.save_settings({"draft_tokens": 11})["ok"])
        self.assertEqual(station.config.draft_tokens, 11)
        self.assertFalse(config_module.settings_path().exists())

    def test_the_endpoints_need_the_token_of_the_page(self):
        station = self.station(mock=True, api_port=free_port(), station_port=free_port())
        station.serve()
        self.addCleanup(station.shutdown)
        self.assertEqual(request(station, "/api/settings", token=False)[0], 403)
        self.assertEqual(request(station, "/api/settings", method="POST", body={"values": {}}, token=False)[0], 403)
        status, body = request(station, "/api/settings")
        loaded = json.loads(body)
        self.assertEqual((status, loaded["values"]["api_host"], loaded["file"]), (200, "127.0.0.1", ""))
        status, body = request(station, "/api/settings", method="POST", body={"values": {"api_port": 80}})
        self.assertEqual((status, json.loads(body)["field"]), (200, "api_port"))
        self.assertEqual(request(station, "/api/settings", method="POST", body={"values": []})[0], 400)


class WindowsNetworkTests(TempData):
    """Before a start in WSL: the firewall rule follows the chosen address."""

    def run_prepare(self, check_code=0, state=False, **overrides):
        config = Config(mode="managed", **overrides)
        supervisor = Supervisor(config, MockLauncher(config))
        calls = []

        def fake(script, arguments, elevated):
            calls.append((arguments, elevated))
            return check_code if not elevated else 0

        if state:
            folder = config_module.data_dir() / "logs"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / LAN_STATE).write_text("{}", encoding="utf-8")
        with mock.patch.object(supervisor, "_run_lan_script", side_effect=fake):
            supervisor._prepare_network()
        return calls, [line["text"] for line in supervisor.log.since(0)]

    def test_an_api_for_this_computer_only_needs_nothing(self):
        self.assertEqual(self.run_prepare()[0], [])

    def test_a_shared_address_that_is_set_up_asks_for_no_rights(self):
        calls, log = self.run_prepare(api_host="192.168.1.20", api_port=8003)
        self.assertEqual(calls, [(["-Check", "-ListenAddress", "192.168.1.20", "-Port8000", "8003"], False)])
        self.assertIn("open for 192.168.1.20, port 8003", log[-1])

    def test_a_new_address_is_set_up_with_administrator_rights(self):
        calls, _ = self.run_prepare(check_code=2, api_host="0.0.0.0")
        self.assertEqual([elevated for _, elevated in calls], [False, True])
        self.assertEqual(calls[1][0], ["-ListenAddress", "0.0.0.0", "-Port8000", "8000"])

    def test_going_back_to_this_computer_only_removes_what_was_set_up(self):
        calls, _ = self.run_prepare(state=True)
        self.assertEqual(calls, [(["-Remove"], True)])

    def test_a_declined_prompt_does_not_stop_the_start(self):
        config = Config(mode="managed", api_host="0.0.0.0")
        supervisor = Supervisor(config, MockLauncher(config))

        def fake(script, arguments, elevated):
            if elevated:
                raise RuntimeError("the administrator prompt was declined")
            return 2

        with mock.patch.object(supervisor, "_run_lan_script", side_effect=fake):
            supervisor._prepare_network()
        self.assertIn("declined", supervisor.log.since(0)[-1]["text"])

    def test_linux_only_says_where_the_api_listens(self):
        config = Config(mode="native", prefix="/p", api_host="0.0.0.0")
        supervisor = Supervisor(config, MockLauncher(config))
        with mock.patch.object(supervisor, "_run_lan_script") as script:
            supervisor._prepare_network()
        script.assert_not_called()
        self.assertIn("all network interfaces", supervisor.log.since(0)[-1]["text"])


class ApiKeyTests(TempData):
    """The key clients must send: where it goes, and where it must not show up."""

    KEY = "qs-0123456789abcdefghijklmnopqrstuv"

    def test_a_key_must_fit_into_a_header(self):
        config = Config()
        for key in ("short", "has a space in it", "quote\"inside-the-key", "x" * 129):
            self.assertEqual(settings.validate(config, settings.values(config), {"api_key": key}, [])[2], "api_key", key)
            with self.assertRaises(ValueError):
                Config(api_key=key).validate()
        for key in ("", self.KEY, "sk-proj_ABC.def~ghi+jkl/mno="):
            self.assertEqual(settings.validate(config, settings.values(config), {"api_key": key}, [])[1], "", key)

    def test_the_server_gets_the_key_through_its_environment_not_its_command_line(self):
        config = Config(mode="managed", api_key=self.KEY)
        supervisor = Supervisor(config, ManagedLauncher(config))
        with mock.patch.dict(os.environ, {"WSLENV": "OTHER/p", "VLLM_API_KEY": "from-the-desktop"}):
            environment = supervisor._server_environment()
            self.assertEqual(environment["VLLM_API_KEY"], self.KEY)
            if os.name == "nt":  # WSLENV names the variables Windows hands on into the distribution
                self.assertEqual(environment["WSLENV"], "OTHER/p:VLLM_API_KEY/u")
            self.assertNotIn(self.KEY, " ".join(supervisor.launcher.command(PRESETS["262k"])))
            config.api_key = ""
            self.assertNotIn("VLLM_API_KEY", supervisor._server_environment())  # only the setting decides

    def test_the_check_for_a_running_model_sends_the_key(self):
        config = Config(api_key=self.KEY)
        supervisor = Supervisor(config, MockLauncher(config))
        with mock.patch.object(supervisor_module, "http_get", return_value=(401, b"")) as get:
            supervisor.probe_api()
        self.assertEqual(get.call_args.kwargs["key"], self.KEY)

    def test_a_saved_key_is_never_shown_again_only_a_hint(self):
        station = self.station(api_key=self.KEY)
        state, menu = station.state_json(), station.settings_json()
        self.assertEqual((state["api"]["key_hint"], menu["key_hint"]), ("qs-01…tuv", "qs-01…tuv"))
        self.assertNotIn("api_key", menu["values"])
        for answer in (state, menu):
            self.assertNotIn(self.KEY, json.dumps(answer))
        self.assertEqual(self.station().settings_json()["key_hint"], "")

    def test_other_settings_can_be_saved_without_touching_the_key(self):
        station = self.station(api_key=self.KEY)
        self.assertEqual(station.save_settings({"draft_tokens": 9})["changed"], ["draft_tokens"])
        self.assertEqual(station.config.api_key, self.KEY)
        self.assertEqual(station.save_settings({"api_key": ""})["changed"], ["api_key"])
        self.assertEqual(station.config.api_key, "")

    def test_a_new_key_waits_for_the_restart_like_the_other_model_settings(self):
        station = self.station()
        station.supervisor.state = "ready"
        self.assertEqual(station.save_settings({"api_key": self.KEY})["pending"], ["api_key"])
        self.assertEqual(station.config.api_key, "")  # the running server still has none
        self.assertEqual(station.settings_json()["key_hint"], "qs-01…tuv")  # the menu shows what was saved
        station.supervisor.before_start()
        self.assertEqual(station.config.api_key, self.KEY)

    def test_opencode_gets_a_reference_to_the_key_file_and_pi_the_key(self):
        key_file = self.home / "api-key"
        opencode, pi = self.home / "opencode.json", self.home / "pi" / "models.json"
        pi.parent.mkdir()
        result = clients.register_all("http://127.0.0.1:8000/v1", 262144, 1, opencode_file=opencode, pi_file=pi,
                                      api_key=self.KEY, key_file=key_file)
        self.assertEqual([entry["status"] for entry in result.values()], ["ok", "ok"])
        self.assertEqual(key_file.read_text(encoding="ascii"), self.KEY)
        written = opencode.read_text(encoding="utf-8")
        self.assertNotIn(self.KEY, written)  # the config names the file, it does not hold the secret
        reference = clients.key_reference(key_file)
        self.assertEqual(json.loads(written)["provider"]["vllm"]["options"]["apiKey"], reference)
        self.assertTrue(reference.startswith("{file:") and "\\" not in reference)
        self.assertEqual(json.loads(pi.read_text(encoding="utf-8"))["providers"]["vllm"]["apiKey"], self.KEY)
        # A second run must not stumble over its own entry (configs with plain secrets are not rewritten).
        again = clients.register_all("http://127.0.0.1:8000/v1", 262144, 1, opencode_file=opencode, pi_file=pi,
                                     api_key=self.KEY, key_file=key_file)
        self.assertEqual([entry["detail"] for entry in again.values()], ["registered", "registered"])

    def test_removing_the_key_takes_it_out_of_the_configs_and_deletes_the_file(self):
        key_file = self.home / "api-key"
        opencode, pi = self.home / "opencode.json", self.home / "pi" / "models.json"
        pi.parent.mkdir()
        paths = {"opencode_file": opencode, "pi_file": pi, "key_file": key_file}
        clients.register_all("http://127.0.0.1:8000/v1", 262144, 1, api_key=self.KEY, **paths)
        clients.register_all("http://127.0.0.1:8000/v1", 262144, 1, api_key="", **paths)
        self.assertFalse(key_file.exists())
        self.assertNotIn("apiKey", json.loads(opencode.read_text(encoding="utf-8"))["provider"]["vllm"]["options"])
        self.assertEqual(json.loads(pi.read_text(encoding="utf-8"))["providers"]["vllm"]["apiKey"], "local")

    def test_a_key_the_user_wrote_into_a_config_stays(self):
        key_file = self.home / "api-key"
        mine = {"provider": {"vllm": {"options": {"apiKey": "{env:MY_OWN_KEY}"}}}}
        self.assertEqual(clients.apply_opencode(mine, "http://x/v1", 262144, 1, key_file=key_file)
                         ["provider"]["vllm"]["options"]["apiKey"], "{env:MY_OWN_KEY}")
        own = {"providers": {"vllm": {"apiKey": "!pass show vllm"}}}
        self.assertEqual(clients.apply_pi(own, "http://x/v1", 262144, 1)["providers"]["vllm"]["apiKey"], "!pass show vllm")

    def test_a_check_without_writing_leaves_the_key_file_alone(self):
        key_file = self.home / "api-key"
        clients.register_all("http://127.0.0.1:8000/v1", 262144, 1, dry_run=True, opencode_file=self.home / "o.json",
                             pi_file=self.home / "p" / "m.json", api_key=self.KEY, key_file=key_file)
        self.assertFalse(key_file.exists())


class VisionTests(TempData):
    """Vision is optional: the image encoder costs VRAM, so it is only loaded when asked for."""

    def command(self, launcher):
        return launcher.command(PRESETS["262k"])

    def test_the_server_only_gets_the_image_arguments_and_its_own_cache_when_vision_is_on(self):
        off = " ".join(self.command(ManagedLauncher(Config(mode="managed"))))
        self.assertNotIn("limit-mm-per-prompt", off)
        self.assertNotIn("VLLM_CACHE_ROOT", off)
        on = self.command(ManagedLauncher(Config(mode="managed", vision=True, extra_args="--seed 7")))
        self.assertIn("EXTRA_ARGS=--disable-access-log-for-endpoints /metrics,/health "
                      "--limit-mm-per-prompt.image 4 --limit-mm-per-prompt.video 0 --seed 7", on)
        # A graph that vLLM compiled without images does not fit a server with images: separate caches.
        self.assertIn("VLLM_CACHE_ROOT=/opt/qwen-station/cache/vllm-vision", on)
        self.assertIn("VLLM_CACHE_ROOT=/data/engine/cache/vllm-vision",
                      self.command(NativeLauncher(Config(mode="native", prefix="/data/engine", vision=True))))
        self.assertFalse([part for part in on if '"' in part])  # nothing that needs quoting on the way into WSL

    def test_vision_takes_its_share_of_the_vram_from_the_context(self):
        free = 32.0 * 1024  # more than a 32 GB card has free: the numbers only show the difference
        self.assertEqual(supervisor_module.fit_context(free, 262144)[0], 262144)
        with_vision = supervisor_module.FIT_RESERVE_GIB + supervisor_module.VISION_RESERVE_GIB
        self.assertEqual(supervisor_module.fit_context(free, 262144, with_vision)[0], 196608)

    def test_the_start_fits_the_context_for_vision_even_when_the_plain_model_would_fit(self):
        config = Config(mode="managed", vision=True, register_clients=False)
        supervisor = Supervisor(config, MockLauncher(config))
        row = {"index": 0, "name": "RTX 5090", "mem_total": 32607, "mem_free": 32.0 * 1024, "mem_used": 0}
        config.mock = False  # the preflight only looks at the card for a real server
        with mock.patch.object(supervisor_module.gpu, "query_once", return_value=[row]), \
                mock.patch.object(supervisor_module.gpu, "pick", return_value=row), \
                mock.patch.object(supervisor_module.gpu, "gpu_processes", return_value=[]):
            checks = supervisor.run_preflight("free")
        vram = next(check for check in checks if check["id"] == "vram")
        self.assertTrue(vram["ok"])
        self.assertEqual(supervisor.effective_preset()["context"], 196608)
        self.assertIn("Vision takes 1.7 GiB", supervisor.warning)

    def test_the_clients_learn_that_the_model_takes_images_and_forget_it_again(self):
        opencode, pi = self.home / "opencode.json", self.home / "pi" / "models.json"
        pi.parent.mkdir()
        paths = {"opencode_file": opencode, "pi_file": pi}

        def entries():
            return (json.loads(opencode.read_text(encoding="utf-8"))["provider"]["vllm"]["models"]["qwen38-gittensor-dflash"],
                    json.loads(pi.read_text(encoding="utf-8"))["providers"]["vllm"]["models"][0])

        clients.register_all("http://127.0.0.1:8000/v1", 262144, 1, **paths)
        plain, _ = entries()
        self.assertNotIn("attachment", plain)  # nothing new in the config of someone who never uses vision
        clients.register_all("http://127.0.0.1:8000/v1", 262144, 1, vision=True, **paths)
        first, second = entries()
        self.assertTrue(first["attachment"])
        self.assertEqual(first["modalities"], {"input": ["text", "image"], "output": ["text"]})
        self.assertEqual(second["input"], ["text", "image"])
        clients.register_all("http://127.0.0.1:8000/v1", 262144, 1, vision=False, **paths)
        first, second = entries()
        self.assertFalse(first["attachment"])
        self.assertEqual((first["modalities"]["input"], second["input"]), (["text"], ["text"]))

    def test_vision_waits_for_the_restart_like_the_other_model_settings(self):
        station = self.station()
        station.supervisor.state = "ready"
        self.assertEqual(station.save_settings({"vision": True})["pending"], ["vision"])
        self.assertFalse(station.state_json()["model"]["vision"])
        station.supervisor.before_start()
        state = station.state_json()["model"]
        self.assertTrue(state["vision"])
        self.assertTrue(state["subtitle"].endswith(", vision"))
        self.assertEqual(settings.validate(station.config, settings.values(station.config), {"vision": "yes"}, [])[2], "vision")


class SlotTests(TempData):
    """Slots: how many requests the server works on at once. They share the context."""

    def row(self, free_gib):
        return {"index": 0, "name": "RTX 5090", "mem_total": 32607, "mem_free": free_gib * 1024, "mem_used": 0}

    def preflight(self, free_gib, **overrides):
        config = Config(**{"mode": "managed", "register_clients": False, **overrides})
        supervisor = Supervisor(config, MockLauncher(config))
        config.mock = False  # the preflight only looks at the card for a real server
        row = self.row(free_gib)
        with mock.patch.object(supervisor_module.gpu, "query_once", return_value=[row]), \
                mock.patch.object(supervisor_module.gpu, "pick", return_value=row), \
                mock.patch.object(supervisor_module.gpu, "gpu_processes", return_value=[]):
            checks = supervisor.run_preflight("free")
        return supervisor, next(check for check in checks if check["id"] == "vram")

    def test_the_server_gets_the_number_of_slots_and_nothing_else_changes(self):
        def parts(config):
            return ManagedLauncher(config).command(PRESETS["262k"])

        one, three = parts(Config(mode="managed")), parts(Config(mode="managed", slots=3))
        self.assertIn("MAX_SEQS=1", one)
        self.assertIn("MAX_SEQS=3", three)
        # A slot reserves nothing in the KV pool, so the context and the pool stay as they are.
        self.assertEqual([part for part in one if not part.startswith("MAX_SEQS")],
                         [part for part in three if not part.startswith("MAX_SEQS")])

    def test_slots_do_not_shrink_the_context(self):
        for slots in (1, 4):
            supervisor, vram = self.preflight(32.0, slots=slots)
            self.assertTrue(vram["ok"])
            self.assertEqual(supervisor.effective_preset()["context"], 262144)
            self.assertEqual(supervisor.warning, "")

    def test_the_full_context_needs_a_card_that_is_really_free(self):
        # Up to 1.5.0 anything above 29.3 GiB free started the full context, and long sessions then paged.
        supervisor, vram = self.preflight(30.08)
        self.assertEqual(supervisor.effective_preset()["context"], 196608)
        self.assertTrue(vram["ok"] and vram["warn"])
        self.assertIn("30.1 GiB of VRAM is free", supervisor.warning)
        self.assertIn("another graphics card", supervisor.warning)
        self.assertEqual(self.preflight(30.9)[0].effective_preset()["context"], 229376)

    def test_linux_can_use_the_card_to_its_end(self):
        # Windows pages before the card is full, Linux does not: the same free memory gives more context there.
        self.assertEqual(self.preflight(31.5)[0].effective_preset()["context"], 229376)
        self.assertEqual(self.preflight(31.5, mode="native", prefix="/p")[0].effective_preset()["context"], 262144)

    def test_the_pool_follows_the_context_and_the_draft_tokens_not_the_preset(self):
        plain = Supervisor(Config(), MockLauncher(Config())).effective_preset()
        self.assertEqual(plain["kv_bytes"], supervisor_module.kv_pool_bytes(262144))
        more = Config(draft_tokens=11)
        self.assertEqual(Supervisor(more, MockLauncher(more)).effective_preset()["kv_bytes"],
                         supervisor_module.kv_pool_bytes(262144, 11))

    def test_without_the_fit_the_preset_starts_as_it_is(self):
        supervisor, vram = self.preflight(30.08, fit_context=False)
        self.assertEqual(supervisor.effective_preset()["context"], 262144)
        self.assertTrue(vram["warn"])  # it starts, with a warning about what can happen

    def test_ready_waits_until_the_server_can_be_reached(self):
        config = Config(mode="managed")
        supervisor = Supervisor(config, MockLauncher(config))
        supervisor.state, supervisor.started_at = "starting", 1.0
        answers = iter([OSError("refused"), OSError("refused"), (200, b"")])

        def health(url, timeout=2.0, key=""):
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return answer

        with mock.patch.object(supervisor_module, "http_get", side_effect=health) as get, \
                mock.patch.object(supervisor_module.time, "sleep"):
            supervisor._ready_when_reachable()
        self.assertEqual((get.call_count, supervisor.state), (3, "ready"))
        self.assertEqual(get.call_args.args[0], "http://127.0.0.1:8000/health")

    def test_the_number_of_slots_has_limits_and_waits_for_the_restart(self):
        config = Config()
        for slots in (0, 5, True):
            self.assertEqual(settings.validate(config, settings.values(config), {"slots": slots}, [])[2], "slots")
        with self.assertRaises(ValueError):
            Config(slots=5).validate()
        station = self.station()
        station.supervisor.state = "ready"
        self.assertEqual(station.save_settings({"slots": 3})["pending"], ["slots"])
        self.assertEqual(station.state_json()["model"]["max_seqs"], 1)
        station.supervisor.before_start()
        self.assertEqual(station.state_json()["model"]["max_seqs"], 3)

    def test_the_old_two_slot_preset_becomes_the_setting(self):
        path = self.home / "old.json"
        path.write_text(json.dumps({"preset": "parallel-96k"}), encoding="utf-8")
        loaded = load_config(str(path))
        self.assertEqual((loaded.preset, loaded.slots), ("262k", 2))
        self.assertNotIn("parallel-96k", PRESETS)

    def test_the_clients_see_the_slots_in_the_name_of_the_model(self):
        self.assertTrue(clients.model_name(229376, 1).endswith("229K (RTX 5090)"))
        self.assertTrue(clients.model_name(229376, 4).endswith("229K (RTX 5090) - 4 slots"))


class RelayTests(unittest.TestCase):
    """The relay passes connections from other devices on to the server behind 127.0.0.1."""

    def setUp(self):
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen()
        self.addCleanup(self.server.close)
        self.relay = Relay("127.0.0.1", free_port(), target_port=self.server.getsockname()[1])
        self.addCleanup(self.relay.stop)

    def connect(self):
        client = socket.create_connection(("127.0.0.1", self.relay.port), timeout=5)
        self.addCleanup(client.close)
        return client

    def test_bytes_pass_unchanged_in_both_directions_and_a_client_that_leaves_ends_the_request(self):
        # The test can only connect from this machine, which the relay refuses on purpose (next test).
        with mock.patch.object(Relay, "_own_machine", return_value=False):
            self.relay.start()
            client = self.connect()
            upstream, _ = self.server.accept()
            self.addCleanup(upstream.close)
            upstream.settimeout(5)
            client.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n\r\n")
            self.assertEqual(upstream.recv(100), b"POST /v1/chat/completions HTTP/1.1\r\n\r\n")
            for chunk in (b"data: one\n\n", b"data: two\n\n"):  # a stream: every chunk arrives by itself
                upstream.sendall(chunk)
                self.assertEqual(client.recv(100), chunk)
            client.close()
            self.assertEqual(upstream.recv(100), b"")  # the server sees that the client is gone

    def test_connections_from_this_machine_are_not_passed_on(self):
        self.relay.start()
        client = self.connect()
        client.settimeout(5)
        try:
            self.assertEqual(client.recv(10), b"")
        except ConnectionError:
            pass  # reset instead of a clean close: refused either way
        self.server.settimeout(0.3)
        with self.assertRaises(socket.timeout):
            self.server.accept()

    def test_stop_ends_the_listener_and_open_connections(self):
        with mock.patch.object(Relay, "_own_machine", return_value=False):
            self.relay.start()
            client = self.connect()
            upstream, _ = self.server.accept()
            self.addCleanup(upstream.close)
            client.sendall(b"x")
            upstream.settimeout(5)
            self.assertEqual(upstream.recv(10), b"x")
            self.relay.stop()
            self.assertFalse(self.relay.running)
            try:
                self.assertEqual(upstream.recv(10), b"")
            except ConnectionError:
                pass  # a reset instead of a clean close: over either way
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", self.relay.port), timeout=2)

    def test_an_address_in_use_is_reported(self):
        with self.assertRaises(OSError):
            Relay("127.0.0.1", self.server.getsockname()[1]).start()


class SharingTests(unittest.TestCase):
    """Windows: the relay runs while a shared model is ready."""

    def supervisor(self, **overrides):
        config = Config(**overrides)
        return Supervisor(config, MockLauncher(config))

    def test_a_ready_model_is_shared_on_the_chosen_address_until_it_stops(self):
        supervisor = self.supervisor(mode="managed", api_host="192.168.1.20", api_port=8003)
        with mock.patch.object(supervisor_module, "Relay") as relay:
            supervisor._share()
            relay.assert_called_once_with("192.168.1.20", 8003)
            relay.return_value.start.assert_called_once_with()
            supervisor._set_state("stopped", "Stopped")
            relay.return_value.stop.assert_called_once_with()

    def test_linux_and_a_private_api_need_no_relay(self):
        for overrides in ({"mode": "native", "prefix": "/p", "api_host": "0.0.0.0"}, {"mode": "managed"}):
            supervisor = self.supervisor(**overrides)
            with mock.patch.object(supervisor_module, "Relay") as relay:
                supervisor._share()
                relay.assert_not_called()

    def test_an_address_that_cannot_be_used_is_reported_and_the_model_keeps_running(self):
        supervisor = self.supervisor(mode="managed", api_host="0.0.0.0")
        with mock.patch.object(supervisor_module, "Relay") as relay:
            relay.return_value.start.side_effect = OSError("address already in use")
            supervisor._share()
        self.assertIn("could not be shared", supervisor.log.since(0)[-1]["text"])
        self.assertIsNone(supervisor._relay)


if __name__ == "__main__":
    unittest.main()
