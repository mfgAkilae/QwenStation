"""End-to-end against the scripted mock server: real subprocess, real HTTP, no GPU."""

import tests.isolation  # noqa: F401  first of all: no test may reach the real agent configs
import json
import os
import socket
import time
import unittest
import urllib.error
import urllib.request

from qwen_station.app import Station
from qwen_station.config import PRESETS, Config
from qwen_station.supervisor import LogBuffer, MockLauncher, Supervisor, WslLauncher


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_for(predicate, timeout=30.0, message="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {message}")


def request(station, path, method="GET", body=None, token=True, host=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Station-Token"] = station.token
    if host:
        headers["Host"] = host
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{station.config.station_port}{path}", data=data,
                                 headers=headers, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class StationTests(unittest.TestCase):
    def make(self, **overrides):
        config = Config(mock=True, api_port=free_port(), station_port=free_port(), autostart=False,
                        open_window=False, register_clients=False, **overrides)
        supervisor = Supervisor(config, MockLauncher(config, speed=25.0))
        station = Station(config, supervisor)
        station.serve()
        self.addCleanup(station.shutdown)
        return station

    def pump(self, station, predicate, timeout=30.0, message="condition"):
        tick = 0
        deadline = time.time() + timeout
        while time.time() < deadline:
            station.poll_once(tick)
            tick += 1
            if predicate():
                return
            time.sleep(0.05)
        raise AssertionError(f"timed out waiting for {message}")

    def test_start_serves_metrics_and_stops_cleanly(self):
        station = self.make()
        supervisor = station.supervisor
        self.assertTrue(supervisor.start())
        self.assertFalse(supervisor.start())  # a second click while starting changes nothing
        self.pump(station, lambda: supervisor.state == "ready", message="ready")
        self.assertEqual(supervisor.facts["kv_tokens"], 264910)
        self.assertEqual(supervisor.facts["vllm_version"], "0.29.0")
        self.pump(station, lambda: len(station.tracker.requests) >= 1, message="a finished request")
        state = json.loads(request(station, "/api/state")[1])
        self.assertEqual(state["state"], "ready")
        self.assertGreater(state["totals"]["generation_tokens"], 0)
        self.assertEqual(len(state["totals"]["accept_per_pos"]), 7)
        self.assertEqual(state["requests"][0]["finish"].keys() <= {"stop", "length"}, True)
        self.assertTrue(any("Ready after" in line["text"] for line in state["log"]))
        newer = json.loads(request(station, f"/api/state?since={state['now']}&log={state['log_seq']}"
                                            f"&req={state['request_seq']}")[1])
        self.assertEqual((newer["samples"], newer["log"], newer["requests"]), ([], [], []))
        self.assertTrue(supervisor.stop())
        self.assertEqual(supervisor.state, "stopped")
        wait_for(lambda: supervisor.probe_api()[0] == "free", message="port released")

    def test_failed_start_reports_the_vram_hint(self):
        os.environ["QWEN_STATION_MOCK_FAIL"] = "oom"
        self.addCleanup(os.environ.pop, "QWEN_STATION_MOCK_FAIL", None)
        station = self.make()
        station.supervisor.start()
        wait_for(lambda: station.supervisor.state == "failed", message="failure")
        error = station.supervisor.error
        self.assertIn("Not enough free VRAM", error["detail"])
        self.assertIn("less than desired GPU memory", error["raw"])
        self.assertEqual(error["code"], 1)
        self.assertTrue(station.supervisor.start())  # a retry is possible from the failed state

    def test_shutdown_during_the_checks_leaves_no_server_behind(self):
        station = self.make()
        supervisor = station.supervisor
        original = supervisor.run_preflight

        def slow_preflight(port_state):
            time.sleep(0.8)  # the station is shut down while the start is still checking
            return original(port_state)

        supervisor.run_preflight = slow_preflight
        supervisor.start()
        time.sleep(0.2)
        station.shutdown()
        self.assertEqual(supervisor.state, "stopped")
        self.assertIsNone(supervisor._process)
        self.assertEqual(supervisor.probe_api()[0], "free")

    def test_running_server_is_adopted_and_port_conflict_blocks(self):
        first = self.make()
        first.supervisor.start()
        self.pump(first, lambda: first.supervisor.state == "ready", message="ready")
        second_config = Config(mock=True, api_port=first.config.api_port, station_port=free_port(),
                               autostart=False, open_window=False, register_clients=False)
        second = Supervisor(second_config, MockLauncher(second_config))
        second.start()
        wait_for(lambda: second.state == "ready", message="adoption")
        self.assertTrue(second.external)
        self.assertFalse(second.owns_server())
        self.assertEqual(second.preset, "262k")

        foreign = socket.socket()  # something that is not our model listens on the API port
        foreign.bind(("127.0.0.1", 0))
        foreign.listen(1)
        self.addCleanup(foreign.close)
        blocked_config = Config(mock=True, api_port=foreign.getsockname()[1], station_port=free_port())
        blocked = Supervisor(blocked_config, MockLauncher(blocked_config))
        blocked.probe_api = lambda: ("foreign", None)
        blocked.start()
        wait_for(lambda: blocked.state == "blocked", message="blocked")
        self.assertFalse(blocked.error["forceable"])

    def test_api_requires_token_and_loopback_host(self):
        station = self.make()
        self.assertEqual(request(station, "/api/start", "POST", {}, token=False)[0], 403)
        self.assertEqual(request(station, "/api/quit", "POST", {}, token=False)[0], 403)
        self.assertFalse(station.quit_event.is_set())
        self.assertEqual(request(station, "/api/state", host="evil.example:80")[0], 403)
        self.assertEqual(request(station, "/api/start", "POST", {"preset": "nope"})[0], 400)
        status, body = request(station, "/")
        self.assertEqual(status, 200)
        self.assertIn(station.token.encode(), body)
        self.assertNotIn(b"__STATION_TOKEN__", body)
        self.assertEqual(json.loads(request(station, "/api/ping", token=False)[1])["app"], "qwen-station")
        self.assertEqual(request(station, "/api/quit", "POST", {})[0], 200)
        self.assertTrue(station.quit_event.is_set())

    def test_window_close_detection(self):
        station = self.make()
        self.assertFalse(station.window_gone())  # no window was opened
        station.window_opened_at = time.time()
        station.browser_tracked = False
        station.last_poll = time.time()
        self.assertFalse(station.window_gone())
        station.last_poll = time.time() - 30  # a hidden tab polls rarely: not gone yet
        self.assertFalse(station.window_gone())
        station.bye_at = time.time() - 20  # the page unloaded after its last poll and stayed away
        self.assertTrue(station.window_gone())
        station.last_poll = time.time()  # a reload: polls resume
        self.assertFalse(station.window_gone())
        self.assertEqual(request(station, "/api/bye?token=wrong", "POST", token=False)[0], 204)
        self.assertLess(station.bye_at, station.last_poll)
        request(station, f"/api/bye?token={station.token}", "POST", token=False)
        self.assertGreaterEqual(station.bye_at, station.last_poll)


class LauncherTests(unittest.TestCase):
    def test_wsl_command_matches_the_existing_starter(self):
        launcher = WslLauncher(Config(root=r"D:\qwen-engine"))
        command = launcher.command({"context": 262144, "kv_bytes": 10350000000})
        self.assertEqual(command, ["wsl.exe", "-d", "Ubuntu", "--exec", "env", "VLLM_HOST=127.0.0.1",
                                   "VLLM_PORT=8000", "CTX=262144", "KV_BYTES=10350000000", "MAX_SEQS=1",
                                   "EXTRA_ARGS=--disable-access-log-for-endpoints /metrics,/health",
                                   "KV_DTYPE=nvfp4", "bash",
                                   "/mnt/d/qwen-engine/run-variant-logged.sh", "dflash"])
        lan = WslLauncher(Config(root=r"D:\qwen-engine", lan=True, gpu_memory_utilization=0.9,
                                 extra_args="--max-model-len 188416", vllm_prefix="/opt/vllm-0.32.0"))
        lan_command = lan.command({"context": 98304, "kv_bytes": 10350000000})
        self.assertIn("VLLM_HOST=127.0.0.1", lan_command)  # in WSL always; other devices come in through the relay
        self.assertIn("GPU_UTIL=0.9", lan_command)
        self.assertIn("EXTRA_ARGS=--disable-access-log-for-endpoints /metrics,/health --max-model-len 188416", lan_command)
        self.assertIn("VLLM_PREFIX=/opt/vllm-0.32.0", lan_command)
        from qwen_station.config import PRESETS
        unsloth = launcher.command(PRESETS["unsloth-82k"])
        self.assertIn("TARGET_NAME=Qwen3.8-27B-NVFP4-Unsloth", unsloth)
        self.assertIn("TARGET_QUANT=compressed-tensors", unsloth)
        self.assertIn("CTX=81920", unsloth)

    def test_xqa_decode_backend_raises_a_warning(self):
        config = Config(mock=True)
        line = "(EngineCore pid=1) INFO [flashinfer.py:910] FlashInfer resolved query dtypes: prefill=torch.bfloat16, decode=torch.bfloat16, decode_backend={}, kv_cache_dtype=torch.float8_e4m3fn, arch=sm120"
        newer = Supervisor(config, MockLauncher(config))
        newer.state = "starting"
        newer._handle_line(b"(APIServer pid=1) INFO [api_utils.py:347]  version 0.32.0", False)
        newer._handle_line(line.format("xqa").encode(), False)
        self.assertEqual(newer.snapshot()["warning"], "")  # on 0.32 XQA is the intended kernel
        newer._handle_line(line.format("flashinfer-native").encode(), False)
        self.assertIn("does not use the XQA decode kernel", newer.snapshot()["warning"])
        supervisor = Supervisor(config, MockLauncher(config))
        supervisor.state = "starting"
        supervisor._handle_line(b"(APIServer pid=1) INFO [api_utils.py:347]  version 0.29.0", False)
        supervisor._handle_line(line.format("flashinfer-native").encode(), False)
        self.assertEqual(supervisor.snapshot()["warning"], "")
        supervisor._handle_line(line.format("xqa").encode(), False)
        supervisor._handle_line(line.format("xqa").encode(), False)
        self.assertEqual(supervisor.facts["decode_backend"], "xqa")
        self.assertEqual(supervisor.snapshot()["warning"].count("XQA decode kernel"), 1)

    def test_context_is_fitted_to_the_free_vram(self):
        from qwen_station.supervisor import fit_context, kv_pool_bytes
        self.assertEqual(fit_context(40000, 262144)[0], 262144)  # never more than the preset
        # Measured 2026-11-17: with 30.08 GiB free the full context ran the card full and a long session paged;
        # with 30.6 GiB free 229,376 tokens stayed fast, also after two long sessions.
        self.assertEqual(fit_context(30.08 * 1024, 262144)[0], 196608)
        context, kv_bytes = fit_context(30.9 * 1024, 262144)
        self.assertEqual(context, 229376)
        self.assertAlmostEqual(kv_bytes / 2 ** 30, 8.87, places=2)
        # The whole context needs 31.9 GiB free: more than a 32 GB card has under Windows.
        self.assertEqual(fit_context(31.84 * 1024, 262144)[0], 229376)
        self.assertEqual(fit_context(31.95 * 1024, 262144)[0], 262144)
        # Measured 2026-11-17: 156,884 tokens filled 97 percent of a 6.617 GiB pool, and a prompt of 99.5
        # percent of the context never finished. The pool for a context must hold a request of that size.
        self.assertGreater(kv_pool_bytes(163840) / 2 ** 30, (1.59 + 163840 / 32500) * 1.01)
        self.assertEqual(fit_context(28.85 * 1024, 262144)[0], 163840)
        self.assertIsNone(fit_context(8000, 262144))  # another model holds the card: do not start at all
        self.assertEqual(fit_context(29.8 * 1024, 81920, 25.9)[0], 65536)  # the larger Unsloth target
        # 163,840 tokens needed 6.51 GiB in the real server; the computed pool must cover that
        self.assertGreater(fit_context(28.85 * 1024, 163840)[1], 6.51 * 2 ** 30)
        config = Config(mock=True)
        supervisor = Supervisor(config, MockLauncher(config))
        self.assertEqual(supervisor.effective_preset()["context"], 262144)
        supervisor.fitted = (200704, 8_300_000_000)
        self.assertEqual((supervisor.effective_preset()["context"], supervisor.effective_preset()["kv_bytes"]),
                         (200704, 8_300_000_000))
        self.assertEqual(supervisor.snapshot()["context"], 200704)

    def test_more_draft_tokens_take_kv_memory_and_so_context(self):
        from qwen_station.supervisor import fit_context, kv_pool_bytes
        # Measured 2026-11-17 for 163,840 tokens: the server needed 7.15 GiB with 11 and 7.84 GiB with 15 draft tokens.
        self.assertGreater(kv_pool_bytes(163840, 11), 7.15 * 2 ** 30)
        self.assertGreater(kv_pool_bytes(163840, 15), 7.84 * 2 ** 30)
        self.assertLess(kv_pool_bytes(163840, 15), 8.3 * 2 ** 30)
        self.assertEqual(kv_pool_bytes(163840, 7), fit_context(28.85 * 1024, 163840)[1])
        self.assertEqual(kv_pool_bytes(163840, 3), kv_pool_bytes(163840, 7))  # fewer never shrinks the pool
        free = 32.0 * 1024
        self.assertEqual(fit_context(free, 262144)[0], 262144)
        self.assertEqual(fit_context(free, 262144, draft_tokens=11)[0], 229376)
        self.assertEqual(fit_context(free, 262144, draft_tokens=15)[0], 196608)
        self.assertEqual(fit_context(30.25 * 1024, 262144, draft_tokens=15)[0], 163840)
        launcher = WslLauncher(Config(root=r"D:\qwen-engine", draft_tokens=11))
        self.assertIn("SPEC_TOKENS=11", launcher.command(PRESETS["262k"]))
        # More slots do not change that: every running request has its own draft states in the shared pool.
        self.assertIn("SPEC_TOKENS=11", WslLauncher(Config(root=r"D:\qwen-engine", draft_tokens=11, slots=3))
                      .command(PRESETS["262k"]))
        self.assertFalse([part for part in WslLauncher(Config(root=r"D:\qwen-engine")).command(PRESETS["262k"])
                          if part.startswith("SPEC_TOKENS")])
        with self.assertRaises(ValueError):
            Config(draft_tokens=16).validate()

    def test_stop_script_only_targets_this_installation(self):
        script = WslLauncher(Config(root=r"D:\qwen-engine")).stop_script()
        self.assertIn(r"pat='vllm serve [^ ]*/Qwen3\.8-27B'", script)
        self.assertIn('kill -TERM -- -"$pg"', script)

    def test_log_buffer_levels_progress_and_cursor(self):
        log = LogBuffer(size=5)
        log.add("(EngineCore pid=1) ERROR boom")
        log.add("\x1b[32m(APIServer pid=2) WARNING careful\x1b[0m")
        log.add("=== step")
        log.add("10%", progress=True)
        log.add("20%", progress=True)
        lines = log.since(0)
        self.assertEqual([line["level"] for line in lines], ["error", "warn", "step", "info"])
        self.assertEqual(lines[1]["text"], "(APIServer pid=2) WARNING careful")
        self.assertEqual(lines[-1]["text"], "20%")
        self.assertEqual(log.since(lines[-1]["seq"]), [])
        self.assertEqual(log.errors(), ["(EngineCore pid=1) ERROR boom"])


if __name__ == "__main__":
    unittest.main()
