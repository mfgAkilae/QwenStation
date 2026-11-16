"""Starts, watches and stops the vLLM server that runs inside WSL."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

from . import clients, gpu
from .config import (DRAFTER_DIR, MODEL_ID, PRESETS, TARGET_DIR, WRAPPER_SCRIPT, Config, app_dir, data_dir,
                     resource_path)

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
READY_MARKER = "Application startup complete."
LAN_SCRIPT = "setup-qwen-lan.ps1"
# Shortfall the start tolerates. Measured 2026-11-16: 26.7 GiB free worked (Windows evicted other
# programs' GPU memory), but the start took 9 minutes instead of 4.
VRAM_TOLERANCE_MIB = 3072  # only used when fit_context is off
# KV pool sizing for this model (FP8 KV, group size 8), measured 16.11.2026: a pool of P GiB holds
# (P - 1.546) * 32730 tokens (9.64 GiB -> 264,910 tokens, 6.61 GiB -> 165,737 tokens).
KV_FIXED_GIB = 1.546
KV_TOKENS_PER_GIB = 32730
# Free VRAM the server needs beside its KV pool: weights and runtime (20.4 GiB measured) plus a margin.
# With less, Windows pages GPU memory and the speed drops to a quarter (prefill 1,825 against 8,800 tok/s).
FIT_RESERVE_GIB = 21.0
MIN_FIT_CONTEXT = 65536
# Coarse steps on purpose: every distinct context is a new server configuration, and each of those runs
# once without the verified kernel tuning (see RETUNE_WARNING). Six sizes instead of fifty.
FIT_STEP = 32768


def fit_context(free_mib: float, max_context: int, reserve_gib: float = FIT_RESERVE_GIB) -> tuple[int, int] | None:
    """Largest context (and its KV pool in bytes) that fits into the free VRAM, or None if too little is free."""
    tokens = (free_mib / 1024 - reserve_gib - KV_FIXED_GIB) * KV_TOKENS_PER_GIB
    context = min(max_context, int(tokens) // FIT_STEP * FIT_STEP)
    if context < MIN_FIT_CONTEXT:
        return None
    return context, int((KV_FIXED_GIB + context / KV_TOKENS_PER_GIB) * 1.01 * 2 ** 30)
# Measured 2026-11-16 on vLLM 0.29.0: 128 against 254 tok/s at 33K context, 51 against 178 tok/s at 129K.
XQA_WARNING = ("The server uses the XQA decode kernel. With spec decode, the speed drops sharply at long context "
               "(measured up to 3.5 times slower). The current start script forces the native kernel.")
# Measured 2026-11-16 on vLLM 0.32.0: 151 against 225 to 289 tok/s at 129K context.
NATIVE_WARNING = ("The server does not use the XQA decode kernel. From vLLM 0.32 on, XQA is the faster kernel. "
                  "Without it, the speed at long context is much lower.")
RETUNE_WARNING = ("FlashInfer measured the kernels for this configuration again. The result varies and can cost "
                  "up to 40% of the speed. The verified values are now stored: stop the model once and start it again.")
# The dashboard polls these once a second; without the filter they bury the log.
_POLL_NOISE = re.compile(r'"GET /(?:metrics|health|ping) HTTP')
QUIET_POLL_ARGS = "EXTRA_ARGS=--disable-access-log-for-endpoints /metrics,/health"
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_PREFIX = re.compile(r"^\((?:APIServer|EngineCore)[^)]*\)\s*")

# key, label, log marker that opens the phase, typical seconds (start log of 2026-09-19)
PHASES = (
    ("prepare", "Prepare and check patches", None, 6),
    ("boot", "Start vLLM", re.compile(r"=== .* (?:Starte|Starting) /|version \d+\.\d+"), 10),
    ("weights", "Load weights", re.compile(r"Loading model from scratch"), 108),
    ("compile", "Compile and warm up", re.compile(r"Model loading took"), 92),
    ("kv", "Allocate the KV cache, warm up kernels", re.compile(r"Initial free memory|GPU KV cache size"), 28),
    ("api", "Start the API", re.compile(r"init engine .* took|Starting vLLM server"), 3),
)
_FACTS = (
    ("kv_tokens", re.compile(r"GPU KV cache size: ([\d,]+) tokens"), lambda text: int(text.replace(",", ""))),
    ("model_gib", re.compile(r"Model loading took ([\d.]+) GiB"), float),
    ("vllm_version", re.compile(r"\bversion (\d+\.\d+\.\d+)"), str),
    ("max_model_len", re.compile(r"Using max model len (\d+)"), int),
    ("free_gib", re.compile(r"Initial free memory ([\d.]+) GiB"), float),
    ("decode_backend", re.compile(r"decode_backend=([a-z0-9_-]+), kv_cache_dtype"), str),
)
_HINTS = (
    (re.compile(r"out of memory|less than desired GPU memory|No available memory for the cache blocks", re.I),
     "Not enough free VRAM on the 5090. Close other GPU programs (for example ComfyUI or LM Studio) and start again."),
    (re.compile(r"address already in use", re.I), "The API port is already in use."),
    (re.compile(r"no distribution with the supplied name|WSL_E_DISTRO_NOT_FOUND", re.I),
     "The WSL distribution was not found."),
    (re.compile(r"Patch fehlgeschlagen|Patch failed"), "A vLLM patch could not be applied. See the log for details."),
)


def http_get(url: str, timeout: float = 2.0) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"Accept": "*/*"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route localhost via a proxy
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _dots(number: int) -> str:
    return f"{number:,}"


def _gib(mib: float) -> str:
    return f"{mib / 1024:.1f}"


def self_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, str(app_dir() / "run-station.py")]


class LogBuffer:
    def __init__(self, size: int = 3000) -> None:
        self._lines: deque[dict] = deque(maxlen=size)
        self._seq = 0
        self._lock = threading.Lock()

    @property
    def seq(self) -> int:
        return self._seq

    def add(self, text: str, level: str | None = None, progress: bool = False) -> None:
        text = _ANSI.sub("", text).rstrip()
        if not text:
            return
        if level is None:
            if re.search(r"\bERROR\b|Traceback|Error:|Exception", text):
                level = "error"
            elif "WARNING" in text:
                level = "warn"
            elif text.startswith(("===", "---")):
                level = "step"
            else:
                level = "info"
        with self._lock:
            if progress and self._lines and self._lines[-1].get("progress"):
                self._lines.pop()  # a progress bar redraws one line instead of filling the log
            self._seq += 1
            self._lines.append({"seq": self._seq, "t": round(time.time(), 3), "text": text[:2000],
                                "level": level, "progress": progress})

    def since(self, seq: int, limit: int = 400) -> list[dict]:
        with self._lock:
            lines = [line for line in self._lines if line["seq"] > seq]
        return lines[-limit:]

    def errors(self, count: int = 6) -> list[str]:
        with self._lock:
            return [line["text"] for line in self._lines if line["level"] == "error"][-count:]


class WslLauncher:
    """Legacy mode: runs the start script of a hand-built engine folder in an existing WSL distribution."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def command(self, preset: dict) -> list[str]:
        config = self.config
        env = [
            f"VLLM_HOST={'0.0.0.0' if config.lan else '127.0.0.1'}",
            f"VLLM_PORT={config.api_port}",
            f"CTX={preset['context']}",
            f"KV_BYTES={preset['kv_bytes']}",
            f"MAX_SEQS={preset['max_seqs']}",
        ]
        if abs(config.gpu_memory_utilization - 0.92) > 1e-9:
            env.append(f"GPU_UTIL={config.gpu_memory_utilization}")
        # The wrapper word-splits EXTRA_ARGS onto the vllm command line.
        env.append(f"{QUIET_POLL_ARGS} {config.extra_args}".strip())
        if config.vllm_prefix:
            env.append(f"VLLM_PREFIX={config.vllm_prefix}")
        if config.model_base:
            env.append(f"MODEL_BASE={config.model_base}")
        env.extend(f"{key}={value}" for key, value in preset.get("env", {}).items())  # e.g. another target
        env.extend(config.extra_env.split())  # KEY=VALUE pairs for the start scripts, e.g. KV_DTYPE=nvfp4
        return [*self.wsl(), "env", *env, *self.script()]

    def wsl(self) -> list[str]:
        return ["wsl.exe", "-d", self.config.distro, "--exec"]

    def script(self) -> list[str]:
        return ["bash", f"{self.config.linux_root}/{WRAPPER_SCRIPT}", "dflash"]

    def status_file(self) -> str:
        return f"{self.config.linux_root}/logs/dflash-status.txt"

    def stop_script(self) -> str:
        # Same match the wrapper uses for its own cleanup: Qwen3.8-27B servers, wherever the weights live.
        pattern = r"vllm serve [^ ]*/Qwen3\.8-27B"
        # SIGTERM also ends the wrapper before it can update its status file, so the script does it.
        status = self.status_file()
        return (
            f"pat='{pattern}'\n"
            f"done_file='{status}'\n"
            'finish() { [ -f "$done_file" ] && echo "exited station-stop" > "$done_file"; exit 0; }\n'
            'for p in $(pgrep -f "$pat"); do\n'
            "  pg=$(ps -o pgid= -p \"$p\" 2>/dev/null | tr -d ' ')\n"
            '  [ -n "$pg" ] && kill -TERM -- -"$pg" 2>/dev/null\n'
            "done\n"
            'for i in $(seq 1 40); do pgrep -f "$pat" >/dev/null || finish; sleep 1; done\n'
            'pgrep -f "$pat" | xargs -r kill -KILL 2>/dev/null\n'
            "finish\n"
        )

    def stop(self, process: subprocess.Popen | None, log: LogBuffer) -> None:
        try:
            # The script travels on stdin, so no quoting can break on the way through wsl.exe.
            subprocess.run([*self.wsl(), "bash", "-s"],
                           input=self.stop_script().encode("utf-8"), capture_output=True,
                           timeout=70, creationflags=NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.add(f"Stop script in WSL failed: {exc}", "error")
        if process is not None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()

    def distro_check(self) -> tuple[bool, str]:
        try:
            result = subprocess.run(["wsl.exe", "-l", "-q"], capture_output=True, timeout=20, creationflags=NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"wsl.exe does not respond: {exc}"
        names = [name.strip() for name in result.stdout.decode("utf-16-le", errors="ignore").splitlines()]
        if self.config.distro in names:
            return True, self.config.distro
        return False, f"'{self.config.distro}' is missing from 'wsl -l'"

    def file_checks(self, preset: dict | None = None) -> list[dict]:
        root = Path(self.config.root)
        target = (preset or {}).get("env", {}).get("TARGET_NAME", TARGET_DIR)
        missing = [name for name in (WRAPPER_SCRIPT, "test-qwen38-gittensor-dflash.sh") if not (root / name).is_file()]
        if not self.config.model_base:
            missing += [name for name in (target, DRAFTER_DIR) if not (root / name).is_dir()]
        # Paths inside WSL (weights on the Linux filesystem, a non-default venv) can only be checked from there.
        inside = [f"{self.config.model_base}/{name}" for name in (target, DRAFTER_DIR)] if self.config.model_base else []
        if self.config.vllm_prefix:
            inside.append(f"{self.config.vllm_prefix}/bin/vllm")
        for path in inside:
            try:
                found = subprocess.run([*self.wsl(), "test", "-e", path],
                                       capture_output=True, timeout=60, creationflags=NO_WINDOW).returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                found = False
            if not found:
                missing.append(f"{path} (in WSL)")
        if inside and missing:
            return [{"id": "files", "label": "Installation", "ok": False, "blocking": True,
                     "detail": "Missing: " + ", ".join(missing)}]
        return [{"id": "files", "label": "Installation", "ok": not missing, "blocking": True,
                 "detail": str(root) if not missing else f"Missing in {root}: {', '.join(missing)}"}]


class ManagedLauncher(WslLauncher):
    """The release: serve.sh from the engine bundle, inside the distribution the app set up."""

    def wsl(self) -> list[str]:
        return ["wsl.exe", "-d", self.config.distro, "-u", "root", "--exec"]

    def script(self) -> list[str]:
        return ["bash", f"{self.config.prefix}/engine/serve.sh"]

    def status_file(self) -> str:
        return f"{self.config.prefix}/logs/status.txt"

    def file_checks(self, preset: dict | None = None) -> list[dict]:
        target = (preset or {}).get("env", {}).get("TARGET_NAME", TARGET_DIR)
        base = self.config.model_base or f"{self.config.prefix}/models"
        needed = {
            f"{self.config.prefix}/installed.json": "The engine is not set up",
            f"{base}/{target}/config.json": f"The model {target} is not installed",
        }
        missing = []
        for path, message in needed.items():
            try:
                found = subprocess.run([*self.wsl(), "test", "-e", path], capture_output=True, timeout=60,
                                       creationflags=NO_WINDOW).returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                found = False
            if not found:
                missing.append(message)
        return [{"id": "files", "label": "Installation", "ok": not missing, "blocking": True,
                 "detail": "; ".join(missing) if missing else f"{self.config.distro}:{self.config.prefix}"}]


class MockLauncher:
    """A scripted fake server for tests and screenshots. No GPU, no WSL."""

    def __init__(self, config: Config, speed: float = 1.0) -> None:
        self.config = config
        self.speed = speed

    def command(self, preset: dict) -> list[str]:
        return [*self_command(), "--mock-server", str(self.config.api_port), "--mock-speed", str(self.speed),
                "--mock-context", str(preset["context"])]

    def stop(self, process: subprocess.Popen | None, log: LogBuffer) -> None:
        if process is not None and process.poll() is None:
            if os.name == "nt":
                # A venv's python.exe is a launcher with the real interpreter as its child: end the whole tree.
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                               creationflags=NO_WINDOW)
            else:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    def distro_check(self) -> tuple[bool, str]:
        return True, "Mock"

    def file_checks(self, preset: dict | None = None) -> list[dict]:
        return [{"id": "files", "label": "Installation", "ok": True, "blocking": True, "detail": "Mock"}]


class Supervisor:
    def __init__(self, config: Config, launcher=None) -> None:
        self.config = config
        self.launcher = launcher or (
            MockLauncher(config) if config.mock
            else ManagedLauncher(config) if config.mode == "managed" else WslLauncher(config))
        self.log = LogBuffer()
        self.state = "idle"
        self.state_since = time.time()
        self.external = False
        self.message = ""
        self.error: dict | None = None
        self.warning = ""
        self.fitted: tuple[int, int | None] | None = None  # (context, kv_bytes) when it differs from the preset
        self.preset = config.preset
        self.preflight: list[dict] = []
        self.gpu_processes: list[dict] = []
        self.facts: dict = {}
        self.clients: dict = {}
        self.started_at: float | None = None
        self.ready_at: float | None = None
        self.generation = 0  # increments per start, so pollers can reset their baselines
        self._phase_index = 0
        self._phase_started: list[float | None] = [None] * len(PHASES)
        self._process: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._op_lock = threading.Lock()
        self._stopping = False
        self._closed = False  # set on shutdown: a start still in its checks must not launch a server

    # ---- state -----------------------------------------------------------------

    def _set_state(self, state: str, message: str = "") -> None:
        with self._lock:
            if state != self.state:
                self.state = state
                self.state_since = time.time()
            self.message = message

    def phases(self) -> list[dict]:
        now = time.time()
        rows = []
        with self._lock:
            for index, (key, label, _, typical) in enumerate(PHASES):
                started = self._phase_started[index]
                following = next((value for value in self._phase_started[index + 1:] if value), None)
                if started is None:
                    status, seconds = ("skipped" if following or self.ready_at else "pending"), None
                elif following or self.ready_at:
                    status, seconds = "done", (following or self.ready_at) - started
                elif self.state == "starting":
                    status, seconds = "active", now - started
                else:
                    status, seconds = "stopped", None
                rows.append({"key": key, "label": label, "status": status, "typical": typical,
                             "seconds": None if seconds is None else round(max(seconds, 0.0), 1)})
        return rows

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "state_since": round(self.state_since, 3),
                "external": self.external,
                "message": self.message,
                "error": self.error,
                "warning": self.warning if self.state in ("starting", "ready") else "",
                "preset": self.preset,
                "context": self.effective_preset()["context"],
                "preflight": self.preflight,
                "gpu_processes": self.gpu_processes,
                "facts": dict(self.facts),
                "clients": self.clients,
                "started_at": self.started_at,
                "ready_at": self.ready_at,
                "startup_seconds": round(self.ready_at - self.started_at, 1)
                if self.ready_at and self.started_at and not self.external else None,
                "generation": self.generation,
            }

    # ---- preflight ---------------------------------------------------------------

    def probe_api(self) -> tuple[str, dict | None]:
        """'free', 'ours' (this model already answers) or 'foreign' (something else owns the port)."""
        try:
            status, body = http_get(f"http://127.0.0.1:{self.config.api_port}/v1/models", timeout=2.0)
        except (OSError, urllib.error.URLError):
            return "free", None
        if status == 200:
            try:
                data = json.loads(body).get("data", [])
            except (ValueError, AttributeError):
                data = []
            for card in data if isinstance(data, list) else []:
                if isinstance(card, dict) and card.get("id") == MODEL_ID:
                    return "ours", card
        return "foreign", None

    def run_preflight(self, port_state: str) -> list[dict]:
        checks = list(self.launcher.file_checks(PRESETS[self.preset]))
        ok, detail = self.launcher.distro_check()
        checks.append({"id": "wsl", "label": "WSL distribution", "ok": ok, "blocking": True, "detail": detail})
        checks.append({"id": "port", "label": f"Port {self.config.api_port}", "ok": port_state != "foreign",
                       "blocking": True,
                       "detail": "free" if port_state == "free" else
                       ("The model is already running" if port_state == "ours" else "in use by another service")})
        processes: list[dict] = []
        warning = ""
        fitted: tuple[int, int] | None = None
        if not self.config.mock:
            try:
                row = gpu.pick(gpu.query_once(), self.config.gpu_match)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                row = None
                checks.append({"id": "vram", "label": "Free VRAM", "ok": False, "blocking": False,
                               "detail": f"Cannot read nvidia-smi: {exc}"})
            if row and row.get("mem_total") and row.get("mem_free") is not None:
                # The model ends up holding about what vLLM budgets (measured 29.6 GiB at 0.92 x 31.8 GiB).
                need = math.ceil(self.config.gpu_memory_utilization * row["mem_total"])
                enough = row["mem_free"] >= need
                preset = PRESETS[self.preset]
                if not enough and self.config.fit_context and preset["max_seqs"] == 1:
                    fitted = fit_context(row["mem_free"], preset["context"],
                                         preset.get("fit_reserve_gib", FIT_RESERVE_GIB))
                tight = not enough and not self.config.fit_context and row["mem_free"] >= need - VRAM_TOLERANCE_MIB
                detail = f"{_gib(row['mem_free'])} GiB free, the model takes about {_gib(need)} GiB"
                if fitted:
                    detail += f"; context reduced to {_dots(fitted[0])} tokens"
                checks.append({"id": "vram", "label": "Free VRAM", "ok": enough or tight or fitted is not None,
                               "warn": tight or fitted is not None, "blocking": False,
                               "free_mib": row["mem_free"], "need_mib": need, "detail": detail})
                if fitted:
                    warning = (f"Only {_gib(row['mem_free'])} GiB of VRAM is free. For {_dots(preset['context'])} "
                               f"tokens the model needs about {_gib(need)} GiB, so the context is reduced to "
                               f"{_dots(fitted[0])} tokens. Otherwise Windows pages out GPU memory and the speed "
                               "drops to a quarter. For the full context, close other GPU programs and start again.")
                if tight:
                    warning = (f"Only {_gib(row['mem_free'])} GiB of VRAM was free at the start, and the model takes "
                               f"about {_gib(need)} GiB. Windows then pages out the GPU memory of other programs. "
                               "The start takes longer and the speed can suffer.")
                if not enough:
                    processes = gpu.gpu_processes(max_mib=row["mem_total"])
        with self._lock:
            self.preflight = checks
            self.gpu_processes = processes
            self.warning = warning
            self.fitted = fitted
        if warning:
            self.log.add(warning, "warn")
        return checks

    # ---- start -------------------------------------------------------------------

    def start(self, preset: str | None = None, force: bool = False) -> bool:
        """Returns False when a start or stop is already in progress."""
        if preset is not None and preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        with self._lock:
            if self.state in ("preflight", "starting", "ready", "stopping"):
                return False
            if preset:
                self.preset = preset
            self.state = "preflight"
            self.state_since = time.time()
            self.message = "Checking requirements"
            self.error = None
            self.external = False
        threading.Thread(target=self._start_worker, args=(force,), name="start", daemon=True).start()
        return True

    def _start_worker(self, force: bool) -> None:
        with self._op_lock:
            try:
                self._start(force)
            except Exception as exc:  # a crash here must end in a visible state, not a silent thread death
                self.log.add(f"Start aborted: {exc!r}", "error")
                self._fail(f"Start aborted: {exc}")

    def effective_preset(self) -> dict:
        """The preset, with context and KV pool replaced when they were fitted to the free VRAM."""
        preset = dict(PRESETS[self.preset])
        if self.fitted:
            preset["context"] = self.fitted[0]
            if self.fitted[1]:
                preset["kv_bytes"] = self.fitted[1]
        return preset

    def _start(self, force: bool) -> None:
        port_state, card = self.probe_api()
        checks = self.run_preflight(port_state)
        preset = self.effective_preset()
        if port_state == "ours":
            self._adopt(card or {})
            return
        failed = [check for check in checks if not check["ok"]]
        blocking = [check for check in failed if check["blocking"]]
        registers = self.config.register_clients and not self.config.mock
        if registers and not blocking:
            # Before the VRAM gate on purpose: the agents should list the model even while the GPU is busy.
            self.refresh_clients(write=True)
        elif registers:
            self.refresh_clients(write=False)
        if blocking or (failed and not force):
            reason = "; ".join(f"{check['label']}: {check['detail']}" for check in (blocking or failed))
            self.log.add(f"Start not possible: {reason}", "error" if blocking else "warn")
            with self._lock:
                self.error = {"title": "Start blocked" if blocking else "Not enough free VRAM",
                              "detail": reason, "forceable": not blocking}
            self._set_state("blocked", reason)
            return
        if self.config.lan and not self.config.mock:
            self._ensure_lan()
        with self._lock:
            self.generation += 1
            self.facts = {}
            self.started_at = time.time()
            self.ready_at = None
            self._phase_index = 0
            self._phase_started = [None] * len(PHASES)
            self._phase_started[0] = self.started_at
            self._stopping = False
        if self._closed:
            self._set_state("stopped", "Stopped")
            return
        command = self.launcher.command(preset)
        self.log.add(f"=== Start: preset {self.preset}, context {preset['context']}, "
                     f"KV {preset['kv_bytes']} bytes, {preset['max_seqs']} slot(s)", "step")
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
        except OSError as exc:
            self._fail(f"The start command could not be run: {exc}")
            return
        with self._lock:
            self._process = process
        self._set_state("starting", "Loading the model")
        threading.Thread(target=self._read_output, args=(process,), name="server-log", daemon=True).start()

    def _adopt(self, card: dict) -> None:
        context = card.get("max_model_len")
        with self._lock:
            self.generation += 1
            self.external = True
            self.started_at = None
            self.ready_at = time.time()
            self.facts = {"max_model_len": context} if context else {}
            for name, values in PRESETS.items():
                if values["context"] == context:
                    self.preset = name
            # A server started elsewhere with another context: the agents must get its real limit.
            self.fitted = (context, None) if context and context != PRESETS[self.preset]["context"] else None
        self.log.add("=== The model is already running on this port (started elsewhere). "
                     "Qwen Station only monitors it.", "step")
        if self.config.register_clients and not self.config.mock:
            self.refresh_clients(write=True)
        self._set_state("ready", "Already running, started elsewhere")

    def refresh_clients(self, write: bool) -> None:
        preset = self.effective_preset()
        result = clients.register_all(self.config.api_base_url, preset["context"], preset["max_seqs"],
                                      dry_run=not write, variant=preset.get("variant", ""))
        for entry in result.values():
            if entry.get("changed"):
                self.log.add(f"--- {entry['client']}: model {entry['model']} {entry['detail']} ({entry['path']})", "step")
            elif entry["status"] == "error":
                self.log.add(f"{entry['client']} registration failed: {entry['detail']}", "warn")
        with self._lock:
            self.clients = result

    def _lan_script(self) -> str:
        if self.config.mode != "managed":
            return str(Path(self.config.root) / LAN_SCRIPT)
        # The script keeps its state next to itself, so it needs a folder that outlives this process.
        source = resource_path(LAN_SCRIPT)
        target = data_dir() / LAN_SCRIPT
        if not target.is_file() or target.read_bytes() != source.read_bytes():
            shutil.copyfile(source, target)
        (data_dir() / "logs").mkdir(exist_ok=True)
        return str(target)

    def _ensure_lan(self) -> None:
        base = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass"]
        try:
            setup = self._lan_script()
            check = subprocess.run([*base, "-File", setup, "-Check", "-Distro", self.config.distro],
                                   capture_output=True, timeout=120, creationflags=NO_WINDOW)
            if check.returncode == 0:
                self.log.add("--- LAN access is set up.", "step")
                return
            self.log.add("--- LAN access is missing: Windows asks for administrator rights (UAC).", "step")
            arguments = f'-NoProfile -ExecutionPolicy Bypass -File "{setup}" -Distro "{self.config.distro}"'
            elevate = ("$p = Start-Process -FilePath powershell.exe -Verb RunAs -WindowStyle Hidden "
                       f"-ArgumentList '{arguments}' -Wait -PassThru; exit $p.ExitCode")
            result = subprocess.run([*base, "-Command", elevate], capture_output=True, timeout=300,
                                    creationflags=NO_WINDOW)
            if result.returncode != 0:
                raise RuntimeError(f"Exit code {result.returncode}")
            self.log.add("--- LAN access set up.", "step")
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            self.log.add(f"LAN access incomplete ({exc}). The local endpoint starts anyway.", "warn")

    # ---- output ------------------------------------------------------------------

    def _read_output(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        descriptor = process.stdout.fileno()
        pending = b""
        while True:
            try:
                chunk = os.read(descriptor, 65536)
            except OSError:
                break
            if not chunk:
                break
            pending += chunk
            while True:
                positions = [position for position in (pending.find(b"\n"), pending.find(b"\r")) if position >= 0]
                if not positions:
                    break
                cut = min(positions)
                is_progress = pending[cut:cut + 1] == b"\r" and pending[cut:cut + 2] != b"\r\n"
                raw, pending = pending[:cut], pending[cut + (2 if pending[cut:cut + 2] == b"\r\n" else 1):]
                self._handle_line(raw, is_progress)
        if pending:
            self._handle_line(pending, False)
        code = process.wait()
        process.stdout.close()
        self._process_ended(process, code)

    def _handle_line(self, raw: bytes, progress: bool) -> None:
        # wsl.exe reports its own errors in UTF-16; the Linux side writes UTF-8.
        text = raw.decode("utf-16-le", errors="replace") if raw.count(b"\x00") > len(raw) // 4 and raw \
            else raw.decode("utf-8", errors="replace")
        text = text.replace("\x00", "")
        if not text.strip() or _POLL_NOISE.search(text):
            return
        self.log.add(text, progress=progress)
        now = time.time()
        with self._lock:
            for index in range(self._phase_index + 1, len(PHASES)):
                marker = PHASES[index][2]
                if marker is not None and marker.search(text):
                    self._phase_index = index
                    self._phase_started[index] = now
                    break
            for key, pattern, convert in _FACTS:
                match = pattern.search(text)
                if match:
                    if key == "max_model_len" and key in self.facts:
                        continue  # the drafter logs its own, larger limit after the target's
                    try:
                        self.facts[key] = convert(match.group(1))
                    except ValueError:
                        pass
            # XQA only hurts on the FlashInfer 0.6 line (vLLM before 0.32); from 0.32 on it is the faster kernel.
            old_stack = str(self.facts.get("vllm_version", "")).startswith(("0.2", "0.30.", "0.31."))
            if old_stack and self.facts.get("decode_backend") == "xqa" and XQA_WARNING not in self.warning:
                self.warning = f"{self.warning} {XQA_WARNING}".strip()
            new_stack = bool(self.facts.get("vllm_version")) and not old_stack
            if new_stack and self.facts.get("decode_backend") not in (None, "xqa") and NATIVE_WARNING not in self.warning:
                self.warning = f"{self.warning} {NATIVE_WARNING}".strip()
            if text.startswith("=== AUTOTUNE:") and RETUNE_WARNING not in self.warning:
                self.warning = f"{self.warning} {RETUNE_WARNING}".strip()
        if READY_MARKER in text:
            self.mark_ready()

    def mark_ready(self) -> None:
        with self._lock:
            if self.state != "starting":
                return
            self.ready_at = time.time()
        self.log.add(f"=== Ready after {self.ready_at - (self.started_at or self.ready_at):.0f} s. "
                     f"API: {self.config.api_base_url}", "step")
        self._set_state("ready", "Ready")

    def _process_ended(self, process: subprocess.Popen, code: int) -> None:
        with self._lock:
            if self._process is process:
                self._process = None
            stopping = self._stopping
            was_ready = self.state == "ready"
        if stopping:
            self._set_state("stopped", "Stopped")
            return
        errors = self.log.errors()
        detail = _PREFIX.sub("", errors[-1]) if errors else f"The server process ended with exit code {code}."
        hint = next((text for pattern, text in _HINTS if any(pattern.search(line) for line in errors[-6:])), None)
        title = "The server crashed" if was_ready else "The start failed"
        self._fail(hint or detail, title=title, detail=detail if hint else None, code=code)

    def _fail(self, message: str, title: str = "The start failed",
              detail: str | None = None, code: int | None = None) -> None:
        with self._lock:
            self.error = {"title": title, "detail": message, "raw": detail, "code": code, "forceable": False}
        self._set_state("failed", message)

    # ---- stop --------------------------------------------------------------------

    def stop(self) -> bool:
        with self._lock:
            if self.state not in ("starting", "ready"):
                return False
            self._stopping = True
            process = self._process
            self.state = "stopping"
            self.state_since = time.time()
            self.message = "Stopping the model"
        self.log.add("=== Stopping the server", "step")
        with self._op_lock:
            self.launcher.stop(process, self.log)
            if process is None:  # adopted server: no reader thread will report the end
                self._set_state("stopped", "Stopped")
            else:
                deadline = time.time() + 20
                while self.state == "stopping" and time.time() < deadline:
                    time.sleep(0.1)
                if self.state == "stopping":
                    self._set_state("stopped", "Stopped")
        with self._lock:
            self.external = False
            self.ready_at = None
        return True

    def external_gone(self) -> None:
        with self._lock:
            self.external = False
            self.ready_at = None
        self._set_state("stopped", "The server started elsewhere has stopped")

    def shutdown(self) -> None:
        """Stop for good. Waits for a start that is still in its checks, so its server cannot outlive us."""
        self._closed = True
        with self._op_lock:
            pass
        if self.owns_server():
            self.stop()

    def owns_server(self) -> bool:
        with self._lock:
            return self.state in ("starting", "ready", "stopping") and not self.external
