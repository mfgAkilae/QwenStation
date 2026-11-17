"""Starts, watches and stops the vLLM server that runs inside WSL."""

from __future__ import annotations

import json
import math
import os
import platform
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
from .relay import Relay
from .config import (ANY_HOST, DEFAULT_DRAFT_TOKENS, DRAFTER_DIR, KEY_FILE, LOCAL_HOST, MODEL_ID, PRESETS, TARGET_DIR,
                     VISION_IMAGES, WRAPPER_SCRIPT, Config,
                     app_dir, data_dir,
                     resource_path)

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
READY_MARKER = "Application startup complete."
LAN_SCRIPT = "setup-qwen-lan.ps1"
LAN_STATE = "setup-qwen-lan.state.json"  # written by the script next to itself, in logs/
# Shortfall the start tolerates. Measured 2026-11-16: 26.7 GiB free worked (Windows evicted other
# programs' GPU memory), but the start took 9 minutes instead of 4.
VRAM_TOLERANCE_MIB = 3072  # only used when fit_context is off
# KV pool sizing for this model (FP8 KV, group size 8). A running request needs a fixed part (its Mamba states
# and the snapshots for the draft tokens) plus its tokens. Measured 2026-11-17 with single cold requests in a
# pool of 6.617 GiB: 128,900 tokens filled 84 percent of it, 143,900 tokens 92, 150,908 tokens 94 and 156,884
# tokens 97, which is 1.59 GiB fixed and 32,500 tokens per GiB. The values before (1.546 and 32,730, taken from
# the size vLLM reports) left the pool 2 percent too small for a request of the whole context: at 99.5 percent
# of the context the request ran out of blocks, lost the ones it had, and started over, without end.
KV_FIXED_GIB = 1.62
KV_TOKENS_PER_GIB = 32400
# On top of that: the cache is handed out in blocks of 1,648 tokens, and a request that needs one more block
# than there is does not fail, it starts over. With 1 percent a prompt of 99.7 percent of the context filled
# the pool to 99 percent; 2 percent leave room for a few blocks more.
KV_MARGIN = 1.02
# Every draft token beyond the default 7 adds a state snapshot per Mamba layer to the fixed part. Measured
# 2026-11-17 at 163,840 tokens: 6.52 GiB with 7 draft tokens, 6.83 with 9, 7.15 with 11, 7.84 with 15,
# which is 0.158 to 0.165 GiB per token. A little more is reserved here.
KV_PER_DRAFT_TOKEN_GIB = 0.17
# Free VRAM the server needs beside its KV pool. Measured 2026-11-17: weights and runtime take 20.3 GiB at the
# start and grow to 21.3 GiB with long sessions, and Windows starts paging GPU memory about 0.65 GiB before
# the card is full (nvidia-smi then stays at 31.9 of 32.6 GiB). With less than this reserve the model still
# starts and short prompts are fast, but a session of 163K tokens decoded at 83 instead of 380 tok/s.
FIT_RESERVE_GIB = 22.0
# Linux has no such paging: the card can be used to its end, and a server that needs more fails at once.
# Not measured on a Linux machine; the 0.15 GiB kept of the 0.65 are for that.
LINUX_RESERVE_RELIEF_GIB = 0.5
# What vision adds to that. Measured 2026-11-17: the image encoder's weights take 0.86 GiB (vLLM reports
# 18.88 instead of 18.02 GiB for the model), and reading the largest image (16.7 million pixels, 16,384 tokens)
# needs another 0.75 GiB for a moment, a 4K screenshot 0.54 GiB. Without this reserve the card ran full with a
# large image, Windows paged, and text got 20 to 55 percent slower.
VISION_RESERVE_GIB = 1.7
MIN_FIT_CONTEXT = 65536
# Coarse steps on purpose: every distinct context is a new server configuration, and each of those runs
# once without the verified kernel tuning (see RETUNE_WARNING). Six sizes instead of fifty.
FIT_STEP = 32768


def kv_fixed_gib(draft_tokens: int = DEFAULT_DRAFT_TOKENS) -> float:
    """The part of the KV pool that does not depend on the context."""
    return KV_FIXED_GIB + KV_PER_DRAFT_TOKEN_GIB * max(0, draft_tokens - DEFAULT_DRAFT_TOKENS)


def kv_pool_bytes(context: int, draft_tokens: int = DEFAULT_DRAFT_TOKENS) -> int:
    return int((kv_fixed_gib(draft_tokens) + context / KV_TOKENS_PER_GIB) * KV_MARGIN * 2 ** 30)


def fit_context(free_mib: float, max_context: int, reserve_gib: float = FIT_RESERVE_GIB,
                draft_tokens: int = DEFAULT_DRAFT_TOKENS) -> tuple[int, int] | None:
    """Largest context (and its KV pool in bytes) that fits into the free VRAM, or None if too little is free."""
    tokens = ((free_mib / 1024 - reserve_gib) / KV_MARGIN - kv_fixed_gib(draft_tokens)) * KV_TOKENS_PER_GIB
    context = min(max_context, int(tokens) // FIT_STEP * FIT_STEP)
    if context < MIN_FIT_CONTEXT:
        return None
    return context, kv_pool_bytes(context, draft_tokens)
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
# The start script ends its command line with these extra arguments, and the last --limit-mm-per-prompt wins.
# The dotted form needs no quotes, which would not survive the way through wsl.exe.
VISION_ARGS = f"--limit-mm-per-prompt.image {VISION_IMAGES} --limit-mm-per-prompt.video 0"
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


def http_get(url: str, timeout: float = 2.0, key: str = "") -> tuple[int, bytes]:
    headers = {"Accept": "*/*", **({"Authorization": f"Bearer {key}"} if key else {})}
    request = urllib.request.Request(url, headers=headers)
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
            f"VLLM_HOST={self.server_host()}",
            f"VLLM_PORT={config.api_port}",
            f"CTX={preset['context']}",
            f"KV_BYTES={preset['kv_bytes']}",
            f"MAX_SEQS={config.slots}",
        ]
        if abs(config.gpu_memory_utilization - 0.92) > 1e-9:
            env.append(f"GPU_UTIL={config.gpu_memory_utilization}")
        # The wrapper word-splits EXTRA_ARGS onto the vllm command line.
        env.append(" ".join(part for part in (QUIET_POLL_ARGS, VISION_ARGS if config.vision else "",
                                              config.extra_args) if part))
        if config.vllm_prefix:
            env.append(f"VLLM_PREFIX={config.vllm_prefix}")
        if config.model_base:
            env.append(f"MODEL_BASE={config.model_base}")
        if config.draft_tokens != DEFAULT_DRAFT_TOKENS:
            env.append(f"SPEC_TOKENS={config.draft_tokens}")
        if config.ram_cache_gib:
            env.append(f"KV_OFFLOAD_GIB={config.ram_cache_gib}")
        if config.vision:
            # vLLM keys its cache of compiled graphs without the multimodal settings, but with images on the
            # model is called with embeddings instead of token ids. A graph compiled in text-only mode then
            # fails at the first run ('NoneType' object has no attribute 'size'). So: a cache of its own.
            env.append(f"VLLM_CACHE_ROOT={self.vision_cache()}")
        env.extend(f"{key}={value}" for key, value in preset.get("env", {}).items())  # e.g. another target
        env.extend(config.extra_env.split())  # KEY=VALUE pairs for the start scripts, e.g. KV_DTYPE=nvfp4
        return [*self.wsl(), "env", *env, *self.script()]

    def vision_cache(self) -> str:
        return "/var/tmp/qwen-station-vllm-vision"  # the hand-built folder may sit on a slow Windows drive

    def server_host(self) -> str:
        # Inside WSL the server always listens on 127.0.0.1, which Windows forwards to its own 127.0.0.1.
        # Other devices come in through the relay in this app (relay.py).
        return LOCAL_HOST

    def wsl(self) -> list[str]:
        return ["wsl.exe", "-d", self.config.distro, "--exec"]

    def script(self) -> list[str]:
        return ["bash", f"{self.config.linux_root}/{WRAPPER_SCRIPT}", "dflash"]

    def status_file(self) -> str:
        return f"{self.config.linux_root}/logs/dflash-status.txt"

    def stop_pattern(self) -> str:
        # Same match the wrapper uses for its own cleanup: Qwen3.8-27B servers, wherever the weights live.
        return r"vllm serve [^ ]*/Qwen3\.8-27B"

    def stop_script(self) -> str:
        pattern = self.stop_pattern()
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

    host_label = "WSL distribution"

    def wsl(self) -> list[str]:
        return ["wsl.exe", "-d", self.config.distro, "-u", "root", "--exec"]

    def vision_cache(self) -> str:
        return f"{self.config.prefix}/cache/vllm-vision"

    def models(self) -> str:
        return self.config.model_base or f"{self.config.prefix}/models"

    def stop_pattern(self) -> str:
        # Only servers of this installation: on a shared Linux machine other vLLM servers are none of our business.
        return "vllm serve " + self.models().replace(".", r"\.") + "/"

    def script(self) -> list[str]:
        return ["bash", f"{self.config.prefix}/engine/serve.sh"]

    def status_file(self) -> str:
        return f"{self.config.prefix}/logs/status.txt"

    def file_checks(self, preset: dict | None = None) -> list[dict]:
        target = (preset or {}).get("env", {}).get("TARGET_NAME", TARGET_DIR)
        base = self.models()
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
                 "detail": "; ".join(missing) if missing else self.location()}]

    def location(self) -> str:
        return f"{self.config.distro}:{self.config.prefix}"


class NativeLauncher(ManagedLauncher):
    """Linux: the same serve.sh, started directly."""

    host_label = "System"
    gpu_index: int | None = None  # set by the preflight: the card that matched, in PCI bus order

    def server_host(self) -> str:
        return self.config.bind_host  # no forwarding in between: the server binds the address itself

    def wsl(self) -> list[str]:
        return []

    def location(self) -> str:
        return self.config.prefix

    def command(self, preset: dict) -> list[str]:
        command = super().command(preset)  # ["env", KEY=VALUE ..., "bash", serve.sh]
        if self.gpu_index is not None:
            command.insert(1, f"GPU_INDEX={self.gpu_index}")
        return command

    def distro_check(self) -> tuple[bool, str]:
        return True, f"{platform.system()} {platform.release()}"


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
            else ManagedLauncher(config) if config.mode == "managed"
            else NativeLauncher(config) if config.mode == "native" else WslLauncher(config))
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
        self.before_start = None  # called before the checks of every start: the station applies saved settings
        self._relay: Relay | None = None  # Windows: passes connections from other devices on to the server in WSL

    # ---- state -----------------------------------------------------------------

    def _set_state(self, state: str, message: str = "") -> None:
        if state != "ready":
            self._unshare()
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
            status, body = http_get(f"http://{self.config.local_host}:{self.config.api_port}/v1/models", timeout=2.0,
                                    key=self.config.api_key)
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
        checks.append({"id": "wsl", "label": getattr(self.launcher, "host_label", "WSL distribution"), "ok": ok,
                       "blocking": True, "detail": detail})
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
                if row and isinstance(self.launcher, NativeLauncher):
                    self.launcher.gpu_index = row.get("index")
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                row = None
                checks.append({"id": "vram", "label": "Free VRAM", "ok": False, "blocking": False,
                               "detail": f"Cannot read nvidia-smi: {exc}"})
            if row and row.get("mem_total") and row.get("mem_free") is not None:
                vision = self.config.vision
                preset = PRESETS[self.preset]
                drafts = self.config.draft_tokens
                more_drafts = drafts > DEFAULT_DRAFT_TOKENS  # the preset's KV pool is sized for the default
                reserve = (preset.get("fit_reserve_gib", FIT_RESERVE_GIB) + (VISION_RESERVE_GIB if vision else 0)
                           - (LINUX_RESERVE_RELIEF_GIB if self.config.mode == "native" else 0))
                # What must be free for the preset's whole context: the reserve and the KV pool for it. Up to
                # 1.5.0 this was 0.92 of the card (29.3 GiB), which let the full context start on a card that
                # then ran full: fast with short prompts, paging with long ones.
                need = math.ceil((reserve + kv_pool_bytes(preset["context"], drafts) / 2 ** 30) * 1024)
                enough = row["mem_free"] >= need
                if self.config.fit_context:
                    fitted = fit_context(row["mem_free"], preset["context"], reserve, drafts)
                reduced = fitted is not None and fitted[0] < preset["context"]
                tight = not enough and not self.config.fit_context and row["mem_free"] >= need - VRAM_TOLERANCE_MIB
                detail = (f"{_gib(row['mem_free'])} GiB free, {_dots(preset['context'])} tokens need about "
                          f"{_gib(need)} GiB")
                if reduced:
                    detail += f"; context reduced to {_dots(fitted[0])} tokens"
                checks.append({"id": "vram", "label": "Free VRAM", "ok": enough or tight or fitted is not None,
                               "warn": tight or reduced, "blocking": False,
                               "free_mib": row["mem_free"], "need_mib": need, "detail": detail})
                if reduced:
                    warning = (f"{_gib(row['mem_free'])} GiB of VRAM is free. For {_dots(preset['context'])} "
                               f"tokens the model needs about {_gib(need)} GiB free, so the context is "
                               f"{_dots(fitted[0])} tokens. "
                               + ("Otherwise the server runs out of GPU memory. " if self.config.mode == "native" else
                                  "Otherwise Windows pages out GPU memory and long sessions get several times "
                                  "slower. ")
                               + "For a larger context, close other GPU programs or run the displays on another "
                                 "graphics card, then start again."
                               + (f" {drafts} draft tokens per round also take part of the memory."
                                  if more_drafts else "")
                               + (f" Vision takes {VISION_RESERVE_GIB} GiB of it; switch it off in the settings "
                                  "for a larger context." if vision else ""))
                if tight:
                    warning = (f"Only {_gib(row['mem_free'])} GiB of VRAM was free at the start, and the model needs "
                               f"about {_gib(need)} GiB. "
                               + ("The start can fail with an out-of-memory error." if self.config.mode == "native" else
                                  "Windows then pages out the GPU memory of other programs. "
                                  "The start takes longer and the speed can suffer."))
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
                if self.before_start:
                    self.before_start()
                self._start(force)
            except Exception as exc:  # a crash here must end in a visible state, not a silent thread death
                self.log.add(f"Start aborted: {exc!r}", "error")
                self._fail(f"Start aborted: {exc}")

    def effective_preset(self) -> dict:
        """The preset, with context and KV pool replaced when they were fitted to the free VRAM."""
        preset = dict(PRESETS[self.preset])
        # The pool always follows the context and the draft tokens; the number in the preset is not used.
        preset["kv_bytes"] = kv_pool_bytes(preset["context"], self.config.draft_tokens)
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
        if not self.config.mock:
            self._prepare_network()
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
                     f"KV {preset['kv_bytes']} bytes, {self.config.slots} slot(s)", "step")
        try:
            # Its own session on Linux: Ctrl+C in the terminal reaches the station only, which then
            # stops the server in order.
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, creationflags=NO_WINDOW,
                                       start_new_session=os.name != "nt", env=self._server_environment())
        except OSError as exc:
            self._fail(f"The start command could not be run: {exc}")
            return
        with self._lock:
            self._process = process
        self._set_state("starting", "Loading the model")
        threading.Thread(target=self._read_output, args=(process,), name="server-log", daemon=True).start()

    def _server_environment(self) -> dict:
        """The environment of the server process. The API key travels here, never on a command line:
        vLLM reads VLLM_API_KEY, and WSLENV carries the variable from Windows into the distribution."""
        environment = dict(os.environ)
        environment.pop("VLLM_API_KEY", None)  # only the setting decides, not what the desktop happens to export
        if self.config.api_key:
            environment["VLLM_API_KEY"] = self.config.api_key
            if os.name == "nt":
                carried = [name for name in environment.get("WSLENV", "").split(":") if name]
                environment["WSLENV"] = ":".join([*carried, "VLLM_API_KEY/u"])
        return environment

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
        self._share()

    def refresh_clients(self, write: bool) -> None:
        preset = self.effective_preset()
        result = clients.register_all(self.config.api_base_url, preset["context"], self.config.slots,
                                      dry_run=not write, variant=preset.get("variant", ""),
                                      api_key=self.config.api_key, key_file=data_dir() / KEY_FILE,
                                      vision=self.config.vision, agents=self.config.agents)
        for entry in result.values():
            if entry.get("changed"):
                self.log.add(f"--- {entry['client']}: model {entry['model']} {entry['detail']} ({entry['path']})", "step")
            elif entry["status"] == "error":
                self.log.add(f"{entry['client']} registration failed: {entry['detail']}", "warn")
        with self._lock:
            self.clients = result

    def _prepare_network(self) -> None:
        """Before the server starts: open the firewall for the chosen address, or take an earlier rule back."""
        config = self.config
        where = "all network interfaces" if config.bind_host == ANY_HOST else config.bind_host
        guard = ("Clients need the API key." if config.api_key
                 else "The API has no key: anyone who reaches it can use the model.")
        if config.mode == "native":
            if config.shared:
                self.log.add(f"--- Network: the API listens on {where}, port {config.api_port}. {guard} "
                             "Opening the port in your firewall is up to you.", "step")
            return
        try:
            script = self._lan_script()
            state = Path(script).parent / "logs" / LAN_STATE
            if not config.shared:
                if state.is_file():
                    self.log.add("--- Network: the API is for this computer only again. Windows asks for "
                                 "administrator rights (UAC) to remove the firewall rule.", "step")
                    self._run_lan_script(script, ["-Remove"], elevated=True)
                    self.log.add("--- Network: firewall rule removed.", "step")
                return
            target = ["-ListenAddress", config.bind_host, "-Port8000", str(config.api_port)]
            if self._run_lan_script(script, ["-Check", *target], elevated=False) == 0:
                self.log.add(f"--- Network: the firewall is open for {where}, port {config.api_port}.", "step")
                return
            self.log.add(f"--- Network: sharing the API on {where}, port {config.api_port}. Windows asks for "
                         "administrator rights (UAC) once, for the firewall rule.", "step")
            self._run_lan_script(script, target, elevated=True)
            self.log.add(f"--- Network: the firewall rule is set up. {guard}", "step")
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            self.log.add(f"The firewall rule was not set up ({exc}). The model starts anyway; other devices may "
                         "not get through.", "warn")

    def _lan_script(self) -> str:
        # The script keeps its state next to itself, so it needs a folder that outlives this process.
        source = resource_path(LAN_SCRIPT)
        target = data_dir() / LAN_SCRIPT
        if not target.is_file() or target.read_bytes() != source.read_bytes():
            shutil.copyfile(source, target)
        (data_dir() / "logs").mkdir(exist_ok=True)
        return str(target)

    def _run_lan_script(self, script: str, arguments: list[str], elevated: bool) -> int:
        """Run the network script. Elevated, a failure (also a declined UAC prompt) raises RuntimeError."""
        base = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass"]
        if not elevated:
            return subprocess.run([*base, "-File", script, *arguments], capture_output=True, timeout=120,
                                  creationflags=NO_WINDOW).returncode
        # One string for Start-Process: every part in double quotes, inside a single-quoted PowerShell string.
        quoted = " ".join('"' + part.replace("'", "''") + '"' for part in ("-NoProfile", "-ExecutionPolicy",
                                                                           "Bypass", "-File", script, *arguments))
        elevate = ("try { $p = Start-Process -FilePath powershell.exe -Verb RunAs -WindowStyle Hidden "
                   f"-ArgumentList '{quoted}' -Wait -PassThru; exit $p.ExitCode }} catch {{ exit 1223 }}")
        code = subprocess.run([*base, "-Command", elevate], capture_output=True, timeout=300,
                              creationflags=NO_WINDOW).returncode
        if code == 1223:
            raise RuntimeError("the administrator prompt was declined")
        if code != 0:
            raise RuntimeError(f"exit code {code}")
        return code

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
            # The server says it listens. On Windows the way to it, WSL's forwarding of 127.0.0.1, appears up to
            # a second later: three requests sent right at "Ready" were refused in a test on 2026-11-17.
            threading.Thread(target=self._ready_when_reachable, name="ready", daemon=True).start()

    def _ready_when_reachable(self, patience: float = 10.0) -> None:
        url = f"http://{self.config.local_host}:{self.config.api_port}/health"
        deadline = time.time() + patience
        while time.time() < deadline and self.state == "starting":
            try:
                if http_get(url, timeout=1.0)[0] == 200:
                    break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.2)
        self.mark_ready()

    def mark_ready(self) -> None:
        with self._lock:
            if self.state != "starting":
                return
            self.ready_at = time.time()
        self.log.add(f"=== Ready after {self.ready_at - (self.started_at or self.ready_at):.0f} s. "
                     f"API: {self.config.api_base_url}", "step")
        self._set_state("ready", "Ready")
        self._share()

    def _share(self) -> None:
        """Windows, the model is up: start passing connections from other devices on to it."""
        config = self.config
        if config.mock or config.mode == "native" or not config.shared:
            return  # on Linux the server binds the chosen address itself
        relay = Relay(config.bind_host, config.api_port)
        try:
            relay.start()
        except OSError as exc:
            self.log.add(f"The API could not be shared on {config.bind_host}, port {config.api_port}: {exc}. "
                         "This computer still reaches it.", "warn")
            return
        with self._lock:
            self._relay = relay
        where = "every address of this computer" if config.bind_host == ANY_HOST else config.bind_host
        self.log.add(f"--- Network: other devices reach the API on {where}, port {config.api_port}.", "step")

    def _unshare(self) -> None:
        with self._lock:
            relay, self._relay = self._relay, None
        if relay is not None:
            relay.stop()

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
            relay, self._relay = self._relay, None
            self.state = "stopping"
            self.state_since = time.time()
            self.message = "Stopping the model"
        if relay is not None:
            relay.stop()
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
