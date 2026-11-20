"""Settings, presets and path handling."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

MODEL_ID = "qwen38-gittensor-dflash"
# The same model under a second name, for subagents. With "main session first" their requests yield to the main
# session and get the context left beside it; without, the name works like the first one. Costs no VRAM.
SUBAGENT_MODEL_ID = MODEL_ID + "-subagent"
# The OpenCode adapter (setting "opencode_adapter"): a LoRA adapter trained for the OpenCode harness, served beside
# the model under this name. Requests for the other names do not use it.
ADAPTER_MODEL_ID = MODEL_ID + "-opencode"
ADAPTER_REPO = "harness-hub/Qwen3.8-27B-opencode-adapted"
MODEL_TITLE = "Qwen3.8 27B"
# How the KV cache stores keys and values: NVFP4 (4 bit, the default since 2.0) or FP8.
KV_CACHE_TYPES = {"nvfp4": "NVFP4", "fp8": "FP8"}
DEFAULT_KV_CACHE = "nvfp4"


def model_subtitle(kv_cache: str = DEFAULT_KV_CACHE) -> str:
    return f"NVFP4, {KV_CACHE_TYPES.get(kv_cache, kv_cache)} KV, DFlash2"


MODEL_SUBTITLE = model_subtitle()
# Legacy mode only (a hand-built engine folder with its own start scripts): set "root" and "distro".
DEFAULT_ROOT = ""
DEFAULT_DISTRO = "Ubuntu"
# Managed mode: the app installs the engine itself into its own WSL distribution.
MANAGED_DISTRO = "QwenStation"
MANAGED_PREFIX = "/opt/qwen-station"
MANAGED_BASE_IMAGE = "Ubuntu-26.04"
CONFIG_NAME = "qwen-station.json"
IS_WINDOWS = os.name == "nt"
TARGET_DIR = "Qwen3.8-27B-NVFP4-RTX5090-Gittensor"
DRAFTER_DIR = "Qwen3.8-27B-DFlash2-FP8"
WRAPPER_SCRIPT = "run-variant-logged.sh"
OUTPUT_LIMIT = 32768
# Draft tokens per round of the DFlash2 drafter. 7 is its trained block; more helps predictable text
# (reasoning, code) and costs KV memory, so context. Measured up to 15, see docs/engine-notes.md.
DEFAULT_DRAFT_TOKENS = 7
# With vision on, a request may carry this many images. The start script switches images off by default,
# and vLLM then does not load the image encoder at all.
VISION_IMAGES = 4
# Requests the server works on at the same time. They share one KV pool: a slot reserves nothing, a running
# request takes its own state (about 1.5 GiB) plus its text. Tested with up to 4.
MAX_SLOTS = 4
# Sessions in RAM (GiB): what leaves the KV pool on the GPU is kept there and copied back. About 3.5 GiB per
# 100K tokens (measured: 4.03 GB copied back for 117,008 tokens).
MAX_RAM_CACHE_GIB = 128
# The coding agents the model can be entered in, in the order the settings show them (labels in clients.py).
AGENT_IDS = ("t3code", "opencode", "pi", "deepseek", "copilot")
LOCAL_HOST = "127.0.0.1"  # the API answers this machine only
ANY_HOST = "0.0.0.0"      # the API listens on every network interface
API_KEY_PATTERN = re.compile(r"[A-Za-z0-9._~+/=-]{8,128}")  # what fits into an Authorization header as it is
KEY_FILE = "api-key"  # the key again, alone in a file of the data folder: OpenCode reads it from there
MAX_DRAFT_TOKENS = 15

# "context" includes the generated output.
PRESETS = {
    "262k": {
        "label": "262K",
        "context": 262144,
        "kv_bytes": 10350000000,
        "note": "Default: the largest context, if the card is free enough for it",
    },
    "237k": {
        "label": "237K",
        "context": 237312,
        "kv_bytes": 9500000000,
        "note": "About 0.8 GB more VRAM headroom",
    },
    # Unsloth's dynamic NVFP4 quant of the same model, with the same DFlash2 drafter. Its weights take 4.3 GiB
    # more, so the context is smaller. Measured 2026-11-16 on vLLM 0.32.0: 242 to 274 tok/s short, 245 to 254 at 33K.
    "unsloth-82k": {
        "label": "Unsloth NVFP4, 82K",
        "context": 81920,
        "kv_bytes": 4390000000,
        "note": "Unsloth Dynamic NVFP4 with DFlash2: 4.3 GiB larger, so less context, about 15% slower",
        "env": {"TARGET_NAME": "Qwen3.8-27B-NVFP4-Unsloth", "TARGET_QUANT": "compressed-tensors"},
        "fit_reserve_gib": 25.9,
        "variant": "Unsloth NVFP4",
    },
}


@dataclass
class Config:
    # "managed": Windows, engine installed by the app into its own WSL distribution (the release).
    # "native": Linux, engine installed by the app into a folder of the user (the release).
    # "legacy": a hand-built folder with its own start scripts (root) in an existing distribution.
    mode: str = "legacy"
    prefix: str = MANAGED_PREFIX  # managed: install folder inside the distribution. native: the engine folder
    engine_location: str = ""  # managed: Windows folder that holds the distribution's disk
    setup_pending: bool = False  # managed: the setup waits for a restart of Windows and continues after it
    root: str = DEFAULT_ROOT
    distro: str = DEFAULT_DISTRO
    preset: str = "262k"
    lan: bool = False  # older switch: the same as api_host "0.0.0.0"
    api_host: str = ""  # where the API listens: 127.0.0.1, 0.0.0.0 or one address of this machine. Empty: by `lan`
    api_key: str = ""  # clients must send it as "Authorization: Bearer <key>". Empty: the API is open
    api_port: int = 8000
    station_port: int = 8010
    autostart: bool = True
    close_action: str = "stop"  # "stop": closing the window stops the model. "keep": only "Quit" does.
    gpu_match: str = "5090"
    gpu_memory_utilization: float = 0.92
    register_clients: bool = True  # off: no agent config is touched at all (--no-register-clients)
    agents: list = field(default_factory=list)  # where the model is entered; only adds. None by default
    open_window: bool = True
    fit_context: bool = True  # shrink the context when too little VRAM is free, instead of paging
    draft_tokens: int = DEFAULT_DRAFT_TOKENS  # per round; single-slot presets only
    vision: bool = False  # load the image encoder: requests may contain images, at the cost of some VRAM
    opencode_adapter: bool = False  # serve the OpenCode LoRA adapter as model ADAPTER_MODEL_ID (a test option)
    slots: int = 1  # requests the server works on at the same time; they share the context
    main_first: bool = False  # with slots > 1: the main session always has room for the full context, subagents
    #                           (model SUBAGENT_MODEL_ID) get what is left and yield to it
    ram_cache_gib: int = 0  # sessions pushed out of the GPU are kept in this much RAM; 0: off
    kv_cache: str = DEFAULT_KV_CACHE  # "nvfp4" (about 1.8 times the context per GiB) or "fp8"
    extra_args: str = ""  # appended to the `vllm serve` command line, for experiments
    vllm_prefix: str = ""  # venv to run, e.g. /opt/vllm-0.32.0; empty: the start script's default
    model_base: str = ""  # WSL folder holding target and drafter; empty: the Windows folder via /mnt
    extra_env: str = ""  # "KEY=VALUE KEY2=VALUE2" for the start scripts, for experiments
    mock: bool = False
    config_path: str = ""
    # Neither the settings nor the command line name the agents. Up to 2.0.0 that meant all of them; the app then
    # keeps the ones that already have the model (clients.entered_agents), so an update changes nothing for them.
    agents_unset: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def linux_root(self) -> str:
        return windows_to_wsl_path(self.root)

    @property
    def bind_host(self) -> str:
        """The address the API listens on."""
        return self.api_host or (ANY_HOST if self.lan else LOCAL_HOST)

    @property
    def shared(self) -> bool:
        """True when other devices can reach the API."""
        return self.bind_host != LOCAL_HOST

    @property
    def local_host(self) -> str:
        """The address this machine itself reaches the API at.

        On Linux a server bound to one address does not answer on 127.0.0.1. Inside WSL the server always
        listens on 127.0.0.1, which Windows forwards; other devices come in through the relay of the app."""
        host = self.bind_host
        return host if self.mode == "native" and not self.mock and host != ANY_HOST else LOCAL_HOST

    @property
    def api_base_url(self) -> str:
        return f"http://{self.local_host}:{self.api_port}/v1"

    def validate(self) -> None:
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown preset: {self.preset}")
        if self.mode not in ("managed", "native", "legacy"):
            raise ValueError("mode must be 'managed', 'native' or 'legacy'")
        if isinstance(self.draft_tokens, bool) or not 1 <= self.draft_tokens <= MAX_DRAFT_TOKENS:
            raise ValueError(f"draft_tokens must be a number from 1 to {MAX_DRAFT_TOKENS}")
        if isinstance(self.slots, bool) or not 1 <= self.slots <= MAX_SLOTS:
            raise ValueError(f"slots must be a number from 1 to {MAX_SLOTS}")
        if isinstance(self.ram_cache_gib, bool) or not 0 <= self.ram_cache_gib <= MAX_RAM_CACHE_GIB:
            raise ValueError(f"ram_cache_gib must be a number from 0 to {MAX_RAM_CACHE_GIB}")
        if self.close_action not in ("stop", "keep"):
            raise ValueError("close_action must be 'stop' or 'keep'")
        if self.kv_cache not in KV_CACHE_TYPES:
            raise ValueError(f"kv_cache must be one of: {', '.join(KV_CACHE_TYPES)}")
        if not isinstance(self.agents, list) or any(agent not in AGENT_IDS for agent in self.agents) \
                or len(set(self.agents)) != len(self.agents):
            raise ValueError(f"agents must be a list of: {', '.join(AGENT_IDS)}")
        if self.api_host:
            try:
                ipaddress.IPv4Address(self.api_host)
            except ValueError:
                raise ValueError("api_host must be an IPv4 address such as 127.0.0.1, 0.0.0.0 or 192.168.1.20") from None
        if self.api_key and not API_KEY_PATTERN.fullmatch(self.api_key):
            raise ValueError("api_key must be 8 to 128 characters: letters, digits and . _ ~ + / = -")
        for name in ("api_port", "station_port"):
            port = getattr(self, name)
            if not isinstance(port, int) or not 1 <= port <= 65535:
                raise ValueError(f"{name} must be a port between 1 and 65535")


def windows_to_wsl_path(path: str) -> str:
    """D:\\models -> /mnt/d/models (the default WSL automount layout)."""
    text = str(path).replace("\\", "/").rstrip("/")
    if len(text) >= 2 and text[1] == ":":
        return f"/mnt/{text[0].lower()}{text[2:]}"
    return text


def app_dir() -> Path:
    """Folder of the .exe (frozen) or of the project (source checkout)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Settings, the station's own log and the browser profile of the window."""
    if IS_WINDOWS:
        path = Path(os.environ.get("LOCALAPPDATA") or str(Path.home())) / "QwenStation"
    else:
        path = Path(os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")) / "qwen-station"
    path.mkdir(parents=True, exist_ok=True)
    return path


def native_prefix() -> str:
    """Default folder of the engine on Linux."""
    return str(Path(os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")) / "qwen-station")


def resource_path(name: str) -> Path:
    """Bundled file: next to this module, or in the PyInstaller unpack folder."""
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle) / "qwen_station" / name
    return Path(__file__).resolve().parent / name


def find_root(candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate and (Path(candidate) / WRAPPER_SCRIPT).is_file():
            return str(Path(candidate))
    return candidates[0] if candidates and candidates[0] else DEFAULT_ROOT


_FILE_KEYS = {
    "mode": str,
    "prefix": str,
    "engine_location": str,
    "setup_pending": bool,
    "root": str,
    "distro": str,
    "preset": str,
    "lan": bool,
    "api_host": str,
    "api_key": str,
    "api_port": int,
    "station_port": int,
    "autostart": bool,
    "close_action": str,
    "gpu_match": str,
    "gpu_memory_utilization": float,
    "register_clients": bool,
    "agents": list,
    "open_window": bool,
    "fit_context": bool,
    "draft_tokens": int,
    "vision": bool,
    "opencode_adapter": bool,
    "slots": int,
    "ram_cache_gib": int,
    "kv_cache": str,
    "main_first": bool,
    "extra_args": str,
    "vllm_prefix": str,
    "model_base": str,
    "extra_env": str,
}


def settings_path() -> Path:
    """Where the installed app keeps its settings (written by the setup)."""
    return data_dir() / CONFIG_NAME


def save_settings(values: dict, path: Path | None = None) -> Path:
    """Merge values into the settings file."""
    path = path or settings_path()
    current: dict = {}
    if path.is_file():
        with path.open("r", encoding="utf-8-sig") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            current = loaded
    current.update(values)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_config(explicit_path: str | None = None, overrides: dict | None = None) -> Config:
    """Defaults < settings file < command line.

    The settings file is the explicit one, else qwen-station.json next to the exe (portable and
    development use), else the one the setup wrote below %LOCALAPPDATA%.
    """
    config = Config()
    path = Path(explicit_path) if explicit_path else app_dir() / CONFIG_NAME
    if not explicit_path and not path.is_file() and not (overrides or {}).get("root"):
        path = settings_path()  # --root means the hand-built folder; the installed app's settings do not apply
    file_values: dict = {}
    if path.is_file():
        with path.open("r", encoding="utf-8-sig") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: The config must be a JSON object")
        file_values = loaded
        config.config_path = str(path)
    elif explicit_path:
        raise FileNotFoundError(explicit_path)
    for key, kind in _FILE_KEYS.items():
        if key not in file_values:
            continue
        value = file_values[key]
        if kind is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise ValueError(f"{path}: '{key}' has the wrong type")
        setattr(config, key, value)
    if file_values.get("register_clients") is False and "agents" not in file_values:
        # Up to 1.7.0 one switch covered OpenCode and Pi. Off meant no agent; the selection says that now.
        config.agents, config.register_clients = [], True
    root_from_file = "root" in file_values
    config.agents_unset = "agents" not in file_values and (overrides or {}).get("agents") is None \
        and file_values.get("register_clients") is not False
    for key, value in (overrides or {}).items():
        if value is not None:
            setattr(config, key, value)
    if config.preset == "parallel-96k":
        # Up to 1.5.0 two slots were a preset of their own, with half the context each. Now slots are a setting.
        config.preset = "262k"
        config.slots = max(config.slots, 2)
    root_given = root_from_file or bool((overrides or {}).get("root"))
    if "mode" not in file_values and not (overrides or {}).get("mode"):
        # A folder with the start scripts means the hand-built setup; everything else is the release.
        legacy = root_given or (app_dir() / WRAPPER_SCRIPT).is_file()
        config.mode = "legacy" if legacy else ("managed" if IS_WINDOWS else "native")
    if config.mode == "legacy" and not root_given:
        config.root = find_root([str(app_dir()), DEFAULT_ROOT])
    if config.mode == "managed" and "distro" not in file_values and not (overrides or {}).get("distro"):
        config.distro = MANAGED_DISTRO
    if config.mode == "native" and "prefix" not in file_values:
        config.prefix = native_prefix()
    config.validate()
    return config
