"""The settings the dashboard can change: which ones, which values are valid, and when they take effect."""

from __future__ import annotations

import dataclasses
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time

from .config import (AGENT_IDS, KV_CACHE_TYPES, ANY_HOST, API_KEY_PATTERN, DEFAULT_DRAFT_TOKENS, LOCAL_HOST, MAX_DRAFT_TOKENS,
                     MAX_RAM_CACHE_GIB, MAX_SLOTS, Config)

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Read when the model starts: a change waits for the next start while the model runs.
MODEL_KEYS = ("api_host", "api_port", "api_key", "draft_tokens", "vision", "slots", "ram_cache_gib", "kv_cache",
              "fit_context", "extra_args", "extra_env")
# Read all the time: a change applies at once.
APP_KEYS = ("autostart", "close_action", "agents")
# The dashboard's own port: the running app keeps it until it is started again.
RESTART_KEYS = ("station_port",)
EDITABLE = MODEL_KEYS + APP_KEYS + RESTART_KEYS

KINDS = {"api_host": str, "api_port": int, "api_key": str, "station_port": int, "draft_tokens": int,
         "fit_context": bool, "vision": bool, "slots": int, "ram_cache_gib": int, "kv_cache": str,
         "extra_args": str, "extra_env": str, "autostart": bool, "close_action": str, "agents": list}
MIN_PORT = 1024
MAX_TEXT = 2000

_cache_lock = threading.Lock()
_cache: tuple[float, list[dict]] = (0.0, [])
_refreshing = False


def values(config: Config) -> dict:
    """The editable settings as the running app uses them."""
    current = {key: getattr(config, key) for key in EDITABLE}
    current["api_host"] = config.bind_host  # the older "lan" switch shows up as its address
    current["agents"] = list(config.agents)
    return current


def primary_address() -> str:
    """The address this machine uses towards its default route, or "". No packet leaves the machine."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))  # TEST-NET-1; a UDP connect only picks the route
            address = probe.getsockname()[0]
    except OSError:
        return ""
    return "" if address.startswith(("127.", "0.")) else address


def _usable(address: str) -> bool:
    return not address.startswith(("127.", "169.254.", "0."))


def _windows_addresses() -> list[dict]:
    script = ("Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.AddressState -eq 'Preferred' } | "
              "ForEach-Object { [pscustomobject]@{ address = $_.IPAddress; name = $_.InterfaceAlias } } | "
              "ConvertTo-Json -Compress")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, text=True, timeout=20, creationflags=NO_WINDOW)
    loaded = json.loads(result.stdout) if result.stdout.strip() else []
    return loaded if isinstance(loaded, list) else [loaded]


def _linux_addresses() -> list[dict]:
    tool = shutil.which("ip")
    if not tool:
        return []
    result = subprocess.run([tool, "-4", "-o", "addr", "show"], capture_output=True, text=True, timeout=10)
    found = re.findall(r"^\d+:\s+(\S+)\s+inet\s+([\d.]+)/", result.stdout, re.M)
    return [{"address": address, "name": name} for name, address in found]


def _quick_addresses() -> list[dict]:
    """Without a subprocess and without interface names: what the name of this machine resolves to."""
    try:
        found = {info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    except OSError:
        found = set()
    return [{"address": address, "name": ""} for address in sorted(found)]


def _ordered(found: list[dict]) -> list[dict]:
    primary = primary_address()
    rows = [{"address": str(row.get("address", "")), "name": str(row.get("name", ""))}
            for row in found if isinstance(row, dict) and _usable(str(row.get("address", "")))]
    if primary and primary not in [row["address"] for row in rows]:
        rows.append({"address": primary, "name": ""})  # the lookup found nothing: at least the main one
    rows.sort(key=lambda row: row["address"] != primary)
    return rows


def refresh_addresses() -> list[dict]:
    """Look the addresses up now. On Windows this starts PowerShell and takes a few seconds."""
    global _cache
    try:
        found = _windows_addresses() if os.name == "nt" else _linux_addresses()
    except (OSError, subprocess.TimeoutExpired, ValueError):
        found = []
    rows = _ordered(found or _quick_addresses())
    with _cache_lock:
        _cache = (time.time(), rows)
    return list(rows)


def local_addresses(max_age: float = 30.0) -> list[dict]:
    """The IPv4 addresses of this machine with the names of their interfaces, the main one first.

    Never waits for PowerShell: it answers from the last lookup and starts a new one in the background when
    that is older than max_age. Before the first lookup is back, the names are missing."""
    global _refreshing
    with _cache_lock:
        age, rows = time.time() - _cache[0], list(_cache[1])
        stale = (not rows or age > max_age) and not _refreshing
        if stale and os.name == "nt":
            _refreshing = True
    if stale and os.name != "nt":
        return refresh_addresses()  # `ip addr` answers at once
    if stale:
        def work() -> None:
            global _refreshing
            try:
                refresh_addresses()
            finally:
                _refreshing = False
        threading.Thread(target=work, name="addresses", daemon=True).start()
    return rows or _ordered(_quick_addresses())


def key_hint(key: str) -> str:
    """Enough of a key to recognize it, for places that anyone on this machine can read."""
    return f"{key[:5]}…{key[-3:]}" if key else ""


def network_url(config: Config) -> str:
    """The address other devices use, or "" while the API is for this machine only."""
    if not config.shared:
        return ""
    host = primary_address() if config.bind_host == ANY_HOST else config.bind_host
    return f"http://{host}:{config.api_port}/v1" if host else ""


def validate(config: Config, current: dict, changes: dict, addresses: list[dict] | None = None) -> tuple[dict, str, str]:
    """Check changed settings against the current ones. Returns (all values after the change, error, field)."""
    merged = dict(current)
    for key, value in changes.items():
        if key not in KINDS:
            return {}, f"Unknown setting: {key}", key
        kind = KINDS[key]
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            return {}, "This value has the wrong type.", key
        merged[key] = value.strip() if kind is str else value

    host = merged["api_host"]
    try:
        ipaddress.IPv4Address(host)
    except ValueError:
        return {}, "Enter an IPv4 address such as 192.168.1.20, or pick one from the list.", "api_host"
    if host.startswith("127.") and host != LOCAL_HOST:
        return {}, f"For this computer only, use {LOCAL_HOST}.", "api_host"
    if host not in (LOCAL_HOST, ANY_HOST) and host != current.get("api_host") and not config.mock:
        known = [row["address"] for row in (local_addresses() if addresses is None else addresses)]
        if host not in known and addresses is None:
            known = [row["address"] for row in refresh_addresses()]  # an interface may have come up just now
        if host not in known:
            return {}, f"This computer has no network address {host}.", "api_host"
    if merged["api_key"] and not API_KEY_PATTERN.fullmatch(merged["api_key"]):
        return {}, "A key has 8 to 128 characters: letters, digits and . _ ~ + / = - (no spaces).", "api_key"
    for key, label in (("api_port", "The API port"), ("station_port", "The dashboard port")):
        if not MIN_PORT <= merged[key] <= 65535:
            return {}, f"{label} must be between {MIN_PORT} and 65535.", key
    if merged["api_port"] == merged["station_port"]:
        return {}, "The API and the dashboard need different ports.", "api_port"
    if not DEFAULT_DRAFT_TOKENS <= merged["draft_tokens"] <= MAX_DRAFT_TOKENS:
        return {}, f"Draft tokens must be between {DEFAULT_DRAFT_TOKENS} and {MAX_DRAFT_TOKENS}.", "draft_tokens"
    if not 1 <= merged["slots"] <= MAX_SLOTS:
        return {}, f"Between 1 and {MAX_SLOTS} requests at the same time.", "slots"
    if not 0 <= merged["ram_cache_gib"] <= MAX_RAM_CACHE_GIB:
        return {}, f"Between 0 (off) and {MAX_RAM_CACHE_GIB} GiB.", "ram_cache_gib"
    if merged["close_action"] not in ("stop", "keep"):
        return {}, "Choose what happens when the window closes.", "close_action"
    if merged["kv_cache"] not in KV_CACHE_TYPES:
        return {}, "Choose NVFP4 or FP8.", "kv_cache"
    agents = merged["agents"]
    if any(agent not in AGENT_IDS for agent in agents) or len(set(agents)) != len(agents):
        return {}, "Choose agents from the list.", "agents"
    merged["agents"] = [agent for agent in AGENT_IDS if agent in agents]  # one order, so a save compares equal
    for key in ("extra_args", "extra_env"):
        if len(merged[key]) > MAX_TEXT or "\n" in merged[key] or "\r" in merged[key]:
            return {}, f"One line, at most {MAX_TEXT} characters.", key
    for pair in merged["extra_env"].split():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", pair):
            return {}, f"Write the environment as KEY=VALUE pairs separated by spaces; “{pair}” is not one.", "extra_env"
    try:
        dataclasses.replace(config, **merged).validate()
    except ValueError as exc:
        return {}, str(exc), ""
    return merged, "", ""
