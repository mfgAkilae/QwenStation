"""GPU readings from nvidia-smi (one long-running child, one line per second)."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
FIELDS = ("index", "name", "utilization.gpu", "memory.used", "memory.total", "memory.free",
          "power.draw", "power.limit", "temperature.gpu", "fan.speed")


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None  # "[N/A]" on some drivers


def parse_line(line: str) -> dict | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != len(FIELDS):
        return None
    index = _number(parts[0])
    if index is None:
        return None
    return {
        "index": int(index),
        "name": parts[1],
        "util": _number(parts[2]),
        "mem_used": _number(parts[3]),
        "mem_total": _number(parts[4]),
        "mem_free": _number(parts[5]),
        "power": _number(parts[6]),
        "power_limit": _number(parts[7]),
        "temp": _number(parts[8]),
        "fan": _number(parts[9]),
    }


def query_once(timeout: float = 10.0) -> list[dict]:
    command = ["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"nvidia-smi exit code {result.returncode}")
    return [row for row in (parse_line(line) for line in result.stdout.splitlines()) if row]


def pick(rows: list[dict], match: str) -> dict | None:
    for row in rows:
        if match.lower() in row["name"].lower():
            return row
    return rows[0] if rows else None


class GpuMonitor:
    def __init__(self, match: str = "5090") -> None:
        self.match = match
        self.latest: dict | None = None
        self.error: str | None = None
        self._stop = threading.Event()
        self._process: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process and process.poll() is None:
            process.terminate()

    def _run(self) -> None:
        command = ["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits", "-l", "1"]
        while not self._stop.is_set():
            try:
                self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                                 text=True, creationflags=NO_WINDOW)
            except OSError as exc:
                self.error = f"nvidia-smi not found: {exc}"
                self.latest = None
                if self._stop.wait(30):
                    return
                continue
            self.error = None
            assert self._process.stdout is not None
            for line in self._process.stdout:
                if self._stop.is_set():
                    break
                row = parse_line(line)
                if row and self.match.lower() in row["name"].lower():
                    row["t"] = round(time.time(), 3)
                    self.latest = row
            self._process.wait()
            if not self._stop.is_set():
                self.error = "nvidia-smi exited, restarting it in 5 s"
                self._stop.wait(5)


_PROCESS_QUERY = r"""
$ErrorActionPreference = 'SilentlyContinue'
$rows = Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory |
  Where-Object { $_.DedicatedUsage -gt 200MB } | Sort-Object DedicatedUsage -Descending | Select-Object -First 8
$out = foreach ($row in $rows) {
  $id = if ($row.Name -match 'pid_(\d+)_') { [int]$Matches[1] } else { 0 }
  $proc = if ($id) { Get-Process -Id $id -ErrorAction SilentlyContinue } else { $null }
  [pscustomobject]@{ pid = $id; name = $(if ($proc) { $proc.ProcessName } else { 'unknown' }); mib = [math]::Round($row.DedicatedUsage / 1MB) }
}
ConvertTo-Json -Compress -InputObject @($out)
"""


def _compute_apps(timeout: float) -> list[dict]:
    """Linux: nvidia-smi knows the memory of every compute process."""
    command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in result.stdout.splitlines() if result.returncode == 0 else []:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit() and _number(parts[2]):
            rows.append({"pid": int(parts[0]), "name": parts[1].rsplit("/", 1)[-1], "mib": round(_number(parts[2]))})
    return sorted(rows, key=lambda row: -row["mib"])


def gpu_processes(timeout: float = 20.0, max_mib: float = 0.0) -> list[dict]:
    """Dedicated GPU memory per Windows process, all adapters. nvidia-smi cannot report this under WDDM.

    `max_mib` drops rows larger than a whole card: the counter reports nonsense for dwm and csrss.
    """
    if os.name != "nt":
        return _compute_apps(timeout)
    try:
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _PROCESS_QUERY],
                                capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW)
        rows = json.loads(result.stdout.strip() or "[]")
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return []
    if isinstance(rows, dict):
        rows = [rows]
    best: dict[int, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("mib") or row.get("name") in ("dwm", "csrss"):
            continue
        if max_mib and row["mib"] > max_mib:
            continue
        if row["pid"] not in best or row["mib"] > best[row["pid"]]["mib"]:
            best[row["pid"]] = row
    return sorted(best.values(), key=lambda row: -row["mib"])
