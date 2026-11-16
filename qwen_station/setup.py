"""First-run setup of the release: the app itself, a dedicated WSL distribution and the engine in it.

The heavy lifting happens in engine/provision.sh inside the distribution; this module creates the
distribution, copies the engine bundle into it and turns the script's progress lines into state for
the dashboard. Every step can be repeated, so a failed or cancelled setup is simply started again.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import string
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import gpu, installer
from .config import MANAGED_BASE_IMAGE, Config, resource_path, save_settings, windows_to_wsl_path

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
MIN_FREE_GB = 45  # measured 31 GB after the setup (distribution, vLLM, two checkpoints, caches), plus headroom
MIN_WSL = (2, 4, 4)  # first version whose `wsl --install` takes --name and --location
MIN_WINDOWS_BUILD = 19044  # Windows 10 21H2: GPU compute inside WSL
MIN_VRAM_MIB = 30000  # the model is built for a 32 GB card
# id, label, share of the total time in percent
STEPS = (
    ("app", "Install the app", 1),
    ("wsl", "Check and set up WSL", 1),
    ("distro", "Create the Linux environment", 4),
    ("files", "Copy the engine files", 1),
    ("system", "Compiler and tools", 5),
    ("uv", "Package manager uv", 1),
    ("venv", "Python 3.12", 2),
    ("vllm", "vLLM with PyTorch and CUDA", 20),
    ("cuda", "CUDA compiler", 3),
    ("patches", "vLLM patches", 1),
    ("kernels", "Prebuilt GPU kernels", 2),
    ("target", "Download the model", 50),
    ("drafter", "Download the drafter", 7),
    ("verify", "Final check", 2),
)
LABELS = {key: label for key, label, _ in STEPS}
WSL_DECLINED = ("WSL cannot be installed without administrator rights. Run the setup again and confirm the "
                "Windows prompt.")
WSL_FAILED = ("WSL could not be installed (exit code {code}). To install it by hand, run "
              "`wsl --install --no-distribution` in a PowerShell as administrator, restart Windows and run the "
              "setup again.")
NO_VIRTUALIZATION = ("Virtualization is not enabled. Turn on \"Virtualization Technology\" in the BIOS/UEFI "
                     "(Intel VT-x or AMD SVM), restart Windows and run the setup again.")
_VIRTUALIZATION = re.compile(r"0x80370102|0x80370114|HCS_E_HYPERV|virtualization|Virtualisierung", re.I)
_COPY_ENGINE = r"""set -e
mkdir -p "$2"
rm -rf "$2/engine"
cp -r "$1" "$2/engine"
find "$2/engine" -type f \( -name '*.sh' -o -name '*.py' -o -name '*.env' \) -exec sed -i 's/\r$//' {} +
chmod +x "$2/engine/"*.sh
"""
_CANCEL = "pkill -TERM -f engine/provision.sh; pkill -TERM -f 'hf download'; pkill -TERM -x uv; pkill -TERM -x apt-get; true\n"


def wsl_env() -> dict:
    """wsl.exe writes its own messages in UTF-16 unless told otherwise."""
    return {**os.environ, "WSL_UTF8": "1"}


def detect_wsl() -> tuple[str, str]:
    """('ok' | 'outdated' | 'missing', version). Windows ships a wsl.exe stub even when WSL is not installed."""
    try:
        status = subprocess.run(["wsl.exe", "--status"], capture_output=True, timeout=60,
                                creationflags=NO_WINDOW, env=wsl_env())
        if status.returncode != 0:
            return "missing", ""
        result = subprocess.run(["wsl.exe", "--version"], capture_output=True, timeout=60,
                                creationflags=NO_WINDOW, env=wsl_env())
    except (OSError, subprocess.TimeoutExpired):
        return "missing", ""
    text = result.stdout.decode("utf-8", "replace").replace("\x00", "")
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text) if result.returncode == 0 else None
    if not match:  # the old WSL that came with Windows knows no --version
        return "outdated", ""
    version = tuple(int(part) for part in match.groups())
    return ("ok" if version >= MIN_WSL else "outdated"), ".".join(match.groups())


def bundled_engine_version() -> str:
    for line in (resource_path("engine") / "engine.env").read_text(encoding="utf-8").splitlines():
        if line.startswith("ENGINE_VERSION="):
            return line.split("=", 1)[1].strip()
    return ""


def is_local_drive(root: str) -> bool:
    """True for a built-in disk. WSL cannot keep its virtual disk on a network share, USB sticks are too slow."""
    if os.name != "nt":
        return True
    import ctypes
    return ctypes.windll.kernel32.GetDriveTypeW(root) == 3  # DRIVE_FIXED


def suggest_location() -> str:
    """A folder on the local drive with the most free space; the distribution's disk grows to about 32 GB."""
    best, free = "C", -1
    for letter in string.ascii_uppercase[2:]:
        if not is_local_drive(f"{letter}:\\"):
            continue
        try:
            usage = shutil.disk_usage(f"{letter}:\\")
        except OSError:
            continue
        if usage.free > free:
            best, free = letter, usage.free
    return f"{best}:\\QwenStation"


def parse_marker(line: str) -> tuple[str, str, list[str]] | None:
    """'::step vllm done' -> ('step', 'vllm', ['done']); '::progress target 12 99' -> ('progress', 'target', ['12', '99'])."""
    if not line.startswith("::"):
        return None
    parts = line[2:].split()
    if len(parts) < 3 or parts[0] not in ("step", "progress"):
        return None
    return parts[0], parts[1], parts[2:]


def split_lines(pending: bytes, final: bool = False) -> tuple[list[tuple[str, bool]], bytes]:
    """Cut complete lines off a byte buffer. A line that ends in a bare \\r is a redrawn progress line."""
    lines = []
    while True:
        positions = [position for position in (pending.find(b"\n"), pending.find(b"\r")) if position >= 0]
        if not positions:
            break
        cut = min(positions)
        if pending[cut:cut + 1] == b"\r" and cut + 1 == len(pending) and not final:
            break  # the \n of a \r\n may still be on its way
        crlf = pending[cut:cut + 2] == b"\r\n"
        progress = pending[cut:cut + 1] == b"\r" and not crlf
        raw, pending = pending[:cut], pending[cut + (2 if crlf else 1):]
        text = raw.decode("utf-8", "replace").replace("\x00", "").strip()
        if text:
            lines.append((text, progress))
    if final:
        text = pending.decode("utf-8", "replace").replace("\x00", "").strip()
        if text:
            lines.append((text, False))
        pending = b""
    return lines, pending


class Setup:
    def __init__(self, config: Config, log) -> None:
        self.config = config
        self.log = log
        self.state = "off"  # off | checking | needed | running | restart | failed | done
        self.reason = ""  # install | update | app
        self.checks: list[dict] = []  # what the machine needs before the setup can run
        self.error = ""
        self.fresh = False  # set once this process finished a setup: the first model start compiles kernels
        self.started_at: float | None = None
        self.on_done = lambda: None
        self._steps = {key: {"status": "pending", "progress": None, "started": None, "seconds": None}
                       for key, _, _ in STEPS}
        self._process: subprocess.Popen | None = None
        self._cancelled = False
        self._lock = threading.Lock()

    # ---- state -----------------------------------------------------------------

    def wsl(self, *command: str) -> list[str]:
        return ["wsl.exe", "-d", self.config.distro, "-u", "root", "--exec", *command]

    def wsl_state(self) -> tuple[str, str]:
        return detect_wsl()

    def prerequisites(self) -> list[dict]:
        """Windows, graphics card and WSL. A missing or old WSL is not a blocker: the setup installs it."""
        checks = []
        build = sys.getwindowsversion().build if os.name == "nt" else MIN_WINDOWS_BUILD
        checks.append({"id": "windows", "label": "Windows", "ok": build >= MIN_WINDOWS_BUILD, "blocking": True,
                       "detail": f"Build {build}" if build >= MIN_WINDOWS_BUILD else
                       f"Build {build} is too old. Windows 11 or Windows 10 version 21H2 or later is required."})
        try:
            card = gpu.pick(gpu.query_once(), self.config.gpu_match)
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            card = None
        if card is None or not card.get("mem_total"):
            checks.append({"id": "gpu", "label": "Graphics card", "ok": False, "blocking": True,
                           "detail": "No NVIDIA graphics card found. Install the current NVIDIA driver "
                                     "(nvidia.com/drivers), then check again."})
        else:
            name = card["name"].replace("NVIDIA GeForce ", "")
            size = f"{name}, {card['mem_total'] / 1024:.0f} GiB"
            if card["mem_total"] < MIN_VRAM_MIB:
                checks.append({"id": "gpu", "label": "Graphics card", "ok": False, "blocking": True,
                               "detail": f"{size}. The model needs an RTX 5090 with 32 GB."})
            else:
                known = self.config.gpu_match.lower() in card["name"].lower()
                checks.append({"id": "gpu", "label": "Graphics card", "ok": True, "warn": not known, "blocking": False,
                               "detail": size if known else f"{size}. Only the RTX 5090 is tested."})
        state, version = self.wsl_state()
        detail = {"ok": f"Version {version}",
                  "outdated": f"Version {version or 'unknown'} is too old. The setup updates WSL.",
                  "missing": "Missing. The setup installs WSL: Windows asks for administrator rights and has to "
                             "restart once afterwards. Qwen Station then continues on its own."}[state]
        checks.append({"id": "wsl", "label": "WSL", "ok": True, "warn": state != "ok", "blocking": False,
                       "state": state, "detail": detail})
        return checks

    def distro_exists(self) -> bool | None:
        """None when wsl.exe itself is missing or broken."""
        try:
            result = subprocess.run(["wsl.exe", "-l", "-q"], capture_output=True, timeout=30,
                                    creationflags=NO_WINDOW, env=wsl_env())
            if result.returncode != 0:
                # Having no distribution at all also ends with an error code; only a missing WSL is fatal.
                status = subprocess.run(["wsl.exe", "--status"], capture_output=True, timeout=30,
                                        creationflags=NO_WINDOW, env=wsl_env())
                return False if status.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
        names = [name.strip() for name in result.stdout.decode("utf-8", "replace").replace("\x00", "").splitlines()]
        return self.config.distro in names

    def installed(self) -> dict | None:
        try:
            result = subprocess.run(self.wsl("cat", f"{self.config.prefix}/installed.json"), capture_output=True,
                                    timeout=90, creationflags=NO_WINDOW, env=wsl_env())
            info = json.loads(result.stdout.decode("utf-8", "replace")) if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return None
        return info if isinstance(info, dict) else None

    def check(self) -> str:
        """Decide whether a setup is needed. Cheap enough to run at every start."""
        if self.config.mode != "managed" or self.config.mock:
            self.state = "off"
            return self.state
        info = self.installed() if self.distro_exists() else None
        engine_current = bool(info) and info.get("engine") == bundled_engine_version()
        checks = [] if engine_current else self.prerequisites()
        with self._lock:
            self.checks = checks
            if engine_current and not installer.install_needed():
                self.state = "done"
            else:
                self.reason = "app" if engine_current else ("update" if info else "install")
                self.state = "needed"
        return self.state

    def install_app_only(self) -> None:
        """The engine is in place and only the app is new: no questions, and a failure never blocks the model."""
        try:
            self.log.add(f"--- App installed: {installer.install_app()}", "step")
        except OSError as exc:
            self.log.add(f"The app could not be installed: {exc}", "warn")
        with self._lock:
            self.state = "done"

    def snapshot(self) -> dict:
        now = time.time()
        with self._lock:
            steps = []
            for key, label, weight in STEPS:
                step = self._steps[key]
                seconds = step["seconds"]
                if step["status"] == "active" and step["started"]:
                    seconds = now - step["started"]
                steps.append({"id": key, "label": label, "weight": weight, "status": step["status"],
                              "progress": step["progress"],
                              "seconds": None if seconds is None else round(seconds, 1)})
            return {"state": self.state, "reason": self.reason, "error": self.error, "fresh": self.fresh,
                    "checks": self.checks if self.state == "needed" else [],
                    "steps": steps if self.state in ("running", "failed") else [],
                    "started_at": self.started_at,
                    "location": self.config.engine_location or suggest_location(),
                    "distro": self.config.distro, "min_free_gb": MIN_FREE_GB}

    def _mark(self, key: str, status: str) -> None:
        now = time.time()
        with self._lock:
            step = self._steps.get(key)
            if step is None:
                return
            if status == "active":
                step.update(status="active", started=now, seconds=None)
                return
            if step["started"]:
                step["seconds"] = now - step["started"]
            step["status"] = status
            if status == "done" and step["progress"]:
                step["progress"] = [step["progress"][1], step["progress"][1]]

    # ---- run -------------------------------------------------------------------

    def start(self, location: str = "") -> str:
        """Returns an error message, or '' when the setup started."""
        if self.config.mode != "managed" or self.config.mock:
            return "This installation is not managed by the app."
        blocked = [check for check in self.prerequisites() if check["blocking"] and not check["ok"]]
        if blocked:
            return "; ".join(f"{check['label']}: {check['detail']}" for check in blocked)
        exists = self.distro_exists()
        location = (location or "").strip() or self.config.engine_location or suggest_location()
        if exists and self.config.engine_location:
            location = self.config.engine_location  # the disk stays where it is
        path = Path(location)
        if len(location) < 4 or location[1:3] != ":\\" or not path.is_absolute():
            return "The folder must be a full Windows path, for example D:\\QwenStation."
        if not exists:
            if not is_local_drive(path.anchor):
                return (f"{path.anchor} is not a built-in drive. The Linux environment needs a local disk, "
                        "ideally an SSD.")
            try:
                free_gb = shutil.disk_usage(path.anchor).free / 2 ** 30
            except OSError:
                return f"The drive {path.anchor} is not reachable."
            if free_gb < MIN_FREE_GB:
                return (f"Only {free_gb:.0f} GB free on {path.anchor}. The setup needs about "
                        f"{MIN_FREE_GB} GB. Choose another drive.")
        with self._lock:
            if self.state == "running":
                return "The setup is already running."
            self.state = "running"
            self.error = ""
            self.started_at = time.time()
            self._cancelled = False
            for step in self._steps.values():
                step.update(status="pending", progress=None, started=None, seconds=None)
        threading.Thread(target=self._run, args=(location,), name="setup", daemon=True).start()
        return ""

    def cancel(self) -> None:
        """Stop a running setup, including what it started inside the distribution."""
        with self._lock:
            if self.state != "running":
                return
            self._cancelled = True
        try:
            subprocess.run(self.wsl("bash", "-s"), input=_CANCEL.encode("utf-8"), capture_output=True, timeout=30,
                           creationflags=NO_WINDOW, env=wsl_env())
        except (OSError, subprocess.TimeoutExpired):
            pass
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def _fail(self, key: str, message: str) -> None:
        self._mark(key, "failed")
        self.log.add(f"Setup failed: {message}", "error")
        with self._lock:
            self.error = message
            self.state = "failed"

    def _distro_hint(self, seen: int) -> str:
        lines = " ".join(line["text"] for line in self.log.since(seen))
        return NO_VIRTUALIZATION if _VIRTUALIZATION.search(lines) else ""

    def _restart(self, location: str) -> None:
        """WSL is installed but Windows has to restart first. Remember where to continue."""
        save_settings({"mode": "managed", "distro": self.config.distro, "prefix": self.config.prefix,
                       "engine_location": location, "setup_pending": True})
        try:
            installer.run_after_restart()
        except OSError as exc:
            self.log.add(f"The automatic start after the restart could not be registered: {exc}", "warn")
        self._mark("wsl", "done")
        self.log.add("=== WSL is installed. Windows has to restart now. The setup continues afterwards.", "step")
        with self._lock:
            self.state = "restart"

    def resume_after_restart(self) -> bool:
        """Continue a setup that was waiting for the restart. True when it started."""
        if not self.config.setup_pending or self.state != "needed":
            return False
        self.config.setup_pending = False
        save_settings({"setup_pending": False})  # once only: a second failure must not loop
        error = self.start(self.config.engine_location)
        if error:
            self.log.add(f"The setup could not continue after the restart: {error}", "warn")
        return not error

    def handle_line(self, text: str, progress: bool = False, markers: bool = False) -> None:
        marker = parse_marker(text) if markers else None
        if marker is None:
            self.log.add(text, progress=progress)
        elif marker[0] == "step":
            status = {"start": "active", "done": "done", "skip": "skipped", "fail": "failed"}.get(marker[2][0])
            if status:
                self._mark(marker[1], status)
            if status == "active":
                self.log.add(f"--- Setup: {LABELS.get(marker[1], marker[1])}", "step")
        elif len(marker[2]) == 2:
            try:
                values = [int(marker[2][0] or 0), int(marker[2][1])]
            except ValueError:
                return
            with self._lock:
                if marker[1] in self._steps:
                    self._steps[marker[1]]["progress"] = values

    def _stream(self, command: list[str], stdin: bytes | None = None, markers: bool = False) -> int:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                                   creationflags=NO_WINDOW, env=wsl_env())
        self._process = process
        assert process.stdout is not None
        if stdin is not None:
            assert process.stdin is not None
            process.stdin.write(stdin)
            process.stdin.close()
        descriptor = process.stdout.fileno()
        pending = b""
        while True:
            try:
                chunk = os.read(descriptor, 65536)
            except OSError:
                chunk = b""
            lines, pending = split_lines(pending + chunk, final=not chunk)
            for text, progress in lines:
                self.handle_line(text, progress, markers)
            if not chunk:
                break
        code = process.wait()
        process.stdout.close()
        self._process = None
        return code

    def _run(self, location: str) -> None:
        try:
            self._run_steps(location)
        except Exception as exc:  # the dashboard must show a failure, never hang in "running"
            self._fail("verify", f"Unexpected error: {exc!r}")

    def _run_steps(self, location: str) -> None:
        config = self.config
        self.log.add(f"=== Setup: distribution {config.distro}, folder {location}", "step")

        self._mark("app", "active")
        if installer.install_needed():
            try:
                self.log.add(f"App installed: {installer.install_app()}")
            except OSError as exc:
                return self._fail("app", f"The app could not be installed: {exc}")
            self._mark("app", "done")
        else:
            self._mark("app", "skipped")

        self._mark("wsl", "active")
        state, _ = self.wsl_state()
        installed_now = False
        if state == "missing":
            self.log.add("--- WSL is missing and will be installed. Windows asks for administrator rights.", "step")
            code = installer.run_elevated("wsl.exe", "--install", "--no-distribution")
            if code == 1223:
                return self._fail("wsl", WSL_DECLINED)
            installed_now = True
            state, _ = self.wsl_state()
            if state == "missing":
                # Windows has switched the features on, but they only work after a restart.
                return self._restart(location) if code == 0 else self._fail("wsl", WSL_FAILED.format(code=code))
        if state == "outdated":
            self.log.add("--- WSL is too old and will be updated.", "step")
            if self._stream(["wsl.exe", "--update"]) != 0 or self.wsl_state()[0] != "ok":
                installer.run_elevated("wsl.exe", "--update")
            if self.wsl_state()[0] != "ok":
                return self._fail("wsl", "WSL could not be updated. Run `wsl --update` in a PowerShell and run "
                                         "the setup again.")
        self._mark("wsl", "done")

        self._mark("distro", "active")
        exists = self.distro_exists()
        seen = self.log.seq
        if not exists:
            Path(location).mkdir(parents=True, exist_ok=True)
            code = self._stream(["wsl.exe", "--install", MANAGED_BASE_IMAGE, "--name", config.distro,
                                 "--location", str(Path(location) / "wsl"), "--no-launch"])
            if self._cancelled:
                return self._fail("distro", "The setup was cancelled.")
            if code != 0 or not self.distro_exists():
                if installed_now:
                    return self._restart(location)
                return self._fail("distro", self._distro_hint(seen) or
                                  f"The Linux environment could not be created (exit code {code}). See the log for details.")
        if self._stream(self.wsl("true")) != 0:
            if installed_now:
                return self._restart(location)
            return self._fail("distro", self._distro_hint(seen) or
                              "The Linux environment does not start. See the log for details.")
        self._mark("distro", "skipped" if exists else "done")

        self._mark("files", "active")
        source = windows_to_wsl_path(str(resource_path("engine")))
        if self._stream(self.wsl("bash", "-s", "--", source, config.prefix), stdin=_COPY_ENGINE.encode("utf-8")) != 0:
            return self._fail("files", "The engine files could not be copied into the Linux environment.")
        self._mark("files", "done")

        code = self._stream(self.wsl("env", f"QWEN_STATION_PREFIX={config.prefix}", "bash",
                                     f"{config.prefix}/engine/provision.sh", "all"), markers=True)
        if code != 0:
            with self._lock:
                failed = next((key for key, step in self._steps.items() if step["status"] in ("failed", "active")),
                              "verify")
            if self._cancelled:
                return self._fail(failed, "The setup was cancelled. Next time it continues where it stopped.")
            return self._fail(failed, f"Step \"{LABELS.get(failed, failed)}\" failed. See the log for details. "
                                      "You can run the setup again; finished steps are kept.")
        config.engine_location = location
        save_settings({"mode": "managed", "distro": config.distro, "prefix": config.prefix,
                       "engine_location": location, "setup_pending": False})
        self.log.add("=== Setup complete", "step")
        with self._lock:
            self.fresh = True
            self.state = "done"
        self.on_done()


def uninstall_message(config: Config) -> str:
    return ("Qwen Station is removed when you close this message. The engine and the models are kept.\n\n"
            f"To delete them too (about 32 GB), run this in a PowerShell: wsl --unregister {config.distro}")
