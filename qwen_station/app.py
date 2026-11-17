"""The station process: HTTP API for the dashboard, pollers, window handling, entry point."""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import __version__, installer, metrics
from .config import (AGENT_IDS, MODEL_ID, SUBAGENT_MODEL_ID, model_subtitle, MODEL_TITLE, PRESETS, Config, data_dir, load_config,
                     resource_path)
from . import clients, settings
from .config import save_settings, settings_path
from .gpu import GpuMonitor
from .setup import make_setup, uninstall_message
from .supervisor import Supervisor, http_get

LOG = logging.getLogger("qwen-station")
SAMPLE_SECONDS = 1.0
HISTORY = 1800  # 30 minutes
MAX_BODY = 64 * 1024


class Station:
    def __init__(self, config: Config, supervisor: Supervisor | None = None) -> None:
        self.config = config
        if config.agents_unset and config.register_clients and not config.mock:
            # No agent is switched on until the user chooses one, so a first start changes no agent config. An
            # update from 2.0.0 or older, where all were on, keeps updating the agents that already have the model.
            config.agents = clients.entered_agents()
        self.token = secrets.token_urlsafe(24)
        self.supervisor = supervisor or Supervisor(config)
        self.setup = make_setup(config, self.supervisor.log)
        self.setup.on_done = self._start_or_idle
        self.auto_setup = False  # --setup: start a needed setup without waiting for the click
        self.setup_location = ""
        self.console = False  # --console: print the log to the terminal
        self.gpu = GpuMonitor(config.gpu_match)
        self.tracker = metrics.Tracker()
        self.samples: deque[dict] = deque(maxlen=HISTORY)
        self.metrics_error: str | None = None
        self.quit_event = threading.Event()
        self.started = time.time()
        self.last_poll = 0.0
        self.bye_at = 0.0
        self.window_opened_at = 0.0
        self.browser: subprocess.Popen | None = None
        self.browser_tracked: bool | None = None  # None: still deciding, False: handed off or no process
        self._generation = -1
        self._metric_failures = 0
        self._lock = threading.Lock()
        self.httpd: ThreadingHTTPServer | None = None
        # Settings saved in the dashboard. Model settings wait in `pending` while the model runs and are
        # applied before its next start; the dashboard's own port waits in `after_restart` for the next app start.
        self.settings_file = Path(config.config_path) if config.config_path else settings_path()
        self.pending: dict = {}
        self.after_restart: dict = {}
        self._settings_lock = threading.Lock()
        self.supervisor.before_start = self.apply_pending

    # ---- polling -----------------------------------------------------------------

    def start_background(self) -> None:
        settings.local_addresses()  # starts the lookup, so the settings open without a wait
        self.gpu.start()
        threading.Thread(target=self._poll_loop, name="poller", daemon=True).start()

    def _poll_loop(self) -> None:
        tick = 0
        while not self.quit_event.is_set():
            started = time.time()
            try:
                self.poll_once(tick)
            except Exception:  # keep sampling whatever a single tick ran into
                LOG.exception("poll tick failed")
            tick += 1
            self.quit_event.wait(max(0.05, SAMPLE_SECONDS - (time.time() - started)))

    def poll_once(self, tick: int = 0) -> None:
        supervisor = self.supervisor
        state = supervisor.state
        base = f"http://{self.config.local_host}:{self.config.api_port}"
        if supervisor.generation != self._generation:
            self._generation = supervisor.generation
            self.tracker.reset()
            self._metric_failures = 0
        live = None
        if state == "starting" and tick % 2 == 0:
            try:
                if http_get(f"{base}/health", timeout=1.5)[0] == 200:
                    supervisor.mark_ready()
            except (OSError, urllib.error.URLError):
                pass
        elif state == "ready":
            try:
                status, body = http_get(f"{base}/metrics", timeout=2.0)
                if status != 200:
                    raise OSError(f"HTTP {status}")
                live = self.tracker.update(metrics.extract(metrics.parse_prometheus(body.decode("utf-8", "replace"))))
                self.metrics_error = None
                self._metric_failures = 0
            except (OSError, urllib.error.URLError, ValueError) as exc:
                self._metric_failures += 1
                self.metrics_error = f"/metrics does not respond: {exc}"
                if supervisor.external and self._metric_failures >= 5:
                    supervisor.log.add("The server started elsewhere no longer responds.", "warn")
                    supervisor.external_gone()
        card = self.gpu.latest
        fresh = card is not None and time.time() - card.get("t", 0) < 5
        sample = {
            "t": round(time.time(), 3),
            "tps": live["tps"] if live else None,
            "running": live["running"] if live else None,
            "kv": live["kv"] if live else None,
            "util": card["util"] if fresh else None,
            "vram": card["mem_used"] if fresh else None,
            "power": card["power"] if fresh else None,
            "temp": card["temp"] if fresh else None,
        }
        with self._lock:
            self.samples.append(sample)

    # ---- state for the dashboard ---------------------------------------------------

    def state_json(self, since: float = 0.0, log_seq: int = 0, request_seq: int = 0) -> dict:
        supervisor = self.supervisor
        snapshot = supervisor.snapshot()
        ready = snapshot["state"] == "ready"
        with self._lock:
            samples = [sample for sample in self.samples if sample["t"] > since]
        card = self.gpu.latest
        preset = PRESETS[snapshot["preset"]]
        return {
            "now": round(time.time(), 3),
            "station": {"version": __version__, "mock": self.config.mock, "close_action": self.config.close_action,
                        "mode": self.config.mode, "platform": self.setup.platform,
                        "root": self.config.root, "distro": self.config.distro,
                        "lan": self.config.lan,
                        "window_poll_age": round(time.time() - self.last_poll, 1) if self.last_poll else None},
            "model": {"id": MODEL_ID, "title": MODEL_TITLE, "vision": self.config.vision,
                      "subtitle": model_subtitle(self.config.kv_cache) + (", vision" if self.config.vision else ""),
                      "kv_cache": self.config.kv_cache,
                      "ram_cache_gib": self.config.ram_cache_gib,
                      "context": snapshot["context"],
                      "max_seqs": self.config.slots,
                      # Main session first: the subagent model's context, and whether it fits beside a full main session
                      "subagent_id": SUBAGENT_MODEL_ID,
                      "subagent_context": snapshot["subagent_context"],
                      "subagent_fits": snapshot["subagent_fits"]},
            # The key itself never leaves the app again once it is saved; a hint tells that one is set.
            "api": {"base_url": self.config.api_base_url, "network_url": settings.network_url(self.config),
                    "host": self.config.bind_host, "key_hint": settings.key_hint(self.config.api_key),
                    "model": MODEL_ID},
            "settings_pending": sorted(self.pending),
            "presets": {name: {key: values[key] for key in ("label", "context", "note")}
                        for name, values in PRESETS.items()},
            **snapshot,
            "setup": self.setup.snapshot(),
            "phases": supervisor.phases() if snapshot["state"] in ("starting", "preflight") else [],
            "live": self.tracker.latest if ready else None,
            "totals": self.tracker.totals() if ready else None,
            "metrics_error": self.metrics_error if ready else None,
            "gpu": card if card and time.time() - card.get("t", 0) < 5 else None,
            "gpu_error": self.gpu.error,
            "samples": samples,
            "requests": [row for row in self.tracker.requests if row["seq"] > request_seq] if ready else [],
            "request_seq": self.tracker.request_seq,
            "log": supervisor.log.since(log_seq),
            "log_seq": supervisor.log.seq,
        }

    # ---- window --------------------------------------------------------------------

    # ---- settings ----------------------------------------------------------------

    def _model_busy(self) -> bool:
        return self.supervisor.state in ("preflight", "starting", "ready", "stopping")

    def settings_json(self) -> dict:
        with self._settings_lock:
            saved = {**settings.values(self.config), **self.after_restart, **self.pending}
            pending, after_restart = sorted(self.pending), sorted(self.after_restart)
        # The key is shown once, by the page that generated it. From here on only a hint leaves the app.
        key = saved.pop("api_key")
        return {"values": saved, "key_hint": settings.key_hint(key), "pending": pending, "after_restart": after_restart,
                "addresses": settings.local_addresses(), "primary": settings.primary_address(),
                "running": self._model_busy(), "platform": self.setup.platform, "mode": self.config.mode,
                "model": MODEL_ID, "subagent_model": SUBAGENT_MODEL_ID,
                "agents": [{"id": agent, "label": clients.AGENTS[agent]} for agent in clients.AGENT_IDS],
                "clients": self.supervisor.clients,
                "file": "" if self.config.mock else str(self.settings_file),
                "limits": {"draft_tokens": [settings.DEFAULT_DRAFT_TOKENS, settings.MAX_DRAFT_TOKENS],
                           "slots": [1, settings.MAX_SLOTS],
                           "ram_cache_gib": [0, settings.MAX_RAM_CACHE_GIB],
                           "port": [settings.MIN_PORT, 65535]}}

    def save_settings(self, changes: dict) -> dict:
        """Check and store changed settings. Returns what the dashboard needs to tell the user."""
        with self._settings_lock:
            saved = {**settings.values(self.config), **self.after_restart, **self.pending}
            merged, error, field_name = settings.validate(self.config, saved, changes)
            if error:
                return {"ok": False, "error": error, "field": field_name}
            changed = {key: value for key, value in merged.items() if saved[key] != value}
            if changed and not self.config.mock:  # the mock is for trying the dashboard: nothing is written
                try:
                    save_settings(changed, self.settings_file)
                except (OSError, ValueError) as exc:
                    return {"ok": False, "error": f"The settings file could not be written: {exc}", "field": ""}
            busy = self._model_busy()
            if not busy:  # the model stopped since they were saved: nothing to wait for any more
                for key, value in self.pending.items():
                    setattr(self.config, key, value)
                self.pending = {}
            for key, value in changed.items():
                if key in settings.RESTART_KEYS:
                    self.after_restart[key] = value
                elif key in settings.MODEL_KEYS and busy:
                    self.pending[key] = value
                else:
                    setattr(self.config, key, value)
            live = settings.values(self.config)
            self.pending = {key: value for key, value in self.pending.items() if live[key] != value}
            self.after_restart = {key: value for key, value in self.after_restart.items() if live[key] != value}
            if changed:
                self.supervisor.log.add("--- Settings saved: " + ", ".join(sorted(changed)), "step")
            return {"ok": True, "changed": sorted(changed), "pending": sorted(self.pending),
                    "after_restart": sorted(self.after_restart), "clients": self._agents_changed(changed)}

    def _agents_changed(self, changed: dict) -> dict:
        """A new agent selection applies at once: the chosen agents get the model now, not at the next start."""
        if "agents" not in changed or self.config.mock or not self.config.register_clients:
            return self.supervisor.clients
        self.supervisor.refresh_clients(write=True)
        return self.supervisor.clients

    def apply_pending(self) -> None:
        """Called before every start of the model: the settings saved while it ran now take effect."""
        with self._settings_lock:
            for key, value in self.pending.items():
                setattr(self.config, key, value)
            self.pending = {}

    def restart_model(self) -> None:
        self.supervisor.stop()
        # The driver needs a moment to give the VRAM back; a start that measures too early shrinks the context.
        self.quit_event.wait(3.0)
        if not self.quit_event.is_set():
            self.supervisor.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.config.station_port}/"

    def open_window(self) -> None:
        self.window_opened_at = time.time()
        self.browser, self.browser_tracked = launch_window(self.url)
        if self.browser is not None:
            threading.Thread(target=self._classify_browser, daemon=True).start()

    def _classify_browser(self) -> None:
        # A browser that exits right away handed the window to an already running instance;
        # then only the page itself can tell us when it closes.
        try:
            self.browser.wait(timeout=8)
            self.browser_tracked = False
        except subprocess.TimeoutExpired:
            self.browser_tracked = True

    def note_poll(self) -> None:
        self.last_poll = time.time()

    def window_gone(self) -> bool:
        if not self.window_opened_at:
            return False
        now = time.time()
        browser_alive = self.browser is not None and self.browser.poll() is None
        if not self.last_poll:  # the page never loaded
            return now - self.window_opened_at > 120 and not browser_alive
        idle = now - self.last_poll
        if idle <= 6:
            return False
        if self.bye_at >= self.last_poll:  # the page said goodbye and did not come back (not a reload)
            return True
        if self.browser_tracked:
            return not browser_alive
        if self.browser_tracked is None:
            return False
        return idle > 150  # a hidden tab still polls about once a minute

    # ---- lifecycle -----------------------------------------------------------------

    def serve(self) -> None:
        handler = type("Handler", (_Handler,), {"station": self})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.config.station_port), handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, name="http", daemon=True).start()

    def shutdown(self) -> None:
        self.quit_event.set()
        try:
            self.setup.cancel()
            self.supervisor.shutdown()
        finally:
            self.gpu.stop()
            if self.httpd is not None:
                self.httpd.shutdown()
                self.httpd.server_close()

    def _boot(self) -> None:
        try:
            if self.setup.check() == "needed" and self.setup.reason == "app":
                self.setup.install_app_only()
            self.setup.resume_after_restart()
            # An update leaves nothing to decide (same folder, only changed parts), so it needs no click.
            if self.setup.state == "needed" and (self.auto_setup or self.setup.reason == "update"):
                error = self.setup.start(self.setup_location)
                if error:
                    self.supervisor.log.add(f"The setup could not start: {error}", "error")
            if self.setup.state in ("off", "done"):
                self._start_or_idle()
        except Exception:
            LOG.exception("boot failed")
            self.supervisor.log.add("The installation check crashed. See station.log for details.", "error")
            self.setup.state = "needed"

    def _start_or_idle(self) -> None:
        if self.quit_event.is_set():
            return
        if self.config.autostart:
            self.supervisor.start()
        elif not self.config.mock:
            self.supervisor.refresh_clients(write=False)

    def _console_loop(self) -> None:
        """The log on the terminal, for machines without a desktop. Downloads report every 15 seconds."""
        seq, reported, state = 0, 0.0, ""
        while not self.quit_event.wait(1.0):
            for line in self.supervisor.log.since(seq):
                seq = line["seq"]
                if not line.get("progress"):
                    print(line["text"], flush=True)
            setup = self.setup.snapshot()
            active = next((step for step in setup["steps"] if step["status"] == "active" and step["progress"]), None)
            if active and time.time() - reported >= 15:
                reported = time.time()
                done, total = active["progress"]
                print(f"    {active['label']}: {done / 1e9:.1f} of {total / 1e9:.1f} GB", flush=True)
            current = setup["state"] if setup["state"] not in ("off", "done") else self.supervisor.state
            if current != state:
                state = current
                if state == "needed":
                    print(f"Setup needed. Open {self.url} and click \"Set up\", or start again with --setup.", flush=True)

    def run(self) -> int:
        self.serve()
        self.start_background()
        if self.config.mode in ("managed", "native") and not self.config.mock:
            self.setup.state = "checking"  # shown until the look at the installation is back
        if self.config.open_window:
            self.open_window()
        if self.console:
            threading.Thread(target=self._console_loop, name="console", daemon=True).start()
        threading.Thread(target=self._boot, name="boot", daemon=True).start()
        try:
            while not self.quit_event.wait(1.0):
                if self.config.close_action == "stop" and self.window_gone():
                    LOG.info("window closed, shutting down")
                    break
        except KeyboardInterrupt:
            pass
        self.shutdown()
        return 0


OTHER_BROWSERS = ("firefox", "firefox-esr", "librewolf", "epiphany", "falkon", "konqueror", "vivaldi", "opera")


def can_show_window() -> bool:
    """Is there a desktop and a browser for the dashboard? A display alone does not say so: WSLg and
    `ssh -X` set one on machines without any browser, and the app then has to behave as on a server."""
    if os.name == "nt":
        return True
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    if any(path.is_file() for path in _app_browsers()) or any(shutil.which(name) for name in OTHER_BROWSERS):
        return True
    tool = shutil.which("xdg-mime")  # some other browser registered itself as the handler for web pages
    if not tool:
        return False
    try:
        handler = subprocess.run([tool, "query", "default", "x-scheme-handler/http"], capture_output=True,
                                 text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return bool(handler.strip())


def _app_browsers() -> list[Path]:
    """Browsers that can show a page as its own window (--app), best first."""
    if os.name != "nt":
        names = ("microsoft-edge", "microsoft-edge-stable", "google-chrome", "google-chrome-stable", "chromium",
                 "chromium-browser", "brave-browser")
        return [Path(found) for found in (shutil.which(name) for name in names) if found]
    roots = [os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")]
    relative = (r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe")
    return [Path(root) / tail for tail in relative for root in roots if root]


def launch_window(url: str) -> tuple[subprocess.Popen | None, bool | None]:
    """Open the dashboard as its own app window (Edge or Chrome), else in the default browser."""
    if not can_show_window():
        return None, False
    for executable in _app_browsers():
        if not executable.is_file():
            continue
        profile = data_dir() / "window-profile"
        arguments = [str(executable), f"--app={url}", f"--user-data-dir={profile}", "--no-first-run",
                     "--no-default-browser-check", "--window-size=1340,920"]
        try:
            return subprocess.Popen(arguments, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL), None
        except OSError:
            LOG.exception("could not start %s", executable)
    webbrowser.open(url)
    return None, False


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"qwen-station/{__version__}"
    station: Station

    def log_message(self, format: str, *args) -> None:
        pass

    def _host_ok(self) -> bool:
        # Only answer requests addressed to this loopback endpoint (blocks DNS rebinding).
        port = self.station.config.station_port
        return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    def do_GET(self) -> None:
        if not self._host_ok():
            self._send(403, b"Forbidden", "text/plain")
            return
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        if url.path == "/":
            page = resource_path("dashboard.html").read_text(encoding="utf-8").replace("__STATION_TOKEN__", self.station.token)
            self.send_response(200)
            body = page.encode("utf-8")
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                             "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/ping":
            self._json(200, {"app": "qwen-station", "version": __version__})
        elif url.path == "/api/state":
            if self.headers.get("X-Station-Token") == self.station.token:
                self.station.note_poll()

            def number(name: str) -> float:
                try:
                    return float(query.get(name, ["0"])[0])
                except ValueError:
                    return 0.0

            self._json(200, self.station.state_json(number("since"), int(number("log")), int(number("req"))))
        elif url.path == "/api/settings":
            if self.headers.get("X-Station-Token") != self.station.token:
                self._send(403, b"Forbidden", "text/plain")
                return
            self._json(200, self.station.settings_json())
        elif url.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._send(404, b"Not found", "text/plain")

    def do_POST(self) -> None:
        if not self._host_ok():
            self._send(403, b"Forbidden", "text/plain")
            return
        url = urlsplit(self.path)
        station = self.station
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            self._send(400, b"Bad request", "text/plain")
            return
        raw = self.rfile.read(length) if length else b""
        if url.path == "/api/bye":
            # Sent by the page while it unloads. A beacon cannot carry headers, so the token is in the query.
            if parse_qs(url.query).get("token", [""])[0] == station.token:
                station.bye_at = time.time()
            self._send(204, b"")
            return
        # The token lives only in the served page; other origins cannot read it or set this header.
        if self.headers.get("X-Station-Token") != station.token:
            self._send(403, b"Forbidden", "text/plain")
            return
        try:
            payload = json.loads(raw) if raw else {}
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            self._json(400, {"error": "Body must be a JSON object"})
            return
        supervisor = station.supervisor
        if url.path == "/api/setup/start":
            error = station.setup.start(str(payload.get("location") or ""))
            self._json(200, {"ok": not error, "error": error})
        elif url.path == "/api/setup/check":
            if station.setup.state in ("needed", "failed", "restart"):
                station.setup.state = "checking"
                threading.Thread(target=station.setup.check, name="setup-check", daemon=True).start()
            self._json(200, {"ok": True})
        elif url.path == "/api/setup/restart":
            if station.setup.state != "restart":
                self._json(409, {"error": "No restart needed"})
                return
            installer.restart_windows()
            self._json(200, {"ok": True})
            station.quit_event.set()
        elif url.path == "/api/setup/cancel":
            threading.Thread(target=station.setup.cancel, name="setup-cancel", daemon=True).start()
            self._json(200, {"ok": True})
        elif url.path == "/api/start":
            if station.setup.state not in ("off", "done"):
                self._json(409, {"error": "Finish the setup first"})
                return
            preset = payload.get("preset")
            if preset is not None and preset not in PRESETS:
                self._json(400, {"error": "Unknown preset"})
                return
            self._json(200, {"ok": supervisor.start(preset, force=bool(payload.get("force")))})
        elif url.path == "/api/stop":
            threading.Thread(target=supervisor.stop, name="stop", daemon=True).start()
            self._json(200, {"ok": True})
        elif url.path == "/api/settings":
            changes = payload.get("values")
            if not isinstance(changes, dict):
                self._json(400, {"error": "values must be an object"})
                return
            self._json(200, station.save_settings(changes))
        elif url.path == "/api/restart":
            if station.setup.state not in ("off", "done"):
                self._json(409, {"error": "Finish the setup first"})
                return
            threading.Thread(target=station.restart_model, name="restart", daemon=True).start()
            self._json(200, {"ok": True})
        elif url.path == "/api/quit":
            self._json(200, {"ok": True})
            station.quit_event.set()
        else:
            self._send(404, b"Not found", "text/plain")


# ---- entry point -------------------------------------------------------------------


def _message_box(text: str, title: str = "Qwen Station", icon: int = 0x10) -> None:
    if os.name != "nt" or (sys.stdout is not None and sys.stdout.isatty()):
        print(text, file=sys.stdout if icon == 0x40 and sys.stdout is not None else sys.stderr)
        return
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, title, icon)
    except Exception:
        LOG.error(text)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _running_instance(port: int) -> bool:
    try:
        status, body = http_get(f"http://127.0.0.1:{port}/api/ping", timeout=1.5)
        return status == 200 and json.loads(body).get("app") == "qwen-station"
    except (OSError, urllib.error.URLError, ValueError):
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="Qwen-Station", description="Starts Qwen3.8 on vLLM and shows a live monitor.")
    parser.add_argument("--config", help="Path to a qwen-station.json")
    parser.add_argument("--root", help="Folder with the models and start scripts")
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument("--api-host", dest="api_host",
                        help="Address the API listens on: 127.0.0.1 (default), 0.0.0.0 or one address of this machine")
    parser.add_argument("--lan", action="store_true", default=None, help="The same as --api-host 0.0.0.0")
    parser.add_argument("--port", type=int, dest="station_port", help="Port of the dashboard")
    parser.add_argument("--api-port", type=int, dest="api_port", help="Port of the vLLM API")
    parser.add_argument("--no-autostart", action="store_false", dest="autostart", default=None)
    parser.add_argument("--no-window", action="store_false", dest="open_window", default=None)
    parser.add_argument("--no-register-clients", action="store_false", dest="register_clients", default=None,
                        help="Leave the configs of all coding agents alone (for experiments)")
    parser.add_argument("--agents", type=lambda text: [part for part in text.split(",") if part],
                        help="Coding agents to enter the model in, comma-separated: " + ",".join(AGENT_IDS)
                        + ". Default: the choice in the settings, none at first")
    parser.add_argument("--no-fit-context", action="store_false", dest="fit_context", default=None,
                        help="Do not shrink the context automatically when VRAM is short")
    parser.add_argument("--ram-cache-gib", type=int, dest="ram_cache_gib",
                        help="Keep sessions that leave the GPU in this much RAM, in GiB (default 0: off)")
    parser.add_argument("--kv-cache", choices=("nvfp4", "fp8"), dest="kv_cache",
                        help="How the KV cache stores keys and values: nvfp4 (default, more context) or fp8")
    parser.add_argument("--slots", type=int, dest="slots",
                        help="Requests the server works on at the same time (default 1); they share the context")
    parser.add_argument("--main-first", action="store_true", dest="main_first", default=None,
                        help="With several slots: the main session always has room for the full context, subagents "
                             "(model " + SUBAGENT_MODEL_ID + ") get what is left and yield to it")
    parser.add_argument("--vision", action="store_true", default=None,
                        help="Load the image encoder, so requests may contain images (costs VRAM, so context)")
    parser.add_argument("--draft-tokens", type=int, dest="draft_tokens",
                        help="Draft tokens per round (default 7). More speeds up reasoning and code, and costs context")
    parser.add_argument("--keep", action="store_const", const="keep", dest="close_action", default=None,
                        help="Keep the model running when the window closes")
    parser.add_argument("--mock", action="store_true", default=None, help="Fake server instead of the GPU, for trying things out")
    parser.add_argument("--extra-args", dest="extra_args", help="Extra arguments for `vllm serve`")
    parser.add_argument("--vllm-prefix", dest="vllm_prefix", help="venv with vLLM, for example /opt/vllm-0.32.0")
    parser.add_argument("--model-base", dest="model_base", help="WSL folder with the target and the drafter")
    parser.add_argument("--extra-env", dest="extra_env", help="KEY=VALUE pairs for the start scripts")
    parser.add_argument("--check", action="store_true", help="Check the requirements and client entries, then exit")
    parser.add_argument("--uninstall", action="store_true",
                        help="Remove the app and its shortcuts. The engine and the models are kept")
    parser.add_argument("--setup", action="store_true",
                        help="Start a needed setup right away instead of waiting for the click in the window")
    parser.add_argument("--location", help="Folder for the engine and the model, for --setup")
    parser.add_argument("--console", action="store_true", default=None,
                        help="Print the log to the terminal (default on Linux without a window)")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--mock-server", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mock-speed", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--mock-context", type=int, default=262144, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mock_server is not None:
        from . import mock
        return mock.serve(args.mock_server, args.mock_speed, args.mock_context)
    if args.version:
        _message_box(f"Qwen Station {__version__}", icon=0x40)
        return 0
    logging.basicConfig(filename=str(data_dir() / "station.log"), level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        overrides = {key: getattr(args, key) for key in
                     ("root", "preset", "lan", "api_host", "station_port", "api_port", "autostart", "open_window",
                      "close_action",
                      "mock", "extra_args", "vllm_prefix", "model_base", "extra_env", "fit_context", "draft_tokens", "vision", "slots", "main_first", "ram_cache_gib", "kv_cache",
                      "register_clients", "agents")}
        config = load_config(args.config, overrides)
        if config.mock:
            if args.api_port is None:
                config.api_port = _free_port()
            if args.station_port is None:
                config.station_port = 8011
        if args.uninstall:
            # The message first: the installed exe can only be deleted once this process has ended.
            _message_box(uninstall_message(config), icon=0x40)
            installer.uninstall_app()
            return 0
        if config.open_window and not can_show_window():
            config.open_window = False  # no desktop: the dashboard is still there for a browser or an SSH tunnel
        station = Station(config)
        station.auto_setup = bool(args.setup)
        station.setup_location = args.location or ""
        station.console = args.console if args.console is not None else (
            os.name != "nt" and not config.open_window and sys.stdout is not None)
        if args.check:
            supervisor = station.supervisor
            checks = supervisor.run_preflight(supervisor.probe_api()[0])
            supervisor.refresh_clients(write=False)
            report = {"mode": config.mode, "prefix": config.prefix, "root": config.root, "distro": config.distro,
                      "setup": station.setup.check(), "preflight": checks, "gpu_processes": supervisor.gpu_processes,
                      "clients": supervisor.clients}
            if sys.stdout is not None:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if all(check["ok"] for check in checks) else 1
        if _running_instance(config.station_port):
            if config.open_window:
                launch_window(f"http://127.0.0.1:{config.station_port}/")  # already running: just show it
            return 0
        if installer.image_dir() is not None and not config.mock:
            arguments = list(sys.argv[1:] if argv is None else argv)
            if not arguments and not can_show_window():
                # Without a desktop there is no window to click in: the setup starts right away.
                arguments, station.auto_setup = ["--setup"], True
            problem = installer.linux_handover(arguments)  # returns only if the app could not be installed
            LOG.warning("app not installed: %s", problem)
            print(f"The app could not be installed ({problem}). It runs from the image this time.", file=sys.stderr)
        def stop(signum, frame) -> None:  # kill, a closed terminal: stop the model in order
            station.quit_event.set()

        for name in ("SIGTERM", "SIGHUP"):
            if hasattr(signal, name):
                try:
                    signal.signal(getattr(signal, name), stop)
                except (ValueError, OSError):
                    pass
        if os.name != "nt" and sys.stdout is not None:
            print(f"Qwen Station {__version__}: dashboard {station.url}, API {config.api_base_url}", flush=True)
        return station.run()
    except OSError as exc:
        LOG.exception("startup failed")
        _message_box(f"Qwen Station could not start:\n{exc}\n\nIs the port of the dashboard already in use?")
        return 1
    except (ValueError, json.JSONDecodeError) as exc:
        LOG.exception("configuration error")
        _message_box(f"Error in the config:\n{exc}")
        return 2
    except Exception as exc:
        LOG.exception("fatal")
        _message_box(f"Qwen Station crashed:\n{exc!r}\n\nDetails: {data_dir() / 'station.log'}")
        return 3
