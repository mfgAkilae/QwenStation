"""Report activity for uv pip install when the station reads its output through a pipe."""

import os
from pathlib import Path
import subprocess
import sys
import time


def activity(proc):
    """Process-wide I/O and CPU counters. A restricted /proc must never interrupt an installation."""
    try:
        counters = dict(line.split(": ") for line in (proc / "io").read_text().splitlines())
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        return int(counters["rchar"]), int(counters["wchar"]), int(stat[11]) + int(stat[12])
    except (OSError, KeyError, IndexError, ValueError):
        return None


def has_build_workers(proc):
    # uv can be waiting for a compiler or a build backend in a child of any of its threads.
    try:
        for task in (proc / "task").iterdir():
            try:
                if (task / "children").read_text().strip():
                    return True
            except FileNotFoundError:
                if task.exists():  # kernel without the children interface
                    return None
        return False
    except OSError:
        return None


def attempt(command, env, interval, idle_limit):
    """Return uv's exit status, or None after stopping an idle process."""
    with subprocess.Popen(command, env=env) as process:
        proc = Path("/proc") / str(process.pid)
        previous = activity(proc)
        last_change = time.monotonic()
        while True:
            try:
                return process.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                pass
            current = activity(proc)
            now = time.monotonic()
            if current is None or current != previous:
                last_change = now
            previous = current
            quiet = now - last_change
            report = (f"uv wrote {current[1] / 2**20:.1f} MiB while preparing packages"
                      if current is not None else "uv is preparing packages; I/O counters unavailable")
            worker = quiet >= interval and has_build_workers(proc)
            if worker:
                report += "; waiting for a subprocess"
            elif worker is None:
                report += "; subprocess status unavailable"
            elif quiet >= interval:
                report += f"; no new uv I/O or CPU activity for {quiet:.0f} s"
            print(report, flush=True)
            if quiet >= idle_limit and worker is False:
                if process.poll() is not None:
                    return process.returncode
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return None


def install(step, command, interval=10, idle_limit=180):
    env = os.environ.copy()
    env.setdefault("UV_CONCURRENT_DOWNLOADS", "4")
    cache = env["UV_CACHE_DIR"]
    print(f"Package cache: {cache}. Completed packages are reused on retry.", flush=True)
    status = attempt(command, env, interval, idle_limit)
    if status is None:
        print(f"No package activity for {idle_limit:g} s. Retrying with one download at a time.", flush=True)
        env["UV_CONCURRENT_DOWNLOADS"] = "1"
        status = attempt(command, env, interval, idle_limit)
    if status is None:
        reason = "The package manager stalled again."
        status = 1
    else:
        reason = "The package manager failed. See its error in the setup log."
    if status:
        print(f"::hint {step} {reason} Check the connection and free disk space. "
              f"Retry in the same folder to reuse {cache}.", flush=True)
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    sys.exit(install(sys.argv[1], sys.argv[2:]))
