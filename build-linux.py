#!/usr/bin/env python3
"""Builds the Linux release: one self-extracting file, dist/Qwen-Station-Setup-<version>-linux.run.

    python build-linux.py              run the tests, then build
    python build-linux.py --skip-tests

The file is packaging/linux-setup.sh followed by a tar.gz of the app (plain Python, standard library only).
It runs on any Linux with Python 3.10 or newer, and this script builds it on Windows or Linux alike.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import make_icon

ROOT = Path(__file__).resolve().parent
TEXT = {".py", ".sh", ".env", ".html", ".json"}  # stored with LF line endings, whatever the checkout did
SKIP_NAMES = {"__pycache__", "setup-qwen-lan.ps1"}  # the LAN script is for Windows port forwarding only


def app_files() -> list[tuple[str, bytes, int]]:
    """(path in the archive, content, mode) for everything the installed app needs."""
    files = [("run-station.py", (ROOT / "run-station.py").read_bytes(), 0o755),
             ("qwen-station.png", make_icon.render(256), 0o644)]
    for path in sorted((ROOT / "qwen_station").rglob("*")):
        if not path.is_file() or SKIP_NAMES.intersection(path.parts) or path.suffix == ".pyc":
            continue
        files.append((path.relative_to(ROOT).as_posix(), path.read_bytes(), 0o755 if path.suffix == ".sh" else 0o644))
    return [(name, data.replace(b"\r\n", b"\n") if Path(name).suffix in TEXT else data, mode)
            for name, data, mode in files]


def payload() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, data, mode in app_files():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def build() -> Path:
    sys.path.insert(0, str(ROOT))
    from qwen_station import __version__

    stub = (ROOT / "packaging" / "linux-setup.sh").read_text(encoding="utf-8").replace("\r\n", "\n")
    if not stub.endswith("__PAYLOAD__\n"):
        raise SystemExit("packaging/linux-setup.sh must end with the line __PAYLOAD__")
    target = ROOT / "dist" / f"Qwen-Station-Setup-{__version__}-linux.run"
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(stub.replace("@VERSION@", __version__).encode("utf-8") + payload())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    Path(f"{target}.sha256").write_text(f"{digest}  {target.name}\n", encoding="ascii", newline="\n")
    print(f"Release: {target} ({target.stat().st_size / 1024:.0f} KB)")
    print(f"SHA256:  {digest}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    if not args.skip_tests:
        if subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=ROOT).returncode != 0:
            raise SystemExit("Tests failed; nothing built.")
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
