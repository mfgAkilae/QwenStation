#!/usr/bin/env python3
"""Builds the Linux release: one executable file, dist/Qwen-Station-<version>-x86_64.AppImage.

    python3 build-linux.py              run the tests, then build
    python3 build-linux.py --skip-tests

The AppImage holds the app (plain Python, standard library only) and a Python of its own, so it needs nothing
from the system it runs on. This script runs on Linux (also inside WSL) and needs mksquashfs from the package
squashfs-tools. It downloads two pinned files: the standalone Python build and the AppImage runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import make_icon

ROOT = Path(__file__).resolve().parent
TEXT = {".py", ".sh", ".env", ".html", ".json"}  # stored with LF line endings, whatever the checkout did
SKIP_NAMES = {"__pycache__", "setup-qwen-lan.ps1"}  # the LAN script is for Windows port forwarding only

# python-build-standalone: runs from any folder and on any distribution with glibc 2.17 or newer.
PYTHON_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/20261110/"
              "cpython-3.12.16%2B20261110-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz")
PYTHON_SHA256 = "6a9e62dad033790aef722dbd370e0ce76ce17b73627ba3ae679f537049350b39"
# The static AppImage runtime: mounts the image with the system's fusermount, needs no libfuse2.
RUNTIME_URL = "https://github.com/AppImage/type2-runtime/releases/download/20251108/runtime-x86_64"
RUNTIME_SHA256 = "2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d"

# Parts of the Python distribution the app never loads. The interpreter in bin/ holds all of libpython itself;
# the shared libraries are for programs that embed Python, and they are half of the size.
PYTHON_DROP = ("include", "share", "lib/pkgconfig", "lib/libpython*", "lib/libtcl*", "lib/tcl*", "lib/tk*",
               "lib/itcl*", "lib/thread*",
               "lib/python3.*/test", "lib/python3.*/idlelib", "lib/python3.*/tkinter", "lib/python3.*/turtledemo",
               "lib/python3.*/ensurepip", "lib/python3.*/lib2to3", "lib/python3.*/pydoc_data",
               "lib/python3.*/config-*", "lib/python3.*/site-packages/pip*",
               "lib/python3.*/lib-dynload/_tkinter*", "bin/pip*", "bin/idle*", "bin/pydoc*", "bin/2to3*",
               "bin/python*-config")


def app_files() -> list[tuple[str, bytes, int]]:
    """(path in the image, content, mode) for everything of the app itself, without the Python."""
    sys.path.insert(0, str(ROOT))
    from qwen_station import __version__, installer

    station = [("station/run-station.py", (ROOT / "run-station.py").read_bytes(), 0o755)]
    for path in sorted((ROOT / "qwen_station").rglob("*")):
        if not path.is_file() or SKIP_NAMES.intersection(path.parts) or path.suffix == ".pyc":
            continue
        mode = 0o755 if path.suffix == ".sh" else 0o644
        station.append((f"station/{path.relative_to(ROOT).as_posix()}", path.read_bytes(), mode))
    station = [(name, data.replace(b"\r\n", b"\n") if Path(name).suffix in TEXT else data, mode)
               for name, data, mode in station]
    start = (ROOT / "packaging" / installer.START_FILE).read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256()
    for name, data, _ in station:
        digest.update(name.encode() + b"\0" + data)
    digest.update(start + PYTHON_SHA256.encode())
    return [(installer.START_FILE, start, 0o755),
            (installer.BUILD_FILE, f"{__version__} {digest.hexdigest()[:12]}\n".encode(), 0o644),
            ("qwen-station.desktop", installer.desktop_entry(installer.START_FILE).encode(), 0o644),
            ("qwen-station.png", make_icon.render(256), 0o644),
            *station]


def fetch(url: str, sha256: str) -> Path:
    """Download a pinned file once into .build/downloads and check its SHA256 every time."""
    target = ROOT / ".build" / "downloads" / url.rsplit("/", 1)[1].replace("%2B", "+")
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as response, open(f"{target}.part", "wb") as output:
            shutil.copyfileobj(response, output)
        os.replace(f"{target}.part", target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != sha256:
        raise SystemExit(f"{target} does not match its pinned SHA256. Delete it and check the pin.")
    return target


def fill(appdir: Path) -> None:
    """The content of the image: the Python, the app, and what the AppImage format asks for at the top."""
    with tarfile.open(fetch(PYTHON_URL, PYTHON_SHA256)) as archive:
        archive.extractall(appdir, filter="tar")  # unpacks into python/
    for pattern in PYTHON_DROP:
        for path in (appdir / "python").glob(pattern):
            shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
    for name, data, mode in app_files():
        path = appdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(mode)
    (appdir / ".DirIcon").symlink_to("qwen-station.png")
    # The trimmed Python must still load everything the app uses.
    modules = sorted(path.stem for path in (appdir / "station" / "qwen_station").glob("*.py") if path.stem != "__init__")
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import ssl, ctypes, sqlite3, lzma, bz2, zlib; "
            f"from qwen_station import {', '.join(modules)}")
    subprocess.run([str(appdir / "python" / "bin" / "python3"), "-I", "-B", "-c", code, str(appdir / "station")],
                   check=True)


def build() -> Path:
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise SystemExit("The AppImage is built on Linux x86_64 (WSL works).")
    if not shutil.which("mksquashfs"):
        raise SystemExit("mksquashfs is missing. Install the package squashfs-tools.")
    from qwen_station import __version__

    runtime = fetch(RUNTIME_URL, RUNTIME_SHA256)
    target = ROOT / "dist" / f"Qwen-Station-{__version__}-x86_64.AppImage"
    target.parent.mkdir(exist_ok=True)
    # A temp folder of the Linux file system: a Windows drive in WSL keeps neither modes nor symlinks.
    with tempfile.TemporaryDirectory(prefix="qwen-station-build-") as work:
        appdir = Path(work) / "AppDir"
        appdir.mkdir()
        fill(appdir)
        squashfs = Path(work) / "image.squashfs"
        subprocess.run(["mksquashfs", str(appdir), str(squashfs), "-root-owned", "-noappend", "-comp", "zstd",
                        "-Xcompression-level", "19", "-quiet", "-no-progress"], check=True)
        image = Path(work) / target.name
        with open(image, "wb") as output:  # an AppImage is the runtime followed by the file system
            for part in (runtime, squashfs):
                with open(part, "rb") as source:
                    shutil.copyfileobj(source, output)
        image.chmod(0o755)
        # --appimage-extract-and-run needs no FUSE, so this check also works in a container. The runtime
        # unpacks into TMPDIR and leaves the folder there; in the work folder it goes away with the rest.
        check = subprocess.run([str(image), "--appimage-extract-and-run", "--version"], capture_output=True,
                               text=True, timeout=120, env={**os.environ, "TMPDIR": work})
        if check.returncode != 0 or __version__ not in check.stdout:
            raise SystemExit(f"The built image does not start: {check.stdout}{check.stderr}")
        shutil.copyfile(image, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    Path(f"{target}.sha256").write_text(f"{digest}  {target.name}\n", encoding="ascii", newline="\n")
    print(f"Release: {target} ({target.stat().st_size / 1e6:.1f} MB)")
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
