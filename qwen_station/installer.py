"""Installs the app itself on Windows: a copy below %LOCALAPPDATA%\\Programs, shortcuts, an uninstall entry.

No admin rights are needed; everything lives in the user's profile.

On Linux the release is an AppImage. Started from there, the app copies the content of the image into the user's
data folder, adds a command and a menu entry, and continues as that installed copy (linux_handover).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__

APP_NAME = "Qwen Station"
EXE_NAME = "Qwen-Station.exe"
UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\QwenStation"
RUN_ONCE_KEY = r"Software\Microsoft\Windows\CurrentVersion\RunOnce"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
IS_WINDOWS = os.name == "nt"


def install_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "Programs" / APP_NAME


def installed_exe() -> Path:
    return install_dir() / EXE_NAME


def start_menu_link() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / f"{APP_NAME}.lnk"


def installed_version() -> str | None:
    if os.name != "nt":
        return None
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as key:
            return str(winreg.QueryValueEx(key, "DisplayVersion")[0])
    except OSError:
        return None


def is_portable() -> bool:
    """A settings file next to the exe marks a portable or development copy: it is never installed."""
    return (Path(sys.executable).resolve().parent / "qwen-station.json").is_file()


def install_needed() -> bool:
    """True for a release exe started from somewhere else (e.g. Downloads) that is newer or not installed."""
    if os.name != "nt" or not getattr(sys, "frozen", False) or is_portable():
        return False
    if Path(sys.executable).resolve() == installed_exe().resolve():
        return False
    return installed_version() != __version__ or not installed_exe().is_file()


_SHORTCUTS = r"""
$shell = New-Object -ComObject WScript.Shell
$links = @($env:QS_START_LINK, (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Qwen Station.lnk'))
foreach ($link in $links) {
  $shortcut = $shell.CreateShortcut($link)
  $shortcut.TargetPath = $env:QS_TARGET
  $shortcut.WorkingDirectory = Split-Path $env:QS_TARGET
  $shortcut.Description = 'Start and monitor Qwen3.8 on vLLM'
  $shortcut.Save()
}
"""

_REMOVE_SHORTCUTS = r"""
Remove-Item -LiteralPath $env:QS_START_LINK -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Qwen Station.lnk') -ErrorAction SilentlyContinue
exit 0
"""


def _powershell(script: str, **env: str) -> None:
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, text=True, timeout=60, creationflags=NO_WINDOW,
                            env={**os.environ, **env})
    if result.returncode != 0:
        raise OSError(result.stderr.strip() or f"PowerShell exit code {result.returncode}")


def run_elevated(program: str, *arguments: str, timeout: int = 1800) -> int:
    """Run a program with administrator rights (Windows asks via UAC) and wait. Returns its exit code,
    or 1223 (ERROR_CANCELLED) when the user declined."""
    quoted = ", ".join("'" + argument.replace("'", "''") + "'" for argument in arguments)
    script = ("try { $p = Start-Process -FilePath '" + program.replace("'", "''") + "' -ArgumentList " + quoted
              + " -Verb RunAs -Wait -PassThru; exit $p.ExitCode } catch { exit 1223 }")
    try:
        return subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                              capture_output=True, timeout=timeout, creationflags=NO_WINDOW).returncode
    except (OSError, subprocess.TimeoutExpired):
        return 1


def run_after_restart() -> None:
    """Open the app once after the next sign-in, so a setup that needed a restart of Windows continues."""
    if os.name != "nt" or not getattr(sys, "frozen", False):
        return
    import winreg
    target = installed_exe() if installed_exe().is_file() else Path(sys.executable)
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_ONCE_KEY) as key:
        winreg.SetValueEx(key, "QwenStation", 0, winreg.REG_SZ, f'"{target}"')


def restart_windows() -> None:
    subprocess.Popen(["shutdown.exe", "/r", "/t", "10", "/c", "Qwen Station: restarting to finish the WSL setup."],
                     creationflags=NO_WINDOW, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


def install_app() -> Path:
    """Copy the running exe to the install folder and register it. Returns the installed exe."""
    import winreg
    target = installed_exe()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sys.executable, target)
    _powershell(_SHORTCUTS, QS_TARGET=str(target), QS_START_LINK=str(start_menu_link()))
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as key:
        for name, value in (
            ("DisplayName", APP_NAME),
            ("DisplayVersion", __version__),
            ("Publisher", "mfgAkilae"),
            ("InstallLocation", str(target.parent)),
            ("DisplayIcon", str(target)),
            ("UninstallString", f'"{target}" --uninstall'),
        ):
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
        winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
        winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)
        winreg.SetValueEx(key, "EstimatedSize", 0, winreg.REG_DWORD, max(1, target.stat().st_size // 1024))
    return target


def linux_paths() -> dict[str, Path]:
    """Where the app lives on Linux: its files, the command, the menu entry and the icon."""
    data = Path(os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share"))
    return {"app": data / "qwen-station" / "app",
            "launcher": Path.home() / ".local" / "bin" / "qwen-station",
            "desktop": data / "applications" / "qwen-station.desktop",
            "icon": data / "icons" / "hicolor" / "256x256" / "apps" / "qwen-station.png"}


START_FILE = "AppRun"      # start file of the image and of the installed copy
BUILD_FILE = "VERSION"     # version and build of an image; decides whether the installed copy is replaced
IMAGE_VARIABLES = ("APPDIR", "APPIMAGE", "ARGV0", "OWD")  # set by the AppImage runtime


def desktop_entry(command: str) -> str:
    """The menu entry. The command is quoted the way the desktop entry specification asks for."""
    quoted = '"' + "".join("\\" + char if char in '"`$\\' else char for char in command) + '"'
    return ("[Desktop Entry]\nType=Application\nName=Qwen Station\nComment=Start and monitor Qwen3.8 on vLLM\n"
            f"Exec={quoted}\nIcon=qwen-station\nTerminal=false\nCategories=Development;Science;\n")


def image_dir() -> Path | None:
    """The mounted or unpacked AppImage this process runs from. None for the installed copy or a checkout."""
    mounted = os.environ.get("APPDIR")
    if IS_WINDOWS or not mounted:
        return None
    root = Path(mounted).resolve()
    return root if root in Path(__file__).resolve().parents and (root / START_FILE).is_file() else None


def _build_of(folder: Path) -> str | None:
    try:
        return (folder / BUILD_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def install_linux_app(source: Path) -> Path:
    """Copy the content of the image to the app folder, then write the command, the menu entry and the icon.
    Returns the start file of the installed copy."""
    paths = linux_paths()
    app = paths["app"]
    staging = app.with_name(app.name + ".new")
    shutil.rmtree(staging, ignore_errors=True)
    app.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, staging, symlinks=True)
    shutil.rmtree(app, ignore_errors=True)
    staging.rename(app)
    start = app / START_FILE
    for path in (paths["launcher"], paths["desktop"], paths["icon"]):
        path.parent.mkdir(parents=True, exist_ok=True)
    quoted = "'" + str(start).replace("'", "'\\''") + "'"
    paths["launcher"].write_text(f'#!/bin/sh\nexec {quoted} "$@"\n', encoding="utf-8", newline="\n")
    paths["launcher"].chmod(0o755)
    paths["desktop"].write_text(desktop_entry(str(paths["launcher"])), encoding="utf-8", newline="\n")
    shutil.copyfile(app / "qwen-station.png", paths["icon"])
    return start


def linux_handover(arguments: list[str]) -> str | None:
    """Started from the AppImage: install its content if it is new, then continue as the installed copy.

    Does not return when that works. Returns None when this process is not an image, and the reason when the
    installation failed; the caller then keeps running from the image."""
    source = image_dir()
    if source is None:
        return None
    start = linux_paths()["app"] / START_FILE
    try:
        if _build_of(start.parent) != _build_of(source) or not start.is_file():
            install_linux_app(source)
            print(f"Qwen Station {__version__} is installed. Start it from the menu or with: "
                  f"{linux_paths()['launcher']}", flush=True)
        environment = {name: value for name, value in os.environ.items() if name not in IMAGE_VARIABLES}
        if source.name.startswith("appimage_extracted_") and os.access(start, os.X_OK):
            # Started with --appimage-extract-and-run: the runtime unpacked the image into the temp folder and
            # never removes it (its cleanup calls nftw with 0 file descriptors, which does nothing in musl).
            # Nothing of it is needed from here on; this process keeps the files it has open.
            shutil.rmtree(source, ignore_errors=True)
        os.execve(str(start), [str(start), *arguments], environment)
    except OSError as exc:
        return str(exc)
    return None


def uninstall_app() -> None:
    """Remove shortcuts, the uninstall entry and the installed exe. The engine and the models stay."""
    if not IS_WINDOWS:
        for name, path in linux_paths().items():
            if name == "app":
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        return
    import winreg
    _powershell(_REMOVE_SHORTCUTS, QS_START_LINK=str(start_menu_link()))
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY)
    except OSError:
        pass
    folder = install_dir()
    if folder.is_dir():
        # A running exe cannot delete itself: a detached shell removes the folder a few seconds later.
        # One command-line string on purpose: as a list, Python would escape the quotes around the path
        # in a way cmd.exe does not understand.
        subprocess.Popen(f'cmd.exe /c "ping -n 6 127.0.0.1 >nul & rmdir /s /q "{folder}""',
                         cwd=os.environ.get("SystemRoot") or None,  # never the folder that is about to go
                         creationflags=NO_WINDOW | getattr(subprocess, "DETACHED_PROCESS", 0),
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
