"""Every home folder the app reads points at a throwaway one, for the whole test run.

Registering the model writes into the configs of coding agents (OpenCode, Pi, DeepSeek Harness, VS Code). A test
that forgets to pass its own paths must land here, never in the real configs of the person running the tests.
Every test module imports this first.
"""
import atexit
import os
import shutil
import tempfile

HOME = tempfile.mkdtemp(prefix="qwen-station-test-home-")
atexit.register(shutil.rmtree, HOME, True)

for name in ("USERPROFILE", "HOME"):
    os.environ[name] = HOME
os.environ["APPDATA"] = os.path.join(HOME, "AppData", "Roaming")
os.environ["XDG_CONFIG_HOME"] = os.path.join(HOME, ".config")
os.environ["DSH_HOME"] = os.path.join(HOME, ".dsh")
os.environ.pop("PI_CODING_AGENT_DIR", None)
