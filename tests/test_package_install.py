"""A quiet wheel install, including a stalled uv and a retry in the same cache."""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


@unittest.skipUnless(sys.platform == "linux", "uses Linux process I/O counters")
class PackageInstallTests(unittest.TestCase):
    def setUp(self):
        helper = Path(__file__).resolve().parents[1] / "qwen_station/engine/package_install.py"
        spec = importlib.util.spec_from_file_location("package_install", helper)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.cache = Path(temp.name) / "package cache"
        self.cache.mkdir()
        patcher = mock.patch.dict(os.environ, {"UV_CACHE_DIR": str(self.cache)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_install(self, code):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            status = self.module.install("vllm", [sys.executable, "-c", code], interval=0.1, idle_limit=0.8)
        return status, log.getvalue()

    def test_an_active_quiet_install_can_outlast_the_idle_limit(self):
        status, log = self.run_install("""
import os, time
with open(os.path.join(os.environ['UV_CACHE_DIR'], 'wheel'), 'wb', buffering=0) as f:
    for _ in range(10):
        f.write(b'x' * 1048576)
        time.sleep(0.15)
""")
        self.assertEqual(status, 0)
        self.assertRegex(log, r"uv wrote [1-9][0-9.]* MiB")
        self.assertNotIn("Retrying", log)

    def test_a_stalled_install_retries_once_reusing_the_cache(self):
        status, log = self.run_install("""
from pathlib import Path
import os, time
cached = Path(os.environ['UV_CACHE_DIR']) / 'completed wheel'
if not cached.exists():
    cached.write_bytes(b'cached')
    time.sleep(30)
else:
    assert cached.read_bytes() == b'cached' and os.environ['UV_CONCURRENT_DOWNLOADS'] == '1'
""")
        self.assertEqual(status, 0)
        self.assertIn("Retrying", log)

    def test_waiting_for_a_build_subprocess_does_not_trigger_a_retry(self):
        status, log = self.run_install("""
import subprocess, sys
subprocess.run([sys.executable, '-c', 'import time; time.sleep(1.4)'], check=True)
""")
        self.assertEqual(status, 0)
        self.assertIn("waiting for a subprocess", log)
        self.assertNotIn("Retrying", log)

    def test_two_stalls_stop_with_a_hint_but_package_errors_are_not_retried(self):
        status, log = self.run_install("import time; time.sleep(30)")
        self.assertNotEqual(status, 0)
        self.assertEqual(log.count("Retrying"), 1)
        self.assertIn("::hint vllm", log)
        self.assertIn(str(self.cache), log)
        status, log = self.run_install("raise SystemExit(7)")
        self.assertEqual(status, 7)
        self.assertNotIn("Retrying", log)
