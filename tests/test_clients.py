import json
import tempfile
import unittest
from pathlib import Path

from qwen_station import clients
from qwen_station.config import windows_to_wsl_path

URL = "http://127.0.0.1:8000/v1"


class ClientRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        self.opencode = root / "opencode" / "opencode.json"
        self.pi = root / "pi" / "models.json"
        self.pi.parent.mkdir()

    def register(self, context=262144, max_seqs=1, dry_run=False):
        return clients.register_all(URL, context, max_seqs, dry_run, opencode_file=self.opencode, pi_file=self.pi)

    def test_creates_both_entries_and_is_idempotent(self):
        first = self.register()
        self.assertEqual((first["opencode"]["status"], first["pi"]["status"]), ("ok", "ok"))
        self.assertTrue(first["opencode"]["changed"])
        opencode = json.loads(self.opencode.read_text(encoding="utf-8"))
        model = opencode["provider"]["vllm"]["models"]["qwen38-gittensor-dflash"]
        self.assertEqual(opencode["provider"]["vllm"]["options"]["baseURL"], URL)
        self.assertEqual(model["limit"], {"context": 262144, "output": 32768})
        self.assertEqual(model["variants"]["high"], {"disabled": True})
        self.assertEqual(model["variants"]["xhigh"], {"reasoningEffort": "xhigh"})
        pi = json.loads(self.pi.read_text(encoding="utf-8"))
        entry = pi["providers"]["vllm"]["models"][0]
        self.assertEqual(entry["contextWindow"], 262144)
        self.assertEqual(entry["thinkingLevelMap"]["high"], None)
        self.assertEqual(pi["providers"]["vllm"]["api"], "openai-completions")
        before = (self.opencode.read_bytes(), self.pi.read_bytes())
        second = self.register()
        self.assertNotIn("changed", second["opencode"])
        self.assertNotIn("changed", second["pi"])
        self.assertEqual(before, (self.opencode.read_bytes(), self.pi.read_bytes()))
        self.assertEqual(list(self.pi.parent.glob("*.bak-station-*")), [])  # nothing existed before the first write

    def test_keeps_foreign_content_makes_one_backup_and_preserves_formatting(self):
        original = {"$schema": "x", "providers": {"other": {"baseUrl": "http://127.0.0.1:47831/v1",
                                                             "apiKey": "!node key.mjs", "models": [{"id": "a"}]}}}
        text = json.dumps(original, indent=2)  # Pi writes no trailing newline
        self.pi.write_text(text, encoding="utf-8")
        self.opencode.parent.mkdir()
        self.opencode.write_text(json.dumps({"mcp": {"x": 1}, "provider": {"other": {"name": "keep"}}}, indent=2) + "\n",
                                 encoding="utf-8")
        self.register()
        pi = json.loads(self.pi.read_text(encoding="utf-8"))
        self.assertEqual(pi["providers"]["other"], original["providers"]["other"])
        self.assertFalse(self.pi.read_text(encoding="utf-8").endswith("\n"))
        opencode = json.loads(self.opencode.read_text(encoding="utf-8"))
        self.assertEqual(opencode["mcp"], {"x": 1})
        self.assertEqual(opencode["provider"]["other"], {"name": "keep"})
        self.assertTrue(self.opencode.read_text(encoding="utf-8").endswith("\n"))
        backups = list(self.pi.parent.glob("models.json.bak-station-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(encoding="utf-8"), text)
        self.register(context=237312)  # a later change must not overwrite the backup of the original
        self.assertEqual(backups[0].read_text(encoding="utf-8"), text)
        self.assertEqual(json.loads(self.pi.read_text(encoding="utf-8"))["providers"]["vllm"]["models"][0]["contextWindow"], 237312)

    def test_preset_change_updates_name_and_context_but_keeps_user_output_limit(self):
        self.register()
        config = json.loads(self.opencode.read_text(encoding="utf-8"))
        config["provider"]["vllm"]["models"]["qwen38-gittensor-dflash"]["limit"]["output"] = 16000
        self.opencode.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        self.register(context=98304, max_seqs=2)
        model = json.loads(self.opencode.read_text(encoding="utf-8"))["provider"]["vllm"]["models"]["qwen38-gittensor-dflash"]
        self.assertEqual(model["limit"], {"context": 98304, "output": 16000})
        self.assertTrue(model["name"].endswith("98K (RTX 5090) - 2 slots"))

    def test_variant_shows_in_the_model_name(self):
        clients.register_all(URL, 81920, 1, opencode_file=self.opencode, pi_file=self.pi, variant="Unsloth NVFP4")
        name = json.loads(self.opencode.read_text(encoding="utf-8"))["provider"]["vllm"]["models"]["qwen38-gittensor-dflash"]["name"]
        self.assertEqual(name, "Qwen3.8 27B Unsloth NVFP4 - FP8 KV - DFlash2 - 81K (RTX 5090)")
        self.assertEqual(clients.model_name(262144, 1), "Qwen3.8 27B - FP8 KV - DFlash2 - 262K (RTX 5090)")

    def test_dry_run_reports_without_writing(self):
        result = self.register(dry_run=True)
        self.assertEqual(result["opencode"]["status"], "outdated")
        self.assertFalse(self.opencode.exists())
        self.assertFalse(self.pi.exists())

    def test_opencode_config_with_plaintext_credential_is_left_alone(self):
        self.opencode.parent.mkdir()
        text = json.dumps({"provider": {"cloud": {"options": {"apiKey": "sk-real-secret"}}}})
        self.opencode.write_text(text, encoding="utf-8")
        result = self.register()
        self.assertEqual(result["opencode"]["status"], "error")
        self.assertEqual(self.opencode.read_text(encoding="utf-8"), text)

    def test_missing_pi_installation_and_broken_json_are_reported(self):
        missing = clients.register_all(URL, 262144, 1, opencode_file=self.opencode,
                                       pi_file=Path(self.folder.name) / "nowhere" / "models.json")
        self.assertEqual(missing["pi"]["status"], "missing")
        self.pi.write_text("{ not json", encoding="utf-8")
        self.assertEqual(self.register()["pi"]["status"], "error")
        self.assertEqual(self.pi.read_text(encoding="utf-8"), "{ not json")


class PathTests(unittest.TestCase):
    def test_windows_to_wsl_path(self):
        self.assertEqual(windows_to_wsl_path(r"D:\qwen-engine"), "/mnt/d/qwen-engine")
        self.assertEqual(windows_to_wsl_path("C:/x/"), "/mnt/c/x")


if __name__ == "__main__":
    unittest.main()
