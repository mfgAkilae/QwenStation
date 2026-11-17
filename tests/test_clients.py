import tests.isolation  # noqa: F401  first of all: no test may reach the real agent configs
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


try:
    import yaml  # only to check the YAML Qwen Station writes; the app itself has no YAML library
except ImportError:  # pragma: no cover
    yaml = None

# The patch file of DeepSeek Harness as it writes it: comments, other plugins, a provider of the user's own.
DSH_PATCH = """# Your patch layer for this dsh profile, applied after every bundle layer:
# a top-level YAML array of loader patch entries.
- id: ui-settings-general
  name: "@deepseek-ai/dsh-client-ui-settings-general"
  config:
    welcomeNoticeVersion: 2026-08-13.1
- id: llm-pi-ai
  name: "@deepseek-ai/dsh-llm-pi-ai"
  config:
    providers:
      openai-codex:
        models:
          - id: gpt-6.1-sol
            name: GPT-6.1 Sol
            contextWindow: 272000
      lmstudio:
        displayName: LM Studio
        api: openai-completions
        baseURL: http://127.0.0.1:1234/v1
        models:
          - id: unsloth/qwen3.8-27b
            name: unsloth/qwen3.8-27b
- id: agent-default-model
  name: "@deepseek-ai/dsh-agent-default-model"
  config:
    provider: openai-codex
    model: gpt-6.1-sol
"""


class AgentTests(unittest.TestCase):
    """The agents beside OpenCode and Pi, and the choice among them. Every writer only adds."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.dsh = self.root / ".dsh"
        self.patch_file = clients.deepseek_path(self.dsh)
        self.patch_file.parent.mkdir(parents=True)
        self.vscode = self.root / "Code" / "User"
        self.vscode.mkdir(parents=True)
        self.t3 = self.root / ".t3"
        self.t3.mkdir()
        self.opencode = self.root / "opencode" / "opencode.json"
        self.pi = self.root / "pi" / "models.json"
        self.pi.parent.mkdir()

    def register(self, agents=clients.AGENT_IDS, **options):
        return clients.register_all(URL, options.pop("context", 196608), 1, opencode_file=self.opencode,
                                    pi_file=self.pi, agents=agents, deepseek_dir=self.dsh,
                                    copilot_folders=[self.vscode], t3_dir=self.t3, **options)

    def provider(self):
        entries = yaml.safe_load(self.patch_file.read_text(encoding="utf-8"))
        return next(entry for entry in entries if entry["id"] == "llm-pi-ai")["config"]["providers"]

    @unittest.skipUnless(yaml, "PyYAML is not installed")
    def test_deepseek_gets_a_provider_of_its_own_and_everything_else_stays(self):
        self.patch_file.write_text(DSH_PATCH, encoding="utf-8")
        result = self.register(agents=["deepseek"])
        self.assertEqual(result["deepseek"]["status"], "ok")
        text = self.patch_file.read_text(encoding="utf-8")
        self.assertTrue(text.startswith(DSH_PATCH.split("- id: llm-pi-ai")[0]))  # the comments and the plugin before
        self.assertTrue(text.endswith(DSH_PATCH.split("unsloth/qwen3.8-27b\n", 2)[2]))  # the plugin after
        before = yaml.safe_load(DSH_PATCH)
        after = yaml.safe_load(text)
        providers = self.provider()
        self.assertEqual({key: providers[key] for key in ("openai-codex", "lmstudio")},
                         next(entry for entry in before if entry["id"] == "llm-pi-ai")["config"]["providers"])
        self.assertEqual([entry for entry in after if entry["id"] != "llm-pi-ai"],
                         [entry for entry in before if entry["id"] != "llm-pi-ai"])
        own = providers["qwen-station"]
        self.assertEqual((own["api"], own["baseURL"]), ("openai-completions", URL))
        self.assertEqual(own["apiKeyEnv"], "QWEN_STATION_API_KEY")  # it refuses a route without a key
        self.assertEqual((self.dsh / ".env").read_text(encoding="utf-8"), "QWEN_STATION_API_KEY=local\n")
        model = own["models"][0]
        self.assertEqual((model["id"], model["contextWindow"], model["input"]), ("qwen38-gittensor-dflash", 196608, ["text"]))
        self.assertEqual(model["reasoningEfforts"], {"low": "low", "medium": "medium", "xhigh": "xhigh"})
        self.assertEqual(len(list(self.patch_file.parent.glob("*.bak-station-*"))), 1)
        # Again: nothing to do. With another context: only our block changes.
        self.assertNotIn("changed", self.register(agents=["deepseek"])["deepseek"])
        self.register(agents=["deepseek"], context=229376)
        self.assertEqual(self.provider()["qwen-station"]["models"][0]["contextWindow"], 229376)
        self.assertEqual(self.patch_file.read_text(encoding="utf-8").count("qwen-station:"), 1)
        self.assertEqual(set(self.provider()), {"openai-codex", "lmstudio", "qwen-station"})

    @unittest.skipUnless(yaml, "PyYAML is not installed")
    def test_deepseek_without_providers_yet(self):
        for text in ("# comment\n[]\n", "", "- id: llm-pi-ai\n  name: \"@deepseek-ai/dsh-llm-pi-ai\"\n",
                     "- id: llm-pi-ai\n  name: x\n  config: {}\n",
                     "- id: llm-pi-ai\n  name: x\n  config:\n    providers: {}\n    retry: 3\n"):
            self.patch_file.write_text(text, encoding="utf-8")
            self.assertEqual(self.register(agents=["deepseek"])["deepseek"]["status"], "ok", text)
            self.assertEqual(list(self.provider()), ["qwen-station"], text)
        self.assertEqual(yaml.safe_load(self.patch_file.read_text(encoding="utf-8"))[0]["config"]["retry"], 3)

    def test_deepseek_leaves_a_file_alone_that_it_cannot_read_safely(self):
        for text in ("- id: llm-pi-ai\n  config: {providers: {lmstudio: {}}}\n", "plugins:\n  - a\n"):
            self.patch_file.write_text(text, encoding="utf-8")
            self.assertEqual(self.register(agents=["deepseek"])["deepseek"]["status"], "error")
            self.assertEqual(self.patch_file.read_text(encoding="utf-8"), text)

    def test_deepseek_finds_the_key_by_name_in_its_env_file(self):
        env = self.dsh / ".env"
        env.write_text("OTHER=1\n", encoding="utf-8")
        self.register(agents=["deepseek"], api_key="qs-0123456789abcdef")
        self.assertIn("apiKeyEnv: QWEN_STATION_API_KEY", self.patch_file.read_text(encoding="utf-8"))
        self.assertEqual(env.read_text(encoding="utf-8"), "OTHER=1\nQWEN_STATION_API_KEY=qs-0123456789abcdef\n")
        self.assertNotIn("qs-0123456789abcdef", self.patch_file.read_text(encoding="utf-8"))
        self.register(agents=["deepseek"], api_key="")  # the key removed in the settings: a placeholder, OTHER stays
        self.assertEqual(env.read_text(encoding="utf-8"), "OTHER=1\nQWEN_STATION_API_KEY=local\n")

    def test_copilot_gets_a_custom_endpoint_group_and_other_groups_stay(self):
        target = self.vscode / "chatLanguageModels.json"
        theirs = [{"name": "llamacpp", "vendor": "customendpoint", "apiType": "chat-completions",
                   "models": [{"id": "", "name": "other", "url": "http://0.0.0.0:8080"}]}]
        target.write_text(json.dumps(theirs, indent=2), encoding="utf-8")
        result = self.register(agents=["copilot"], vision=True)
        self.assertEqual(result["copilot"]["status"], "ok")
        groups = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(groups[0], theirs[0])
        model = groups[1]["models"][0]
        self.assertEqual((groups[1]["name"], groups[1]["vendor"]), ("Qwen Station", "customendpoint"))
        self.assertEqual(model["url"], URL + "/chat/completions")
        self.assertEqual((model["id"], model["contextWindow"], model["vision"]), ("qwen38-gittensor-dflash", 196608, True))
        self.assertEqual(model["supportsReasoningEffort"], ["low", "medium", "xhigh"])
        self.assertNotIn("changed", self.register(agents=["copilot"], vision=True)["copilot"])
        self.assertEqual(clients.register_all(URL, 196608, 1, agents=["copilot"], copilot_folders=[])["copilot"]["status"],
                         "missing")

    def test_t3_code_keeps_the_entries_of_opencode_and_pi_up_to_date(self):
        result = self.register(agents=["t3code"])
        self.assertEqual((result["t3code"]["status"], result["t3code"]["detail"]), ("ok", "through OpenCode and Pi"))
        self.assertTrue(self.opencode.is_file() and self.pi.is_file())
        self.assertEqual([result[agent]["status"] for agent in ("deepseek", "copilot")], ["off", "off"])
        self.assertFalse(self.patch_file.exists())
        self.assertFalse((self.vscode / "chatLanguageModels.json").exists())
        self.t3.rmdir()
        self.assertEqual(self.register(agents=["t3code"])["t3code"]["status"], "missing")

    def test_the_agents_that_already_have_the_model(self):
        def found():
            return clients.entered_agents(opencode_file=self.opencode, pi_file=self.pi, t3_dir=self.t3,
                                          deepseek_dir=self.dsh, copilot_folders=[self.vscode])

        self.assertEqual(found(), [])
        self.pi.write_text(json.dumps({"providers": {"vllm": {"models": [{"id": "own-model"}]}}}), encoding="utf-8")
        self.patch_file.write_text(DSH_PATCH, encoding="utf-8")
        self.assertEqual(found(), [])  # a provider vllm of the user's own does not count
        self.register(agents=["pi", "copilot"])
        self.assertEqual(found(), ["t3code", "pi", "copilot"])
        self.register(agents=["opencode", "deepseek"])
        self.assertEqual(found(), list(clients.AGENT_IDS))
        self.opencode.write_text("{ broken", encoding="utf-8")  # unreadable: counts as without the model
        self.assertEqual(found(), ["t3code", "pi", "deepseek", "copilot"])

    def test_a_deselected_agent_keeps_its_entry(self):
        self.patch_file.write_text(DSH_PATCH, encoding="utf-8")
        self.register()
        files = [self.opencode, self.pi, self.patch_file, self.vscode / "chatLanguageModels.json"]
        written = [path.read_bytes() for path in files]
        result = self.register(agents=[], context=98304)
        self.assertEqual({entry["status"] for entry in result.values()}, {"off"})
        self.assertEqual([path.read_bytes() for path in files], written)  # nothing removed, nothing changed
        self.assertEqual(list(result), list(clients.AGENT_IDS))

    @unittest.skipUnless(yaml, "PyYAML is not installed")
    def test_main_session_first_adds_a_subagent_model_that_then_stays(self):
        sub = "qwen38-gittensor-dflash-subagent"

        def contexts():
            opencode = json.loads(self.opencode.read_text(encoding="utf-8"))["provider"]["vllm"]["models"]
            pi = json.loads(self.pi.read_text(encoding="utf-8"))["providers"]["vllm"]["models"]
            copilot = json.loads((self.vscode / "chatLanguageModels.json").read_text(encoding="utf-8"))[0]["models"]
            deepseek = self.provider()["qwen-station"]["models"]
            return {
                "opencode": {key: entry["limit"]["context"] for key, entry in opencode.items()},
                "pi": {entry["id"]: entry["contextWindow"] for entry in pi},
                "copilot": {entry["id"]: entry["contextWindow"] for entry in copilot},
                "deepseek": {entry["id"]: entry["contextWindow"] for entry in deepseek},
            }

        # Off and never on: no second model in any agent.
        self.register()
        for agent, models in contexts().items():
            self.assertEqual(models, {clients.MODEL_ID: 196608}, agent)
        # On: the subagent model with its own context, the main model as it was.
        result = self.register(subagent=98304)
        self.assertTrue(all(result[agent]["status"] == "ok" for agent in ("opencode", "pi", "deepseek", "copilot")))
        for agent, models in contexts().items():
            self.assertEqual(models, {clients.MODEL_ID: 196608, sub: 98304}, agent)
        pi = json.loads(self.pi.read_text(encoding="utf-8"))["providers"]["vllm"]["models"]
        self.assertEqual(pi[1]["name"], "Qwen3.8 27B subagent - FP8 KV - DFlash2 - 98K (RTX 5090)")
        self.assertEqual(pi[1]["thinkingLevelMap"], pi[0]["thinkingLevelMap"])  # everything else as the main model
        # Off again: the entry stays, since a subagent setting may name it, and is the same model again.
        self.register()
        for agent, models in contexts().items():
            self.assertEqual(models, {clients.MODEL_ID: 196608, sub: 196608}, agent)
        self.assertEqual(self.register(dry_run=True)["opencode"]["status"], "ok")  # and that is stable

    def test_the_default_paths_are_not_the_real_home(self):
        # tests/isolation.py moved every home folder; a forgotten path in a test lands there.
        from tests import isolation
        for path in (clients.opencode_path(), clients.pi_path(), clients.deepseek_path(), clients.t3_home()):
            self.assertTrue(str(path).startswith(isolation.HOME), path)


class PathTests(unittest.TestCase):
    def test_windows_to_wsl_path(self):
        self.assertEqual(windows_to_wsl_path(r"D:\qwen-engine"), "/mnt/d/qwen-engine")
        self.assertEqual(windows_to_wsl_path("C:/x/"), "/mnt/c/x")


if __name__ == "__main__":
    unittest.main()
