"""Register the local model with the coding agents the user chose.

The agents: T3 Code (it drives OpenCode and Pi), OpenCode, Pi, DeepSeek Harness and Copilot in VS Code.

Every writer only adds: it enters or updates the entry of this model and leaves everything else in the file as
it is. An agent the user deselects is no longer updated; its entry stays. An entry that already matches is left
alone, and the first real change keeps a backup of the previous file next to it.
"""

from __future__ import annotations

import copy
import functools
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from .config import ADAPTER_MODEL_ID, AGENT_IDS, KV_CACHE_TYPES, MODEL_ID, OUTPUT_LIMIT, SUBAGENT_MODEL_ID

PROVIDER_ID = "vllm"  # OpenCode and Pi; T3 Code lists the model as vllm/<model>
PROVIDER_NAME = "vLLM RTX 5090"
GENERATED_PREFIX = "qs-"  # keys the settings menu generates start like this
REASONING_LEVELS = ("low", "medium", "xhigh")  # the only values this Qwen chat template accepts
PLACEHOLDERS = {"EMPTY", "dummy", "not-needed", "lm-studio", "ollama", "local"}
_CREDENTIAL_KEY = re.compile(
    r"(?:x[-_])?api[_-]?key|password|secret|token|(?:auth|access|refresh|id)[_-]?token|"
    r"client[_-]?secret|private[_-]?key|authorization|cookies?|set[-_]cookies?",
    re.I,
)

# The agents, in the order the settings show them.
AGENTS = {
    "t3code": "T3 Code",
    "opencode": "OpenCode",
    "pi": "Pi",
    "deepseek": "DeepSeek Harness",
    "copilot": "Copilot (VS Code)",
}
assert tuple(AGENTS) == AGENT_IDS

# DeepSeek Harness and Copilot get a provider of their own, so an update never touches one the user made.
OWN_PROVIDER = "qwen-station"
OWN_PROVIDER_NAME = "Qwen Station"
DSH_PLUGIN_ID = "llm-pi-ai"
DSH_PLUGIN_NAME = "@deepseek-ai/dsh-llm-pi-ai"
DSH_KEY_NAME = "QWEN_STATION_API_KEY"  # DeepSeek Harness reads keys by name, from its .env file


def model_name(context: int, max_seqs: int, variant: str = "", kv_cache: str = "fp8", role: str = "") -> str:
    model = " ".join(part for part in ("Qwen3.8 27B", variant, role) if part)
    slots = f" - {max_seqs} slots" if max_seqs > 1 else ""
    return f"{model} - {KV_CACHE_TYPES.get(kv_cache, kv_cache)} KV - DFlash2 - {context // 1000}K (RTX 5090){slots}"


def subagent_entry(subagent: int, context: int, present: bool) -> int:
    """The context to enter for the subagent model, 0 for none. With "main session first" it has its own; once
    entered the entry stays, since a subagent setting of the user may name it, and while the setting is off the
    name is the same model with the full context."""
    return subagent or (context if present else 0)


def _home() -> Path:
    return Path(os.environ.get("USERPROFILE") or str(Path.home()))


def opencode_path() -> Path:
    return _home() / ".config" / "opencode" / "opencode.json"


def pi_path() -> Path:
    agent_dir = os.environ.get("PI_CODING_AGENT_DIR")
    if agent_dir:
        return Path(agent_dir) / "models.json"
    return _home() / ".pi" / "agent" / "models.json"


def t3_home() -> Path:
    return _home() / ".t3"


def deepseek_home() -> Path:
    return Path(os.environ["DSH_HOME"]) if os.environ.get("DSH_HOME") else _home() / ".dsh"


def deepseek_path(home: Path | None = None) -> Path:
    """The user's own patch layer of the desktop profile: the one place DeepSeek Harness keeps providers."""
    return (home or deepseek_home()) / "profiles" / "desktop" / "cordis.patch.yml"


def copilot_dirs() -> list[Path]:
    """The user folders of VS Code (stable and Insiders) that exist on this machine."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or _home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = _home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or _home() / ".config")
    return [base / name / "User" for name in ("Code", "Code - Insiders") if (base / name / "User").is_dir()]


def has_plaintext_credential(value) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if _CREDENTIAL_KEY.fullmatch(key) and isinstance(item, str) and item \
                    and not item.startswith(("{env:", "{file:")) and item not in PLACEHOLDERS:
                return True
            if has_plaintext_credential(item):
                return True
    if isinstance(value, list):
        return any(has_plaintext_credential(item) for item in value)
    return False


def _load(path: Path, root: type = dict):
    if not path.exists():
        return root(), ""
    text = path.read_text(encoding="utf-8-sig")
    value = json.loads(text) if text.strip() else root()
    if not isinstance(value, root):
        raise ValueError(f"{path.name}: The root must be a JSON {'object' if root is dict else 'array'}")
    return value, text


def _child(parent: dict, key: str) -> dict:
    value = parent.get(key)
    if value is None:
        value = {}
        parent[key] = value
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be a JSON object")
    return value


def _write_text(path: Path, original_text: str, rendered: str) -> str | None:
    """Atomic write with readback. Returns the backup path if one was made."""
    backup = None
    if original_text:
        backup_path = path.with_name(f"{path.name}.bak-station-{datetime.now():%Y%m%d}")
        if not backup_path.exists():
            backup_path.write_text(original_text, encoding="utf-8", newline="")
            backup = str(backup_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(rendered)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    if path.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"{path.name}: The readback does not match what was written")
    return backup


def _write(path: Path, original_text: str, updated, trailing_newline: bool) -> str | None:
    rendered = json.dumps(updated, ensure_ascii=False, indent=2) + ("\n" if trailing_newline else "")
    return _write_text(path, original_text, rendered)


def key_reference(key_file: Path | None) -> str:
    """How OpenCode's config names the key: by the file that holds it, so the config stays free of secrets."""
    return f"{{file:{key_file.as_posix()}}}" if key_file else ""


# ---- OpenCode and Pi ------------------------------------------------------------------------------------------

def apply_opencode(config: dict, base_url: str, context: int, max_seqs: int, variant: str = "",
                   api_key: str = "", key_file: Path | None = None, vision: bool = False, kv_cache: str = "fp8",
                   subagent: int = 0, adapter: bool = False) -> dict:
    """Same entry shape as register-qwen-opencode.py, so both starters agree.
    adapter: the OpenCode adapter is on; it gets an entry of its own. Switched off, the entry stays, like any other."""
    updated = copy.deepcopy(config)
    provider = _child(_child(updated, "provider"), PROVIDER_ID)
    provider["npm"] = "@ai-sdk/openai-compatible"
    provider["name"] = PROVIDER_NAME
    options = _child(provider, "options")
    options["baseURL"] = base_url
    if api_key and key_file:
        options["apiKey"] = key_reference(key_file)
    elif options.get("apiKey") == key_reference(key_file):
        del options["apiKey"]  # the key was removed in the settings; a key the user wrote here stays
    models = _child(provider, "models")
    sub = subagent_entry(subagent, context, SUBAGENT_MODEL_ID in models)
    _opencode_model(_child(models, MODEL_ID), model_name(context, max_seqs, variant, kv_cache), context, vision)
    if sub:
        _opencode_model(_child(models, SUBAGENT_MODEL_ID),
                        model_name(sub, max_seqs, variant, kv_cache, "subagent"), sub, vision)
    if adapter:
        _opencode_adapter_model(_child(models, ADAPTER_MODEL_ID),
                                model_name(context, max_seqs, variant, kv_cache, "OpenCode adapter"), context)
    return updated


def _opencode_adapter_model(entry: dict, name: str, context: int) -> None:
    """The adapter was trained and evaluated with thinking off, and the server runs it that way: no reasoning
    levels to choose from. Text only, like its training (vLLM does not put LoRA on the image encoder here)."""
    entry["name"] = name
    entry["reasoning"] = False
    limit = _child(entry, "limit")
    limit["context"] = context
    if not isinstance(limit.get("output"), int):
        limit["output"] = OUTPUT_LIMIT


def _opencode_model(entry: dict, name: str, context: int, vision: bool) -> None:
    entry["name"] = name
    entry["reasoning"] = True
    # Images: OpenCode only offers attachments for a model that says it takes them.
    if vision or "attachment" in entry:
        entry["attachment"] = vision
    if vision or "modalities" in entry:
        entry["modalities"] = {"input": ["text", "image"] if vision else ["text"], "output": ["text"]}
    _child(entry, "options").setdefault("reasoningEffort", "medium")
    variants = _child(entry, "variants")
    for level in REASONING_LEVELS:
        variant = _child(variants, level)
        variant["reasoningEffort"] = level
        variant.pop("disabled", None)
    _child(variants, "high")["disabled"] = True  # OpenCode's generic "high" is not a valid value here
    limit = _child(entry, "limit")
    limit["context"] = context
    if not isinstance(limit.get("output"), int):
        limit["output"] = OUTPUT_LIMIT


def apply_pi(config: dict, base_url: str, context: int, max_seqs: int, variant: str = "",
             api_key: str = "", key_file: Path | None = None, vision: bool = False, kv_cache: str = "fp8",
             subagent: int = 0) -> dict:
    updated = copy.deepcopy(config)
    updated.setdefault("$schema", "https://pi.dev/schemas/models.schema.json")
    provider = _child(_child(updated, "providers"), PROVIDER_ID)
    provider["name"] = PROVIDER_NAME
    provider["baseUrl"] = base_url
    provider["api"] = "openai-completions"
    if api_key:
        provider["apiKey"] = api_key
    elif str(provider.get("apiKey", "")).startswith(GENERATED_PREFIX):
        provider["apiKey"] = "local"  # a key generated here and removed again
    provider.setdefault("apiKey", "local")  # without a key in vLLM, Pi still needs any non-empty one
    compat = _child(provider, "compat")
    compat["supportsReasoningEffort"] = True
    compat["thinkingFormat"] = "openai"
    compat["supportsDeveloperRole"] = False
    compat["maxTokensField"] = "max_tokens"
    compat["supportsStore"] = False  # not a vLLM field; keep it out of the request
    models = provider.get("models")
    if models is None:
        models = []
        provider["models"] = models
    if not isinstance(models, list):
        raise ValueError("'models' must be a list")
    sub = subagent_entry(subagent, context, _listed(models, SUBAGENT_MODEL_ID) is not None)
    _pi_model(_entry(models, MODEL_ID), model_name(context, max_seqs, variant, kv_cache), context, vision)
    if sub:
        _pi_model(_entry(models, SUBAGENT_MODEL_ID), model_name(sub, max_seqs, variant, kv_cache, "subagent"),
                  sub, vision)
    return updated


def _listed(models: list, model_id: str) -> dict | None:
    return next((item for item in models if isinstance(item, dict) and item.get("id") == model_id), None)


def _entry(models: list, model_id: str) -> dict:
    """The entry of a model in a list of models, added at the end when it is missing."""
    entry = _listed(models, model_id)
    if entry is None:
        entry = {"id": model_id}
        models.append(entry)
    return entry


def _pi_model(entry: dict, name: str, context: int, vision: bool) -> None:
    entry["name"] = name
    entry["reasoning"] = True
    # Pi levels -> reasoning_effort. null marks a level this checkpoint cannot do; Pi then clamps.
    entry["thinkingLevelMap"] = {"off": None, "minimal": None, "low": "low", "medium": "medium",
                                 "high": None, "xhigh": "xhigh", "max": None}
    entry["input"] = ["text", "image"] if vision else ["text"]
    entry["cost"] = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    entry["contextWindow"] = context
    entry["maxTokens"] = OUTPUT_LIMIT


# ---- Copilot in VS Code ---------------------------------------------------------------------------------------

def apply_copilot(groups: list, base_url: str, context: int, max_seqs: int, variant: str = "",
                  api_key: str = "", key_file: Path | None = None, vision: bool = False, kv_cache: str = "fp8",
                  subagent: int = 0) -> list:
    """VS Code's chatLanguageModels.json: a list of provider groups. Ours is a "Custom Endpoint" group."""
    updated = copy.deepcopy(groups)
    group = next((item for item in updated if isinstance(item, dict) and item.get("name") == OWN_PROVIDER_NAME
                  and item.get("vendor") == "customendpoint"), None)
    if group is None:
        group = {"name": OWN_PROVIDER_NAME, "vendor": "customendpoint"}
        updated.append(group)
    group["apiType"] = "chat-completions"
    models = group.get("models")
    if not isinstance(models, list):
        models = []
        group["models"] = models
    sub = subagent_entry(subagent, context, _listed(models, SUBAGENT_MODEL_ID) is not None)
    _copilot_model(_entry(models, MODEL_ID), model_name(context, max_seqs, variant, kv_cache), context, vision,
                   base_url)
    if sub:
        _copilot_model(_entry(models, SUBAGENT_MODEL_ID), model_name(sub, max_seqs, variant, kv_cache, "subagent"),
                       sub, vision, base_url)
    return updated


def _copilot_model(entry: dict, name: str, context: int, vision: bool, base_url: str) -> None:
    entry["name"] = name
    entry["url"] = f"{base_url}/chat/completions"  # an explicit path: VS Code then sends exactly there
    entry["toolCalling"] = True
    entry["vision"] = vision
    entry["contextWindow"] = context
    entry["maxOutputTokens"] = OUTPUT_LIMIT
    entry["thinking"] = True
    entry["supportsReasoningEffort"] = list(REASONING_LEVELS)
    entry["reasoningEffortFormat"] = "chat-completions"


# ---- DeepSeek Harness -----------------------------------------------------------------------------------------
# Its providers live in a YAML file that the harness and its users both write. Qwen Station has no YAML library
# (standard library only), so it edits the text: it finds its own provider block under the pi-ai plugin entry
# and replaces or adds only that block. Every other line, comments included, stays as it was.

def deepseek_provider(base_url: str, context: int, max_seqs: int, variant: str = "", api_key: str = "",
                      vision: bool = False, kv_cache: str = "fp8", subagent: int = 0) -> dict:
    # DeepSeek Harness refuses a route without a key, also when the server needs none; the .env then holds "local".
    provider: dict = {"displayName": PROVIDER_NAME, "api": "openai-completions", "baseURL": base_url,
                      "apiKeyEnv": DSH_KEY_NAME}
    provider["compat"] = {"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": True,
                          "maxTokensField": "max_tokens", "thinkingFormat": "openai"}
    provider["models"] = [{"id": model_id, "name": model_name(size, max_seqs, variant, kv_cache, role),
                           "contextWindow": size, "maxTokens": OUTPUT_LIMIT,
                           "input": ["text", "image"] if vision else ["text"],
                           "reasoningEfforts": {level: level for level in REASONING_LEVELS}}
                          for model_id, size, role in ((MODEL_ID, context, ""), (SUBAGENT_MODEL_ID, subagent, "subagent"))
                          if size]
    return provider


def _yaml_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict) and not value:
        return "{}"
    if isinstance(value, list) and not value:
        return "[]"
    text = str(value)
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", text) and text.lower() not in ("true", "false", "null", "yes", "no",
                                                                             "on", "off", "y", "n", "~"):
        return text
    return json.dumps(text, ensure_ascii=False)  # a JSON string is a valid double-quoted YAML scalar


def yaml_lines(value, indent: int) -> list[str]:
    """Block-style YAML for plain data (dicts, lists, scalars)."""
    pad = " " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{pad}{key}:")
                lines += yaml_lines(item, indent + 2)
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(item)}")
    else:
        for item in value:
            if isinstance(item, dict) and item:
                nested = yaml_lines(item, indent + 2)
                nested[0] = f"{pad}- {nested[0].lstrip()}"
                lines += nested
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
    return lines


_YAML_KEY = re.compile(r"^(?P<lead>\s*(?:-\s+)?)(?P<key>[^\s:#'\"][^:#]*?|\"[^\"]*\"|'[^']*')\s*:(?:\s+(?P<rest>.*))?$")


def _meaningful(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _key_at(line: str) -> tuple[int, str, str] | None:
    """(indent of the key, key, rest of the line without a comment) for a 'key: value' line."""
    match = _YAML_KEY.match(line)
    if not match:
        return None
    rest = (match.group("rest") or "").strip()
    if rest.startswith("#"):
        rest = ""
    return len(match.group("lead")), match.group("key").strip("'\""), rest


def _block_end(lines: list[str], start: int, indent: int, stop: int) -> int:
    """The index after the last meaningful line that belongs below lines[start] (indented deeper than indent)."""
    end = start + 1
    for index in range(start + 1, stop):
        if not _meaningful(lines[index]):
            continue
        if _indent(lines[index]) <= indent:
            break
        end = index + 1
    return end


def _find_key(lines: list[str], start: int, stop: int, indent: int, key: str) -> int | None:
    for index in range(start, stop):
        if _meaningful(lines[index]):
            found = _key_at(lines[index])
            if found and found[0] == indent and found[1] == key:
                return index
    return None


def _child_indent(lines: list[str], line: int, end: int, parent_indent: int) -> int:
    for index in range(line + 1, end):
        if _meaningful(lines[index]):
            return _indent(lines[index])
    return parent_indent + 2


def _open_block(lines: list[str], index: int, indent: int, key: str) -> None:
    """Turn 'key: {}' into 'key:' so a block can go below it. Any other inline value is not ours to change."""
    rest = _key_at(lines[index])[2]
    if rest in ("{}", "", "null", "~"):
        lines[index] = " " * indent + f"{key}:"
        return
    raise ValueError(f"'{key}' is written inline; Qwen Station only edits block style. Add the provider by hand.")


def apply_deepseek(text: str, provider: dict) -> str:
    """The patch file with our provider block entered under the pi-ai plugin's providers. Only adds or updates."""
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    if [line.strip() for line in lines if _meaningful(line)] == ["[]"]:
        lines = [line for line in lines if line.strip() != "[]"]  # an empty list written inline
    items = [index for index, line in enumerate(lines) if re.match(r"-(\s|$)", line)]
    if any(_meaningful(line) and _indent(line) == 0 and not line.startswith("-") for line in lines):
        raise ValueError("The patch file is not a list of plugin entries; it is left as it is")
    entry = None
    for position, start in enumerate(items):
        stop = items[position + 1] if position + 1 < len(items) else len(lines)
        names = {}
        for index in range(start, stop):
            found = _key_at(lines[index]) if _meaningful(lines[index]) else None
            if found and found[0] == 2 and found[1] in ("id", "name"):
                names[found[1]] = found[2].strip("'\"")
        if names.get("id") == DSH_PLUGIN_ID or ("id" not in names and names.get("name") == DSH_PLUGIN_NAME):
            entry = (start, stop)
    block = yaml_lines({OWN_PROVIDER: provider}, 0)

    def placed(indent: int) -> list[str]:
        return [" " * indent + line for line in block]

    if entry is None:
        while lines and not lines[-1].strip():
            lines.pop()
        lines += [f"- id: {DSH_PLUGIN_ID}", f"  name: {_yaml_scalar(DSH_PLUGIN_NAME)}", "  config:", "    providers:",
                  *placed(6)]
        return newline.join(lines) + newline
    start, stop = entry
    config = _find_key(lines, start, stop, 2, "config")
    if config is None:
        end = _block_end(lines, start, 0, stop)
        lines[end:end] = ["  config:", "    providers:", *placed(6)]
        return newline.join(lines) + newline
    _open_block(lines, config, 2, "config")
    config_end = _block_end(lines, config, 2, stop)
    inner = _child_indent(lines, config, config_end, 2)
    providers = _find_key(lines, config + 1, config_end, inner, "providers")
    if providers is None:
        lines[config_end:config_end] = [" " * inner + "providers:", *placed(inner + 2)]
        return newline.join(lines) + newline
    _open_block(lines, providers, inner, "providers")
    providers_end = _block_end(lines, providers, inner, stop)
    child = _child_indent(lines, providers, providers_end, inner)
    own = _find_key(lines, providers + 1, providers_end, child, OWN_PROVIDER)
    if own is None:
        lines[providers_end:providers_end] = placed(child)
    else:
        lines[own:_block_end(lines, own, child, stop)] = placed(child)
    return newline.join(lines) + newline


def apply_dotenv(text: str, name: str, value: str) -> str:
    """The .env text with one variable set, or removed when value is empty. Other lines stay as they are."""
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = [line for line in text.splitlines() if not re.match(rf"\s*(export\s+)?{re.escape(name)}\s*=", line)]
    if value:
        lines.append(f"{name}={value}")
    return newline.join(lines) + newline if lines else ""


# ---- registration ---------------------------------------------------------------------------------------------

def _status(name: str, path: Path, model: str) -> dict:
    return {"client": name, "path": str(path), "model": model}


def _register(name: str, path: Path, apply, base_url: str, context: int, max_seqs: int,
              dry_run: bool, must_exist: bool, guard_credentials: bool, variant: str = "",
              api_key: str = "", key_file: Path | None = None, vision: bool = False, root: type = dict,
              model: str = f"{PROVIDER_ID}/{MODEL_ID}", kv_cache: str = "fp8", subagent: int = 0) -> dict:
    status = _status(name, path, model)
    try:
        if must_exist and not path.parent.is_dir():
            return {**status, "status": "missing", "detail": f"{name} is not set up here"}
        original, text = _load(path, root)
        updated = apply(original, base_url, context, max_seqs, variant, api_key, key_file, vision, kv_cache,
                        subagent)
        if updated == original:
            return {**status, "status": "ok", "detail": "registered"}
        if dry_run:
            return {**status, "status": "outdated", "detail": "The entry is missing or differs"}
        if guard_credentials and has_plaintext_credential(original):
            return {**status, "status": "error",
                    "detail": "The config holds a plaintext credential, so it is not rewritten"}
        backup = _write(path, text, updated, trailing_newline=text.endswith("\n") or not text)
        detail = "registered" + (f", backup: {Path(backup).name}" if backup else "")
        return {**status, "status": "ok", "detail": detail, "changed": True}
    except (OSError, ValueError, RuntimeError) as exc:
        return {**status, "status": "error", "detail": str(exc)}


def _register_deepseek(home: Path, base_url: str, context: int, max_seqs: int, dry_run: bool, variant: str,
                       api_key: str, vision: bool, kv_cache: str = "fp8", subagent: int = 0) -> dict:
    path = deepseek_path(home)
    status = _status(AGENTS["deepseek"], path, f"{OWN_PROVIDER}/{MODEL_ID}")
    try:
        if not path.parent.is_dir():
            return {**status, "status": "missing", "detail": f"{AGENTS['deepseek']} is not set up here"}
        text = path.read_text(encoding="utf-8-sig") if path.exists() else ""
        # Our block is written whole each time; the subagent model stays in it once it was there.
        sub = subagent_entry(subagent, context, f"id: {SUBAGENT_MODEL_ID}" in text)
        updated = apply_deepseek(text, deepseek_provider(base_url, context, max_seqs, variant, api_key, vision,
                                                         kv_cache, sub))
        env_path = home / ".env"
        env_text = env_path.read_text(encoding="utf-8-sig") if env_path.exists() else ""
        env_updated = apply_dotenv(env_text, DSH_KEY_NAME, api_key or "local")
        if updated == text and env_updated == env_text:
            return {**status, "status": "ok", "detail": "registered"}
        if dry_run:
            return {**status, "status": "outdated", "detail": "The entry is missing or differs"}
        backup = _write_text(path, text, updated) if updated != text else None
        if env_updated != env_text:
            # The key by name, where DeepSeek Harness looks it up; only our own line changes.
            env_path.write_text(env_updated, encoding="utf-8", newline="")
            if os.name != "nt":
                env_path.chmod(0o600)
        detail = "registered" + (f", backup: {Path(backup).name}" if backup else "")
        return {**status, "status": "ok", "detail": detail, "changed": True}
    except (OSError, ValueError, RuntimeError) as exc:
        return {**status, "status": "error", "detail": str(exc)}


def _register_copilot(folders: list[Path], base_url: str, context: int, max_seqs: int, dry_run: bool, variant: str,
                      api_key: str, vision: bool, kv_cache: str = "fp8", subagent: int = 0) -> dict:
    name = AGENTS["copilot"]
    if not folders:
        return {**_status(name, Path("chatLanguageModels.json"), f"{OWN_PROVIDER_NAME}/{MODEL_ID}"),
                "status": "missing", "detail": "VS Code is not set up here"}
    results = [_register(name, folder / "chatLanguageModels.json", apply_copilot, base_url, context, max_seqs,
                         dry_run, must_exist=True, guard_credentials=False, variant=variant, vision=vision,
                         root=list, model=f"{OWN_PROVIDER_NAME}/{MODEL_ID}", kv_cache=kv_cache, subagent=subagent)
               for folder in folders]
    worst = next((result for state in ("error", "outdated", "missing") for result in results
                  if result["status"] == state), results[0])
    combined = {**worst, "path": "; ".join(result["path"] for result in results)}
    if any(result.get("changed") for result in results):
        combined["changed"] = True
    if api_key and combined["status"] == "ok":
        combined["detail"] += ". VS Code keeps keys itself: enter the API key once under Chat, Manage Models"
    return combined


def write_key_file(key_file: Path, api_key: str) -> None:
    """Keep the file with the key in step with the setting: only the key, readable by this user only."""
    if not api_key:
        key_file.unlink(missing_ok=True)
        return
    if key_file.is_file() and key_file.read_text(encoding="ascii", errors="replace") == api_key:
        return
    key_file.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "w", encoding="ascii", newline="") as stream:
        stream.write(api_key)


def _not_selected(name: str, path) -> dict:
    return {"client": name, "path": str(path), "model": "", "status": "off",
            "detail": "Not selected. Qwen Station leaves its config as it is."}


def register_all(base_url: str, context: int, max_seqs: int, dry_run: bool = False,
                 opencode_file: Path | None = None, pi_file: Path | None = None, variant: str = "",
                 api_key: str = "", key_file: Path | None = None, vision: bool = False,
                 agents=AGENT_IDS, deepseek_dir: Path | None = None, copilot_folders: list[Path] | None = None,
                 t3_dir: Path | None = None, kv_cache: str = "fp8", subagent: int = 0, adapter: bool = False) -> dict:
    """Enter the model in every chosen agent. Returns one status per agent, in the order of AGENTS.
    subagent: the context of the subagent model with "main session first", else 0.
    adapter: the OpenCode adapter is on. Only OpenCode (and T3 Code through it) gets its entry: it was trained there."""
    chosen = set(agents)
    common = {"variant": variant, "api_key": api_key, "vision": vision, "kv_cache": kv_cache, "subagent": subagent}
    # T3 Code runs OpenCode and Pi; choosing it keeps their entries up to date.
    use_opencode = bool(chosen & {"opencode", "t3code"})
    use_pi = bool(chosen & {"pi", "t3code"})
    if use_opencode and key_file and not dry_run:
        try:
            write_key_file(key_file, api_key)
        except OSError:
            key_file = None  # OpenCode then gets no key; its entry reports the difference at the next check
    opencode_file = opencode_file or opencode_path()
    pi_file = pi_file or pi_path()
    result: dict = {}
    if use_opencode:
        opencode = _register(AGENTS["opencode"], opencode_file, functools.partial(apply_opencode, adapter=adapter),
                             base_url, context, max_seqs, dry_run, must_exist=False, guard_credentials=True,
                             key_file=key_file, **common)
    else:
        opencode = _not_selected(AGENTS["opencode"], opencode_file)
    if use_pi:
        # Pi's models.json legitimately holds "apiKey" entries (dummy keys, "!command"), so no credential guard.
        pi = _register(AGENTS["pi"], pi_file, apply_pi, base_url, context, max_seqs, dry_run,
                       must_exist=True, guard_credentials=False, key_file=key_file, **common)
    else:
        pi = _not_selected(AGENTS["pi"], pi_file)
    t3 = t3_dir or t3_home()
    if "t3code" not in chosen:
        result["t3code"] = _not_selected(AGENTS["t3code"], t3)
    elif not t3.is_dir():
        result["t3code"] = {**_status(AGENTS["t3code"], t3, ""), "status": "missing",
                            "detail": "T3 Code is not set up here"}
    else:
        usable = [entry for entry in (opencode, pi) if entry["status"] in ("ok", "outdated")]
        state = "ok" if any(entry["status"] == "ok" for entry in usable) else "outdated" if usable else "missing"
        through = " and ".join(entry["client"] for entry in usable)
        result["t3code"] = {**_status(AGENTS["t3code"], t3, f"{PROVIDER_ID}/{MODEL_ID}"), "status": state,
                            "detail": f"through {through}" if usable
                            else "needs OpenCode or Pi, and neither is set up here"}
    result["opencode"] = opencode
    result["pi"] = pi
    if "deepseek" in chosen:
        result["deepseek"] = _register_deepseek(deepseek_dir or deepseek_home(), base_url, context, max_seqs, dry_run,
                                                variant, api_key, vision, kv_cache, subagent)
    else:
        result["deepseek"] = _not_selected(AGENTS["deepseek"], deepseek_path(deepseek_dir))
    if "copilot" in chosen:
        folders = copilot_dirs() if copilot_folders is None else copilot_folders
        result["copilot"] = _register_copilot(folders, base_url, context, max_seqs, dry_run, variant, api_key, vision,
                                              kv_cache, subagent)
    else:
        result["copilot"] = _not_selected(AGENTS["copilot"], "chatLanguageModels.json")
    for agent_id, entry in result.items():
        entry["id"] = agent_id
        entry["selected"] = agent_id in chosen
    return result


def entered_agents(opencode_file: Path | None = None, pi_file: Path | None = None, t3_dir: Path | None = None,
                   deepseek_dir: Path | None = None, copilot_folders: list[Path] | None = None) -> list[str]:
    """The agents whose configs already hold the model. Up to 2.0.0 every agent was switched on by default, so these
    are the ones an earlier version entered it in. Reads only; a config it cannot read counts as without the model."""
    def loaded(path: Path, root: type = dict):
        try:
            return _load(path, root)[0]
        except (OSError, ValueError):
            return root()

    def section(parent, key: str) -> dict:
        value = parent.get(key) if isinstance(parent, dict) else None
        return value if isinstance(value, dict) else {}

    found = set()
    if MODEL_ID in section(section(section(loaded(opencode_file or opencode_path()), "provider"), PROVIDER_ID),
                           "models"):
        found.add("opencode")
    pi_models = section(section(loaded(pi_file or pi_path()), "providers"), PROVIDER_ID).get("models")
    if isinstance(pi_models, list) and _listed(pi_models, MODEL_ID) is not None:
        found.add("pi")
    if found and (t3_dir or t3_home()).is_dir():
        found.add("t3code")  # its model is the one of OpenCode or Pi
    try:
        text = deepseek_path(deepseek_dir).read_text(encoding="utf-8-sig")
    except (OSError, ValueError):
        text = ""
    if f"{OWN_PROVIDER}:" in text and f"id: {MODEL_ID}" in text:
        found.add("deepseek")
    folders = copilot_dirs() if copilot_folders is None else copilot_folders
    if any(isinstance(group, dict) and group.get("name") == OWN_PROVIDER_NAME
           for folder in folders for group in loaded(folder / "chatLanguageModels.json", list)):
        found.add("copilot")
    return [agent for agent in AGENT_IDS if agent in found]


def installed_agents() -> dict:
    """Which agents this machine has, for the settings. Cheap: folders and the PATH only."""
    return {
        "t3code": t3_home().is_dir(),
        "opencode": opencode_path().parent.is_dir() or bool(shutil.which("opencode")),
        "pi": pi_path().parent.is_dir(),
        "deepseek": deepseek_path().parent.is_dir(),
        "copilot": bool(copilot_dirs()),
    }
