"""Register the local model with OpenCode and Pi, the two agents T3 Code drives.

Both writers are idempotent: an entry that already matches is left alone, and the
first real change keeps a backup of the previous file next to it.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from .config import MODEL_ID, OUTPUT_LIMIT

PROVIDER_ID = "vllm"
PROVIDER_NAME = "vLLM RTX 5090"
REASONING_LEVELS = ("low", "medium", "xhigh")  # the only values this Qwen chat template accepts
PLACEHOLDERS = {"EMPTY", "dummy", "not-needed", "lm-studio", "ollama", "local"}
_CREDENTIAL_KEY = re.compile(
    r"(?:x[-_])?api[_-]?key|password|secret|token|(?:auth|access|refresh|id)[_-]?token|"
    r"client[_-]?secret|private[_-]?key|authorization|cookies?|set[-_]cookies?",
    re.I,
)


def model_name(context: int, max_seqs: int, variant: str = "") -> str:
    model = f"Qwen3.8 27B {variant}".strip()
    if max_seqs == 2:
        return f"{model} - FP8 KV - DFlash2 - {context // 1024}K (RTX 5090) - 2 Slots (experimental)"
    return f"{model} - FP8 KV - DFlash2 - {context // 1000}K (RTX 5090)"


def opencode_path() -> Path:
    profile = os.environ.get("USERPROFILE") or str(Path.home())
    return Path(profile) / ".config" / "opencode" / "opencode.json"


def pi_path() -> Path:
    agent_dir = os.environ.get("PI_CODING_AGENT_DIR")
    if agent_dir:
        return Path(agent_dir) / "models.json"
    profile = os.environ.get("USERPROFILE") or str(Path.home())
    return Path(profile) / ".pi" / "agent" / "models.json"


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


def _load(path: Path) -> tuple[dict, str]:
    if not path.exists():
        return {}, ""
    text = path.read_text(encoding="utf-8-sig")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: The root must be a JSON object")
    return value, text


def _child(parent: dict, key: str) -> dict:
    value = parent.get(key)
    if value is None:
        value = {}
        parent[key] = value
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be a JSON object")
    return value


def _write(path: Path, original_text: str, updated: dict, trailing_newline: bool) -> str | None:
    """Atomic write with readback. Returns the backup path if one was made."""
    backup = None
    if original_text:
        backup_path = path.with_name(f"{path.name}.bak-station-{datetime.now():%Y%m%d}")
        if not backup_path.exists():
            backup_path.write_text(original_text, encoding="utf-8", newline="")
            backup = str(backup_path)
    rendered = json.dumps(updated, ensure_ascii=False, indent=2) + ("\n" if trailing_newline else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(rendered)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    if json.loads(path.read_text(encoding="utf-8")) != updated:
        raise RuntimeError(f"{path.name}: The readback does not match what was written")
    return backup


def apply_opencode(config: dict, base_url: str, context: int, max_seqs: int, variant: str = "") -> dict:
    """Same entry shape as register-qwen-opencode.py, so both starters agree."""
    updated = copy.deepcopy(config)
    provider = _child(_child(updated, "provider"), PROVIDER_ID)
    provider["npm"] = "@ai-sdk/openai-compatible"
    provider["name"] = PROVIDER_NAME
    _child(provider, "options")["baseURL"] = base_url
    entry = _child(_child(provider, "models"), MODEL_ID)
    entry["name"] = model_name(context, max_seqs, variant)
    entry["reasoning"] = True
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
    return updated


def apply_pi(config: dict, base_url: str, context: int, max_seqs: int, variant: str = "") -> dict:
    updated = copy.deepcopy(config)
    updated.setdefault("$schema", "https://pi.dev/schemas/models.schema.json")
    provider = _child(_child(updated, "providers"), PROVIDER_ID)
    provider["name"] = PROVIDER_NAME
    provider["baseUrl"] = base_url
    provider["api"] = "openai-completions"
    provider.setdefault("apiKey", "local")  # vLLM runs without auth; Pi needs any non-empty key
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
    entry = next((item for item in models if isinstance(item, dict) and item.get("id") == MODEL_ID), None)
    if entry is None:
        entry = {"id": MODEL_ID}
        models.append(entry)
    entry["name"] = model_name(context, max_seqs, variant)
    entry["reasoning"] = True
    # Pi levels -> reasoning_effort. null marks a level this checkpoint cannot do; Pi then clamps.
    entry["thinkingLevelMap"] = {"off": None, "minimal": None, "low": "low", "medium": "medium",
                                 "high": None, "xhigh": "xhigh", "max": None}
    entry["input"] = ["text"]
    entry["cost"] = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    entry["contextWindow"] = context
    entry["maxTokens"] = OUTPUT_LIMIT
    return updated


def _register(name: str, path: Path, apply, base_url: str, context: int, max_seqs: int,
              dry_run: bool, must_exist: bool, guard_credentials: bool, variant: str = "") -> dict:
    status = {"client": name, "path": str(path), "model": f"{PROVIDER_ID}/{MODEL_ID}"}
    try:
        if must_exist and not path.parent.is_dir():
            return {**status, "status": "missing", "detail": f"{name} is not set up here"}
        original, text = _load(path)
        updated = apply(original, base_url, context, max_seqs, variant)
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


def register_all(base_url: str, context: int, max_seqs: int, dry_run: bool = False,
                 opencode_file: Path | None = None, pi_file: Path | None = None, variant: str = "") -> dict:
    return {
        "opencode": _register("OpenCode", opencode_file or opencode_path(), apply_opencode,
                              base_url, context, max_seqs, dry_run, must_exist=False, guard_credentials=True,
                              variant=variant),
        # Pi's models.json legitimately holds "apiKey" entries (dummy keys, "!command"), so no credential guard.
        "pi": _register("Pi", pi_file or pi_path(), apply_pi,
                        base_url, context, max_seqs, dry_run, must_exist=True, guard_credentials=False,
                        variant=variant),
    }
