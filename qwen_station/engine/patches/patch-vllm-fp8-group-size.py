#!/usr/bin/env python3
"""Opt-in, reversible patch for the Qwen38 FP8 KV-cache group size."""

import argparse
import os
import difflib
import shutil
import sys
from pathlib import Path


_CORE = Path(os.environ.get("VLLM_PREFIX", "/opt/vllm")) / "lib/python3.12/site-packages/vllm/v1/core"
# vLLM 0.33.0 moved the grouping code from kv_cache_utils.py to kv_cache_planning.py; the patched lines are the same.
DEFAULT_SOURCE = _CORE / ("kv_cache_planning.py" if (_CORE / "kv_cache_planning.py").is_file() else "kv_cache_utils.py")
BACKUP_SUFFIX = ".pre-fp8-group8.bak"
MARKER = "# codex-vllm-qwen38-kv-group-size-v1"
ANCHOR = "    group_size = min_num_layers\n"
INJECTION = """    # codex-vllm-qwen38-kv-group-size-v1
    _qwen38_group_size = os.environ.get("VLLM_QWEN38_KV_GROUP_SIZE")
    if _qwen38_group_size is not None:
        if _qwen38_group_size != "8":
            raise ValueError("VLLM_QWEN38_KV_GROUP_SIZE must be exactly 8")
        _qwen38_layer_counts = sorted(len(layers) for layers in layer_buckets)
        if _qwen38_layer_counts != [5, 16, 48]:
            raise ValueError(f"VLLM_QWEN38_KV_GROUP_SIZE=8 requires [5, 16, 48], got {_qwen38_layer_counts}")
        logger.info("VLLM_QWEN38_KV_GROUP_SIZE=8 enabled for layer buckets %s", _qwen38_layer_counts)
        group_size = 8
"""


def _backup_path(source: Path) -> Path:
    return source.with_name(source.name + BACKUP_SUFFIX)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc


def _with_os_import(source: str) -> str:
    """kv_cache_planning.py (vLLM 0.33.0) does not import os, which the injected lines need."""
    if "\nimport os\n" in "\n" + source:
        return source
    if "\nimport math\n" not in source:
        raise RuntimeError("no place found for the os import")
    return source.replace("\nimport math\n", "\nimport math\nimport os\n", 1)


def _candidate(source: str) -> tuple[str, bool]:
    if MARKER in source:
        if INJECTION not in source:
            raise RuntimeError("existing patch marker has unexpected contents")
        fixed = _with_os_import(source)
        return fixed, fixed != source
    if source.count(ANCHOR) != 1:
        raise RuntimeError("expected exactly one group_size = min_num_layers anchor")
    if "layer_buckets: list[list[str]] = []" not in source or "for layers in layer_buckets:" not in source:
        raise RuntimeError("layer_buckets source check failed")
    return _with_os_import(source.replace(ANCHOR, ANCHOR + INJECTION, 1)), True


def _compile(source: str, path: Path) -> None:
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        raise RuntimeError(f"generated source does not compile: {exc}") from exc


def apply(source_path: Path, dry_run: bool = False) -> int:
    original = _read(source_path)
    candidate, changed = _candidate(original)
    if not changed:
        print(f"already patched: {source_path}")
        return 0
    _compile(candidate, source_path)
    if dry_run:
        print("".join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True), str(source_path), str(source_path))))
        return 0
    backup = _backup_path(source_path)
    if not backup.exists():
        try:
            shutil.copy2(source_path, backup)
        except OSError as exc:
            raise RuntimeError(f"cannot create one-time backup {backup}: {exc}") from exc
    try:
        source_path.write_text(candidate, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot write {source_path}: {exc}") from exc
    print(f"patched: {source_path}")
    print(f"backup: {backup}")
    return 0


def check(source_path: Path) -> int:
    source = _read(source_path)
    if MARKER not in source:
        print(f"unpatched: {source_path}")
        return 0
    if INJECTION not in source:
        raise RuntimeError("existing patch marker has unexpected contents")
    _compile(source, source_path)
    print(f"patched and compilable: {source_path}")
    return 0


def restore(source_path: Path) -> int:
    backup = _backup_path(source_path)
    current = _read(source_path)
    saved = _read(backup)
    if MARKER not in current:
        raise RuntimeError("source has no patch marker; refusing restore")
    candidate, changed = _candidate(saved)
    if not changed or candidate != current:
        raise RuntimeError("source or backup does not match this patch; refusing restore")
    _compile(saved, backup)
    try:
        source_path.write_text(saved, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot write {source_path}: {exc}") from exc
    print(f"restored: {source_path}")
    print(f"backup retained: {backup}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--restore", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.check:
            return check(args.source)
        if args.restore:
            return restore(args.source)
        return apply(args.source, dry_run=args.dry_run)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
