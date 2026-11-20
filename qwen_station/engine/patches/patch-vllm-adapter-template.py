"""Local change for vLLM 0.32: the OpenCode adapter gets its own chat template, and thinking off.

Qwen Station's setting "OpenCode adapter" serves the LoRA adapter harness-hub/Qwen3.8-27B-opencode-adapted next to
the model, under a model name of its own (serve.sh). vLLM renders every request with the one chat template of the
server. The target's template (qwen3.8-gittensor-safe-v2) words the tool instructions differently from the
upstream Qwen template the adapter was trained with: it forbids text before a tool call, and it leaves out the empty
thinking block in earlier turns. Rendered side by side for an OpenCode conversation, the two prompts differ in the
system part and in every assistant turn.

With this change a chat request for the adapter's model name

1. is rendered with the adapter's template (the file it ships), unless the request brings a template of its own;
2. runs without thinking, unless the request asks for it (chat_template_kwargs.enable_thinking or
   reasoning_effort). The adapter was trained and evaluated with thinking off; with it on, the model card measured
   no significant gain at about six times the tokens.

Environment, set by serve.sh: VLLM_QWEN38_ADAPTER_MODEL (the model name), VLLM_QWEN38_ADAPTER_TEMPLATE (the file).
Without them every request is rendered as before.

Idempotent. Backup: protocol.py.pre-adapter-template.bak
"""
import glob, os, pathlib, shutil, sys

ROOT = glob.glob(os.environ.get('VLLM_PREFIX', '/opt/vllm') + '/lib/python3*/site-packages/vllm')
if not ROOT:
    print('PATCH_NOT_APPLICABLE: no vLLM found'); sys.exit(0)
path = os.path.join(ROOT[0], 'entrypoints/openai/chat_completion/protocol.py')
if not os.path.exists(path):
    print('PATCH_NOT_APPLICABLE: missing', path); sys.exit(0)
MARK = 'qwen-station: OpenCode adapter (local change)'

THINKING = (  # after the anchor
    """        if self.reasoning_effort is not None and "enable_thinking" not in user_kwargs:
            extra_kwargs["enable_thinking"] = self.reasoning_effort != "none"
""",
    """        # --- qwen-station: OpenCode adapter (local change) ---
        if (self.reasoning_effort is None and "enable_thinking" not in user_kwargs
                and _qs_adapter(self.model)):
            extra_kwargs["enable_thinking"] = False
""")
TEMPLATE = (  # replaces the anchor
    """            chat_template=self.chat_template or default_template,
""",
    """            chat_template=self.chat_template or _qs_adapter_template(self.model) or default_template,
""")
HELPER = """

# --- qwen-station: OpenCode adapter (local change) ---
_qs_template_cache = {}


def _qs_adapter(model):
    \"\"\"True for requests to the adapter's model name.\"\"\"
    import os

    name = os.environ.get("VLLM_QWEN38_ADAPTER_MODEL")
    return bool(name) and model == name


def _qs_adapter_template(model):
    \"\"\"The adapter's own chat template for requests to its model name, else None.\"\"\"
    import os

    if not _qs_adapter(model):
        return None
    file = os.environ.get("VLLM_QWEN38_ADAPTER_TEMPLATE", "")
    if file not in _qs_template_cache:
        try:
            with open(file, encoding="utf-8") as handle:
                _qs_template_cache[file] = handle.read()
        except OSError:
            _qs_template_cache[file] = None  # no file: the server's template, as without this change
    return _qs_template_cache[file]
"""

src = open(path).read()
if MARK in src:
    print('PATCH_ALREADY_APPLIED', path); sys.exit(0)
for anchor, _ in (THINKING, TEMPLATE):
    if src.count(anchor) != 1:
        # A newer vLLM restructured this code: the adapter then runs with the server's template and its defaults.
        print('PATCH_NOT_APPLICABLE: anchor not found in', path, repr(anchor.strip()[:60])); sys.exit(0)
if not os.path.exists(path + '.pre-adapter-template.bak'):
    shutil.copy2(path, path + '.pre-adapter-template.bak')
src = src.replace(THINKING[0], THINKING[0] + THINKING[1]).replace(TEMPLATE[0], TEMPLATE[1]) + HELPER
open(path, 'w').write(src)
for pyc in pathlib.Path(path).parent.glob('__pycache__/protocol.*.pyc'):
    pyc.unlink()
print('PATCH_APPLIED', path)
