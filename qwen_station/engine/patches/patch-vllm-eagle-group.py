"""Local fix for vLLM 0.29 to 0.32 (vllm/v1/core/kv_cache_utils.py, _annotate_eagle_groups).

With speculative decoding of the Eagle family (DFlash included) the KV cache group of the drafter's layers
has to be flagged as "eagle group". vLLM only detects it through the MLA flag
non_causal_multi_token_decode (Kimi-K3) or a special case for DeepseekV4. The DFlash2 drafter (plain attention
with a sliding window) is not detected. The coordinator then flags ALL groups as drafter groups, Mamba groups
ask for a window of two chunks that the align mode never provides, and the prefix cache hits nothing, without
any warning. On 0.32 the KV cache also needs 2.2 GiB more.

Fix: for method "dflash", flag the groups whose layers are all sliding-window attention (the target model has
no such layers).

Idempotent. Backup: kv_cache_utils.py.pre-eagle-group.bak
"""
import glob, os, pathlib, shutil, sys
paths = glob.glob(os.environ.get('VLLM_PREFIX', '/opt/vllm') + '/lib/python3*/site-packages/vllm/v1/core/kv_cache_utils.py')
if not paths:
    sys.exit('kv_cache_utils.py not found')
path = paths[0]
src = open(path).read()
MARK = 'dflash: flag sliding-window draft groups (local fix)'
if MARK in src:
    print('PATCH_ALREADY_APPLIED', path); sys.exit(0)
ANCHOR = """    if not use_deepseek_v4_fallback:
        return
    last_layer = next(reversed(kv_cache_spec))
"""
# vLLM 0.32.0 restructured the function (rule 2 is now use_trailing_layer_fallback), the bug stayed:
# without this fix 0.32.0 reports "no KV cache group could be identified as the draft model's" and needs
# 11.84 instead of 9.61 GiB of KV cache for 262,144 tokens (checked 2026-11-16).
ANCHOR_032 = """    if not use_trailing_layer_fallback:
        return
    if not _groups_partition_layers_exactly(kv_cache_spec, kv_cache_groups):
"""
if src.count(ANCHOR) != 1 and src.count(ANCHOR_032) == 1:
    ANCHOR = ANCHOR_032
PATCH = """    # --- dflash: flag sliding-window draft groups (local fix) ---
    # DFlash2 drafters use plain sliding-window Attention (no MLA marker), so
    # rule 1 never fires and the coordinator flags every group, which kills
    # prefix-cache reuse for hybrid (Mamba) targets. The Qwen3.5 target has no
    # sliding-window layers, so a group made only of SlidingWindowSpec layers
    # is the drafter's.
    if getattr(spec_config, "method", None) == "dflash" and not any(
        g.is_eagle_group for g in kv_cache_groups
    ):
        for group in kv_cache_groups:
            specs = list(iter_layer_specs(group.kv_cache_spec))
            if specs and all(isinstance(s, SlidingWindowSpec) for s in specs):
                group.is_eagle_group = True
                logger.info(
                    "Flagging sliding-window KV cache group %s as DFlash draft "
                    "group (local fix).",
                    sorted(group.layer_names),
                )
"""
if 'def _annotate_eagle_groups' not in src:
    # vLLM 0.33.0 moved the KV planning to kv_cache_planning.py and restructured it; this fix is not
    # ported there. The server reports it itself when it cannot identify the drafter's group.
    print('PATCH_NOT_APPLICABLE: _annotate_eagle_groups is missing in', path); sys.exit(0)
if src.count(ANCHOR) != 1:
    sys.exit('PATCH_ANCHOR_NOT_FOUND: the installed vLLM version does not fit (' + path + ')')
if 'SlidingWindowSpec' not in src.split('def _annotate_eagle_groups')[0]:
    sys.exit('PATCH_IMPORT_MISSING: kv_cache_utils.py does not import SlidingWindowSpec')
if not os.path.exists(path + '.pre-eagle-group.bak'):
    shutil.copy2(path, path + '.pre-eagle-group.bak')
open(path, 'w').write(src.replace(ANCHOR, PATCH + ANCHOR))
for pyc in pathlib.Path(path).parent.glob('__pycache__/kv_cache_utils.*.pyc'):
    pyc.unlink()
print('PATCH_APPLIED', path)
