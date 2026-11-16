"""Local fix for vLLM 0.32.x (vllm/model_executor/layers/attention/attention.py, get_kv_cache_spec).

Sliding-window layers (here: the five layers of the DFlash2 drafter) get the largest kernel block that fits
into the page. FlashInfer offers 16, 32 and 64 on SM120, so 64 is chosen. The primary block size of this
hybrid model is 1648 tokens (aligned to the Mamba side), and 64 does not divide 1648. The pages then cannot be
scaled by an integer ratio: every 64-token block is padded to a full page. Measured 2026-11-16: about 2.3 GiB
more KV memory (11.84 instead of 9.61 GiB for 262,144 tokens) and prefix cache hits only in steps of 6,592
instead of 1,648 tokens.

Fix: for unpadded sliding-window layers take the largest kernel block that divides the primary block size
(here 16). vLLM then scales it up to 1648 by an integer ratio, as 0.29.0 did.

Only for vLLM 0.32.x. From 0.33.0 on upstream uses uniform blocks (primary 1664, kernel block 64 divides it).
Idempotent. Backup: attention.py.pre-swblock-divisor.bak
"""
import glob
import os
import pathlib
import shutil
import sys

prefix = os.environ.get('VLLM_PREFIX', '/opt/vllm')
paths = glob.glob(prefix + '/lib/python3*/site-packages/vllm/model_executor/layers/attention/attention.py')
if not paths:
    sys.exit('attention.py not found')
path = paths[0]
versions = glob.glob(prefix + '/lib/python3*/site-packages/vllm-*.dist-info')
version = pathlib.Path(versions[0]).name.split('-')[1].replace('.dist', '') if versions else ''
if not version.startswith('0.32.'):
    print('PATCH_NOT_NEEDED: vLLM', version or '?', 'in', prefix)
    sys.exit(0)
src = open(path).read()
MARK = 'sliding-window block must divide the primary block (local fix)'
if MARK in src:
    print('PATCH_ALREADY_APPLIED', path)
    sys.exit(0)
ANCHOR = """            sw_block_size = _largest_kernel_block_within(
                self.attn_backend,
                sw_per_token,
                page_budget,
                block_size,
                kv_cache_spec,
            )
"""
PATCH = ANCHOR + """            # --- sliding-window block must divide the primary block (local fix) ---
            # A kernel block that does not divide the primary block cannot be scaled
            # by an integer ratio; every small block is then padded to a full page.
            if shared_page is None and block_size % sw_block_size != 0:
                _dividing = [
                    s
                    for s in self.attn_backend.get_supported_kernel_block_sizes(kv_cache_spec)
                    if isinstance(s, int) and block_size % s == 0
                ]
                if _dividing:
                    sw_block_size = max(_dividing)
"""
if src.count(ANCHOR) != 1:
    sys.exit('PATCH_ANCHOR_NOT_FOUND: the installed vLLM version does not fit (' + path + ')')
if not os.path.exists(path + '.pre-swblock-divisor.bak'):
    shutil.copy2(path, path + '.pre-swblock-divisor.bak')
open(path, 'w').write(src.replace(ANCHOR, PATCH))
for pyc in pathlib.Path(path).parent.glob('__pycache__/attention.*.pyc'):
    pyc.unlink()
print('PATCH_APPLIED', path)
