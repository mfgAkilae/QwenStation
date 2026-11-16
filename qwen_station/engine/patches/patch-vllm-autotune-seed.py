"""Local addition for vLLM from 0.32 on (vllm/model_executor/warmup/kernel_warmup.py, flashinfer_autotune).

FlashInfer picks kernel variants by timing them and stores the result per server configuration (a hash over
the vLLM config, so also over the context size). The measurement varies from run to run; a bad result costs up
to 40 % decode speed (measured 2026-11-16: 146 against 249 tok/s; on 2026-11-17 in a fresh installation 216 to
245 against 287 to 297 tok/s).

The start script therefore replaces the file with verified values after the start. That only takes effect on
the next start, and every new context size begins with a fresh measurement again.

This patch puts the verified file in place before it is read: if the environment variable
VLLM_QWEN38_AUTOTUNE_SEED points to a file, it is copied to the cache path before vLLM loads it. Without the
variable nothing changes.

Idempotent. Backup: kernel_warmup.py.pre-autotune-seed.bak
"""
import glob
import os
import pathlib
import shutil
import sys

prefix = os.environ.get('VLLM_PREFIX', '/opt/vllm')
paths = glob.glob(prefix + '/lib/python3*/site-packages/vllm/model_executor/warmup/kernel_warmup.py')
if not paths:
    print('PATCH_NOT_NEEDED: kernel_warmup.py is missing in', prefix, '(vLLM before 0.32)')
    sys.exit(0)
path = paths[0]
src = open(path).read()
MARK = 'VLLM_QWEN38_AUTOTUNE_SEED'
if MARK in src:
    print('PATCH_ALREADY_APPLIED', path)
    sys.exit(0)
ANCHOR = """    if is_leader:
        logger.info_once("Using FlashInfer autotune cache file: %s", cache_path)
"""
PATCH = ANCHOR + """    # --- verified tuning in place before it is read (local addition) ---
    import os as _os

    _seed = _os.environ.get("VLLM_QWEN38_AUTOTUNE_SEED")
    if _seed and is_leader and _os.path.isfile(_seed):
        import filecmp as _filecmp
        import shutil as _shutil

        try:
            if not (cache_path.is_file() and _filecmp.cmp(_seed, cache_path, shallow=False)):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                _shutil.copyfile(_seed, cache_path)
                logger.info("Seeded FlashInfer autotune cache from %s", _seed)
        except OSError as _exc:
            logger.warning("Could not seed the FlashInfer autotune cache: %s", _exc)
"""
if src.count(ANCHOR) != 1:
    # Not fatal: without the seed the start scripts still pin the file after the first start.
    print('PATCH_NOT_APPLIED: anchor missing in', path, '- the tuning is only replaced after the first start')
    sys.exit(0)
if not os.path.exists(path + '.pre-autotune-seed.bak'):
    shutil.copy2(path, path + '.pre-autotune-seed.bak')
open(path, 'w').write(src.replace(ANCHOR, PATCH))
for pyc in pathlib.Path(path).parent.glob('__pycache__/kernel_warmup.*.pyc'):
    pyc.unlink()
print('PATCH_APPLIED', path)
