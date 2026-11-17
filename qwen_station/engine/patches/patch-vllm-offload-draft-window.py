"""Local fix for vLLM 0.32 (vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py).

KV offloading to RAM (--kv-offloading-size) lets a session that was pushed out of the GPU come back by a copy
instead of being read again. With the DFlash2 drafter vLLM 0.32 writes the blocks to RAM and never finds them
again: every later visit reads the whole prompt, and the counter for copies back to the GPU stays at 0.

Two things come together (traced on 2026-11-17 with a lookup that printed its result per KV cache group):

1. The drafter's layers form a sliding-window group that is flagged as an "eagle group"
   (patch-vllm-eagle-group.py does that; without the flag the prefix cache of the GPU does not work). The
   offloading code treats eagle groups as volatile: it wants the window plus one more chunk in a row when it
   looks a request up, and stores next to none of them. The lookup of that one group returns 0, and one group
   with 0 makes the whole request a miss, although the attention groups of the model hit completely.
2. The Mamba groups of the model only hit up to one chunk before the last full one, so the boundary a
   request can be restored at moves one chunk to the left. A sliding-window group, however, only stores the
   chunks its window needs at the last boundary, so the window for the boundary that is really used is missing.

Fix, for the method "dflash" only:

1. For offloading, the drafter's group is a plain sliding-window group, not a volatile one.
2. Such a group stores every chunk as it is completed, so a window exists for every boundary.

What it costs: the drafter's KV of the newest chunk can be one speculation step stale when it is stored. That
can only lower how many draft tokens are kept right after a restore; the model itself verifies every draft
(vLLM's own code says the same about drafter KV served one chunk stale). Storing all chunks of the drafter's
five layers adds a few percent to what goes to RAM.

Measured with both changes: two sessions of 119K tokens in turns, 1.1 instead of 27 seconds to the first token
with 98 percent of the prompt from RAM, the same answers as before, 370 to 410 tok/s.

Idempotent. Backup: scheduler.py.pre-offload-draft-window.bak
"""
import glob, os, pathlib, shutil, sys
paths = glob.glob(os.environ.get('VLLM_PREFIX', '/opt/vllm') + '/lib/python3*/site-packages/vllm/distributed/'
                  'kv_transfer/kv_connector/v1/offloading/scheduler.py')
if not paths:
    # Older vLLM versions have no such offloading; the switch for it then does nothing.
    print('PATCH_NOT_APPLICABLE: no offloading scheduler in this vLLM'); sys.exit(0)
path = paths[0]
src = open(path).read()
MARK = 'qwen-station: draft window groups for offloading (local fix)'
if MARK in src:
    print('PATCH_ALREADY_APPLIED', path); sys.exit(0)

ANCHOR_GROUPS = """        if eagle_groups:
            logger.info(
                "KV offloading: EAGLE/MTP draft attention groups %s "
"""
PATCH_GROUPS = """        # --- qwen-station: draft window groups for offloading (local fix) ---
        # DFlash2 drafter groups are plain sliding-window attention. Treated as
        # volatile eagle groups their lookup needs window + 1 chunks that are
        # never stored, and one group without a hit zeroes the whole request.
        if eagle_groups and getattr(
            vllm_config.speculative_config, "method", None
        ) == "dflash":
            logger.info(
                "KV offloading: treating DFlash draft groups %s as "
                "sliding-window groups (local fix).",
                sorted(eagle_groups),
            )
            _QS_DRAFT_WINDOW_GROUPS.update(eagle_groups)
            eagle_groups = set()

"""
ANCHOR_STORE = """                prompt_horizon_block_mask: list[bool] | None = None
                if reconsider_final_swa_tail:
"""
PATCH_STORE = """                # qwen-station (local fix): the Mamba groups move the boundary a
                # request is restored at one chunk to the left, so the draft
                # window has to exist for every boundary, not only the last one.
                if group_config.group_idx in _QS_DRAFT_WINDOW_GROUPS:
                    block_mask = None

"""
ANCHOR_GLOBAL = """class GroupOffloadConfig(NamedTuple):
"""
PATCH_GLOBAL = """# qwen-station (local fix): KV cache groups of a DFlash drafter, filled in
# SchedulerOffloadConfig.from_spec.
_QS_DRAFT_WINDOW_GROUPS: set[int] = set()


"""
for name, anchor in (('groups', ANCHOR_GROUPS), ('store', ANCHOR_STORE), ('global', ANCHOR_GLOBAL)):
    if src.count(anchor) != 1:
        # A newer vLLM restructured this code. Offloading then works as that version does; nothing breaks.
        print('PATCH_NOT_APPLICABLE: anchor "%s" not found in %s' % (name, path)); sys.exit(0)
if not os.path.exists(path + '.pre-offload-draft-window.bak'):
    shutil.copy2(path, path + '.pre-offload-draft-window.bak')
src = src.replace(ANCHOR_GLOBAL, PATCH_GLOBAL + ANCHOR_GLOBAL)
src = src.replace(ANCHOR_GROUPS, PATCH_GROUPS + ANCHOR_GROUPS)
src = src.replace(ANCHOR_STORE, PATCH_STORE + ANCHOR_STORE)
open(path, 'w').write(src)
for pyc in pathlib.Path(path).parent.glob('__pycache__/scheduler.*.pyc'):
    pyc.unlink()
print('PATCH_APPLIED', path)
