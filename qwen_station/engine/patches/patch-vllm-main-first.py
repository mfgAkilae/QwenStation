"""Local change for vLLM 0.32: the main session first, subagents in the room it leaves.

With several slots the requests share one KV pool. vLLM serves them first come, first served: when the pool runs
out, the request that came last is pushed out, and that can be the main session of a coding agent (the
orchestrator) while its subagents keep running. Qwen Station's setting "Main session first" turns that around.
The server then answers to a second model name for subagents (serve.sh), and with SUBAGENT_CTX set:

1. Requests for the subagent model get priority 1, the main session keeps 0 (generate/base/serving.py).
   vLLM's priority policy (--scheduling-policy priority) already pushes the lowest priority out when a running
   request needs more blocks.
2. A main request that waits does not wait for a subagent to finish: when it found no slot or no room for its
   blocks, the next step pushes out the subagent request that came last, one per step, until it fits. Within a
   step the main session is served before the subagents (v1/core/sched/scheduler.py). vLLM itself pushes out
   requests only for requests that are already running.
3. The subagent model has a context of its own, the room beside a main session of the full context. A longer
   prompt gets vLLM's usual error ("This model's maximum context length is ..."), which the coding agents read
   as the signal to compact (openai/*/protocol.py).

A subagent that is pushed out continues where it was when room is free again; with sessions in RAM its KV is
copied back instead of read again.

Environment, set by serve.sh: VLLM_QWEN38_SUBAGENT_MODEL (the second model name), VLLM_QWEN38_SUBAGENT_CONTEXT.
Without them every request keeps priority 0 and the full context, and the scheduler change does nothing
unless the priority policy is on.

Idempotent per file. Backups: <file>.pre-main-first.bak
"""
import glob, os, pathlib, shutil, sys

ROOT = glob.glob(os.environ.get('VLLM_PREFIX', '/opt/vllm') + '/lib/python3*/site-packages/vllm')
if not ROOT:
    print('PATCH_NOT_APPLICABLE: no vLLM found'); sys.exit(0)
ROOT = ROOT[0]
MARK = 'qwen-station: main session first (local change)'

SCHEDULER_HOOK = (  # before the anchor
    """        # First, schedule the RUNNING requests.
        req_index = 0
""",
    """        # --- qwen-station: main session first (local change) ---
        if self.policy == SchedulingPolicy.PRIORITY and self.running:
            preempted_reqs.extend(self._qs_main_first(scheduled_timestamp))

""")
SCHEDULER_BLOCKED = (  # after the anchor
    """                if new_blocks is None:
                    # The request cannot be scheduled.

""",
    """                    # qwen-station: main session first (local change). Remembered so
                    # the next step can make room for it, see _qs_main_first.
                    self._qs_blocked = request.request_id
""")
SCHEDULER_METHOD = (  # before the anchor
    """    def _preempt_request(
""",
    """    # --- qwen-station: main session first (local change) ---
    def _qs_main_first(self, timestamp: float) -> list[Request]:
        \"\"\"Priority policy: the main session (priority 0) is served before subagents
        (priority 1), and a main request that waits for a slot, or found no room
        for its blocks at the last step, takes them from the subagent request that
        came last. One request per step; the preempted one waits for room again.
        \"\"\"
        self.running.sort(key=lambda r: r.priority)  # stable: first come within a priority
        blocked = getattr(self, "_qs_blocked", None)
        self._qs_blocked = None
        if self._pause_state != PauseState.UNPAUSED or not self.waiting:
            return []
        head = self.waiting.peek_request()
        lower = [r for r in self.running if r.priority > head.priority]
        if not lower:
            return []
        no_slot = (
            len(self.running) + self.num_waiting_for_streaming_input
            >= self.max_num_active_reqs
        )
        # For blocks: not while blocks of a request pushed out before are still on
        # their way back to the pool (deferred until the step in flight is done).
        if not no_slot and (blocked != head.request_id or self.deferred_frees):
            return []
        victim = max(lower, key=lambda r: (r.priority, r.arrival_time))
        # No _request_blocks_can_be_freed check, unlike vLLM's preemption for a
        # running request: nothing here needs the blocks at once, and
        # _preempt_request defers their return while a step may still write them.
        # With async scheduling and sessions in RAM a request that decodes is in
        # every step, so that check would never let it go.
        self.running.remove(victim)
        self._preempt_request(
            victim, timestamp, drop_stale_output=self.requires_kv_delivery
        )
        return [victim]

""")

SERVING_PRIORITY = (
    """                    pass
        return request.priority
""",
    """                    pass
        # qwen-station: main session first (local change)
        return max(request.priority, _qs_subagent_priority(request))
""")
SERVING_HELPER = """

# --- qwen-station: main session first (local change) ---
def _qs_subagent_priority(request) -> int:
    \"\"\"Requests for the subagent model yield to the main session.\"\"\"
    import os

    name = os.environ.get("VLLM_QWEN38_SUBAGENT_MODEL")
    return 1 if name and getattr(request, "model", None) == name else 0
"""

PROTOCOL_LIMIT = (
    """            max_total_tokens=model_config.max_model_len,
""",
    """            max_total_tokens=_qs_context(self.model, model_config.max_model_len),
""")
PROTOCOL_HELPER = """

# --- qwen-station: main session first (local change) ---
def _qs_context(model, max_model_len):
    \"\"\"The subagent model has the room beside a main session of the full context.\"\"\"
    import os

    name = os.environ.get("VLLM_QWEN38_SUBAGENT_MODEL")
    limit = os.environ.get("VLLM_QWEN38_SUBAGENT_CONTEXT")
    if name and limit and model == name:
        return min(max_model_len, int(limit))
    return max_model_len
"""


def patch(relative, before=(), after=(), replaces=(), append='', needs=()):
    """before/after: (anchor, text) puts text before or after the anchor; replaces: (anchor, text) replaces it."""
    path = os.path.join(ROOT, relative)
    if not os.path.exists(path):
        print('PATCH_NOT_APPLICABLE: missing', path); return
    src = open(path).read()
    if MARK in src:
        print('PATCH_ALREADY_APPLIED', path); return
    for anchor, _ in (*before, *after, *replaces):
        if src.count(anchor) != 1:
            # A newer vLLM restructured this code; this part of the change is then left out, nothing breaks.
            print('PATCH_NOT_APPLICABLE: anchor not found in', path, repr(anchor.strip()[:60])); return
    for text in needs:
        if text not in src:
            print('PATCH_NOT_APPLICABLE: no', repr(text), 'in', path); return
    if not os.path.exists(path + '.pre-main-first.bak'):
        shutil.copy2(path, path + '.pre-main-first.bak')
    for anchor, text in before:
        src = src.replace(anchor, text + anchor)
    for anchor, text in after:
        src = src.replace(anchor, anchor + text)
    for anchor, text in replaces:
        src = src.replace(anchor, text)
    src += append
    open(path, 'w').write(src)
    name = pathlib.Path(path).stem
    for pyc in pathlib.Path(path).parent.glob(f'__pycache__/{name}.*.pyc'):
        pyc.unlink()
    print('PATCH_APPLIED', path)


patch('v1/core/sched/scheduler.py', before=(SCHEDULER_HOOK, SCHEDULER_METHOD), after=(SCHEDULER_BLOCKED,),
      needs=('self.deferred_frees', 'self.requires_kv_delivery', 'max_num_active_reqs',
             'num_waiting_for_streaming_input', 'PauseState', 'SchedulingPolicy'))
patch('entrypoints/generate/base/serving.py', replaces=(SERVING_PRIORITY,), append=SERVING_HELPER)
for endpoint in ('chat_completion', 'completion', 'responses'):
    patch(f'entrypoints/openai/{endpoint}/protocol.py', replaces=(PROTOCOL_LIMIT,), append=PROTOCOL_HELPER)
