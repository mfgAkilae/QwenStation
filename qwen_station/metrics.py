"""vLLM /metrics (Prometheus text format) -> live speed, per-request rows, spec-decode stats.

No proxy sits in the request path: every number comes from counter deltas.
"""

from __future__ import annotations

import re
import time
from collections import deque

_SAMPLE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$")
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    samples: dict[str, list[tuple[dict[str, str], float]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        name, label_text, value_text = match.groups()
        try:
            value = float(value_text)
        except ValueError:
            continue
        if value != value:  # NaN
            continue
        labels = {key: val.replace('\\"', '"') for key, val in _LABEL.findall(label_text or "")}
        samples.setdefault(name, []).append((labels, value))
    return samples


def _total(samples, *names: str) -> float | None:
    """Sum over all label sets of the first metric name that exists."""
    for name in names:
        rows = samples.get(name)
        if rows:
            return sum(value for _, value in rows)
    return None


def _by_label(samples, label: str, *names: str) -> dict[str, float]:
    for name in names:
        rows = samples.get(name)
        if rows:
            out: dict[str, float] = {}
            for labels, value in rows:
                key = labels.get(label, "")
                out[key] = out.get(key, 0.0) + value
            return out
    return {}


COUNTER_KEYS = (
    "prompt_tokens", "generation_tokens", "cached_tokens", "prefix_queries", "prefix_hits",
    "preemptions", "requests", "ttft_sum", "ttft_count", "e2e_sum", "e2e_count",
    "prefill_sum", "decode_sum", "queue_sum", "drafts", "draft_tokens", "accepted_tokens",
)


def extract(samples) -> dict:
    """Flatten the vLLM metric families this tool uses. Missing families become 0."""
    finish = {key: value for key, value in _by_label(samples, "finished_reason", "vllm:request_success_total").items()}
    cached = _total(samples, "vllm:prompt_tokens_cached_total")
    if cached is None:
        by_source = _by_label(samples, "source", "vllm:prompt_tokens_by_source_total")
        cached = sum(value for key, value in by_source.items() if "cache" in key and "hit" in key) if by_source else None
    per_pos_raw = _by_label(samples, "position", "vllm:spec_decode_num_accepted_tokens_per_pos_total")
    per_pos: dict[int, float] = {}
    for key, value in per_pos_raw.items():
        try:
            per_pos[int(key)] = value
        except ValueError:
            continue

    def number(*names: str) -> float:
        return _total(samples, *names) or 0.0

    return {
        "running": number("vllm:num_requests_running"),
        "waiting": number("vllm:num_requests_waiting"),
        "kv_usage": number("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
        "prompt_tokens": number("vllm:prompt_tokens_total"),
        "generation_tokens": number("vllm:generation_tokens_total"),
        "cached_tokens": cached,
        "prefix_queries": number("vllm:prefix_cache_queries_total"),
        "prefix_hits": number("vllm:prefix_cache_hits_total"),
        "preemptions": number("vllm:num_preemptions_total"),
        "requests": sum(finish.values()),
        "finish": finish,
        "ttft_sum": number("vllm:time_to_first_token_seconds_sum"),
        "ttft_count": number("vllm:time_to_first_token_seconds_count"),
        "e2e_sum": number("vllm:e2e_request_latency_seconds_sum"),
        "e2e_count": number("vllm:e2e_request_latency_seconds_count"),
        "prefill_sum": number("vllm:request_prefill_time_seconds_sum"),
        "decode_sum": number("vllm:request_decode_time_seconds_sum"),
        "queue_sum": number("vllm:request_queue_time_seconds_sum"),
        "drafts": number("vllm:spec_decode_num_drafts_total"),
        "draft_tokens": number("vllm:spec_decode_num_draft_tokens_total"),
        "accepted_tokens": number("vllm:spec_decode_num_accepted_tokens_total"),
        "accepted_per_pos": per_pos,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator and denominator > 0 else None


class Tracker:
    """Turns successive snapshots into live rates and one row per finished request."""

    SPEED_WINDOW_SECONDS = 3.0

    def __init__(self, max_requests: int = 60) -> None:
        self.requests: deque[dict] = deque(maxlen=max_requests)
        self.request_seq = 0
        self._previous: dict | None = None
        self._mark: dict | None = None  # counters at the last finished request
        self._window: deque[tuple[float, float]] = deque()
        self._activity = "idle"
        self._activity_since = time.time()
        self.latest: dict | None = None

    def reset(self) -> None:
        self._previous = None
        self._mark = None
        self._window.clear()
        self.latest = None
        self._set_activity("idle", time.time())

    def _set_activity(self, activity: str, now: float) -> None:
        if activity != self._activity:
            self._activity = activity
            self._activity_since = now

    def update(self, snapshot: dict, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        previous = self._previous
        if previous is not None and any(snapshot[key] is not None and previous.get(key) is not None
                                        and snapshot[key] < previous[key] for key in COUNTER_KEYS):
            previous = None  # the server restarted: counters began again at zero
            self._window.clear()
        if previous is None or self._mark is None:
            self._mark = dict(snapshot)
        self._window.append((now, snapshot["generation_tokens"]))
        while len(self._window) > 2 and now - self._window[0][0] > self.SPEED_WINDOW_SECONDS:
            self._window.popleft()
        oldest_time, oldest_tokens = self._window[0]
        span = now - oldest_time
        speed = (snapshot["generation_tokens"] - oldest_tokens) / span if span > 0 else 0.0
        generated = snapshot["generation_tokens"] - previous["generation_tokens"] if previous else 0.0

        if snapshot["running"] > 0:
            # No new token in this tick but some in the last seconds: a pause inside decode, not a prefill.
            self._set_activity("decode" if generated > 0 or speed > 0 else "prefill", now)
        else:
            self._set_activity("idle", now)

        if previous is not None:
            finished = snapshot["requests"] - self._mark["requests"]
            if finished > 0:
                self._record(snapshot, now, int(finished), aborted=False)
            elif snapshot["running"] == 0 and previous["running"] > 0:
                moved = (snapshot["generation_tokens"] - self._mark["generation_tokens"]
                         + snapshot["prompt_tokens"] - self._mark["prompt_tokens"])
                if moved > 0:
                    self._record(snapshot, now, 1, aborted=True)

        self._previous = dict(snapshot)
        self.latest = {
            "t": round(now, 3),
            "tps": round(max(speed, 0.0), 1) if snapshot["running"] > 0 or speed > 0 else 0.0,
            "running": int(snapshot["running"]),
            "waiting": int(snapshot["waiting"]),
            "kv": round(snapshot["kv_usage"], 4),
            "activity": self._activity,
            "activity_since": round(self._activity_since, 3),
        }
        return self.latest

    def _record(self, snapshot: dict, now: float, count: int, aborted: bool) -> None:
        mark = self._mark or snapshot

        def delta(key: str) -> float:
            if snapshot.get(key) is None or mark.get(key) is None:
                return 0.0
            return snapshot[key] - mark[key]

        output_tokens = delta("generation_tokens")
        decode_seconds = delta("decode_sum")
        finish = {key: int(value - mark.get("finish", {}).get(key, 0.0))
                  for key, value in snapshot.get("finish", {}).items()
                  if value - mark.get("finish", {}).get(key, 0.0) > 0}
        if aborted:
            finish = {"abort": 1}
        self.request_seq += 1
        self.requests.appendleft({
            "seq": self.request_seq,
            "t": round(now, 3),
            "count": count,
            "prompt_tokens": int(delta("prompt_tokens")),
            "cached_tokens": int(delta("cached_tokens")) if snapshot.get("cached_tokens") is not None else None,
            "output_tokens": int(output_tokens),
            "ttft_s": _round(_ratio(delta("ttft_sum"), delta("ttft_count"))),
            "prefill_s": _round(delta("prefill_sum") / count) if delta("prefill_sum") > 0 else None,
            "decode_s": _round(decode_seconds / count) if decode_seconds > 0 else None,
            "e2e_s": _round(_ratio(delta("e2e_sum"), delta("e2e_count"))),
            "decode_tps": _round(_ratio(output_tokens, decode_seconds), 1),
            "accept_rate": _round(_ratio(delta("accepted_tokens"), delta("draft_tokens")), 4),
            "finish": finish,
        })
        self._mark = dict(snapshot)

    def totals(self) -> dict | None:
        snapshot = self._previous
        if snapshot is None:
            return None
        drafts = snapshot["drafts"]
        positions = snapshot.get("accepted_per_pos") or {}
        per_pos = [_round(_ratio(positions[index], drafts), 4) for index in sorted(positions)] if drafts else []
        return {
            "requests": int(snapshot["requests"]),
            "prompt_tokens": int(snapshot["prompt_tokens"]),
            "generation_tokens": int(snapshot["generation_tokens"]),
            "preemptions": int(snapshot["preemptions"]),
            "prefix_hit_rate": _round(_ratio(snapshot["prefix_hits"], snapshot["prefix_queries"]), 4),
            "ttft_avg_s": _round(_ratio(snapshot["ttft_sum"], snapshot["ttft_count"])),
            "accept_rate": _round(_ratio(snapshot["accepted_tokens"], snapshot["draft_tokens"]), 4),
            # Every draft round emits one verified token plus the accepted draft tokens.
            "accept_length": _round(1 + snapshot["accepted_tokens"] / drafts, 2) if drafts else None,
            "accept_per_pos": per_pos,
            "finish": {key: int(value) for key, value in snapshot.get("finish", {}).items()},
        }


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)
