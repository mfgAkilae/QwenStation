#!/usr/bin/env python3
"""Benchmark for a running Qwen Station (or any OpenAI-compatible server with vLLM's /metrics).

Fixed prompts in three classes, because speculative decoding depends on what the model writes:
  prose      an essay in plain sentences
  code       a Python module with tests
  reasoning  a multi-part problem worked through step by step

Each class runs with a short prompt and behind a long context (33K and 129K tokens of source code by default).
The long context is read once cold; the tasks after it reuse the cached prefix. Every cold prompt starts with a
fresh nonce, so the prefix cache cannot help it.
Decode speed is output tokens after the first one, divided by the time from the first to the last token.

    python tools/bench.py                          # everything, about five minutes
    python tools/bench.py --contexts 0             # the short prompts only
    python tools/bench.py --out result.json --markdown

Plain Python, standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import time
import urllib.request
import uuid

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
TOKENS_PER_FUNCTION = 36.7  # of synthetic_module(), measured with the Qwen3.8 tokenizer

PROMPTS = {
    "prose": (
        "Write an essay of about 600 words on why port cities became centers of trade and ideas, from the "
        "Mediterranean of antiquity to the container ports of today. Use plain flowing prose with concrete "
        "examples. No lists, no headings."
    ),
    "code": (
        "Write a complete Python module duration.py: parse_duration(text) converts strings such as '1h30m', '45s' "
        "and '2d4h' to seconds, format_duration(seconds) does the reverse. Add error handling, docstrings and "
        "twelve pytest tests. Code only."
    ),
    "reasoning": (
        "A bakery sells three kinds of bread. A rye loaf costs 4.20, a wheat loaf 3.10 and a spelt loaf 5.40. On "
        "Monday it sold 212 loaves for 846.50 in total, and twice as many wheat loaves as spelt loaves. "
        "1. How many loaves of each kind were sold? Show every step. "
        "2. The baker wants to raise one price by 10 percent and keep the others. Assuming sales stay the same, "
        "which choice adds the most revenue, and by how much? "
        "3. Explain in a short paragraph which assumption in part 2 is the weakest and why."
    ),
}
# Asked behind the long contexts. The first one runs cold, the others reuse its cached prefix.
LONG_TASKS = {
    "code": ("\n\nFor this module, write a class HelperRegistry with register, call and a stats method, with "
             "detailed docstrings. Then ten pytest tests. Code only."),
    "prose": ("\n\nDescribe in about 500 words of plain prose what kind of code this module contains, what is "
              "questionable about its design, and how you would restructure it. No lists, no code."),
    "reasoning": ("\n\nCompute helper_17(5), helper_40(3) and helper_123(11) by hand. Show every arithmetic step, "
                  "then explain which of the three results is the largest and why the modulus matters."),
}
METRICS = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
           "vllm:spec_decode_num_accepted_tokens_total", "vllm:num_preemptions_total",
           "vllm:num_requests_running", "vllm:request_success_total")
PER_POSITION = "vllm:spec_decode_num_accepted_tokens_per_pos_total"


def synthetic_module(functions: int) -> str:
    """Source-like filler for the long contexts: repetitive on purpose, so its size is predictable."""
    return "\n".join(
        f"def helper_{i}(value):\n    # utility {i}\n    return (value * {i} + {i % 7}) % 9973\n" for i in range(functions))


def counters(base: str) -> tuple[dict[str, float], dict[int, float]]:
    text = OPENER.open(f"{base}/metrics", timeout=10).read().decode()
    totals = {name: 0.0 for name in METRICS}
    positions: dict[int, float] = {}
    for line in text.splitlines():
        match = re.match(r"^([a-z_:]+)\{([^}]*)\}\s+(\S+)$", line)
        if not match:
            continue
        name, labels, value = match.group(1), match.group(2), float(match.group(3))
        if name in totals:
            totals[name] += value
        elif name == PER_POSITION:
            position = re.search(r'position="(\d+)"', labels)
            if position:
                positions[int(position.group(1))] = positions.get(int(position.group(1)), 0.0) + value
    return totals, positions


def run(args: argparse.Namespace, name: str, kind: str, prompt: str, max_tokens: int, seed: int = 7) -> dict:
    body = {"model": args.model, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
            "temperature": args.temperature, "seed": seed, "messages": [{"role": "user", "content": prompt}]}
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {args.api_key}"} if args.api_key else {})}
    request = urllib.request.Request(f"{args.base}/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    before, positions_before = counters(args.base)
    started = time.perf_counter()
    first = last = None
    usage = None
    with OPENER.open(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or delta.get("tool_calls"):
                    last = time.perf_counter()
                    first = first or last
    after, positions_after = counters(args.base)
    drafts = after[METRICS[0]] - before[METRICS[0]]
    draft_tokens = after[METRICS[1]] - before[METRICS[1]]
    accepted = after[METRICS[2]] - before[METRICS[2]]
    prompt_tokens = usage["prompt_tokens"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    output = usage["completion_tokens"]
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    ttft = first - started
    decode_seconds = last - first
    row = {
        "name": name, "class": kind, "prompt_tokens": prompt_tokens, "cached_tokens": cached, "output_tokens": output,
        "reasoning_tokens": reasoning, "ttft_s": round(ttft, 3),
        "prefill_tps": round((prompt_tokens - cached) / ttft, 1) if ttft > 0 else None,
        "decode_tps": round((output - 1) / decode_seconds, 1) if decode_seconds > 0 else None,
        "rounds_per_s": round(drafts / decode_seconds, 1) if decode_seconds > 0 and drafts else None,
        "accept_length": round(1 + accepted / drafts, 2) if drafts else None,
        "accept_rate": round(accepted / draft_tokens, 3) if draft_tokens else None,
        "accept_per_position": [round((positions_after.get(i, 0.0) - positions_before.get(i, 0.0)) / drafts, 3)
                                for i in sorted(positions_after)] if drafts else [],
        "preemptions": int(after[METRICS[3]] - before[METRICS[3]]),
        # Somebody else used the server during this run: the numbers of this row are then too low.
        "disturbed": after[METRICS[5]] - before[METRICS[5]] > 1,
    }
    print(f"{name:<18} prompt {prompt_tokens:>7} cached {cached:>7} out {output:>5}  TTFT {ttft:>7.2f} s  "
          f"prefill {row['prefill_tps'] or 0:>8.0f} tok/s  decode {row['decode_tps'] or 0:>6.1f} tok/s  "
          f"AL {row['accept_length']}  rounds/s {row['rounds_per_s']}{'  DISTURBED' if row['disturbed'] else ''}", flush=True)
    time.sleep(1.5)
    return row


def summarize(rows: list[dict]) -> list[dict]:
    """Median per test name, without the repeat suffix."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(re.sub(r"-\d+$", "", row["name"]), []).append(row)
    summary = []
    for name, members in groups.items():
        decode = [member["decode_tps"] for member in members if member["decode_tps"]]
        lengths = [member["accept_length"] for member in members if member["accept_length"]]
        summary.append({
            "name": name, "runs": len(members), "prompt_tokens": members[0]["prompt_tokens"],
            "cached_tokens": members[0]["cached_tokens"],
            "decode_tps_median": round(statistics.median(decode), 1) if decode else None,
            "decode_tps_min": min(decode) if decode else None, "decode_tps_max": max(decode) if decode else None,
            "accept_length": round(statistics.median(lengths), 2) if lengths else None,
            "ttft_s": round(statistics.median(member["ttft_s"] for member in members), 2),
            "prefill_tps": members[0]["prefill_tps"],
        })
    return summary


def markdown(summary: list[dict]) -> str:
    lines = ["| Test | Prompt tokens | Cached | Decode tok/s (median, min to max) | Tokens per round | TTFT |",
             "|---|---|---|---|---|---|"]
    for row in summary:
        lines.append(f"| {row['name']} | {row['prompt_tokens']:,} | {row['cached_tokens']:,} | "
                     f"{row['decode_tps_median']} ({row['decode_tps_min']} to {row['decode_tps_max']}) | "
                     f"{row['accept_length']} | {row['ttft_s']} s |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="server root, without /v1")
    parser.add_argument("--model", default="qwen38-gittensor-dflash")
    parser.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY", ""),
                        help="the API key, if the settings have one (default: the variable QWEN_API_KEY)")
    parser.add_argument("--label", default="", help="a name for this run, stored in the result")
    parser.add_argument("--out", help="write the result as JSON")
    parser.add_argument("--markdown", action="store_true", help="print the summary as a Markdown table")
    parser.add_argument("--classes", default="prose,code,reasoning", help="which of prose, code, reasoning to run")
    parser.add_argument("--contexts", default="0,33000,129000",
                        help="context sizes in tokens; 0 is the short prompts, the others put the code task behind filler")
    parser.add_argument("--repeats", type=int, default=3, help="repeats of every short prompt")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 is greedy")
    parser.add_argument("--reasoning-effort", default="", help="low, medium or xhigh; empty keeps the server default")
    args = parser.parse_args()
    nonce = uuid.uuid4().hex
    classes = [name for name in args.classes.split(",") if name]
    for name in classes:
        if name not in PROMPTS:
            parser.error(f"unknown class {name}")
    contexts = [int(value) for value in args.contexts.split(",") if value]
    rows = []
    print(f"=== {args.label or args.base}", flush=True)
    run(args, "warmup", "warmup", "Reply with OK.", 16)
    if 0 in contexts:
        for kind in classes:
            for index in range(args.repeats):
                rows.append(run(args, f"short-{kind}-{index + 1}", kind, f"[run {nonce}-{kind}-{index}]\n{PROMPTS[kind]}",
                                args.max_tokens, 7 + index))
    for size in [value for value in contexts if value > 0]:
        label = f"{round(size / 1000)}k"
        filler = f"# run {nonce}-{label}\n" + synthetic_module(round(size / TOKENS_PER_FUNCTION))
        rows.append(run(args, f"{label}-cold-code", "code", filler + LONG_TASKS["code"], args.max_tokens))
        for kind in classes:
            task = LONG_TASKS[kind] if kind != "code" else LONG_TASKS["code"].replace("ten pytest", "six pytest")
            rows.append(run(args, f"{label}-warm-{kind}", kind, filler + task, args.max_tokens))
    summary = summarize(rows)
    if args.markdown:
        print("\n" + markdown(summary))
    if any(row["disturbed"] for row in rows):
        print("\nSome rows are marked DISTURBED: other requests ran on the server at the same time.")
    if args.out:
        result = {"label": args.label, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "system": platform.system(),
                  "settings": {"temperature": args.temperature, "max_tokens": args.max_tokens,
                               "reasoning_effort": args.reasoning_effort or "server default", "repeats": args.repeats},
                  "summary": summary, "rows": rows}
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
