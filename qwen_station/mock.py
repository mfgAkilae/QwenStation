"""A scripted stand-in for the vLLM server: same log markers, same metric names, no GPU.

Used by the tests, by `--mock` for trying the dashboard, and to check what a client
(OpenCode, Pi) really sends.
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import MODEL_ID

PER_POSITION = (0.87, 0.74, 0.62, 0.56, 0.53, 0.50, 0.47)

STARTUP = (
    (0.0, "=== Start wrapper, variant: dflash (mock)"),
    (0.3, "--- vLLM fix for the DFlash drafter quantization:"),
    (0.3, "=== Starting the mock server"),
    (0.6, "(APIServer pid=101) INFO [api_utils.py:347]  version 0.29.0"),
    (0.2, "(APIServer pid=101) INFO [model.py:2021] Using max model len {context}"),
    (0.8, "(EngineCore pid=202) INFO [model_runner.py:382] Loading model from scratch..."),
    (3.0, "(EngineCore pid=202) INFO [model_runner.py:404] Model loading took 18.24 GiB memory and 108.4 seconds"),
    (2.5, "(EngineCore pid=202) INFO [gpu_worker.py:544] Initial free memory 30.2 GiB, reserved 9.64 GiB memory for KV Cache"),
    (0.2, "(EngineCore pid=202) INFO [kv_cache_utils.py:2060] GPU KV cache size: 264,910 tokens, Maximum concurrency: 1.01x"),
    (0.8, "(EngineCore pid=202) INFO [core.py:361] init engine (profile, create kv cache, warmup model) took 119.52 s"),
    (0.3, "(APIServer pid=101) INFO [entry.py:139] Starting vLLM server on http://127.0.0.1:{port}"),
    (0.2, "(APIServer pid=101) INFO:     Application startup complete."),
)


class MockState:
    def __init__(self, context: int) -> None:
        self.context = context
        self.lock = threading.Lock()
        self.ready = False
        self.running = 0
        self.kv_usage = 0.0
        self.prompt_tokens = 0
        self.generation_tokens = 0
        self.cached_tokens = 0
        self.prefix_queries = 0
        self.prefix_hits = 0
        self.preemptions = 0
        self.finished = {"stop": 0, "length": 0}
        self.ttft_sum = 0.0
        self.e2e_sum = 0.0
        self.prefill_sum = 0.0
        self.decode_sum = 0.0
        self.count = 0
        self.drafts = 0
        self.draft_tokens = 0
        self.accepted_tokens = 0
        self.per_pos = [0] * len(PER_POSITION)
        self.last_request: dict | None = None

    def generate(self, tokens: int) -> None:
        """Account `tokens` output tokens as speculative rounds of 7 draft tokens."""
        with self.lock:
            done = 0
            while done < tokens:
                accepted = 0
                for index, rate in enumerate(PER_POSITION):
                    # PER_POSITION is cumulative (share of rounds reaching position n), so step conditionally.
                    if random.random() > rate / (PER_POSITION[index - 1] if index else 1.0):
                        break
                    accepted += 1
                    self.per_pos[index] += 1
                self.drafts += 1
                self.draft_tokens += len(PER_POSITION)
                self.accepted_tokens += accepted
                done += accepted + 1
            self.generation_tokens += done

    def finish(self, prompt: int, cached: int, ttft: float, prefill: float, decode: float, reason: str = "stop") -> None:
        with self.lock:
            self.prefix_queries += prompt
            self.prefix_hits += cached
            self.cached_tokens += cached
            self.finished[reason] = self.finished.get(reason, 0) + 1
            self.ttft_sum += ttft
            self.prefill_sum += prefill
            self.decode_sum += decode
            self.e2e_sum += ttft + decode
            self.count += 1

    def metrics(self) -> str:
        label = f'{{engine="0",model_name="{MODEL_ID}"}}'
        with self.lock:
            lines = [
                f"vllm:num_requests_running{label} {self.running}",
                f"vllm:num_requests_waiting{label} 0",
                f"vllm:kv_cache_usage_perc{label} {self.kv_usage:.4f}",
                f"vllm:prompt_tokens_total{label} {self.prompt_tokens}",
                f"vllm:generation_tokens_total{label} {self.generation_tokens}",
                f"vllm:prompt_tokens_cached_total{label} {self.cached_tokens}",
                f"vllm:prefix_cache_queries_total{label} {self.prefix_queries}",
                f"vllm:prefix_cache_hits_total{label} {self.prefix_hits}",
                f"vllm:num_preemptions_total{label} {self.preemptions}",
                f"vllm:time_to_first_token_seconds_sum{label} {self.ttft_sum:.4f}",
                f"vllm:time_to_first_token_seconds_count{label} {self.count}",
                f"vllm:e2e_request_latency_seconds_sum{label} {self.e2e_sum:.4f}",
                f"vllm:e2e_request_latency_seconds_count{label} {self.count}",
                f"vllm:request_prefill_time_seconds_sum{label} {self.prefill_sum:.4f}",
                f"vllm:request_decode_time_seconds_sum{label} {self.decode_sum:.4f}",
                f"vllm:spec_decode_num_drafts_total{label} {self.drafts}",
                f"vllm:spec_decode_num_draft_tokens_total{label} {self.draft_tokens}",
                f"vllm:spec_decode_num_accepted_tokens_total{label} {self.accepted_tokens}",
            ]
            for reason, value in self.finished.items():
                lines.append(f'vllm:request_success_total{{engine="0",finished_reason="{reason}",'
                             f'model_name="{MODEL_ID}"}} {value}')
            for index, value in enumerate(self.per_pos):
                lines.append(f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",'
                             f'model_name="{MODEL_ID}",position="{index}"}} {value}')
        return "# HELP vllm:num_requests_running Number of requests in model execution batches.\n" + "\n".join(lines) + "\n"


def _workload(state: MockState, speed: float) -> None:
    """Endless fake traffic: a pause, a prefill, then decoding at a code-like pace."""
    history = 0
    while True:
        time.sleep(random.uniform(3.0, 7.0) / speed)
        fresh = random.randint(900, 9000)
        prompt = history + fresh
        cached = history
        prefill = max(fresh / 2300.0, 0.25)
        with state.lock:
            state.running = 1
            state.kv_usage = min(prompt / state.context, 1.0)
        time.sleep(prefill / speed)
        with state.lock:
            state.prompt_tokens += prompt
        output = random.randint(250, 2400)
        rate = random.uniform(120.0, 260.0)
        started = time.time()
        produced = 0
        while produced < output:
            time.sleep(0.2 / speed)
            step = min(int(rate * 0.2), output - produced)
            state.generate(step)
            produced += step
            with state.lock:
                state.kv_usage = min((prompt + produced) / state.context, 1.0)
        decode = (time.time() - started) * speed
        state.finish(prompt, cached, ttft=prefill + 0.08, prefill=prefill, decode=decode,
                     reason="length" if random.random() < 0.06 else "stop")
        with state.lock:
            state.running = 0
            state.kv_usage = 0.0
        history = min(prompt + output, state.context - 40000)
        if random.random() < 0.15:
            history = 0  # a new conversation


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: MockState

    def log_message(self, format: str, *args) -> None:
        pass

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/ping"):
            self._send(200 if self.state.ready else 503, b"")
        elif path == "/metrics":
            self._send(200, self.state.metrics().encode(), "text/plain; version=0.0.4")
        elif path == "/v1/models":
            card = {"id": MODEL_ID, "object": "model", "owned_by": "vllm", "max_model_len": self.state.context}
            self._send(200, json.dumps({"object": "list", "data": [card]}).encode())
        elif path == "/mock/last-request":
            self._send(200, json.dumps(self.state.last_request).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._send(400, b'{"error":"invalid json"}')
            return
        if not self.path.split("?", 1)[0].endswith("/chat/completions"):
            self._send(404, b'{"error":"not found"}')
            return
        self.state.last_request = payload
        if payload.get("min_p") or payload.get("logit_bias"):
            # The real server refuses these while speculative decoding is active.
            self._send(400, b'{"error":{"message":"min_p and logit_bias are not supported with speculative decoding"}}')
            return
        text = "MOCK_OK"
        prompt = sum(len(str(message.get("content", ""))) for message in payload.get("messages", [])) // 4 + 8
        with self.state.lock:
            self.state.prompt_tokens += prompt
        self.state.generate(3)
        self.state.finish(prompt, 0, ttft=0.05, prefill=0.04, decode=0.02)
        usage = {"prompt_tokens": prompt, "completion_tokens": 3, "total_tokens": prompt + 3}
        base = {"id": "chatcmpl-mock", "created": int(time.time()), "model": MODEL_ID}
        if payload.get("stream"):
            chunks = [
                {**base, "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
                {**base, "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {**base, "object": "chat.completion.chunk", "choices": [], "usage": usage},
            ]
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            self._send(200, body.encode(), "text/event-stream")
        else:
            message = {"role": "assistant", "content": text}
            body = {**base, "object": "chat.completion", "usage": usage,
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}
            self._send(200, json.dumps(body).encode())


def serve(port: int, speed: float = 1.0, context: int = 262144, traffic: bool = True) -> int:
    def say(text: str) -> None:
        if sys.stdout is not None:
            print(text, flush=True)
        else:  # windowed build: no Python-level stdout, but the inherited pipe still works
            try:
                os.write(1, (text + "\n").encode("utf-8"))
            except OSError:
                pass

    state = MockState(context)
    failure = os.environ.get("QWEN_STATION_MOCK_FAIL", "")
    handler = type("Handler", (_Handler,), {"state": state})
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as exc:
        say(f"ERROR: [Errno 98] error while attempting to bind on address ('127.0.0.1', {port}): address already in use ({exc})")
        return 1
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for delay, line in STARTUP:
        time.sleep(delay / speed)
        if failure == "oom" and "Initial free memory" in line:
            say("(EngineCore pid=202) ERROR [core.py:1139] ValueError: Free memory on device (27.1/31.82 GiB) on startup "
                "is less than desired GPU memory utilization (0.92, 29.27 GiB).")
            return 1
        say(line.format(context=context, port=port))
    state.ready = True
    if traffic:
        threading.Thread(target=_workload, args=(state, speed), daemon=True).start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8000))
