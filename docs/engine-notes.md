# Engine notes

Why the engine is set up the way it is. Everything here was measured on one machine: RTX 5090 (32 GB),
Windows 11, WSL 2, vLLM 0.32.0 with FlashInfer 0.7.0.post1. Numbers are decode speeds with greedy sampling
unless stated otherwise.

## The stack

| Part | Choice |
|---|---|
| Target model | `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`: NVFP4 weights (ModelOpt format), 18.0 GiB on the GPU |
| Drafter | `TechPrototyper/Qwen3.8-27B-DFlash2-fp8-vllm`: DFlash2, FP8, block size 8, so up to 7 speculative tokens |
| KV cache | FP8, fixed pool, layers in groups of eight |
| Decode kernel | XQA (TRT-LLM attention) |
| Context | up to 262,144 tokens, one slot |

Typical acceptance length of the drafter is 4.5 to 5 tokens per round.

## Decisions and their measurements

**Fixed KV pool instead of the automatic one.** `--kv-cache-memory-bytes` is set for every start. With the
automatic pool vLLM takes all the memory it sees, Windows starts paging GPU memory, and the speed drops to a
quarter (54 against 243 tok/s). The symptom is a GPU at 99 % utilization that draws only about 240 W.

**The context follows the free VRAM.** A pool of P GiB holds (P − 1.546) × 32,730 tokens. The server needs
21.0 GiB beside its pool. Qwen Station computes the largest context that fits, in steps of 32,768 tokens, and
refuses to start below 65,536. Without this, half a GiB too little is enough to trigger the paging above
(prefill 1,825 instead of 8,800 tok/s).

**XQA decode kernel from vLLM 0.32 on.** With FlashInfer 0.7, XQA is faster at every context size: 214 to 289
tok/s at 129K context against about 150 with the native kernel. On vLLM 0.29 (FlashInfer 0.6) it was the other
way round, XQA collapsed with long context (51 against 178 tok/s at 129K), so `serve.sh` chooses by version.

**KV groups of eight** (`patch-vllm-fp8-group-size.py`). The drafter's five layers would otherwise force groups
of five for all 69 layers.

**The drafter's KV group must be flagged** (`patch-vllm-eagle-group.py`). vLLM does not recognize the DFlash2
drafter's sliding-window layers as the draft group. Without the fix the prefix cache never hits, and on 0.32 the
KV cache needs 11.84 instead of 9.61 GiB for 262,144 tokens.

**Sliding-window block size must divide the primary block** (`patch-vllm-swblock-divisor.py`, vLLM 0.32.x only).
Otherwise every 64-token block is padded to a full page: 2.3 GiB more KV memory and prefix cache hits only in
steps of 6,592 instead of 1,648 tokens.

**FP8 drafter loading** (`patch-vllm-dflash-quant.py`). The drafter's quantization config lacks the mapping for
fused layers; loading fails without it.

**Verified kernel tuning, in place before the first start** (`patch-vllm-autotune-seed.py`, `autotune/`).
FlashInfer picks kernel variants by timing them, and the result varies between runs. A bad pick costs up to
40 %: 216 to 245 tok/s in a fresh installation against 287 to 297 with the verified file. vLLM stores the result
per configuration hash, which includes the context size, so every new context size would measure again. The
patch copies the verified file to the cache path before vLLM reads it. The autotuner itself also runs out of GPU
memory once weights and the full KV pool are loaded, so `fp4_gemm` is skipped
(`VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS`).

**Prebuilt GPU kernels** (`flashinfer-jit-cache-sm120f`). Without them FlashInfer compiles each kernel on the
first start, one after the other. The largest one alone took 8 minutes; the whole first start more than half an
hour. With the package the first start takes about two minutes: only a few XQA variants are still compiled, at
6 to 10 seconds each. `vllm download-kernels` does the same but needs pip, which a uv venv does not have, and it
downloads the kernels of six GPU generations.

**The CUDA compiler needs linker names.** The pip packages `nvidia-cuda-nvcc` and friends are a runtime, not a
toolkit. `provision.sh` adds what a kernel build needs at its link step: `lib64`, unversioned names such as
`libcudart.so`, and `libcuda.so` from the GPU driver.

**A Python of uv's own, not the system's.** A distribution's `python3` comes without its headers, and the
kernel build needs them. `provision.sh` therefore creates the venv with `--managed-python`: the same Python 3.12
with headers on every distribution.

**Weights on the Linux filesystem.** Loading from a Windows drive through `/mnt` took about four minutes;
from the distribution's own disk a start takes 45 to 80 seconds.

## Tried and dropped

- **NVFP4 KV cache.** Loads only with the drafter's layers excluded, and then the prefix cache is unusable
  (block sizes 2,912 and 816 tokens only meet every 148,512 tokens).
- **vLLM 0.33.0.** Runs after pointing the drafter's attention to the Triton backend (on XQA its acceptance
  length falls from 4.5 to 3.0), but brings no speed gain over 0.32.0.
- **MTP head instead of the drafter.** 119 to 126 tok/s, less than half.
- **Unsloth's NVFP4 build as target.** Works with the same drafter, 242 to 274 tok/s, but its weights take
  4.3 GiB more, which limits the context to about 82K tokens. Available as the preset `unsloth-82k`.
  Unsloth's Dynamic 3.1 quants exist as GGUF only, which vLLM cannot load.

## Reference numbers

Fresh installation with the release exe, first start, 2026-11-17, context limit 163,840 tokens (28.1 GiB free
at start):

| Run | Prompt tokens | TTFT | Prefill | Decode |
|---|---|---|---|---|
| short | 141 | 0.11 to 0.14 s | | 296 to 323 tok/s |
| 33K cold | 33,075 | 3.8 s | 8,755 tok/s | 288 tok/s |
| 33K warm (31,312 cached) | 33,075 | 0.48 s | | 280 tok/s |
| 129K cold | 128,775 | 30.8 s | 4,181 tok/s | 223 tok/s |
| 129K warm (126,896 cached) | 128,775 | 0.97 s | | 204 to 260 tok/s |

A second run in the same installation gave 287 to 297 tok/s short and 231 to 305 tok/s at 129K: the spread
between runs is about 10 %.

Native Linux mode (release file 1.1.0, Ubuntu 24.04 inside WSL 2, user without root rights), 2026-11-17, context
limit 196,608 tokens: 272 to 346 tok/s short, 261 to 292 tok/s at 33K, 203 to 211 tok/s at 129K. Setup 381 seconds,
28 GB on disk, first start 199 seconds, second start 81 seconds.

Setup time on a 35 MB/s connection: 446 seconds from the click to a complete installation (model download 17.9 GB
included), 31 GB on disk afterwards. First start: 166 seconds. Later starts: 47 to 80 seconds.
