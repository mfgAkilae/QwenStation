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
| Context | what the free VRAM holds: up to 229,376 tokens on Windows, 262,144 on Linux; one slot by default, up to four that share it |

Typical acceptance length of the drafter is 4.5 to 5 tokens per round.

## Decisions and their measurements

**Fixed KV pool instead of the automatic one.** `--kv-cache-memory-bytes` is set for every start. With the
automatic pool vLLM takes all the memory it sees, Windows starts paging GPU memory, and the speed drops to a
quarter (54 against 243 tok/s). The symptom is a GPU at 99 % utilization that draws only about 240 W.

**The context follows the free VRAM.** A request of L tokens needs 1.62 GiB + L / 32,400 GiB in the KV pool;
the pool gets 2 percent on top. Qwen Station keeps 22.0 GiB beside the pool (21.5 on Linux), computes the
largest context that fits, in steps of 32,768 tokens, and refuses to start below 65,536.

The pool figures were 1.546 GiB and 32,730 tokens per GiB up to version 1.5.0, taken from the size vLLM
reports for its cache. That left the pool 2 percent too small for one request of the whole context. Measured
2026-11-17 with single cold requests in a pool of 6.617 GiB (context 163,840): 128,900 tokens filled 84 percent
of the pool, 143,900 tokens 92, 150,908 tokens 94, 156,884 tokens 97. A prompt of 163,000 tokens reached 99
percent, lost its blocks and started over, without end and without an error; vLLM does not count that as a
preemption. With the new figures and 1 percent on top, a prompt of 163,388 tokens (99.7 percent of the
context) was answered after 47 seconds with the pool at 99 percent, and a longer one got HTTP 400 at once. The
2 percent are for the blocks of 1,648 tokens the cache is handed out in.

The 22.0 GiB were 21.0 up to version 1.5.0, and the full context started without any fit whenever 0.92 of the
card (29.3 GiB) was free. Measured 2026-11-17, that was too little. Weights and runtime take 20.3 GiB at the
start and 20.8 to 21.3 GiB after one or two long sessions. Windows pages before the card is full: `nvidia-smi`
stayed at 31.9 of 32.6 GiB in every run that paged. And the paging hides: vLLM allocates the whole pool at the
start, Windows moves the untouched part out, and short prompts stay fast. With 30.1 GiB free and the full
context, a session of 163K tokens decoded at 83 tok/s; with 229,376 tokens of context (30.6 GiB free) the same
session ran at 380 to 400 tok/s, also after two long sessions, with `nvidia-smi` at 31.5 to 31.9 GiB. So the
full context needs 31.9 GiB free, more than a 32 GB card has under Windows: there the largest context is
229,376 tokens. On Linux nothing pages, so 0.5 GiB less is kept and the full context fits a free card; that
figure is not measured on a Linux machine.

**Slots share the KV pool, and a slot costs nothing.** `--max-num-seqs` only limits how many requests run at
once. vLLM reports its pool as "tokens" and "maximum concurrency", and both follow from one rule: a running
request needs a fixed part plus its tokens. With a pool of 9.64 GiB it reported 264,910 tokens and 1.01x for
requests of 262,144 tokens, 180,577 tokens and 2.76x for 65,536, and 45,471 tokens and 5.55x for 8,192, the
same with 3 and with 4 slots. That fits a fixed part of 1.48 GiB per running request (the Mamba states and the
snapshots for the draft tokens) and 32,500 tokens per GiB. So Qwen Station sizes the pool for one request of
the full context, as before, and several requests share it: each one that runs takes the room of about 52,000
tokens for its fixed part (1.62 GiB as the pool counts it). Requests that do not fit together wait; vLLM counted no preemption in any test.
Two requests at once ran at 332 to 347 tok/s each (653 to 680 together), four at 344 to 449 each (1,365 to
1,376 together). The verified kernel tuning applies with every number of slots (96 configs loaded, nothing
measured again). The earlier preset with two slots and half the context each is gone.

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

**Seven draft tokens per round by default, more as an option** (`draft_tokens`, `SPEC_TOKENS`). vLLM takes the
block size of the DFlash2 drafter from the setting, not from the checkpoint, so the drafter can be asked for more
than the 7 tokens it was trained for. Its acceptance does not stop there: with 15 draft tokens, position 8 is
still kept in 60 percent of the rounds of a reasoning text and position 15 in 14 to 38 percent. Measured
2026-11-17 (short prompt, 33K, 137K; tok/s): reasoning 451, 391, 323 with 7 draft tokens, 527, 443, 403 with 11
and 554, 461, 419 with 15. Prose keeps about 2.5 tokens per round either way and gets up to 14 percent slower.
Two costs: a round verifies more tokens (65 rounds per second with 7, 60 with 15; behind 137K tokens 54 and 46),
and every extra draft token needs one more state snapshot per Mamba layer in the KV pool, 0.16 GiB or about
5,400 tokens of context. Qwen Station sizes the pool for the setting.

**Vision is off unless asked for, and has a cache of its own.** The checkpoint ships its image encoder
(`model.visual`, 27 layers, bf16). `serve.sh` sets `--limit-mm-per-prompt` to 0 for images and video, and vLLM
then does not load the encoder: 18.02 instead of 18.88 GiB for the model. With the setting `vision`, Qwen
Station appends `--limit-mm-per-prompt.image 4 --limit-mm-per-prompt.video 0`; the start script puts extra
arguments last and the last one wins, so the engine needs no change. Two things had to be found out, both on
2026-11-17:

- vLLM keys its cache of compiled graphs without the multimodal settings (`limit_mm_per_prompt` is on the list
  of ignored factors in `ModelConfig.compute_hash`). With images on, the model is called with embeddings
  instead of token ids, so a graph compiled in text-only mode fails at the first run of the engine with
  `'NoneType' object has no attribute 'size'`. Qwen Station therefore sets `VLLM_CACHE_ROOT` to
  `cache/vllm-vision` below the engine folder when vision is on. The verified kernel tuning arrives there too,
  because the seed patch writes to whatever path vLLM computes.
- Memory: 0.86 GiB for the weights, up to 0.75 GiB while the largest image is read. With the context forced to
  the same size as without vision, the card ran full, Windows paged GPU memory, and plain text got slower:
  259 instead of 328 tok/s for code, 147 instead of 372 behind 33K tokens, the cold read of 33K tokens 20
  instead of 4 seconds. With 1.7 GiB reserved the speed of text is unchanged.

The DFlash2 drafter is text-only; vLLM says so at the start ("Embeddings from the target model will not be
passed to the drafter") and falls back to drafting from the text alone. XQA stays the decode kernel.

**An NVFP4 KV cache by default** (`kv_cache`, `KV_DTYPE=nvfp4`; FP8 as the setting; measured 2026-11-17). vLLM 0.32
offers `--kv-cache-dtype nvfp4` for FlashInfer: keys and values in 4 bit with an FP8 scale per 16 values, so a
token of an attention layer takes 0.5625 of its FP8 size. FlashInfer reads it for the prefill through its FA2
path and, on SM12x only, with the XQA decode kernel; the start log says `decode_backend=xqa,
kv_cache_dtype=torch.uint8`. The `turboquant` and `ultraquant` types run with Triton backends only and were not
tried: without XQA the decode is much slower here.

- Size. vLLM's "KV cache size" is its concurrency times the context, so it grows with the context for the same
  pool: a pool of 5.78 GiB showed 201,850 tokens at a context of 131,072, 234,105 at 196,608 and 245,304 at
  229,376. Per request that is 1.55 GiB fixed (the Mamba states, unchanged) and 59,500 tokens per GiB. The same
  reading for FP8 gives 1.50 GiB and 32,560. Qwen Station keeps 1.62 GiB and 58,000 tokens per GiB for NVFP4. A
  prompt of 99.5 percent of 262,144 tokens then filled the pool to 95 percent and was answered.
- Speed, against FP8 (single runs): draft rounds per second 65 and 52 with a short prompt and behind 137K
  tokens, against 65 and 54; tokens kept per round 6.8, 5.8 and 6.1 for reasoning (short, 33K, 137K) against
  6.9, 6.4 and 6.1; reading 137K tokens cold 34.4 against 34 seconds; a session of 250K tokens decoded at 343
  tok/s.
- Recall: 8 and 16 codes hidden at spread depths of a 200K-token log and 16 in a 100K one came back complete,
  with NVFP4 as with FP8.
- What it gives: with the 28.6 GiB a desktop leaves free, the full context of 262,144 tokens instead of 131,072.
- Several slots get a pool of all the VRAM the reserve leaves (`fit_context(fill=True)`), not one sized for a
  single request of the context. With 30.4 GiB free that was 8.35 GiB, which vLLM reported as 1.39 times the
  context. Two sessions of 149K tokens then ran at once without preemption; their prefills share the GPU (both
  read after 79 s), and the one decoding while the other still reads gets about 20 tok/s. Sessions in RAM work
  with 2 and 4 slots: 98 and 85 percent came back from RAM, the same answers.

**Sessions in RAM, with a patch to vLLM's offloading** (`ram_cache_gib`, `KV_OFFLOAD_GIB`; off by default;
measured 2026-11-17). The KV pool holds one large session. A second one pushes the first out of the GPU, and
every switch between two sessions of 119K tokens then reads the prompt again: 27 seconds, 0 percent from the
cache. vLLM's native CPU offloading (`--kv-offloading-size`) copies blocks that leave the pool to RAM and back.

- It needs `--enable-cumem-allocator`: vLLM refuses a KV connector together with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which `serve.sh` sets. The RAM is taken at the start, as a
  file in `/dev/shm` (24 GiB asked for, 25.8 GB taken, about 14 seconds longer to start). `serve.sh` leaves
  the offloading off, with a warning in the log, when less than the asked size plus 8 GiB is free.
- Without the drafter it works out of the box: two sessions of 149K tokens in turns, 1.1 to 1.2 seconds
  instead of 40 for the second visit, 99 percent from the cache. But then the model decodes at 70 tok/s.
- With the DFlash2 drafter, unpatched, nothing comes back: the blocks are written, every later visit still
  reads the whole prompt, and `CPU_to_GPU` stays at 0. A lookup that printed its result per KV cache group
  showed why. The model has nine groups here: six Mamba groups, two full-attention groups, and the drafter's
  five layers as a sliding-window group (window 2,048 tokens, two chunks of 1,648).
  1. `patch-vllm-eagle-group.py` flags the drafter's group as an "eagle group", and the offloading code treats
     eagle groups as volatile: its lookup wants the window plus one more chunk in a row and stores next to
     none of them. That group returned 0 hits, and one group with 0 makes the whole request a miss.
  2. The Mamba groups hit only up to one chunk before the last full one, so the boundary a request is restored
     at moves one chunk to the left. A sliding-window group stores only the chunks its window needs at the
     last boundary, so the window for the boundary that is really used was missing.
- `patch-vllm-offload-draft-window.py` changes both, for the method `dflash` only: for offloading the
  drafter's group is a plain sliding-window group, and it stores every chunk as it is completed. The first
  change alone was not enough. The drafter's KV of the newest chunk can be one speculation step stale when it
  is stored; that can only lower how many draft tokens are kept right after a restore, because the model
  verifies every draft.
- With the patch: two sessions of 119K tokens in turns, 1.1 seconds to the first token instead of 26.5, with
  117,008 of 118,928 tokens from RAM, the right code word from each document and the same answers as on the
  first visit, 407 to 417 tok/s. Two conversations of 109K tokens that grow turn by turn, in turns: 1.0 second
  per switch instead of 23, about 1,900 tokens read again each time. No preemptions, no errors. Two sessions
  of 174K tokens with a context of 196,608: 1.8 seconds instead of 52. With 2 slots and 15 draft tokens, three
  sessions of 59K tokens in turns: 0.6 seconds instead of 8.1. Copied back: 4.03 GB for 117,008 tokens.
- `blocks_per_chunk` 2, which should have made the window fit into one chunk, changed nothing. Decode with
  offloading on is within a few percent of the speed without (single runs: 372 to 417 against 417 to 437).

## Tried and dropped

- **B12X** (`b12x` 1.5.1, the version vLLM 0.32 pins; measured 2026-11-17 in rounds per second, which do not
  depend on the text: 65 with a short prompt, 62 behind 33K and 54.5 behind 137K tokens as the baseline).
  - As attention backend it is not released for hybrid models: vLLM aligns the KV storage block to the Mamba
    page (1,664 tokens) and B12X accepts only 64 or 128. With the two checks relaxed to "a multiple of 64" and
    `--kernel-config.enable_cutedsl_warmup false` (the warmup of this version pair fails with
    `'Tensor' object has no attribute 'fake_mode'`) it runs and keeps the same tokens per round. Reading a long
    context gets faster: 9,838 against 9,188 tok/s for 33K tokens, 4,761 against 4,057 tok/s for 137K (29 instead
    of 34 seconds). Decode gets slower behind long contexts: 57 against 62 rounds per second at 33K, 49.5 against
    54.5 at 137K. XQA stays the better decode kernel, and a backend can only be chosen per layer group, not
    separately for prefill and decode.
  - Its NVFP4 linear kernels bring nothing: `--linear-backend flashinfer_b12x` 64 and 53.6 rounds per second
    (short, 137K), the native kernel (`--kernel-config.linear_backend_per_quant.nvfp4_w4a4 b12x`) 62 and 52.8,
    against 65 and 54.5 with FlashInfer's CUTLASS kernel and the verified tuning. `--linear-backend b12x` for
    everything fails at the drafter's FP8 layers, and `flashinfer_cutedsl` picks a W4A16 kernel and runs out of
    memory.
- **DSpark drafter** (`gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4`, trained against this very checkpoint). vLLM
  0.32 has its own DSpark code and loads the drafter once the exclude patterns in its `hf_quant_config.json` start
  with `*` (vLLM prefixes the layer names with `model.`, so `markov_head*` matches nothing and the unquantized
  heads are built as NVFP4). It keeps fewer tokens per round than DFlash2 on every kind of text: prose 1.9
  against 2.5, code 3.7 against 5.0, reasoning 5.3 against 6.9; in tok/s 122, 237 and 342 against 155, 302 and
  424. More draft tokens do not help it, and neither does reasoning effort `low`.

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

The AppImage of 1.3.0 in a fresh Ubuntu 24.04 inside WSL 2, 2026-11-17, context limit 262,144 tokens: setup 364
seconds, first start 154 seconds, a later start 44 seconds. Short prompts: 162 tok/s for prose, 320 for code, 423
for reasoning, at 65 rounds per second.

Setup time on a 35 MB/s connection: 446 seconds from the click to a complete installation (model download 17.9 GB
included), 31 GB on disk afterwards. First start: 166 seconds. Later starts: 47 to 80 seconds.
