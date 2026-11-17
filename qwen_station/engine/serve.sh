#!/usr/bin/env bash
# Starts the vLLM server for Qwen Station: Qwen3.8 27B (NVFP4) with the DFlash2 drafter and an FP8 KV cache.
# Everything is found relative to the install prefix, so the folder can live anywhere.
#
# Settings come from the environment (Qwen Station sets them):
#   CTX, KV_BYTES, MAX_SEQS       context window (includes the output), KV pool in bytes, parallel slots
#   VLLM_HOST, VLLM_PORT          bind address and port
#   TARGET_NAME, TARGET_QUANT     another target below models/ (default: the Gittensor NVFP4 build, modelopt)
#   EFFORT                        default reasoning_effort (low | medium | xhigh)
#   EXTRA_ARGS                    appended to `vllm serve`; later arguments win
#   XQA_DECODE, SPEC_MODE, SPEC_TOKENS, KV_GROUP_SIZE, AUTOTUNE_PIN   switches for experiments, see below
#
# The reasons behind the non-obvious choices are measured and written up in docs/engine-notes.md.
set -uo pipefail

ENGINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${QWEN_STATION_PREFIX:-$(dirname "$ENGINE_DIR")}"
# shellcheck source=engine.env
. "$ENGINE_DIR/engine.env"

VENV="${VLLM_PREFIX:-$PREFIX/venv}"
MODELS="${MODEL_BASE:-$PREFIX/models}"
LOGDIR="$PREFIX/logs"
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$LOGDIR/serve-$TS.log"
STATUS="$LOGDIR/status.txt"
echo "$LOG" > "$LOGDIR/latest.txt"
echo starting > "$STATUS"
log() { printf '%s\n' "$*" | tee -a "$LOG"; }

TARGET_NAME="${TARGET_NAME:-$TARGET_DIR}"
TARGET_QUANT="${TARGET_QUANT:-modelopt}"
MAX_SEQS="${MAX_SEQS:-1}"
PREFILL_BATCH="${PREFILL_BATCH:-2048}"
VLLM_VERSION_INSTALLED=$(basename "$(ls -d "$VENV"/lib/python3*/site-packages/vllm-*.dist-info 2>/dev/null | head -1)" .dist-info)
VLLM_VERSION_INSTALLED="${VLLM_VERSION_INSTALLED#vllm-}"

log "=== $(date '+%F %T') Qwen Station Engine $ENGINE_VERSION, vLLM ${VLLM_VERSION_INSTALLED:-?}, Target $TARGET_NAME"
[ -x "$VENV/bin/vllm" ] || { log "ERROR: vLLM is missing in $VENV. Run the setup again."; echo "exited 4" > "$STATUS"; exit 4; }
[ -f "$MODELS/$TARGET_NAME/config.json" ] || { log "ERROR: The model is missing: $MODELS/$TARGET_NAME"; echo "exited 4" > "$STATUS"; exit 4; }

# A server left over from an earlier run of this installation would hold the port and the VRAM.
# Only this installation's: other vLLM servers on the machine are left alone.
PATTERN="vllm serve $(printf '%s' "$MODELS" | sed 's/[].[*^$]/\\&/g')/"
if pgrep -f "$PATTERN" >/dev/null; then
  log "--- Stopping a leftover server"
  for pid in $(pgrep -f "$PATTERN"); do
    group=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$group" ] && [ "$group" != "$$" ] && kill -TERM -- -"$group" 2>/dev/null
  done
  for _ in $(seq 1 30); do pgrep -f "$PATTERN" >/dev/null || break; sleep 1; done
  pgrep -f "$PATTERN" | xargs -r kill -KILL 2>/dev/null
fi

# Local patches for vLLM. Each script is idempotent and skips versions it does not apply to.
for patch in "$ENGINE_DIR"/patches/*.py; do
  VLLM_PREFIX="$VENV" "$VENV/bin/python" "$patch" 2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" != 0 ]; then log "Patch failed - aborting."; echo patch-failed > "$STATUS"; exit 3; fi
done

# Fixed KV pool instead of the automatic one: an oversized pool pages under Windows/WSL (54 against 243 tok/s).
KV_ARGS=(--kv-cache-dtype "${KV_DTYPE:-fp8}" --kv-cache-memory-bytes "${KV_BYTES:-10350000000}")
case "${KV_GROUP_SIZE:-8}" in
  8) export VLLM_QWEN38_KV_GROUP_SIZE=8 ;;   # the drafter's five layers would otherwise force groups of five
  auto) unset VLLM_QWEN38_KV_GROUP_SIZE ;;
  *) log "KV_GROUP_SIZE must be 8 or auto"; echo "exited 2" > "$STATUS"; exit 2 ;;
esac

# Decode kernel: with FlashInfer 0.7 (vLLM 0.32 and later) XQA is the faster one at every context size
# (129K: 214 to 289 tok/s against about 150 native). On older versions it collapses with long context.
case "$VLLM_VERSION_INSTALLED" in
  0.2*|0.30.*|0.31.*) XQA_DEFAULT=0 ;;
  *) XQA_DEFAULT=1 ;;
esac
if [ "${XQA_DECODE:-$XQA_DEFAULT}" != 1 ]; then
  KV_ARGS+=(--attention-config.use_trtllm_attention false)
fi
# vLLM 0.33: the drafter's attention must not run on XQA there (acceptance length 3.0 instead of 4.5).
case "$VLLM_VERSION_INSTALLED" in
  0.33.*) [ "${XQA_DECODE:-$XQA_DEFAULT}" = 1 ] && KV_ARGS+=(--attention-config.backend_per_kind.sliding_window TRITON_ATTN) ;;
esac
# FlashInfer 0.7 autotune runs out of GPU memory once weights and the full KV pool are loaded. Skipping the
# op keeps the stored values in use; the block at the end puts the verified file in place.
if [ "$XQA_DEFAULT" = 1 ]; then
  export VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS="${VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS-fp4_gemm}"
fi

# SPEC_MODE=mtp uses the MTP head inside the checkpoint instead of the drafter (needs KV_GROUP_SIZE=auto).
SPEC_JSON='{"method":"dflash","model":"'"$MODELS/$DRAFTER_DIR"'","quantization":"compressed-tensors","num_speculative_tokens":7}'
if [ "${SPEC_MODE:-dflash}" = mtp ]; then
  SPEC_JSON='{"method":"mtp","num_speculative_tokens":'"${SPEC_TOKENS:-2}"'}'
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Parallel jobs of the kernel build on the first start. Each compiler process takes several GB of RAM,
# so: one job per 8 GB of free memory, at most half the cores, at most 8.
if [ -z "${MAX_JOBS:-}" ]; then
  MAX_JOBS=$(awk '/MemAvailable/ { print int($2 / 1048576 / 8) }' /proc/meminfo 2>/dev/null)
  CORES=$(( $(nproc 2>/dev/null || echo 2) / 2 ))
  [ "${MAX_JOBS:-0}" -gt "$CORES" ] && MAX_JOBS=$CORES
  [ "${MAX_JOBS:-0}" -gt 8 ] && MAX_JOBS=8
  [ "${MAX_JOBS:-0}" -lt 1 ] && MAX_JOBS=1
fi
export MAX_JOBS
export CUDA_HOME="$PREFIX/cuda-sdk/nvidia/cu13"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_INDEX:-0}"
export VLLM_NO_USAGE_STATS=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V2_MODEL_RUNNER=1
# vLLM switches pinned memory off under WSL2; the V2 runner needs it.
grep -qi microsoft /proc/version 2>/dev/null && export VLLM_WSL2_ENABLE_PIN_MEMORY=1

# Verified kernel tuning, put in place before vLLM reads its cache (patch-vllm-autotune-seed.py). Without it
# every new configuration, and so every new context size, would start with a fresh and unreliable measurement.
GOLDEN_DIR="$ENGINE_DIR/autotune"
FLASHINFER_VERSION=$(basename "$(ls -d "$VENV"/lib/python3*/site-packages/flashinfer_python-*.dist-info 2>/dev/null | head -1)" .dist-info)
FLASHINFER_VERSION="${FLASHINFER_VERSION#flashinfer_python-}"
SEED="$GOLDEN_DIR/flashinfer-$FLASHINFER_VERSION-sm120-qwen38-gittensor-dflash.json"
if [ "${AUTOTUNE_PIN:-1}" = 1 ] && [ -f "$SEED" ]; then
  export VLLM_QWEN38_AUTOTUNE_SEED="$SEED"
fi

[ -n "${EXTRA_ARGS:-}" ] && log "--- EXTRA_ARGS: $EXTRA_ARGS"
log "=== $(date '+%F %T') Starting $VENV/bin/vllm serve (context ${CTX:-262144}, KV ${KV_BYTES:-10350000000} bytes, $MAX_SEQS slot(s))"
echo running > "$STATUS"

# shellcheck disable=SC2086
"$VENV/bin/vllm" serve "$MODELS/$TARGET_NAME" \
  --host "${VLLM_HOST:-127.0.0.1}" --port "${VLLM_PORT:-8000}" --served-model-name qwen38-gittensor-dflash \
  --quantization "$TARGET_QUANT" --linear-backend auto "${KV_ARGS[@]}" \
  --speculative-config "$SPEC_JSON" \
  --max-model-len "${CTX:-262144}" --gpu-memory-utilization "${GPU_UTIL:-0.92}" \
  --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$PREFILL_BATCH" --long-prefill-token-threshold "$PREFILL_BATCH" \
  --enable-prefix-caching --enable-prompt-tokens-details --cudagraph-capture-sizes 1 2 4 8 16 32 64 128 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --default-chat-template-kwargs "{\"reasoning_effort\":\"${EFFORT:-medium}\"}" \
  ${EXTRA_ARGS:-} 2>&1 | tee -a "$LOG" | while IFS= read -r line; do
  printf '%s\n' "$line"
  case "$line" in
    *"Using FlashInfer autotune cache file: "*)
      TUNE_FILE="${line##*autotune cache file: }"; TUNE_FILE="${TUNE_FILE%$'\r'}" ;;
    *"Application startup complete."*)
      echo ready > "$STATUS"
      # Safety net for the seed above (a vLLM whose code the seed patch does not fit): the autotuner picks
      # kernel settings by timing them, and a bad pick costs 40 % decode speed. Put the verified settings in
      # place; vLLM has read the file already, so this takes effect from the next start of this configuration.
      if [ "${AUTOTUNE_PIN:-1}" = 1 ] && [ -n "${TUNE_FILE:-}" ]; then
        case "$TUNE_FILE" in
          */flashinfer_autotune_cache/*/120f/*/autotune_configs*.json)
            TUNE_VERSION="${TUNE_FILE#*/flashinfer_autotune_cache/}"; TUNE_VERSION="${TUNE_VERSION%%/*}"
            GOLDEN="$GOLDEN_DIR/flashinfer-$TUNE_VERSION-sm120-qwen38-gittensor-dflash.json"
            if [ -f "$GOLDEN" ] && ! cmp -s "$GOLDEN" "$TUNE_FILE"; then
              [ -f "$TUNE_FILE" ] && cp -p "$TUNE_FILE" "$TUNE_FILE.tuned-$TS"
              mkdir -p "$(dirname "$TUNE_FILE")" && cp "$GOLDEN" "$TUNE_FILE" &&
                echo "=== AUTOTUNE: Kernel tuning for this configuration replaced by the verified values. Full speed from the next start." | tee -a "$LOG"
            fi ;;
        esac
      fi ;;
  esac
done
CODE=${PIPESTATUS[0]}
log "=== $(date '+%F %T') Server ended, exit code $CODE"
echo "exited $CODE" > "$STATUS"
exit "$CODE"
