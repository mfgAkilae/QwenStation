#!/usr/bin/env bash
# Installs everything the engine needs below one prefix: Python, vLLM, the CUDA compiler for the
# kernel JIT, the local patches and the model weights. Every step is idempotent, so a failed or
# interrupted run can simply be started again.
#
#   provision.sh all              full installation
#   provision.sh <step> [...]     single steps: system uv venv vllm cuda patches kernels target drafter verify
#   provision.sh model-unsloth    optional second target
#
# Lines starting with "::" are machine-readable progress for Qwen Station:
#   ::step <id> start|done|skip|fail      ::progress <id> <bytes> <total>      ::hint <id> <what the user can do>
set -uo pipefail

ENGINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${QWEN_STATION_PREFIX:-$(dirname "$ENGINE_DIR")}"
# shellcheck source=engine.env
. "$ENGINE_DIR/engine.env"

VENV="$PREFIX/venv"
UV="$PREFIX/bin/uv"
export UV_CACHE_DIR="$PREFIX/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PREFIX/python"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export DEBIAN_FRONTEND=noninteractive
mkdir -p "$PREFIX/bin" "$PREFIX/models" "$PREFIX/logs" "$PREFIX/cache"

say() { printf '%s\n' "$*"; }

step_system() {
  if command -v c++ >/dev/null && command -v curl >/dev/null && command -v make >/dev/null; then
    say "Compiler and curl are present."; return 0
  fi
  # Something is missing. Installing it is the one thing here that needs root rights.
  local install hint
  if command -v apt-get >/dev/null; then
    install="apt-get update -q && apt-get install -y -q --no-install-recommends build-essential curl ca-certificates"
    hint="sudo apt-get update && sudo apt-get install -y build-essential curl ca-certificates"
  elif command -v dnf >/dev/null; then
    install="dnf install -y gcc-c++ make curl ca-certificates"; hint="sudo $install"
  elif command -v pacman >/dev/null; then
    install="pacman -Sy --noconfirm --needed base-devel curl ca-certificates"; hint="sudo $install"
  elif command -v zypper >/dev/null; then
    install="zypper --non-interactive install gcc-c++ make curl ca-certificates"; hint="sudo $install"
  else
    say "A C++ compiler, make or curl is missing, and no known package manager was found."
    printf '::hint system %s\n' "A C++ compiler, make or curl is missing. Install them with your package manager, then run the setup again."
    return 1
  fi
  if [ "$(id -u)" = 0 ]; then
    sh -c "$install"; return
  fi
  if command -v sudo >/dev/null && sudo -n true 2>/dev/null; then
    sudo -n sh -c "$install" && return 0
  fi
  if command -v pkexec >/dev/null && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    say "A compiler is missing. The system asks for your password to install it."
    pkexec sh -c "$install" && return 0
  fi
  say "A C++ compiler, make or curl is missing, and installing them needs root rights."
  printf '::hint system %s\n' "A compiler is missing. Run this once in a terminal, then start the setup again: $hint"
  return 1
}

step_uv() {
  if [ -x "$UV" ] && "$UV" --version | grep -q "$UV_VERSION"; then say "uv $UV_VERSION is present."; return 0; fi
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR="$PREFIX/bin" UV_NO_MODIFY_PATH=1 sh
}

step_venv() {
  # A Python of uv's own, never the system's: it brings its headers, which the GPU kernel build needs
  # (a distribution's python3 comes without them), and it is the same on every distribution.
  if [ -x "$VENV/bin/python" ] && readlink -f "$VENV/bin/python" | grep -q "^$PREFIX/python/"; then
    say "The Python environment is present."; return 0
  fi
  rm -rf "$VENV"
  "$UV" venv "$VENV" --python "$PYTHON_VERSION" --managed-python
}

step_vllm() {
  if [ -x "$VENV/bin/vllm" ] && ls "$VENV"/lib/python3*/site-packages | grep -q "^vllm-$VLLM_VERSION\.dist-info$"; then
    say "vLLM $VLLM_VERSION is installed."; return 0
  fi
  "$UV" pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION" --torch-backend=auto
}

step_cuda() {
  local sdk="$PREFIX/cuda-sdk/nvidia/cu13" lib name driver
  if [ -x "$sdk/bin/nvcc" ]; then
    say "The CUDA compiler is present."
  else
    # shellcheck disable=SC2086
    "$UV" pip install --python "$VENV/bin/python" --target "$PREFIX/cuda-sdk" $CUDA_SDK_PACKAGES || return 1
    [ -x "$sdk/bin/nvcc" ] || return 1
  fi
  # The pip packages are a runtime, not a toolkit: the linker looks in lib64 and wants unversioned names
  # (-lcudart, -lcublas, -lcuda). Without these links the first kernel build fails at its link step.
  ln -sfn lib "$sdk/lib64"
  for lib in "$VENV"/lib/python3*/site-packages/nvidia/cu13/lib/*.so*; do   # the libraries torch brought
    [ -e "$lib" ] || continue
    name=$(basename "$lib")
    [ -e "$sdk/lib/$name" ] || ln -s "$lib" "$sdk/lib/$name"
  done
  for lib in "$sdk"/lib/lib*.so.*; do
    [ -e "$lib" ] || continue
    name=$(basename "$lib")
    [ -e "$sdk/lib/${name%%.so.*}.so" ] || ln -s "$name" "$sdk/lib/${name%%.so.*}.so"
  done
  # libcuda belongs to the GPU driver: WSL mounts it, a native Linux has it in the system library folder (Arch:
  # /usr/lib), NixOS below /run/opengl-driver/lib. Last, whatever the loader's cache knows.
  driver=""
  for lib in /usr/lib/wsl/lib/libcuda.so /usr/lib/wsl/lib/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so \
      /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib64/libcuda.so.1 /usr/lib/libcuda.so.1 \
      /run/opengl-driver/lib/libcuda.so.1; do
    if [ -e "$lib" ]; then driver=$lib; break; fi
  done
  if [ -z "$driver" ]; then
    driver=$( (ldconfig -p || /sbin/ldconfig -p) 2>/dev/null | awk '$1 == "libcuda.so.1" && /x86-64/ { print $NF; exit }')
  fi
  if [ -z "$driver" ]; then say "libcuda.so of the NVIDIA driver is missing. Is the driver installed?"; return 1; fi
  ln -sfn "$driver" "$sdk/lib/libcuda.so"
}

step_patches() {
  local patch
  for patch in "$ENGINE_DIR"/patches/*.py; do
    VLLM_PREFIX="$VENV" "$VENV/bin/python" "$patch" || return 1
  done
}

step_kernels() {
  # Prebuilt FlashInfer kernels for this GPU generation. Without them the first start compiles each kernel
  # itself, one after the other: more than half an hour instead of two minutes (measured 2026-11-17).
  # `vllm download-kernels` would do the same, but it needs pip (a uv venv has none) and fetches all six
  # GPU generations. A failure is not an error: the compiler from step "cuda" is the fallback.
  local version
  version=$(basename "$(ls -d "$VENV"/lib/python3*/site-packages/flashinfer_python-*.dist-info 2>/dev/null | head -1)" .dist-info)
  version="${version#flashinfer_python-}+$KERNEL_CUDA"
  if ls "$VENV"/lib/python3*/site-packages | grep -q "^flashinfer_jit_cache_${KERNEL_ARCH}-${version}\.dist-info$"; then
    say "Prebuilt kernels ($KERNEL_ARCH, $version) are installed."; return 0
  fi
  "$UV" pip install --python "$VENV/bin/python" --no-deps --index-url "$KERNEL_INDEX" \
    "flashinfer-jit-cache==$version" "flashinfer-jit-cache-$KERNEL_ARCH==$version" ||
    say "No prebuilt kernels for $KERNEL_ARCH and FlashInfer $version; the first start compiles them itself."
  return 0
}

download_model() {
  local id="$1" repo="$2" rev="$3" dir="$4" total="$5"
  local dest="$PREFIX/models/$dir"
  if [ -f "$dest/.complete-$rev" ]; then say "$dir is complete."; return 0; fi
  mkdir -p "$dest"
  ( while :; do
      printf '::progress %s %s %s\n' "$id" "$(du -sb "$dest" 2>/dev/null | cut -f1)" "$total"
      sleep 3
    done ) &
  local reporter=$!
  "$VENV/bin/hf" download "$repo" --revision "$rev" --local-dir "$dest"
  local code=$?
  kill "$reporter" 2>/dev/null; wait "$reporter" 2>/dev/null
  [ "$code" = 0 ] || return "$code"
  printf '::progress %s %s %s\n' "$id" "$total" "$total"
  touch "$dest/.complete-$rev"
}

step_target()  { download_model target  "$TARGET_REPO"  "$TARGET_REV"  "$TARGET_DIR"  "$TARGET_BYTES"; }
step_drafter() { download_model drafter "$DRAFTER_REPO" "$DRAFTER_REV" "$DRAFTER_DIR" "$DRAFTER_BYTES"; }
step_unsloth() { download_model unsloth "$UNSLOTH_REPO" "$UNSLOTH_REV" "$UNSLOTH_DIR" "$UNSLOTH_BYTES"; }

step_verify() {
  "$VENV/bin/python" -c "import vllm, torch; print('vLLM', vllm.__version__, '| torch', torch.__version__, '| CUDA visible:', torch.cuda.is_available())" || return 1
  [ -f "$PREFIX/models/$TARGET_DIR/config.json" ] && [ -f "$PREFIX/models/$DRAFTER_DIR/config.json" ] || {
    say "Model files are missing."; return 1; }
  local models="\"$TARGET_DIR\", \"$DRAFTER_DIR\""
  [ -f "$PREFIX/models/$UNSLOTH_DIR/.complete-$UNSLOTH_REV" ] && models="$models, \"$UNSLOTH_DIR\""
  printf '{"engine": "%s", "vllm": "%s", "models": [%s]}\n' "$ENGINE_VERSION" "$VLLM_VERSION" "$models" > "$PREFIX/installed.json"
  say "Installation complete: $PREFIX"
}

run() {
  local id="$1"
  printf '::step %s start\n' "$id"
  if "step_$id"; then printf '::step %s done\n' "$id"; else printf '::step %s fail\n' "$id"; exit 1; fi
}

[ "$#" -gt 0 ] || set -- all
for request in "$@"; do
  case "$request" in
    all) for id in system uv venv vllm cuda patches kernels target drafter verify; do run "$id"; done ;;
    model-unsloth) run unsloth; run verify ;;
    system|uv|venv|vllm|cuda|patches|kernels|target|drafter|verify) run "$request" ;;
    *) say "Unknown step: $request"; exit 2 ;;
  esac
done
