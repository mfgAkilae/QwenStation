"""Two local fixes for vLLM so that a compressed-tensors FP8 DFlash drafter loads and runs.

Part 1: The drafter gets its own quantization config (get_draft_quant_config), but without
        packed_modules_mapping. qkv_proj and gate_up_proj then stay unquantized and loading *.weight_scale
        fails ("'MergedColumnParallelLinear' object has no attribute 'data'").
Part 2: The fused context-KV precomputation calls F.linear on the raw qkv_proj.weight; with FP8 that ends in
        "expected mat1 and mat2 to have the same dtype". The fix dequantizes with the per-channel scale.
        Only needed before vLLM 0.32.0.

Idempotent. Backup of the original: qwen3_dflash.py.pre-dflash-quant.bak (never overwritten).
Undo: cp <file>.pre-dflash-quant.bak <file>
"""
import glob, os, pathlib, shutil, sys
paths = glob.glob(os.environ.get('VLLM_PREFIX', '/opt/vllm') + '/lib/python3*/site-packages/vllm/model_executor/models/qwen3_dflash.py')
if not paths:
    sys.exit('qwen3_dflash.py not found')
path = paths[0]
src = open(path).read()
MARK1 = 'draft quant config fused-layer mapping (local fix)'
MARK2 = 'FP8 drafter support for the fused context-KV GEMM (local fix)'
ANCHOR1 = "        self.quant_config = get_draft_quant_config(vllm_config)\n"
PATCH1 = ANCHOR1 + """        # --- draft quant config fused-layer mapping (local fix) ---
        # The draft's own quant config never receives the model's
        # packed_modules_mapping (configure_quant_config only touches the
        # target's vllm_config.quant_config), so compressed-tensors cannot
        # match qkv_proj / gate_up_proj against per-projection targets.
        if self.quant_config is not None:
            self.quant_config.packed_modules_mapping.update(
                {
                    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                    "gate_up_proj": ["gate_proj", "up_proj"],
                }
            )
"""
ANCHOR2 = """        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
"""
PATCH2 = """        # --- FP8 drafter support for the fused context-KV GEMM (local fix) ---
        # Runs before process_weights_after_loading, so weight is still the
        # raw [out, in] checkpoint layout and weight_scale is per-channel
        # [out, 1] (or a per-tensor scalar) for compressed-tensors FP8.
        kv_weights = []
        for a in layers_attn:
            w = a.qkv_proj.weight[a.q_size :]
            if w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                scale = getattr(a.qkv_proj, "weight_scale", None)
                if scale is not None and scale.dim() >= 1 and scale.shape[0] == a.qkv_proj.weight.shape[0]:
                    scale = scale[a.q_size :].reshape(-1, 1).float()
                elif scale is not None:
                    scale = scale.float()
                else:
                    scale = 1.0
                w = (w.float() * scale).to(a.k_norm.weight.dtype)
            kv_weights.append(w)
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
"""
changed = False
if MARK1 not in src:
    if src.count(ANCHOR1) != 1:
        sys.exit('PATCH1_ANCHOR_NOT_FOUND: the installed vLLM version does not fit (' + path + ')')
    src = src.replace(ANCHOR1, PATCH1); changed = True
# From vLLM 0.32.0 on upstream runs quantized projections through their quant method itself
# (_build_context_kv_buffers), so part 2 is not needed there. Part 1 is still missing upstream
# (checked 2026-11-16).
UPSTREAM2 = 'Run each projection through its quant method'
if MARK2 not in src and UPSTREAM2 not in src:
    if src.count(ANCHOR2) != 1:
        sys.exit('PATCH2_ANCHOR_NOT_FOUND: the installed vLLM version does not fit (' + path + ')')
    src = src.replace(ANCHOR2, PATCH2); changed = True
if not changed:
    print('PATCH_ALREADY_APPLIED', path); sys.exit(0)
if not os.path.exists(path + '.pre-dflash-quant.bak'):
    shutil.copy2(path, path + '.pre-dflash-quant.bak')
open(path, 'w').write(src)
for pyc in pathlib.Path(path).parent.glob('__pycache__/qwen3_dflash.*.pyc'):
    pyc.unlink()
print('PATCH_APPLIED', path, '(part 2: %s)' % ('solved upstream' if UPSTREAM2 in src else 'local'))
