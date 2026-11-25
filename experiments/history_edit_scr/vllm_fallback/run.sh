#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
experiment=$(cd "$here/.." && pwd)
repo=$(cd "$here/../../.." && pwd)
source "$experiment/pins.env"
MODEL_REVISION=${CLM_MODEL_REVISION:-$MODEL_REVISION}
py=${PYTHON:-python}
action=${1:-}
mode=${2:-}
out_root=${3:-}

if [[ $# != 3 || ( $action != serve && $action != smoke && $action != report ) || ( $mode != off && $mode != on ) ]]; then
  echo "Usage: bash experiments/history_edit_scr/vllm_fallback/run.sh {serve|smoke|report} {off|on} OUT_ROOT" >&2
  exit 2
fi

port=${VLLM_OFF_PORT:-32000}
enabled=0
if [[ $mode == on ]]; then
  port=${VLLM_ON_PORT:-32001}
  enabled=1
fi
out="$out_root/$mode"
mkdir -p "$out"

case "$action" in
  serve)
    : "${CLM_MODEL_DIR:?stage the pinned model on local scratch and export CLM_MODEL_DIR}"
    model_dir=$(PYTHONPATH="$experiment${PYTHONPATH:+:$PYTHONPATH}" "$py" - "$CLM_MODEL_DIR" <<'PY'
import sys
from storage import model_ready
print(model_ready(sys.argv[1]))
PY
)
    set -o noclobber
    exec >"$out/server.log" 2>&1
    # FlashInfer's shared JIT cache may still contain include paths from a removed Conda env.
    # Keep this run's regenerated kernels on local scratch, outside published eval artifacts.
    export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$(dirname "$(dirname "$model_dir")")}"
    mkdir -p "$FLASHINFER_WORKSPACE_BASE"
    "$py" - <<'PY'
from importlib.util import find_spec
from pathlib import Path
import sys

spec = find_spec("flashinfer")
if spec is None or not spec.submodule_search_locations:
    sys.exit("FlashInfer is unavailable in the active Python environment")
header = Path(next(iter(spec.submodule_search_locations))) / "data/include/flashinfer/flat/prefill/prefill_kernel_delta_rule_sm90.cuh"
if not header.is_file():
    sys.exit(f"FlashInfer SM90 GDN header missing from active environment: {header}")
print(f"FlashInfer SM90 GDN header: {header}", flush=True)
PY
    export KVREUSE_ENABLED=$enabled KVREUSE_MAX_BLOCKS=6 KVREUSE_SSM_MODE=fork
    export KVREUSE_SIDE_SESSIONS=${KVREUSE_SIDE_SESSIONS:-4}
    export KVREUSE_MAX_SESSIONS=${KVREUSE_MAX_SESSIONS:-$KVREUSE_SIDE_SESSIONS}
    export KVREUSE_SIDE_TOKENS=${KVREUSE_SIDE_TOKENS:-12000}
    export KVREUSE_TRACE=1
    "$py" - "$repo" "$MODEL_ID" "$MODEL_REVISION" "$model_dir" "$mode" <<'PY'
import json
import os
import subprocess
import sys
from importlib.metadata import version

import torch

repo, model, revision, snapshot, mode = sys.argv[1:]
vllm = version("vllm")
if vllm != "0.19.1":
    sys.exit(f"Expected vLLM 0.19.1, found {vllm}")
if not torch.cuda.is_available():
    sys.exit("No CUDA GPU is visible to Torch")
print("[history-edit-vllm-config] " + json.dumps({
    "commit": subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip(),
    "backend": "vllm", "mode": mode, "vllm": vllm, "model": model,
    "model_revision": revision, "model_source": os.environ.get("CLM_MODEL_SOURCE", "huggingface"),
    "source_revision": os.environ.get("CLM_SOURCE_REVISION", revision),
    "snapshot": snapshot, "torch": torch.__version__,
    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
    "python_prefix": sys.prefix,
    "flashinfer_workspace_base": os.environ["FLASHINFER_WORKSPACE_BASE"],
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "kvreuse_env": {key: value for key, value in os.environ.items() if key.startswith("KVREUSE_")},
}, sort_keys=True, default=str), flush=True)
PY
    exec "$py" -m vllm_scr.serve "$model_dir" \
      --served-model-name qwen36-27b --enable-prefix-caching \
      --language-model-only --reasoning-parser qwen3 \
      --host 127.0.0.1 --port "$port" --dtype bfloat16 \
      --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.80}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" --max-num-seqs 1
    ;;
  smoke)
    "$py" "$repo/suffix_cache_reuse/vllm/tests/smoke.py" \
      --port "$port" --model qwen36-27b --rid-prefix "vllm-$mode" | tee "$out/smoke.log"
    ;;
  report)
    "$py" "$repo/suffix_cache_reuse/vllm/analysis/scr_report.py" "$out/server.log" \
      --csv "$out/requests.csv" --json "$out/summary.json" | tee "$out/report.log"
    "$py" - "$out/summary.json" "$mode" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
mode = sys.argv[2]
if summary["requests"] < 4 or summary["edited_requests"] < 1:
    sys.exit("Official smoke did not produce the expected edited requests")
if mode == "on" and summary["edited_requests_relocated"] < 1:
    sys.exit("SCR on did not relocate an edited request; inspect server.log")
if mode == "off" and summary.get("relocated", 0) != 0:
    sys.exit("SCR off unexpectedly relocated tokens")
print(f"vLLM SCR {mode}: official smoke and relocation accounting passed")
PY
    ;;
esac
