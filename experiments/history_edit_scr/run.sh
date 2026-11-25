#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
scr="$repo/suffix_cache_reuse/sglang"
source "$here/pins.env"
py=${PYTHON:-python}
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

usage() {
  cat <<'EOF'
Usage from the CLM checkout:
  bash experiments/history_edit_scr/run.sh download
  bash experiments/history_edit_scr/run.sh snapshot
  bash experiments/history_edit_scr/run.sh {serve|smoke|contextbench|bcp} {off|on} RUN_DIR
contextbench requires CB_REPO. bcp requires BCP_REPO, BCP_TASK_ROOT, BCP_TASK_LIST,
and BCP_PROVENANCE. See README.md for the setup and the required provenance fields.
EOF
}

snapshot() {
  PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}" "$py" - "$MODEL_ID" "$MODEL_REVISION" <<'PY'
import os
import sys
from storage import model_ready

path = os.environ.get("CLM_MODEL_DIR")
if not path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True)
print(model_ready(path))
PY
}

action=${1:-help}
case "$action" in
  download)
    if [[ -n ${CLM_MODEL_DIR:-} ]]; then
      target=$(PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}" "$py" -c \
        'import sys; from storage import model_location; print(model_location(sys.argv[1]))' "$CLM_MODEL_DIR")
      exec hf download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$target"
    fi
    exec hf download "$MODEL_ID" --revision "$MODEL_REVISION"
    ;;
  snapshot) snapshot; exit ;;
  serve|smoke|contextbench|bcp) ;;
  help|-h|--help) usage; exit ;;
  *) usage >&2; exit 2 ;;
esac

if [[ $# != 3 || ( $2 != off && $2 != on ) ]]; then
  usage >&2
  exit 2
fi
mode=$2
mkdir -p "$3/$mode"
out=$(cd "$3/$mode" && pwd)
port=${OFF_PORT:-31000}
enabled=0
if [[ $mode == on ]]; then
  port=${ON_PORT:-31001}
  enabled=1
fi
api_base="http://127.0.0.1:$port/v1"

case "$action" in
  serve)
    model_dir=$(snapshot)
    mem_fraction=${MEM_FRACTION_STATIC:-0.75}
    # One slot would strand each new conversation until the LRU drops the old one.
    export KVREUSE_ENABLED=$enabled KVREUSE_MAX_BLOCKS=6 KVREUSE_SSM_MODE=fork
    export KVREUSE_SIDE_SESSIONS=${KVREUSE_SIDE_SESSIONS:-4}
    export KVREUSE_MAX_SESSIONS=${KVREUSE_MAX_SESSIONS:-$KVREUSE_SIDE_SESSIONS}
    export KVREUSE_SIDE_TOKENS=${KVREUSE_SIDE_TOKENS:-12000}
    export KVREUSE_TRACE=1 KVREUSE_DEBUG=1
    args=(--host 127.0.0.1 --dtype bfloat16 --mem-fraction-static "$mem_fraction" --max-running-requests 1
          --reasoning-parser qwen3)
    set -o noclobber
    exec >"$out/server.log" 2>&1
    PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}" "$py" - "$repo" "$CLM_BASE" "$SGLANG_VERSION" "$MODEL_ID" "$MODEL_REVISION" "$model_dir" "${args[@]}" <<'PY'
import json
import os
import subprocess
import sys
from importlib.metadata import version

import torch
from probe import tokenizer_files
from suffix_cache_reuse.config import apply_defaults

repo, clm_base, expected_sglang, model_id, revision, snapshot, *args = sys.argv[1:]
sglang = version("sglang")
if sglang != expected_sglang:
    sys.exit(f"Expected SGLang {expected_sglang}, found {sglang}")
if not torch.cuda.is_available():
    sys.exit("No CUDA GPU is visible to Torch")
apply_defaults()
gpu = torch.cuda.get_device_properties(0)
config = {
    "commit": subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip(),
    "clm_base": clm_base,
    "model": model_id,
    "model_revision": revision,
    "tokenizer_revision": revision,
    "tokenizer_files_sha256": tokenizer_files(snapshot),
    "snapshot": snapshot,
    "sglang": sglang,
    "torch": str(torch.__version__),
    "cuda": torch.version.cuda,
    "transformers": version("transformers"),
    "gpu": {"name": gpu.name, "memory_bytes": gpu.total_memory},
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "args": args,
    "kvreuse_env": {key: value for key, value in os.environ.items()
                    if key.startswith("KVREUSE_") and key != "KVREUSE_ENABLED"},
}
print("[history-edit-config] " + json.dumps(config, sort_keys=True), flush=True)
PY
    nvidia-smi
    cd "$scr"
    MODEL="$model_dir" PORT="$port" PYTHON="$py" exec bash examples/serve_qwen36.sh "${args[@]}"
    ;;
  smoke)
    [[ -f "$out/server.log" ]] || { echo "Missing $out/server.log" >&2; exit 2; }
    [[ ! -e "$out/smoke.log" ]] || { echo "Already exists: $out/smoke.log" >&2; exit 2; }
    "$py" "$scr/tests/smoke.py" --port "$port" --rid-prefix "smoke-$mode" | tee "$out/smoke.log"
    if [[ $mode == on ]]; then
      "$py" "$scr/tests/regression.py" --port "$port" --log "$out/server.log" | tee "$out/regression.log"
    fi
    ;;
  contextbench)
    : "${CB_REPO:?point to the pinned ContextBench checkout}"
    [[ $(git -C "$CB_REPO" rev-parse HEAD) == "$CB_REVISION" ]] || { echo "ContextBench revision differs from pins.env" >&2; exit 2; }
    task="$CB_REPO/tasks/kv-store-R1000-seed42"
    [[ -f "$task/task.toml" ]] || { echo "Missing original task: $task" >&2; exit 2; }
    exec harbor trials start -p "$task" -e "${HARBOR_ENV:-docker}" \
      -a clm_contextbench:ClmStream -m openai/qwen36-27b \
      --agent-kwarg "api_base=$api_base" \
      --trial-name "contextbench-kv-store-R1000-seed42-$mode" --trials-dir "$out/contextbench"
    ;;
  bcp)
    : "${BCP_REPO:?point to the pinned BrowseComp-Plus checkout}"
    : "${BCP_TASK_ROOT:?point to the prepared Harbor task root}"
    : "${BCP_TASK_LIST:?point to a list of 8–16 original task directory names, one per line}"
    : "${BCP_PROVENANCE:?point to a record of the index, retrieval tools and judge}"
    [[ $(git -C "$BCP_REPO" rev-parse HEAD) == "$BCP_REVISION" ]] || { echo "BrowseComp-Plus revision differs from pins.env" >&2; exit 2; }
    "$py" "$here/bcp_manifest.py" "$BCP_TASK_ROOT" "$BCP_TASK_LIST" "$BCP_PROVENANCE" "$out/bcp-manifest.json"
    other_mode=on
    if [[ $mode == on ]]; then other_mode=off; fi
    other="$out/../$other_mode/bcp-manifest.json"
    if [[ -f $other ]]; then cmp "$out/bcp-manifest.json" "$other"; fi
    while IFS= read -r task || [[ -n "$task" ]]; do
      mkdir -p "$out/bcp-server"
      size=$(wc -c <"$out/server.log")
      MODEL=openai/qwen36-27b API_BASE="$api_base" TRIALS_DIR="$out/bcp" PYTHON="$py" \
        bash "$repo/clm/examples/run_harbor.sh" "$BCP_TASK_ROOT/$task" bcp --trial-name "bcp-$task-$mode"
      tail -c "+$((size + 1))" "$out/server.log" > "$out/bcp-server/$task.log"
      "$py" "$scr/analysis/scr_report.py" "$out/bcp-server/$task.log" --num-tasks 1 \
        --csv "$out/bcp-server/$task.csv" --json "$out/bcp-server/$task.json"
    done <"$BCP_TASK_LIST"
    ;;
esac
