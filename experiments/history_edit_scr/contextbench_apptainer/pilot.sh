#!/usr/bin/env bash
# Run one original ContextBench KV Store task with the vLLM SCR fallback.
set -euo pipefail

if [[ $# != 1 || $1 == -h || $1 == --help ]]; then
  echo "Usage: CLM_MODEL_DIR=/tmp/clm/.../model/modelscope-master-v1 bash $0 RUN_ID" >&2
  exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
experiment=$(cd "$here/.." && pwd)
repo=$(cd "$experiment/../.." && pwd)
source "$experiment/pins.env"
source "$experiment/vllm_fallback/modelscope.env"

run_id=$1
cb_repo=${CB_REPO:-"$(dirname "$repo")/ContextBench"}
image_cache=${CLM_CB_IMAGE_CACHE:-/tmp/clm/contextbench-image-cache}
apptainer_home=${CLM_APPTAINER_HOME:-"$HOME/.local/opt/apptainer-1.5.5"}
port=${VLLM_OFF_PORT:-32000}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-40960}
export VLLM_ENABLE_AUTO_TOOL_CHOICE=1
export VLLM_TOOL_CALL_PARSER=qwen3_coder
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-/tmp/clm/apptainer-cache}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-/tmp/clm/apptainer-tmp}
export APPTAINER_NO_MOUNT=proc
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
export PATH="$apptainer_home/bin:$PATH"
export PYTHONPATH="$repo/clm:$cb_repo${PYTHONPATH:+:$PYTHONPATH}"

[[ $VLLM_MAX_MODEL_LEN =~ ^[0-9]+$ ]] && (( VLLM_MAX_MODEL_LEN >= 40960 )) || {
  echo "VLLM_MAX_MODEL_LEN must be at least 40960 for the paper's 32768 + 8192 budget" >&2
  exit 2
}
: "${CLM_MODEL_DIR:?Set CLM_MODEL_DIR to the staged ModelScope model on local scratch}"
[[ -f $CLM_MODEL_DIR/model.safetensors.index.json ]] || {
  echo "Model snapshot is absent: $CLM_MODEL_DIR" >&2
  exit 2
}
[[ $(git -C "$cb_repo" rev-parse HEAD) == "$CB_REVISION" ]] || {
  echo "ContextBench checkout differs from pins.env: $cb_repo" >&2
  exit 2
}
task="$cb_repo/tasks/kv-store-R1000-seed42"
[[ -f $task/task.toml ]] || { echo "Original task missing: $task" >&2; exit 2; }
[[ -f $image_cache/python_3.12-slim.sif ]] || {
  echo "Harbor-ready image missing: $image_cache/python_3.12-slim.sif" >&2
  exit 2
}
command -v singularity >/dev/null || { echo "singularity launcher not in PATH" >&2; exit 2; }
python - <<'PY'
from importlib.metadata import version
assert version('harbor') == '0.16.1', 'Expected harbor==0.16.1'
import contextbench, clm_contextbench
print('Harbor and ContextBench imports OK')
PY
harbor_help=$(harbor trials start --help)
[[ $harbor_help == *--environment-kwarg* ]] || {
  echo "This Harbor CLI does not support --environment-kwarg" >&2
  exit 2
}

run=$(python "$experiment/storage.py" run-dir "$run_id")
out="$run/eval/contextbench-kv-s42-vllm-off"
[[ ! -e $out/off/server.log && ! -e $out/off/trials ]] || {
  echo "Existing trial output: $out/off; choose a fresh RUN_ID" >&2
  exit 2
}
mkdir -p "$out/off" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
echo "Run: $run_id"
echo "Output: $out/off"
echo "GPU: $CUDA_VISIBLE_DEVICES; port: $port; model context: $VLLM_MAX_MODEL_LEN"

server_pid=
cleanup() {
  if [[ -n $server_pid ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

bash "$experiment/vllm_fallback/run.sh" serve off "$out" &
server_pid=$!
ready=0
for (( attempt=1; attempt<=180; attempt++ )); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM exited during startup; see $out/off/server.log" >&2
    exit 1
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >"$out/off/models.json" 2>/dev/null; then
    ready=1
    break
  fi
  if (( attempt % 6 == 0 )); then
    echo "Waiting for vLLM ($(( attempt * 5 ))s); log: $out/off/server.log"
  fi
  sleep 5
done
(( ready == 1 )) || { echo "vLLM did not become ready; see $out/off/server.log" >&2; exit 1; }

harbor trials start -p "$task" -e singularity \
  --environment-kwarg "singularity_image_cache_dir=$image_cache" \
  --environment-kwarg "singularity_no_mount=home,tmp,bind-paths,proc" \
  -a clm_contextbench:ClmStream -m openai/qwen36-27b \
  --agent-kwarg "api_base=http://127.0.0.1:$port/v1" \
  --trial-name contextbench-kv-store-R1000-seed42-vllm-off \
  --trials-dir "$out/off/trials"

python - "$out/off/trials" <<'PY'
import json
from pathlib import Path
import sys

results = list(Path(sys.argv[1]).rglob("result.json"))
if len(results) != 1:
    sys.exit(f"Expected one Harbor trial result, found {len(results)} in {sys.argv[1]}")
result = json.loads(results[0].read_text(encoding="utf-8"))
error = result.get("exception_info")
if error:
    sys.exit(f"Harbor trial failed: {error.get('exception_type')}: {error.get('exception_message')}")
if result.get("verifier_result") is None:
    sys.exit(f"Harbor trial has no verifier result: {results[0]}")
print(f"Harbor trial and verifier completed. Inspect: {results[0]}")
PY
