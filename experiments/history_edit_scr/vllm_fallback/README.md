# glibc 2.32 节点的 vLLM SCR 备选 smoke

此目录是独立的 vLLM 服务后端验收，固定 `vllm==0.19.1`。已接入官方四轮 smoke 和 C 组固定 token 探针；vLLM 与原 SGLang 路线的数值不可直接合并。模型来自 ModelScope，第一次上传到 OSS 后以文件 SHA-256 清单冻结权重与 tokenizer；`pins.env` 的 Hugging Face revision 只属于原 SGLang 路线。

## 一次准备

在 GPU 节点的 fork 仓库根目录执行。若分支已经 checkout，先 `git pull --ff-only`。持久 Conda 环境按名称 `clm` 建在用户的 `~/.conda/envs/`；下载缓存、权重和评估输出放在本机 scratch。确保 OSS 配置可用，模型下载及归档细节见 [服务器与 OSS](../SERVER_OSS.md)。

```bash
git fetch origin
git switch experiments/history-edit-scr
git pull --ff-only
getconf GNU_LIBC_VERSION
E=experiments/history_edit_scr
RUN_ID="vllm-$(date +%Y%m%d-%H%M%S)"
RUN=$(python3.12 "$E/storage.py" run-dir "$RUN_ID")
mkdir -p "$RUN/tmp" "$RUN/hf" "$RUN/pip-cache" "$RUN/conda-pkgs" "$RUN/modelscope-cache" "$RUN/modelscope-home" "$RUN/eval"
export TMPDIR="$RUN/tmp" HF_HOME="$RUN/hf" PIP_CACHE_DIR="$RUN/pip-cache"
export CONDA_PKGS_DIRS="$RUN/conda-pkgs"
export MODELSCOPE_CACHE="$RUN/modelscope-cache" MODELSCOPE_HOME="$RUN/modelscope-home"
df -h "$RUN"
df -h /home/liuyanlin.lyl
conda create -n clm python=3.12 pip -y
conda activate clm
python -c 'import sys; print(sys.prefix)'
python -m pip install -e ./suffix_cache_reuse/vllm
python -m pip check
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -c 'import torch, vllm; assert torch.cuda.is_available(); print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python suffix_cache_reuse/vllm/tests/unit_test.py
```

如果安装未成功，停止在这一行并保留 pip 错误；`pip check` 和 CPU 单测不能替代安装与 CUDA 验收。默认 pip 源缺少 0.19.1 时，先确认该节点能访问官方 PyPI，再给安装命令加 `--index-url https://pypi.org/simple`。不要在这个环境里安装 `suffix_cache_reuse/sglang`，它固定另一套服务依赖。

若已按照旧说明将环境建在 `$RUN/conda`，先让正在运行的 smoke 完成，再迁移到持久的命名环境。Conda 支持从已有环境克隆。克隆后重装两个轻量的入口包，检查实际 Python 路径、依赖和 GPU；这些通过后才清理旧环境。模型和评估文件位于 `$RUN/model` 与 `$RUN/eval`，不随旧环境删除。

```bash
conda deactivate
df -h /home/liuyanlin.lyl
conda create -n clm --clone "$RUN/conda" -y
conda activate clm
python -m pip install --no-deps --force-reinstall -e ./suffix_cache_reuse/vllm 'modelscope-hub==0.7.2'
python -m pip check
python -c 'import sys, torch, vllm; assert torch.cuda.is_available(); assert vllm.__version__ == "0.19.1"; print(sys.prefix, vllm.__version__, torch.__version__)'
python suffix_cache_reuse/vllm/tests/unit_test.py
conda info --envs
```

确认 `sys.prefix` 指向 `/home/liuyanlin.lyl/.conda/envs/clm`、GPU 可见，并用新环境完成后续服务检查后，可用 `conda env remove -p "$RUN/conda"` 清理旧环境。这个命令只移除旧 Conda 环境，不移除整个 `$RUN`。

## 从 ModelScope 归档模型并运行

CPU 单测不需要权重。GPU smoke 使用 ModelScope 官方 `Qwen/Qwen3.6-27B` 的 BF16 权重与同一快照的 tokenizer，约 55.59 GB。先加载独立的 [ModelScope 归档设置](modelscope.env)。其中 `master` 是第一次下载的来源分支，`modelscope-master-v1` 是 OSS 快照标签；归档后的实际内容由文件 SHA-256 清单固定，不声称它等于 Hugging Face 的提交号。

```bash
source "$E/vllm_fallback/modelscope.env"
source "$E/pins.env"
ossutil stat "oss://industry-algo/yanlin/clm/models/$MODEL_ID/$CLM_MODEL_REVISION/_SUCCESS"
```

若标记存在，快照已归档到 OSS，执行：

```bash
CLM_MODEL_DIR=$(python "$E/storage.py" stage-model "$RUN_ID") || exit
export CLM_MODEL_DIR
python -m pip freeze > "$RUN/eval/pip-freeze.txt"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$E/vllm_fallback/pilot.py" "$RUN/eval/vllm-smoke"
python "$E/storage.py" publish-eval "$RUN_ID"
```

如果 `stat` 明确返回 `NoSuchKey`，在当前 Conda 环境中安装轻量的 ModelScope Hub CLI，下载到本机 scratch，并由 `ossutil` 归档到 OSS。然后执行上方的 smoke 与评估归档命令，跳过其中的 `stage-model`：

```bash
python -m pip install 'modelscope-hub==0.7.2'
python -m pip check
CLM_MODEL_DIR=$(python "$E/storage.py" model-dir "$RUN_ID") || exit
export CLM_MODEL_DIR
ms-hub download "$MODEL_ID" --revision "$CLM_SOURCE_REVISION" --local-dir "$CLM_MODEL_DIR"
bash "$E/run.sh" snapshot
python "$E/storage.py" publish-model "$RUN_ID"
python -m pip freeze > "$RUN/eval/pip-freeze.txt"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$E/vllm_fallback/pilot.py" "$RUN/eval/vllm-smoke"
python "$E/storage.py" publish-eval "$RUN_ID"
```

下载前检查 `df -h "$RUN"` 及实际配额；安装缓存和权重均留在该 run 的本机目录。上传流程逐文件核对大小，完成标记最后写入；后续运行可由 OSS 暂存同一快照。遇到 OSS 权限或 Endpoint 错误时先处理配置，不将其当作模型不存在。完整路径规则见 [服务器与 OSS](../SERVER_OSS.md#权重和运行目录)。

`pilot.py` 用一张卡依次启动 off 和 on 服务，端口默认 32000 与 32001。它对每组运行仓库内的官方 `suffix_cache_reuse/vllm/tests/smoke.py`，停止服务后调用官方 `analysis/scr_report.py`。验收条件包括 on 组至少一条编辑请求 `relocated > 0`，off 组搬迁量为 0，以及官方 token 守恒检查通过。两个服务除 SCR 开关、端口和输出路径外使用同一组参数。每组的 `server.log`、`smoke.log`、`requests.csv`、`summary.json` 和 `report.log` 留在各自输出目录。

## C 组两例配对探针

已有 `(clm)` 持久 Conda 环境和本机暂存的同一模型时，沿用原来的 `RUN_ID`；不需重新下载权重或重建环境。下面以先前 `vllm-20261125-153822` 为例，若实际 run ID 不同只改第一行。先生成两个反驳型样本的冻结输入，再用同一个 `CLM_MODEL_DIR` 顺序跑 off/on。此次输出使用新目录 `vllm-c2`，官方 smoke 会在每个服务启动后再次跑一遍，作为本轮服务健康和搬迁对照。

```bash
git pull --ff-only
conda activate clm
E=experiments/history_edit_scr
RUN_ID=vllm-20261125-153822
source "$E/vllm_fallback/modelscope.env"
RUN=$(python "$E/storage.py" run-dir "$RUN_ID")
CLM_MODEL_DIR=$(python "$E/storage.py" model-dir "$RUN_ID")
export CLM_MODEL_DIR
test -f "$CLM_MODEL_DIR/model.safetensors.index.json" || { echo '模型不在本机 scratch；先按上节从 OSS stage-model' >&2; exit 1; }
python "$E/probe.py" prepare "$RUN/eval/vllm-c2-manifest.json" --snapshot "$CLM_MODEL_DIR" --cases 2 --stage clean
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$E/vllm_fallback/pilot.py" "$RUN/eval/vllm-c2" --manifest "$RUN/eval/vllm-c2-manifest.json"
python -m json.tool "$RUN/eval/vllm-c2/probe-summary.json" | head -80
cp "$RUN/eval/vllm-c2-manifest.json" "$RUN/eval/vllm-c2/manifest-input.json"
python "$E/storage.py" publish-eval "$RUN_ID" --subdir vllm-c2
```

结果中的 `valid_pairs` 应等于 `expected_pairs`；`clean_on_hits` 记录有多少 clean 编辑请求在 on 组真正搬迁。即使 `clean_on_hits=0`，脚本也只判定输入/服务证据是否有效，不把零命中误写为行为效应。`probe-joined.jsonl` 保留每个 off/on 分支的 token 数、搬迁量、输出和错误；`probe-summary.json` 汇总组别和配对效应。两例只用于检查管线；正式估计需要扩大到 30–50 个基础样本及重复 seed。上面的 `publish-eval --subdir vllm-c2` 上传到独立的 `eval/<RUN_ID>/vllm-c2/` 前缀，逐文件核对大小、写清单，最后写 `_SUCCESS`；已归档的 `eval/<RUN_ID>/` 不会被改写。

## C 组八卡分片

两例试跑确认 `valid_pairs == expected_pairs`、输入 ID 校验通过，并记录 clean 实际命中后，再扩大样本。下面生成 36 个反证样本和 8 个旧判断仍有效的对照，共 44 个基础样本，按基础样本轮流分到 8 个 shard；同一样本的 A/B 历史、clean/visible/cold 终态和三个 seed 全留在同一个 shard。每张卡只启动一个服务，先 off 再 on，各 shard 使用独立端口和唯一缓存隔离键。所有 GPU 只读同一份本机 scratch 模型文件，不复制 8 份权重到磁盘。

```bash
git pull --ff-only
conda activate clm
E=experiments/history_edit_scr
RUN_ID=vllm-20261125-153822
source "$E/vllm_fallback/modelscope.env"
RUN=$(python "$E/storage.py" run-dir "$RUN_ID")
CLM_MODEL_DIR=$(python "$E/storage.py" model-dir "$RUN_ID")
export CLM_MODEL_DIR
python "$E/probe.py" prepare "$RUN/eval/vllm-c44-manifest.json" \
  --snapshot "$CLM_MODEL_DIR" --cases 36 --stable-controls 8 --stage full
if python "$E/vllm_fallback/sharded.py" "$RUN/eval/vllm-c44" \
  --manifest "$RUN/eval/vllm-c44-manifest.json" \
  --gpus 0,1,2,3,4,5,6,7 --workers 8 --seeds 42 43 44; then
  python "$E/storage.py" publish-eval "$RUN_ID" --subdir vllm-c44
  cat "$RUN/eval/vllm-c44/probe-summary.json"
fi
```

`sharded.py` 创建全局输入清单、各 shard 的输入清单、`driver.log`、off/on 服务日志和配对报告。所有 shard 完成后，汇总脚本核对基础样本不重复、不缺失，off/on 参数与 token 输入配对，重新以**基础样本**为单位汇总效应及置信区间。若某个 shard 失败，保留其日志，不生成通过标记或上传；修复后用新的结果目录运行。`--workers 4` 可让同样的 8 个 shard 分两批执行，降低同时加载权重时的主机内存和 IO 压力。8 卡并发适合加速行为/正确性实验；并发服务共享主机资源，不能将各 shard 的耗时直接当作独占 GPU 的延迟基准。

运行失败时，先看对应 `server.log` 的首个异常。若是显存不足，可调低 `VLLM_GPU_MEMORY_UTILIZATION` 或 `KVREUSE_SIDE_SESSIONS`、`KVREUSE_SIDE_TOKENS`，并从新 `RUN_ID` 重跑两组；不要只调其中一组。`VLLM_MAX_MODEL_LEN` 默认 32768，仅用于这轮 smoke。安装状态和服务日志成功之前，结果仍标记为「GPU 待验证」。

### Conda 环境迁移后的 FlashInfer JIT 缓存

若 `server.log` 的 nvcc 命令仍带已删除的 `/tmp/clm/<run_id>/conda/.../flashinfer/data/include`，但当前 Python 位于 `~/.conda/envs/clm`，这是旧的共享 FlashInfer JIT 工作目录保留了旧环境绝对路径。`run.sh serve` 现在默认将 `FLASHINFER_WORKSPACE_BASE` 设为模型所在 run 的本机 scratch；FlashInfer 0.6.6 会在其下新建 `.cache/flashinfer/`，不再读取 `~/.cache/flashinfer/0.6.6/90a` 的旧构建文件。若该变量已由用户设置，脚本尊重用户设置。新目录不会作为 eval 或模型产物上传 OSS。

先在 `(clm)` 中确认 FlashInfer 的 SM90 GDN 头文件确实位于新环境：

```bash
python - <<'PY'
import sys
from importlib.util import find_spec
from pathlib import Path
spec = find_spec('flashinfer')
assert spec and spec.submodule_search_locations, 'flashinfer not installed'
header = Path(next(iter(spec.submodule_search_locations))) / 'data/include/flashinfer/flat/prefill/prefill_kernel_delta_rule_sm90.cuh'
print('Python:', sys.prefix)
print('GDN header:', header, 'exists:', header.is_file())
assert header.is_file(), 'current environment is missing FlashInfer package data'
PY
```

若头文件存在，沿用已冻结的 `vllm-c2-manifest.json`，在**新的**结果目录重跑。旧失败目录和全局缓存先保留，不需要删除、重装 Conda 或重新下载模型：

```bash
export FLASHINFER_WORKSPACE_BASE="$RUN"
export MAX_JOBS=4
if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$E/vllm_fallback/pilot.py" \
  "$RUN/eval/vllm-c2-jitfix" --manifest "$RUN/eval/vllm-c2-manifest.json"; then
  cp "$RUN/eval/vllm-c2-manifest.json" "$RUN/eval/vllm-c2-jitfix/manifest-input.json"
  python "$E/storage.py" publish-eval "$RUN_ID" --subdir vllm-c2-jitfix
fi
```

第一次新缓存编译可能较慢。若头文件在当前环境中不存在，停止重跑并检查当前 `flashinfer-python` 安装，而不是反复清理 JIT 缓存。

[vLLM wheel](https://pypi.org/project/vllm/0.19.1/#files) · [官方 vLLM SCR 说明](https://github.com/facebookresearch/context-language-models/tree/3c89408be27d34e9e8d14eade365a8d440b041e9/suffix_cache_reuse/vllm)
