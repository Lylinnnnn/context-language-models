# glibc 2.32 节点的 vLLM SCR 备选 smoke

此目录是独立的服务后端验收，不替代主目录固定 SGLang 0.5.16 的 A/C/B 结果。CLM 仓库自带官方 vLLM SCR，固定 `vllm==0.19.1`。该版 x86_64 wheel 标注 glibc 2.31+，但完整依赖能否安装、H20 上能否加载 Qwen3.6-27B，仍以 GPU 节点实际运行结果为准。保留 `pins.env` 中的模型及 tokenizer revision；这轮先验证官方四轮 smoke 和日志搬迁，随后才移植 C 组固定 token 探针。

## 一次准备

在 GPU 节点的 fork 仓库根目录执行。若分支已经 checkout，先 `git pull --ff-only`。下面的 Conda 环境、下载缓存和权重都放在本机 scratch。确保 OSS 配置可用，模型下载及归档细节见 [服务器与 OSS](../SERVER_OSS.md)。

```bash
git fetch origin
git switch experiments/history-edit-scr
git pull --ff-only
getconf GNU_LIBC_VERSION
E=experiments/history_edit_scr
RUN_ID="vllm-$(date +%Y%m%d-%H%M%S)"
RUN=$(python3.12 "$E/storage.py" run-dir "$RUN_ID")
mkdir -p "$RUN/tmp" "$RUN/hf" "$RUN/pip-cache" "$RUN/conda-pkgs" "$RUN/eval"
export TMPDIR="$RUN/tmp" HF_HOME="$RUN/hf" PIP_CACHE_DIR="$RUN/pip-cache"
export CONDA_PKGS_DIRS="$RUN/conda-pkgs"
df -h "$RUN"
conda create --prefix "$RUN/conda" python=3.12 pip -y
conda activate "$RUN/conda"
python -m pip install -e ./suffix_cache_reuse/vllm
python -m pip check
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -c 'import torch, vllm; assert torch.cuda.is_available(); print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python suffix_cache_reuse/vllm/tests/unit_test.py
```

如果安装未成功，停止在这一行并保留 pip 错误；`pip check` 和 CPU 单测不能替代安装与 CUDA 验收。默认 pip 源缺少 0.19.1 时，先确认该节点能访问官方 PyPI，再给安装命令加 `--index-url https://pypi.org/simple`。不要在这个环境里安装 `suffix_cache_reuse/sglang`，它固定另一套服务依赖。

## 固定模型并运行

本实验固定的是 `Qwen/Qwen3.6-27B` 的 BF16 原始权重及同 revision 的 tokenizer，revision 见 [`pins.env`](../pins.env)。CPU 单测不需要下载权重。先检查该 revision 的 OSS 完成标记：

```bash
source "$E/pins.env"
ossutil stat "oss://industry-algo/yanlin/clm/models/$MODEL_ID/$MODEL_REVISION/_SUCCESS"
```

若标记存在，固定 revision 已由授权身份归档到 OSS，执行：

```bash
CLM_MODEL_DIR=$(python "$E/storage.py" stage-model "$RUN_ID") || exit
export CLM_MODEL_DIR
python -m pip freeze > "$RUN/eval/pip-freeze.txt"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$E/vllm_fallback/pilot.py" "$RUN/eval/vllm-smoke"
python "$E/storage.py" publish-eval "$RUN_ID"
```

如果 `stat` 明确返回 `NoSuchKey`，且节点可以访问 Hugging Face，按固定 revision 下载并首次归档，再执行上方的 smoke 与评估归档命令：

```bash
CLM_MODEL_DIR=$(python "$E/storage.py" model-dir "$RUN_ID") || exit
export CLM_MODEL_DIR
bash "$E/run.sh" download
bash "$E/run.sh" snapshot
python "$E/storage.py" publish-model "$RUN_ID"
```

下载前检查 `df -h "$RUN"` 及实际配额，模型约 56 GB；安装缓存和权重均留在该 run 的本机目录。遇到 OSS 权限或 Endpoint 错误时先处理配置，不将其当作模型不存在。完整路径规则见 [服务器与 OSS](../SERVER_OSS.md#权重和运行目录)。

`pilot.py` 用一张卡依次启动 off 和 on 服务，端口默认 32000 与 32001。它对每组运行仓库内的官方 `suffix_cache_reuse/vllm/tests/smoke.py`，停止服务后调用官方 `analysis/scr_report.py`。验收条件包括 on 组至少一条编辑请求 `relocated > 0`，off 组搬迁量为 0，以及官方 token 守恒检查通过。两个服务除 SCR 开关、端口和输出路径外使用同一组参数。每组的 `server.log`、`smoke.log`、`requests.csv`、`summary.json` 和 `report.log` 留在各自输出目录。

运行失败时，先看对应 `server.log` 的首个异常。若是显存不足，可调低 `VLLM_GPU_MEMORY_UTILIZATION` 或 `KVREUSE_SIDE_SESSIONS`、`KVREUSE_SIDE_TOKENS`，并从新 `RUN_ID` 重跑两组；不要只调其中一组。`VLLM_MAX_MODEL_LEN` 默认 32768，仅用于这轮 smoke。安装状态和服务日志成功之前，结果仍标记为「GPU 待验证」。

[vLLM wheel](https://pypi.org/project/vllm/0.19.1/#files) · [官方 vLLM SCR 说明](https://github.com/facebookresearch/context-language-models/tree/3c89408be27d34e9e8d14eade365a8d440b041e9/suffix_cache_reuse/vllm)
