# ContextBench 单题 Apptainer 试跑

节点已验证 Apptainer 1.5.5 普通容器执行。Harbor 0.16.1 的 `--pid` 在该节点需要额外禁用自动 `/proc` 挂载。`pilot.sh` 通过 Harbor 自带的 `singularity_no_mount` 选项实现，不改 Harbor 或 ContextBench 源码。

本入口运行固定原版任务 `kv-store-R1000-seed42`，使用现有 `clm` Conda 环境、ModelScope 归档的 Qwen3.6-27B 和 vLLM SCR fallback 的 **off** 服务。它把 vLLM 上下文上限设为至少 40,960 tokens，与 ContextBench 的 32,768 预算和 8,192 最大输出相加一致；同一任务的论文结果还需要核对完整任务时限、模型 revision、SGLang 后端和镜像基础层。此试跑只验收端到端 agent、任务流和评分。

ContextBench agent 使用 `tool_choice="auto"`，入口只对本次服务启用 vLLM 的 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`。Harbor 可能在 trial 出错时仍以状态码 0 退出；入口会检查唯一的 `result.json`，要求没有 `exception_info` 且已经生成 `verifier_result`，否则以失败退出。

先把带 `/usr/bin/python3` 和 `/opt/harbor-server` 的 ContextBench 镜像放在 `/tmp/clm/contextbench-image-cache/python_3.12-slim.sif`。原版任务的镜像名是 `python:3.12-slim`，Harbor 在指定缓存目录中寻找该文件。镜像构建失败时保留日志，不使用普通 Python smoke SIF 代替。

```bash
conda activate clm
CB_REPO="$(dirname "$(git rev-parse --show-toplevel)")/ContextBench"
python -m pip install -e . -e "$CB_REPO[tokenizer]"
python -m pip check

E=experiments/history_edit_scr
source "$E/vllm_fallback/modelscope.env"
MODEL_RUN_ID=vllm-20261125-153822
CLM_MODEL_DIR=$(python "$E/storage.py" model-dir "$MODEL_RUN_ID")
export CLM_MODEL_DIR
if [ ! -f "$CLM_MODEL_DIR/model.safetensors.index.json" ]; then
  CLM_MODEL_DIR=$(python "$E/storage.py" stage-model "$MODEL_RUN_ID")
  export CLM_MODEL_DIR
fi

RUN_ID="cb-kv-s42-$(date -u +%Y%m%dT%H%M%SZ)"
bash "$E/contextbench_apptainer/pilot.sh" "$RUN_ID"
```

`pilot.sh` 自动启动 off 服务，等 `/v1/models` 就绪后只跑一题，并在退出时停止自己启动的服务。日志在 `/tmp/clm/<RUN_ID>/eval/contextbench-kv-s42-vllm-off/off/`；Harbor trial 与评分在其 `trials/` 子目录。成功检查评分和轨迹之后，可用 `python "$E/storage.py" publish-eval "$RUN_ID" --subdir contextbench-kv-s42-vllm-off` 独立归档到 OSS。若构建镜像时 `/proc` 警告但命令成功，可继续；若 trial 报错，先查看 `server.log` 和 trial 日志的首个异常。

每次重试使用新的 `RUN_ID`，保留上次的 `server.log` 与失败现场。入口在启动 GPU 服务前检查当前 Harbor CLI 是否支持所需的 `--environment-kwarg`。
