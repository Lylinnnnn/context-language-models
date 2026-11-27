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

## 原版四类任务：单 seed 八卡

`full_seed42.py` 使用冻结的 ContextBench checkout 里全部 **seed 42** 任务：KV Store 8 个级别、Log Triage 7 个、Needle Retention 10 个、Sudoku Sketchpad 7 个，共 32 题。官方完整集合为 8 个 seed、256 题；这里是节省算力的单 seed 子集。脚本按原始 `environment/stream/chunks.jsonl` 文件大小分配 8 个 shard，每张卡 4 题。每张卡只启动一个 vLLM 服务，依次执行自己的 4 题。四类仍用原版 `task.toml`、任务流和 verifier，不修改官方任务。

在仓库根目录执行；复用已有的 Conda 环境和 Apptainer 镜像。若之前的本机模型还在，就让八个 shard 共用它；否则入口从 OSS 暂存一份到本次 `$RUN_ID/model/`，本批任务结束后自动核验并清理该份暂存：

```bash
conda activate clm
E=experiments/history_edit_scr
source "$E/vllm_fallback/modelscope.env"
MODEL_RUN_ID=vllm-20261125-153822
CLM_MODEL_DIR=$(python "$E/storage.py" model-dir "$MODEL_RUN_ID")
if [ -f "$CLM_MODEL_DIR/model.safetensors.index.json" ]; then
  export CLM_MODEL_DIR
else
  unset CLM_MODEL_DIR
fi
RUN_ID="cb-full-s42-$(date -u +%Y%m%dT%H%M%SZ)"
python "$E/contextbench_apptainer/full_seed42.py" "$RUN_ID" --gpus 0,1,2,3,4,5,6,7
```

运行中每题的 Harbor 输出在 `/tmp/clm/<RUN_ID>/eval/contextbench-s42-vllm-off/logs/`，原版 trial、评分和轨迹在 `trials/`，服务日志在 `servers/`。`manifest.json` 记录固定任务和 shard，`summary.json` 汇总完成数与平均 reward。脚本只有 32/32 题都有正式评分，且 OSS 上传核验和本机结果清理均成功，才返回 0。中途失败后先修复原因，再以**同一个** `RUN_ID` 加 `--resume`，已通过的题会从保存的 `result.json` 重新核验并跳过：

```bash
python "$E/contextbench_apptainer/full_seed42.py" "$RUN_ID" --gpus 0,1,2,3,4,5,6,7 --resume
```

32/32 完成后脚本自动把该评测子目录归档到 OSS，重新下载远端 manifest 和 `_SUCCESS`、核对本机逐文件 SHA-256 清单与远端逐文件大小；全部通过才删除该 run 下的 `eval/contextbench-s42-vllm-off`、上传中转文件及这次八张卡的 FlashInfer 临时编译缓存。若模型由此入口暂存到本次 run，它也会核验 OSS 原权重并删除本次模型暂存。Conda 环境和其他评测目录保持可用。可用下面的只读命令核对完成标记：

```bash
ossutil stat "oss://industry-algo/yanlin/clm/eval/$RUN_ID/contextbench-s42-vllm-off/_SUCCESS"
```

已经由旧版脚本跑完但还未清理的同类评测，可单独执行通用收尾命令；它也能核验之前已上传的同一结果后再清理：

```bash
python "$E/storage.py" publish-eval "$RUN_ID" --subdir contextbench-s42-vllm-off --cleanup
```

若沿用 `CLM_MODEL_DIR=/tmp/clm/<MODEL_RUN_ID>/model/modelscope-master-v1` 的共享本机模型，脚本不会在单次 ContextBench 结束时删除它，以免影响随后使用该模型的任务。**所有**依赖该共享暂存的任务都结束后，执行一次 `python "$E/storage.py" publish-model "$MODEL_RUN_ID" --cleanup`，核验 OSS 原权重再释放本机副本。

这组保持单题试跑验证过的 vLLM 0.19.1、Qwen3.6-27B、`ClmStream` 和 SCR **off** 配置；论文 ContextBench 主图使用 GPT-5.4，每个难度运行四个 seed，部分难度运行八个 seed。这组单 seed、不同模型的结果不能直接当作主图的数值复现。若要做 SCR on 对照，在另一轮使用新的 `RUN_ID` 和 `--mode on`，同一份 seed 42 任务会按相同 shard 规则运行，输出目录是 `contextbench-s42-vllm-on`。

## 不编辑历史的对照

`SCR off` 仍允许 CLM 自编辑。`--agent no-edit` 使用同一模型、任务、seed、预算、vLLM SCR off 和八卡分片，移除 CLM 编辑方法说明，改用无历史编辑提示，并在执行层跳过历史镜像读写。每题结束时要求 `usage.json` 的 `n_ctx_syncs=0`，避免把只改提示的试验误记作无编辑。历史不能压缩，长任务可能先到上下文预算墙；比较时同时看 reward、送达操作数和结束原因。此对照同时改变编辑能力与编辑提示，估计的是整套 CLM 方法相对不编辑方法的差异，不是仅由历史文本编辑造成的因果效应。单 seed 与模型替换的局限同上。

先用一题试跑。它在评分与零编辑校验通过后归档到 OSS 并清理本次本机评测目录：

```bash
conda activate clm
E=experiments/history_edit_scr
source "$E/vllm_fallback/modelscope.env"
MODEL_RUN_ID=vllm-20261125-153822
CLM_MODEL_DIR=$(python "$E/storage.py" model-dir "$MODEL_RUN_ID")
if [ ! -f "$CLM_MODEL_DIR/model.safetensors.index.json" ]; then
  CLM_MODEL_DIR=$(python "$E/storage.py" stage-model "$MODEL_RUN_ID")
fi
export CLM_MODEL_DIR
SMOKE_RUN_ID="cb-noedit-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
CUDA_VISIBLE_DEVICES=0 bash "$E/contextbench_apptainer/pilot.sh" "$SMOKE_RUN_ID" --no-edit
ossutil stat "oss://industry-algo/yanlin/clm/eval/$SMOKE_RUN_ID/contextbench-kv-s42-vllm-off-no-edit/_SUCCESS"
```

随后跑 32 题，完成后自动核验上传并清理本次本机评测目录；失败时用同一个 `RUN_ID` 加 `--resume`：

```bash
RUN_ID="cb-noedit-s42-$(date -u +%Y%m%dT%H%M%SZ)"
python -u "$E/contextbench_apptainer/full_seed42.py" "$RUN_ID" \
  --agent no-edit --mode off --gpus 0,1,2,3,4,5,6,7
ossutil cat "oss://industry-algo/yanlin/clm/eval/$RUN_ID/contextbench-s42-vllm-off-no-edit/files/summary.json"
```

## 已归档结果的编辑与案例审计

完整评测成功归档后，运行以下命令。它从 OSS 读取每题较小的 `usage.json`、`timing.json`，把未满分或任务流未结束的题列为 bad case，并从每类挑一题完整满分的 good case。仅对选中的题读取完整 `trajectory.ctx.json`，不拉取模型权重、数据集或全部逐轮快照，也不启动 GPU 服务。

```bash
conda activate clm
E=experiments/history_edit_scr
RUN_ID=cb-full-s42-20261126T122915Z
python -u "$E/contextbench_apptainer/audit_run.py" "$RUN_ID"
cat "tmp_result/$RUN_ID-contextbench-s42-vllm-off-analysis.md"
```

审计只生成一份 Markdown，保留在仓库的 `tmp_result/<RUN_ID>-contextbench-s42-vllm-off-analysis.md` 供直接打开；同一份上传至原评测的 `eval/<RUN_ID>/contextbench-s42-vllm-off/files/analysis.md`。上传后重新下载并核对 SHA-256，再写 `files/analysis.md._SUCCESS`。原评测的 manifest 和 `_SUCCESS` 不覆盖；新报告及其标记是附加文件。下载的轨迹暂存在 `/tmp/clm/<RUN_ID>/tmp/`，审计结束自动清除，报告本地副本保留。

报告顶部是一张案例分数表；每个案例用三列横向展示一次历史编辑的前后片段，并用删除线和新增高亮标记变化。对比时忽略自动重编号的回合头，优先显示正文差异。下方折叠区提供该次编辑的 Qwen 原始回复、完整 bash 命令和工具返回。bad case 展示最后一次编辑，good case 展示缩减最多的一次编辑。标签只用于分拣，不声称该次编辑导致最终分数。`trajectory.json` 只保留最终历史；完整追溯以 `trajectory.ctx.json` 的分段和逐轮快照为准。工具返回可能在进入模型上下文前被截断，因此报告展示的是模型实际收到的工具文本。
