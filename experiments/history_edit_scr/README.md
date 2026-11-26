# CLM 历史编辑与后缀缓存复现

本目录需要仓库根目录、官方 CLM Harbor harness 和官方 SCR 启动与分析脚本。A 组运行官方 smoke 和回归。C 组固定原生 `/generate` 的输入 ID，测合成 KV Store 记录中的可见计划和复用状态。B 组通过官方 Harbor 入口运行原版任务。D1 提供静态编辑分支和行动判分。

## 服务器环境

先在实际 GPU 节点确认 Python、驱动、本机可用空间及 OSS 配置，再选 Python 环境和权重路径。检查命令与存储流程见 [服务器与 OSS](SERVER_OSS.md)。

从 fork 取实验分支。已有 checkout 时可从该 checkout 执行 `git fetch origin` 和 `git switch experiments/history-edit-scr`。

```bash
git clone --branch experiments/history-edit-scr https://github.com/Lylinnnnn/context-language-models.git
cd context-language-models
getconf GNU_LIBC_VERSION
```

Linux x86_64 上，SGLang 0.5.16 的 Python 3.12 wheel 需要 glibc 2.34 或更新版本。已确认旧 Notebook 节点为 glibc 2.32 时，先切换到满足要求的 GPU Notebook 或容器镜像；在旧节点里仅新建 Conda 环境无法改变系统 glibc。原安装失败后，可在仓库根目录退出并删除旧 venv，避免后续混用：

```bash
cd "$(git rev-parse --show-toplevel)"
if [ "${VIRTUAL_ENV:-}" = "$PWD/.venv" ]; then deactivate; fi
if [ -f .venv/pyvenv.cfg ]; then rm -rf -- .venv; fi
```

这个命令只删除仓库里的 `.venv`，不会删除 Conda base、代码、权重或 OSS 文件。pip 候选版本停在 0.5.10.post1 与新版 wheel 被 glibc 过滤的现象吻合。 [SGLang 0.5.16 发布文件](https://pypi.org/project/sglang/0.5.16/#files)

如果平台没有现成的新 GPU 镜像，先检查 `command -v docker podman apptainer singularity`。有容器运行能力时，可基于 [SGLang 0.5.16 官方 Dockerfile](https://github.com/sgl-project/sglang/blob/v0.5.16/docker/Dockerfile) 使用的 Ubuntu 24.04/CUDA 13.0.1 基础镜像制作运行环境，并验证容器内 glibc、Python 与 GPU。没有容器运行能力时，先按独立的 [vLLM SCR 备选 smoke](vllm_fallback/README.md) 验收。它固定 vLLM 0.19.1，其 Linux x86_64 wheel 要求 glibc 2.31+，但仍须检查依赖和 GPU。此后才为现有 SGLang 专用的 C 组探针另做适配。[vLLM 0.19.1 发布文件](https://pypi.org/project/vllm/0.19.1/#files)

不要仅将 SGLang 降到 0.5.10.post1：官方 SCR 包固定 0.5.16，补丁调用 SGLang 内部调度器、缓存池及混合模型状态接口。降级需要移植补丁并重新跑 GPU 回归，不能当作原版复现。

确认新镜像的 glibc 满足要求后，若已安装 Conda，创建并激活一个 Python 3.12 环境：

```bash
getconf GNU_LIBC_VERSION
conda create -n clm-scr python=3.12 pip -y
conda activate clm-scr
python --version
```

新镜像没有 Conda 时也可以创建 venv：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

使用选定的同一个环境安装两个项目，并在安装成功后验证依赖和 GPU。

```bash
python -m pip install -e . -e ./suffix_cache_reuse/sglang &&
python -m pip check &&
python -c 'import torch, sglang; assert torch.cuda.is_available(); print(sglang.__version__, torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))' &&
python suffix_cache_reuse/sglang/tests/unit_test.py
```

glibc 版本满足要求而现有 pip 源仍缺 0.5.16 时，若节点能访问官方 PyPI，对上面的安装命令加 `--index-url https://pypi.org/simple`。也可请管理员同步该版本及其依赖。一次失败的安装仍可能让 `pip check` 或纯 CPU 单测返回成功，验收以安装成功且 Torch、SGLang 和 CUDA 检查通过为准。

在 Linux/NVIDIA 服务器上安装。SCR 包固定 SGLang 0.5.16，官方该版本的依赖使用 CUDA 13 系列，驱动需要与所装 wheel 兼容。用一张能放下 Qwen3.6-27B BF16、运行时缓存和 side buffer 的卡。Harbor 验收另外需要 Docker 或 Apptainer。软件安装和首次加载模型的错误会分别出现在安装终端和各组的 `server.log`。

`pins.env` 固定了 CLM fork 的上游基线、SGLang、Qwen 权重与同一 snapshot 的 tokenizer，以及外部两个 benchmark 的源码。Qwen revision 是本次运行选用的公开 snapshot，论文没有给出用于逐字节核对的权重 revision。每个服务日志的首条配置记录保留实验 commit、Torch/CUDA、GPU、模型和 tokenizer revision、tokenizer 文件 hash、启动参数和 SCR 环境。

## A 与 C 组首轮

在仓库根目录的同一个 Bash 会话执行。权重准备和设置 `CLM_MODEL_DIR`、`OUT` 的命令见 [服务器与 OSS](SERVER_OSS.md)。先冻结两个基础样本，再依次启动 off 和 on。脚本每次只启动一个服务，并在这一组请求结束后关闭服务。两端口默认 31000 和 31001，执行前保持空闲。

```bash
E=experiments/history_edit_scr
: "${OUT:?set OUT to the local run eval directory}"
mkdir -p "$OUT"
python -m pip freeze > "$OUT/pip-freeze.txt"
SNAPSHOT=$(bash "$E/run.sh" snapshot)

python "$E/probe.py" prepare "$OUT/clean-2.json" --snapshot "$SNAPSHOT" --cases 2
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python "$E/pilot.py" "$OUT/clean-2" --manifest "$OUT/clean-2.json"
```

只做 A 组时运行 `CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python "$E/pilot.py" "$OUT/smoke"`。每次使用新的结果目录。用 `tail -f "$OUT/clean-2/on/server.log"` 查看启动与请求日志。

两组使用同一个官方入口、BF16、单并发、64K 服务上下文、标准 radix 缓存和相同基础配置。SCR 开启使用 K=6、SSM fork。启动默认复用官方显存比例 0.75。实验入口默认把 side buffer 调成 4 个会话 × 12,000 tokens，按官方 64 KiB/token 估算约 2.93 GiB。4 个槽允许官方隔离回归同时保有近期会话，追踪会话数默认等于槽数。原始服务默认 12 × 30,000 tokens，约 22 GiB。若服务器显存池或任务输入需要调整，给 off/on 统一设置 `MEM_FRACTION_STATIC`、`KVREUSE_SIDE_SESSIONS` 和 `KVREUSE_SIDE_TOKENS`。启动器对两组统一启用 Qwen reasoning parser 以兼容 Harbor 工具续跑。A/C 的模板参数禁用新生成的 thinking。C 的旧判断是可见助手笔记。

`summary.md` 和 `summary.json` 会检查 A 的有效生成和 on/off 实际搬迁，以及 C 的 4 条基础配对是否输入一致、预热状态归属正确。至少一个基础样本的两条 clean 历史在 on 侧均实际搬迁，C 首轮才能通过。实际搬迁需要官方日志出现对应的 `v6 splice OK` 且统计为正。两组原始服务日志、官方 per-RID CSV/JSON、原生请求和响应、合并判分均保存在结果目录。

### 正式 pilot

首轮通过后冻结 36 个基础样本，加 4 个旧计划仍有效的控制样本。常规样本另带 cold-only 分支。用独立目录保存运行。

```bash
python "$E/probe.py" prepare "$OUT/full-36.json" --snapshot "$SNAPSHOT" \
  --cases 36 --stage full --stable-controls 4
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python "$E/pilot.py" "$OUT/full-36" --manifest "$OUT/full-36.json"
```

生成器检查 W_A/W_B 只差一个 token、warm/probe 共同前缀至少 1,024 tokens、后缀包含至少 1,024 个逐项一致的 token，以及两条 clean 终态完全相同。每条分支以独立 `extra_key` 保存预热和决策，预热至少生成 8 tokens。决策仅判完整的 `ACTION: A` 至 `ACTION: D`，容许首尾空白和大小写差异。无效请求计入无效配对，无法解析的有效输出计作错误。H1 比较 off 中可见计划和 clean。H2 比较同题 clean 的 A/B 旧标签跟随差，再计算 on 减 off，并单列两条 warm 历史都实际命中的样本。区间以基础样本为单位做 percentile bootstrap。默认使用贪心解码和 seed 42。

### 手动运行

需要每组复用同一服务跑 B 时，可在一个终端启动 `bash "$E/run.sh" serve off "$OUT/manual"`。在另一个激活同一 venv 的终端运行对应命令。单卡完成 off 后，用 Ctrl-C 停服务，再用同样环境启动 on，改端口、组名和输出路径。运行中不要把其他客户端请求混到这台服务。

```bash
curl -fsS http://127.0.0.1:31000/health
bash "$E/run.sh" smoke off "$OUT/manual"
python "$E/probe.py" run "$OUT/clean-2.json" "$OUT/manual/off/requests.jsonl" \
  --arm off --url http://127.0.0.1:31000

# 停掉 off 并在服务终端启动 on 后执行
curl -fsS http://127.0.0.1:31001/health
bash "$E/run.sh" smoke on "$OUT/manual"
python "$E/probe.py" run "$OUT/clean-2.json" "$OUT/manual/on/requests.jsonl" \
  --arm on --url http://127.0.0.1:31001
python "$E/report.py" "$OUT/manual" --manifest "$OUT/clean-2.json" --require-clean-hit
```

## B 组原版任务

### ContextBench KV Store

在当前 glibc 2.32 节点使用无特权 Apptainer 和已验证的 vLLM fallback 时，先按 [单题 Apptainer 试跑](contextbench_apptainer/README.md) 验收 Harbor、原版任务流与评分。该入口只运行 off 组的一题，不等同于论文 SGLang SCR 数值复现。

用冻结的源码 checkout 保留原任务、方法 skill 和容器里的官方 scorer。在服务存活时运行，保存原版 trial 与 `verifier/` 输出。该命令用公开 `ClmStream` 默认预算，样本为仓库内预打包的 `kv-store-R1000-seed42`。

```bash
source "$E/pins.env"
export CB_REPO="$PWD/../ContextBench"
git clone https://github.com/RulinShao/ContextBench.git "$CB_REPO"
git -C "$CB_REPO" checkout --detach "$CB_REVISION"
python -m pip install -e "$CB_REPO[tokenizer]"
bash "$E/run.sh" contextbench off "$OUT/manual"
```

### BrowseComp-Plus 配对

先准备官方数据。ContextBench 的原版任务随其源码仓库的 `tasks/` 提供；BCP 的原题、语料和预建索引分别在 Hugging Face 的 `Tevatron/browsecomp-plus`、`Tevatron/browsecomp-plus-corpus`、`Tevatron/browsecomp-plus-indexes`。下面的下载器使用 `huggingface_hub`，默认经 HF-Mirror 获取文件，在本机 scratch 暂存，再按逐文件大小核验上传到 OSS 的 `data/<DATA_RUN>/`，写 `_SUCCESS` 后删除该次暂存目录。失败时保留暂存，使用相同 `DATA_RUN` 重试；若镜像失效，设置 `HF_ENDPOINT=https://huggingface.co` 重试。不会上传解密后的题目或 Hugging Face 下载缓存。尚未验证这些数据在 ModelScope 上有同版本发布，不以第三方同名数据替代。模型权重已经另行归档，不在此重复下载。

```bash
conda activate clm
E=experiments/history_edit_scr
source "$E/pins.env"
CLM_REPO=$(git rev-parse --show-toplevel)
CB_REPO="$(dirname "$CLM_REPO")/ContextBench"
BCP_REPO="$(dirname "$CLM_REPO")/BrowseComp-Plus"
test -e "$CB_REPO" || git clone https://github.com/RulinShao/ContextBench.git "$CB_REPO"
git -C "$CB_REPO" checkout --detach "$CB_REVISION"
test -e "$BCP_REPO" || git clone https://github.com/texttron/BrowseComp-Plus.git "$BCP_REPO"
git -C "$BCP_REPO" checkout --detach "$BCP_REVISION"
test -d "$CB_REPO/tasks"
DATA_RUN="bcp-data-$(date -u +%Y%m%dT%H%M%SZ)"
python "$E/official_data.py" "$DATA_RUN"
ossutil stat "oss://industry-algo/yanlin/clm/data/$DATA_RUN/_SUCCESS"
```

源码 checkout 是小文件，位于持久 home；BCP 大数据只在 OSS 留正式副本。若服务器中已有这两个源码 checkout，先用 `git -C "$CB_REPO" rev-parse HEAD` 等命令检查版本，不覆盖现有修改。BCP 的原始数据到位不代表 Harbor 任务包、检索工具和判分器已对齐；这些仍需另行接入和验收。

按 BrowseComp-Plus 的官方说明准备原始题、固定 corpus、检索器和索引、judge。CLM 的公开入口接收现成 Harbor 任务目录，仓库未提供含这些资源的 BCP Harbor 任务包。将已经准备的任务目录放在 `BCP_TASK_ROOT`，每个任务包含 `task.toml`、`instruction.md`、`environment/` 和 `tests/`。在 `BCP_TASK_LIST` 每行写一个目录名，冻结 8–16 题。用 `BCP_PROVENANCE` 指向 JSON，填写 `corpus_revision`、`queries_revision`、`retriever`、`index_revision`、`judge_revision` 的真实值。

```bash
source "$E/pins.env"
export BCP_REPO="$PWD/../BrowseComp-Plus"
git clone https://github.com/texttron/BrowseComp-Plus.git "$BCP_REPO"
git -C "$BCP_REPO" checkout --detach "$BCP_REVISION"
export BCP_TASK_ROOT=/path/to/prepared/bcp-harbor-tasks
export BCP_TASK_LIST=/path/to/frozen-task-names.txt
export BCP_PROVENANCE=/path/to/bcp-provenance.json
bash "$E/run.sh" bcp off "$OUT/manual"
# 在 on 服务存活时执行同一清单
bash "$E/run.sh" bcp on "$OUT/manual"
```

包装脚本核对两组任务文件 hash 和外部资源记录，原样保存 Harbor 的 judge、工具和轨迹输出，并按照每题独占服务的时间窗切分请求日志，用官方解析器计算 forwarded 和 PFLOPs。公开 `bcp.yaml` 的 28,672 token 和 500 task steps 用于试跑配对，对照论文 §5.3 时同时记录检索、读文档、采样、超限回滚与 judge 条件。附录 E 的 23,560 token、100 turns 是另一项公开描述的零训练设置。自由续跑的 token 轨迹预计会分叉。

## D1 静态分支与 ALFWorld 协议

```bash
PYTHONPATH=clm python "$E/edit_fixtures.py" prepare "$OUT/d1-static.jsonl"
PYTHONPATH=clm python "$E/edit_fixtures.py" score \
  "$OUT/d1-static.jsonl" /path/to/events.jsonl "$OUT/d1-scores.jsonl"
```

生成 12 个单次反证更新快照、8 个不使计划失效的单次对照，以及 8 条含 2–4 个更新的序列。快照含原始观察、稳定 span ID 和 keep、drop-all、oracle、近似等长随机删除分支。文件用官方 context 文本格式做 round-trip。事件输入每行包含 `snapshot_id`、`policy` 和 `events`，先记录各 lane 的 `{"action":"dispatch","lane":"north","station":"A"}` 事件，再记录 `{"action":"handoff","credential":"permit-123456"}` 事件。判分输出首个 dispatch 是否陈旧、是否使用独立约束和任务是否成功。这里的轨迹和观察是静态合成材料，真实 Harness 回接、零训练自编辑和环境续跑待验证。

CLM 有 `context_snapshots`、`trajectory.ctx.json` 和用于恢复工作目录的 `utils/resume.py`。固定基线的 `context_string.parse_back` 会扁平化已编辑历史的工具结构和 reasoning 字段。正式接真实轨迹时，先用无语义编辑核对未编辑消息的角色与字段，再比较显式删笔记和模板自动剥离。

ICLR 附件 §3.1–3.3、Figure 2、附录 C.1 和 Table 3 使用 ALFWorld 非终止切点。把 30-turn 源轨迹按 1–10、11–20、21–30 分组，每组最多 32 个切点，同一源 episode 最多贡献 3 个。重放切点之前已实际执行的动作，验证最后的观测和当前环境状态。完整视图和动作—观测视图共享任务、动作、观测、环境以及每次 continuation 的 seed。后者仅移除既有 reasoning，两组之后都生成新的 reasoning 和动作。原实验每分支采 4 次，最多续跑 30 turns、每 turn 512 tokens，temperature 1.0、top-p 1.0、top-k -1，以任务成功判分并按源 episode bootstrap。本实验扩展时以同一个 CLM checkpoint 读取自己的历史。CLM 基线未包含该 ALFWorld 的导出或环境重放器。附件给出的公开匿名代码链接访问返回 HTTP 403，轨迹与恢复接口待核验。

- [官方 SCR 配置与日志](https://github.com/facebookresearch/context-language-models/tree/3c89408be27d34e9e8d14eade365a8d440b041e9/suffix_cache_reuse/sglang)
- [CLM 论文](https://arxiv.org/html/2609.37725v1)
- [ContextBench 任务与评分](https://github.com/RulinShao/ContextBench/tree/d91c442a56bde006ed19b2a3070b61c1881c6fae)
- [BrowseComp-Plus 数据和索引](https://github.com/texttron/BrowseComp-Plus/tree/046949032b0328319cc9a02663a759ec601d9402)
- [SGLang 0.5.16 依赖](https://github.com/sgl-project/sglang/blob/v0.5.16/python/pyproject.toml)
