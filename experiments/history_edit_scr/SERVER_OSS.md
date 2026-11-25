# 服务器确认与 OSS 存储

## 先在 GPU 节点检查

在实际分配到 GPU 的节点执行以下命令。根据输出选择 Python 环境和本机盘，已有可用环境或配置可以复用。

```bash
uname -sm
getconf GNU_LIBC_VERSION
grep '^PRETTY_NAME=' /etc/os-release
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
python3 --version
python3.12 --version
python3.12 -m ensurepip --version
command -v conda mamba srun sbatch docker apptainer ossutil
df -h /tmp
findmnt -T /tmp -o TARGET,FSTYPE,AVAIL
findmnt -T /data/oss_bucket_0/yanlin -o TARGET,FSTYPE
ossutil version
```

确认当前系统或容器的 glibc 至少为 2.34 后再创建环境。机器有 Python 3.12 且能创建 venv 时可以用 venv。若机器已经使用 Conda 管理 Python，可以用 `conda create -n clm-scr python=3.12 pip` 和 `conda activate clm-scr`。激活选定环境后执行 [依赖安装](README.md#服务器环境) 中的 pip 命令。若从 Slurm 获得设备，保留作业分配的 `CUDA_VISIBLE_DEVICES`。

## 确认 OSS

请管理员确认 bucket `industry-algo` 和候选内网 Endpoint `cn-zhangjiakou.oss-internal.aliyun-inc.com`、挂载是否映射到 `yanlin/`、执行身份是否有权限访问 `yanlin/clm/*`。挂载路径本身不能验证 Endpoint 和地域。OSS 前缀可随首个对象创建。

保留已有可用的 ossutil 配置。需要新增配置时按所装版本选择已授予的实例 RAM 角色，或通过 `umask 077` 后交互式运行 `ossutil config` 配置被授权的 RAM 用户。2.x 需要实际地域 ID。若使用默认配置文件，以 `chmod 600 ~/.ossutilconfig` 限制文件权限。实例角色与挂载可能各用各的身份。项目变量只指定路径，凭据由服务器的 ossutil 配置或角色提供。

```bash
ossutil ls oss://industry-algo/yanlin/clm/
```

列表命令成功后，可用 `ossutil stat` 查询固定模型前缀里的 `_SUCCESS`。对象存在时走下文的 OSS 暂存流程。返回 `NoSuchKey` 时，先准备并归档该 revision。遇到权限或 Endpoint 错误时，先核对服务器的现有配置和管理员授权。

管理员可将 bucket 的 ListObjects 权限限定到 `yanlin/clm/` 下的请求前缀，并将 GetObject、PutObject 授予 `acs:oss:*:*:industry-algo/yanlin/clm/*`。分片上传还需核对 ListParts 和 AbortMultipartUpload。本流程只需要读、写和查询，对象删除权限按实际清理需求单独授予。

[ossutil 配置](https://help.aliyun.com/zh/oss/developer-reference/ossutil-overview/) · [RAM 前缀授权](https://help.aliyun.com/en/oss/user-guide/access-control-base-on-ram-policy) · [OSS 目录](https://help.aliyun.com/en/oss/user-guide/manage-directories) · [ossutil 版本差异](https://help.aliyun.com/en/oss/developer-reference/1-0-and-2-0-functional-differences)

## 权重和运行目录

[默认路径](storage.env) 可通过环境变量覆盖。以挂载读取固定数据，随机读吞吐不足时缓存到本机。运行时权重、训练状态、合并与传输临时文件写在本机 scratch，完成的产物再由 ossutil 归档。下面的传输脚本按 ossutil 2.x 的 `cp`、`stat` 及续传参数编写。模型约 56 GB，换用本地 NVMe 时先设置 `CLM_SCRATCH_ROOT` 为实际挂载位置，再检查可用字节、配额和临时目录是否由作业回收。

```bash
E=experiments/history_edit_scr
RUN_ID="clm-$(date -u +%Y%m%dT%H%M%SZ)-$$"
# 默认 /tmp/clm。若 /tmp 空间不足，先 export CLM_SCRATCH_ROOT 到本机 NVMe。
umask 077
RUN=$(python "$E/storage.py" run-dir "$RUN_ID")
mkdir -p "$RUN"
df -h "$RUN"
findmnt -T "$RUN" -o TARGET,FSTYPE,AVAIL
export TMPDIR="$RUN/tmp"
export HF_HOME="$RUN/hf"
mkdir -p "$TMPDIR"
OUT="$RUN/eval"
mkdir -p "$OUT"
```

如固定 revision 已经归档到 OSS，先暂存权重，再按主 README 的「A 与 C 组首轮」运行。脚本读取完成标记和文件清单，逐文件验证 SHA-256，通过后才将暂存目录改名为正式本机权重目录。继续使用同一个 run 时可通过 `model-dir` 找到已经暂存好的权重，再用 `run.sh snapshot` 检查必需文件。

```bash
CLM_MODEL_DIR=$(python "$E/storage.py" stage-model "$RUN_ID") || exit
export CLM_MODEL_DIR
```

若该 revision 尚未归档，在能访问 Hugging Face 且本机盘足够的机器上执行一次下面的命令。已按固定 revision 准备好的完整权重目录也可放到 `model-dir` 输出的位置。共享模型前缀由一个上传者负责。

```bash
CLM_MODEL_DIR=$(python "$E/storage.py" model-dir "$RUN_ID") || exit
export CLM_MODEL_DIR
bash "$E/run.sh" download
bash "$E/run.sh" snapshot
python "$E/storage.py" publish-model "$RUN_ID"
```

固定权重归档到 `models/Qwen/Qwen3.6-27B/<revision>/`，评估和训练产物分别归档到 `eval/<run_id>/`、`ckpt/<run_id>/`。本机盘连一份完整权重和运行开销都放不下时，需要先安排本机 NVMe 或另一台 GPU 节点。

## 归档结果

评估结束后上传 `$RUN/eval`。如果将来需要续训，把完整训练状态放在 `$RUN/ckpt` 并用 `publish-ckpt` 单独归档。确认文件已停止写入，且目录里只有预期产物。环境、下载缓存和传输断点应放在各自的 scratch 路径，凭据保留在服务器的凭据配置中。

```bash
python "$E/storage.py" publish-eval "$RUN_ID"
```

上传按清单逐文件进行，之后用 `ossutil stat` 核对每个对象的字节数，最后写入 SHA-256 绑定的 `manifest.json` 和 `_SUCCESS`。续传元数据位于 `$RUN/transfer`。传输中断可用同一个 `RUN_ID` 重试，已完成前缀会拒绝覆写。只有验证成功并确认该 run 的产物齐全后，才手动清理它自己的 scratch。推理权重或 LoRA adapter 只提供模型参数，续训还要保留训练框架要求的优化器、step、随机数等状态。

[上传和续传](https://www.alibabacloud.com/help/en/oss/developer-reference/cp-upload-file) · [对象大小核对](https://www.alibabacloud.com/help/en/oss/developer-reference/stat)
