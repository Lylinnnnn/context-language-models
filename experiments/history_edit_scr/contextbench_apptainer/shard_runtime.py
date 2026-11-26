#!/usr/bin/env python3
"""One ContextBench shard: vLLM service, Harbor trials, and durable task status."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen
from urllib.error import URLError

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPO = HERE.parents[2]


def load_result(path):
    if not path.is_file():
        raise RuntimeError(f"Harbor result missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    error = result.get("exception_info")
    if error:
        raise RuntimeError(f"{error.get('exception_type')}: {error.get('exception_message')}")
    verifier = result.get("verifier_result")
    if verifier is None:
        raise RuntimeError(f"Harbor verifier result missing: {path}")
    rewards = verifier.get("rewards") or {}
    if "reward" not in rewards:
        raise RuntimeError(f"Harbor reward missing: {path}")
    return rewards


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def ready(port, process, log):
    for _ in range(360):
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup; see {log}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(5)
    raise RuntimeError(f"vLLM startup timed out; see {log}")


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def worker(shard_id, gpu, names, args, root, cb_repo, image_cache, apptainer_home, run):
    status_dir = root / "status"
    pending = []
    for name in names:
        status_file = status_dir / f"{name}.json"
        if args.resume and status_file.is_file():
            previous = json.loads(status_file.read_text(encoding="utf-8"))
            if previous.get("ok"):
                try:
                    load_result(root / previous["result"])
                    print(f"[GPU {gpu}] skip completed {name}", flush=True)
                    continue
                except (KeyError, RuntimeError, ValueError):
                    pass
        pending.append(name)
    if not pending:
        return

    port = (32000 if args.mode == "off" else 32100) + gpu
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu), "VLLM_OFF_PORT": str(port),
        "VLLM_ON_PORT": str(port), "VLLM_MAX_MODEL_LEN": "40960",
        "VLLM_ENABLE_AUTO_TOOL_CHOICE": "1", "VLLM_TOOL_CALL_PARSER": "qwen3_coder",
        "APPTAINER_CACHEDIR": env.get("APPTAINER_CACHEDIR", "/tmp/clm/apptainer-cache"),
        "APPTAINER_TMPDIR": env.get("APPTAINER_TMPDIR", "/tmp/clm/apptainer-tmp"),
        "APPTAINER_NO_MOUNT": "proc", "OPENAI_API_KEY": env.get("OPENAI_API_KEY", "EMPTY"),
        "FLASHINFER_WORKSPACE_BASE": str(run / "tmp" / f"flashinfer-cb-gpu{gpu}"),
        "PATH": f"{apptainer_home}/bin:{env['PATH']}",
        "PYTHONPATH": f"{REPO}/clm:{cb_repo}" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""),
    })
    Path(env["FLASHINFER_WORKSPACE_BASE"]).mkdir(parents=True, exist_ok=True)
    Path(env["APPTAINER_CACHEDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["APPTAINER_TMPDIR"]).mkdir(parents=True, exist_ok=True)
    server_root = root / "servers" / f"gpu{gpu}" / f"attempt-{int(time.time())}"
    server_root.mkdir(parents=True)
    # Stagger eight model loads and FlashInfer JIT builds.
    time.sleep(5 * shard_id)
    process = subprocess.Popen(
        ["bash", str(EXPERIMENT / "vllm_fallback/run.sh"), "serve", args.mode, str(server_root)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    log = server_root / args.mode / "server.log"
    try:
        ready(port, process, log)
        print(f"[GPU {gpu}] ready; {len(pending)} tasks; log {log}", flush=True)
        for name in pending:
            trial_dir = root / "trials"
            trial_dir.mkdir(exist_ok=True)
            attempt = 0
            while (trial_dir / f"{name}-vllm-{args.mode}-attempt{attempt}").exists():
                attempt += 1
            trial_name = f"{name}-vllm-{args.mode}-attempt{attempt}"
            result_path = trial_dir / trial_name / "result.json"
            output = root / "logs" / f"{trial_name}.log"
            output.parent.mkdir(exist_ok=True)
            command = [
                "harbor", "trials", "start", "-p", str(cb_repo / "tasks" / name),
                "-e", "singularity",
                "--environment-kwarg", f"singularity_image_cache_dir={image_cache}",
                "--environment-kwarg", "singularity_no_mount=home,tmp,bind-paths,proc",
                "-a", "clm_contextbench:ClmStream", "-m", "openai/qwen36-27b",
                "--agent-kwarg", f"api_base=http://127.0.0.1:{port}/v1",
                "--trial-name", trial_name, "--trials-dir", str(trial_dir),
            ]
            print(f"[GPU {gpu}] start {name}", flush=True)
            with output.open("w", encoding="utf-8") as log_file:
                completed = subprocess.run(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            try:
                if completed.returncode:
                    raise RuntimeError(f"Harbor exit {completed.returncode}; see {output}")
                rewards = load_result(result_path)
                status = {"ok": True, "task": name, "gpu": gpu, "reward": rewards["reward"],
                          "rewards": rewards, "result": str(result_path.relative_to(root)),
                          "log": str(output.relative_to(root))}
                print(f"[GPU {gpu}] done {name}: reward={rewards['reward']}", flush=True)
            except (RuntimeError, ValueError) as exc:
                status = {"ok": False, "task": name, "gpu": gpu, "error": str(exc),
                          "result": str(result_path.relative_to(root)),
                          "log": str(output.relative_to(root))}
                print(f"[GPU {gpu}] FAILED {name}: {exc}", flush=True)
            write_json(status_dir / f"{name}.json", status)
            if process.poll() is not None:
                raise RuntimeError(f"vLLM exited during task execution; see {log}")
    finally:
        stop(process)

