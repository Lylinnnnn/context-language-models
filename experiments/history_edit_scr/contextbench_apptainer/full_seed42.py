#!/usr/bin/env python3
"""Run the 32 original ContextBench seed-42 tasks on eight GPU shards."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

from shard_runtime import worker, write_json


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPO = HERE.parents[2]
FAMILIES = ("kv-store", "log-triage", "needle-retention", "sudoku-sketchpad")
EXPECTED_COUNTS = (8, 7, 10, 7)
COMPATIBLE_RUNNER_COMMITS = {
    "cfadf111058ef7547af7c3e5bbdca97425d7c2df",
    "6dc5d1009bae9275f0d9c454e9815f0af48be961",
}


def git_head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def load_pins():
    return dict(line.split("=", 1) for line in (EXPERIMENT / "pins.env").read_text().splitlines())


def family(name):
    return next((value for value in FAMILIES if name.startswith(value + "-")), None)


def tasks_and_shards(cb_repo):
    tasks = sorted(p for p in (cb_repo / "tasks").iterdir()
                   if p.is_dir() and p.name.endswith("-seed42") and (p / "task.toml").is_file())
    counts = tuple(sum(family(p.name) == value for p in tasks) for value in FAMILIES)
    if counts != EXPECTED_COUNTS or len(tasks) != 32:
        raise RuntimeError(f"Expected 32 pinned paper tasks for seed 42, found {len(tasks)}; "
                           f"family counts {dict(zip(FAMILIES, counts))}")
    sizes = {}
    for task in tasks:
        stream = task / "environment/stream/chunks.jsonl"
        if not stream.is_file():
            raise RuntimeError(f"Missing original task stream: {stream}")
        sizes[task.name] = stream.stat().st_size
    shards = [[] for _ in range(8)]
    totals = [0] * 8
    for task in sorted(tasks, key=lambda p: (-sizes[p.name], p.name)):
        shard = min((i for i in range(8) if len(shards[i]) < 4), key=lambda i: (totals[i], i))
        shards[shard].append(task.name)
        totals[shard] += sizes[task.name]
    return shards, sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--mode", choices=("off", "on"), default="off")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if len(gpus) != 8 or len(set(gpus)) != 8 or min(gpus) < 0:
        parser.error("--gpus must list eight distinct nonnegative GPU indices")
    if version("harbor") != "0.16.1":
        parser.error("Expected harbor==0.16.1")
    pins = load_pins()
    cb_repo = Path(os.environ.get("CB_REPO", str(REPO.parent / "ContextBench"))).resolve()
    sys.path[:0] = [str(REPO / "clm"), str(cb_repo)]
    import contextbench, clm_contextbench  # noqa: F401 - preflight imports
    if git_head(cb_repo) != pins["CB_REVISION"]:
        parser.error(f"ContextBench checkout differs from pinned {pins['CB_REVISION']}: {cb_repo}")
    image_cache = Path(os.environ.get("CLM_CB_IMAGE_CACHE", "/tmp/clm/contextbench-image-cache"))
    if not (image_cache / "python_3.12-slim.sif").is_file():
        parser.error(f"Harbor-ready image missing: {image_cache}/python_3.12-slim.sif")
    apptainer_home = Path(os.environ.get("CLM_APPTAINER_HOME", str(Path.home() / ".local/opt/apptainer-1.5.5")))
    if not (apptainer_home / "bin/singularity").is_file():
        parser.error(f"Apptainer launcher missing: {apptainer_home}/bin/singularity")
    help_text = subprocess.check_output(["harbor", "trials", "start", "--help"], text=True)
    if "--environment-kwarg" not in help_text:
        parser.error("Harbor CLI lacks --environment-kwarg")
    shards, sizes = tasks_and_shards(cb_repo)
    run = Path(subprocess.check_output(
        [sys.executable, str(EXPERIMENT / "storage.py"), "run-dir", args.run_id], text=True).strip())
    root = run / "eval" / f"contextbench-s42-vllm-{args.mode}"
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not args.resume:
        parser.error(f"Output already exists: {root}; use --resume")
    if args.resume and not manifest_path.exists():
        parser.error(f"Cannot resume without manifest: {manifest_path}")
    prior = json.loads(manifest_path.read_text()) if args.resume and manifest_path.exists() else None
    model_setting = os.environ.get("CLM_MODEL_DIR") or (prior or {}).get("model_dir")
    if model_setting:
        model = Path(model_setting).resolve()
    else:
        if not os.environ.get("CLM_MODEL_REVISION"):
            parser.error("Source vllm_fallback/modelscope.env before staging model weights")
        model = run / "model" / os.environ["CLM_MODEL_REVISION"]
        if not (model / "model.safetensors.index.json").is_file():
            model = Path(subprocess.check_output([
                sys.executable, str(EXPERIMENT / "storage.py"), "stage-model", args.run_id,
            ], text=True).strip())
    if not (model / "model.safetensors.index.json").is_file():
        parser.error(f"Staged model is missing: {model}")
    os.environ["CLM_MODEL_DIR"] = str(model)
    owned_model = model.parent == run / "model"
    manifest = {"seed": 42, "mode": args.mode, "gpus": gpus, "shards": shards,
                "task_stream_bytes": sizes, "contextbench_commit": git_head(cb_repo),
                "clm_commit": git_head(REPO), "model_dir": str(model.resolve()),
                "model_revision": os.environ.get("CLM_MODEL_REVISION", ""),
                "max_model_len": 40960, "tool_call_parser": "qwen3_coder"}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        same_except_commit = all(previous.get(key) == value for key, value in manifest.items()
                                 if key != "clm_commit")
        compatible_old_runner = (previous.get("clm_commit") in COMPATIBLE_RUNNER_COMMITS
                                 and same_except_commit)
        if not args.resume or (previous != manifest and not compatible_old_runner):
            parser.error(f"Output already exists or configuration changed: {root}; use --resume with identical inputs")
        if compatible_old_runner and previous != manifest:
            print("Resuming run created before automatic OSS publication; trial inputs are unchanged", flush=True)
    else:
        if args.resume:
            parser.error(f"Cannot resume without manifest: {manifest_path}")
        root.mkdir(parents=True)
        write_json(manifest_path, manifest)
    for gpu in gpus:
        port = (32000 if args.mode == "off" else 32100) + gpu
        with socket.socket() as connection:
            if connection.connect_ex(("127.0.0.1", port)) == 0:
                parser.error(f"Port {port} already in use; stop that service before running")
    print(f"32 original ContextBench seed-42 tasks, 8 shards, output {root}", flush=True)
    errors = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i, gpu, names, args, root, cb_repo, image_cache,
                               apptainer_home, run) for i, (gpu, names) in enumerate(zip(gpus, shards))]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
                print(f"Shard failed: {exc}", file=sys.stderr, flush=True)
    statuses = {}
    for names in shards:
        for name in names:
            path = root / "status" / f"{name}.json"
            if path.is_file():
                statuses[name] = json.loads(path.read_text())
    passed = {name: item for name, item in statuses.items() if item.get("ok")}
    family_summary = {}
    for value, expected in zip(FAMILIES, EXPECTED_COUNTS):
        rows = [item for name, item in passed.items() if family(name) == value]
        family_summary[value] = {
            "expected": expected, "completed": len(rows),
            "mean_reward": (sum(float(item["reward"]) for item in rows) / len(rows) if rows else None),
        }
    summary = {"expected": 32, "completed": len(passed), "failed": len(statuses) - len(passed),
               "missing": sorted(set(sizes) - set(statuses)), "shard_errors": errors,
               "mean_reward": (sum(float(item["reward"]) for item in passed.values()) / len(passed)
                               if passed else None), "families": family_summary, "tasks": statuses}
    write_json(root / "summary.json", summary)
    print(f"Completed {len(passed)}/32; summary {root / 'summary.json'}", flush=True)
    if len(passed) != 32 or errors:
        print("Local results retained for --resume; OSS publication skipped", flush=True)
        return 1
    subprocess.run([
        sys.executable, str(EXPERIMENT / "storage.py"), "publish-eval", args.run_id,
        "--subdir", root.name, "--cleanup",
    ], check=True)
    if owned_model:
        subprocess.run([
            sys.executable, str(EXPERIMENT / "storage.py"), "publish-model", args.run_id,
            "--cleanup",
        ], check=True)
    if (run / "tmp").is_symlink():
        raise RuntimeError(f"Refusing to remove caches through symlinked temp directory: {run / 'tmp'}")
    for gpu in gpus:
        cache = run / "tmp" / f"flashinfer-cb-gpu{gpu}"
        if cache.is_symlink():
            raise RuntimeError(f"Refusing to remove symlinked JIT cache: {cache}")
        if cache.is_dir():
            shutil.rmtree(cache)
    print("OSS publication verified; local eval copy removed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
