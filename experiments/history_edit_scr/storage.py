#!/usr/bin/env python3
"""Stage pinned weights and publish selected CLM artifacts with ossutil."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

from probe import HERE, PINS

DEFAULTS = dict(line.split("=", 1) for line in (HERE / "storage.env").read_text().splitlines())
REVISION = os.environ.get("CLM_MODEL_REVISION") or PINS["MODEL_REVISION"]
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", REVISION):
    raise ValueError(f"invalid model artifact revision: {REVISION!r}")
MODEL_PREFIX = f"models/{PINS['MODEL_ID']}/{REVISION}"
EXCLUDED = {".cache", ".venv", ".git", "__pycache__", ".ossutil_checkpoint"}
SHA256 = re.compile(r"[0-9a-f]{64}")
SUCCESS = "_SUCCESS"
MANIFEST = "manifest.json"


def setting(name):
    return os.environ.get(name) or DEFAULTS[name]


def local_path(path):
    path = Path(path).resolve()
    mount = Path(setting("CLM_MOUNT_PREFIX")).resolve()
    if path.is_relative_to(mount):
        raise ValueError(f"use local scratch instead of the OSS mount for {path}")
    return path


def model_location(path):
    path = local_path(path)
    if path.name != REVISION:
        raise ValueError(f"model directory must end in the pinned revision {REVISION}: {path}")
    return path


def model_ready(path):
    path = model_location(path)
    check_model_files(path)
    return path


def check_model_files(path):
    index = path / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values())
    if not shards or not all(valid_name(name) for name in shards):
        raise ValueError(f"invalid weight shard index: {index}")
    files = shards | {"config.json", "tokenizer_config.json", "tokenizer.json"}
    missing = sorted(name for name in files if not (path / name).is_file())
    if missing:
        raise ValueError(f"incomplete model at {path}: {missing}")


def valid_name(name):
    path = PurePosixPath(name)
    return (
        bool(path.parts) and not path.is_absolute() and path.as_posix() == name
        and not any(part in {"..", "."} or "\\" in part or any(ord(c) < 32 for c in part) for part in path.parts)
    )


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect_files(root):
    if root.is_symlink():
        raise ValueError(f"publish a local artifact directory: {root}")
    entries = {}
    for folder, directories, names in os.walk(root):
        directories[:] = sorted(set(directories) - EXCLUDED)
        for name in sorted([*directories, *names]):
            if name == ".gitattributes":
                continue
            path = Path(folder) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or name.startswith(".") or name.endswith(".ossutilconfig"):
                raise ValueError(f"exclude private entries and symlinks from the artifact directory: {path}")
            if path.is_dir():
                continue
            if not path.is_file() or not valid_name(relative):
                raise ValueError(f"unsupported artifact entry: {path}")
            entries[relative] = {"size": path.stat().st_size, "sha256": file_sha(path)}
    if not entries:
        raise ValueError(f"no artifacts to publish in {root}")
    return entries


def ossutil(*args, capture=False, check=True):
    # Only the server's existing ossutil configuration or role supplies authentication.
    return subprocess.run(
        [os.environ.get("OSSUTIL", "ossutil"), *map(str, args)],
        stdout=subprocess.PIPE if capture else sys.stderr,
        stderr=subprocess.PIPE if capture else None, check=check,
    )


def oss_url(suffix):
    return f"{setting('CLM_OSS_PREFIX').rstrip('/')}/{suffix}"


def confirm_unpublished(prefix):
    result = ossutil("stat", f"{prefix}/{SUCCESS}", capture=True, check=False)
    if result.returncode == 0:
        raise ValueError(f"already published: {prefix}")
    error = (result.stdout + result.stderr).decode(errors="replace")
    if not re.search(r"\b(?:NoSuchKey|404)\b", error, re.IGNORECASE):
        raise ValueError(f"could not check {prefix}/{SUCCESS}: {error.strip()}")


def copy_file(source, destination, checkpoints):
    ossutil("cp", source, destination, "-f", "--checkpoint-dir", checkpoints)


def verify_size(url, size):
    result = ossutil("stat", url, capture=True)
    match = re.search(rb"^\s*Content-Length\s*:\s*(\d+)\s*$", result.stdout, re.MULTILINE | re.IGNORECASE)
    if not match or int(match[1]) != size:
        raise ValueError(f"OSS object size verification failed: {url} (expected {size} bytes)")


def publish(run, kind):
    root = run / kind
    if kind == "model":
        root = model_ready(root / REVISION)
    prefix = oss_url(MODEL_PREFIX if kind == "model" else f"{kind}/{run.name}")
    confirm_unpublished(prefix)
    files = collect_files(root)
    workspace = run / "transfer" / f"publish-{kind}"
    workspace.mkdir(parents=True, exist_ok=True)
    manifest = workspace / MANIFEST
    marker = workspace / SUCCESS
    manifest.write_text(json.dumps({
        "kind": kind, "model_revision": REVISION if kind == "model" else None, "files": files,
        "source": ({"provider": os.environ.get("CLM_MODEL_SOURCE", "huggingface"),
                    "repo": PINS["MODEL_ID"],
                    "revision": os.environ.get("CLM_SOURCE_REVISION", REVISION)}
                   if kind == "model" else None),
    }, indent=2, sort_keys=True) + "\n")

    for name, entry in files.items():
        url = f"{prefix}/files/{name}"
        copy_file(root / name, url, workspace)
        verify_size(url, entry["size"])
    copy_file(manifest, f"{prefix}/{MANIFEST}", workspace)
    verify_size(f"{prefix}/{MANIFEST}", manifest.stat().st_size)
    marker.write_text(file_sha(manifest) + "\n")
    copy_file(marker, f"{prefix}/{SUCCESS}", workspace)
    verify_size(f"{prefix}/{SUCCESS}", marker.stat().st_size)
    print(f"{prefix}/{SUCCESS}")


def stage_model(run):
    target = run / "model" / REVISION
    if target.exists():
        raise ValueError(f"model directory already exists: {target}; use run.sh snapshot to reuse it")
    partial = target.with_name(f"{REVISION}.partial")
    partial.mkdir(parents=True, exist_ok=True)
    if partial.is_symlink():
        raise ValueError(f"staging directory must be local and materialized: {partial}")
    partial_root = partial.resolve()
    workspace = run / "transfer" / "stage-model"
    workspace.mkdir(parents=True, exist_ok=True)
    manifest, marker = workspace / MANIFEST, workspace / SUCCESS
    prefix = oss_url(MODEL_PREFIX)
    copy_file(f"{prefix}/{SUCCESS}", marker, workspace)
    copy_file(f"{prefix}/{MANIFEST}", manifest, workspace)
    if marker.read_text().strip() != file_sha(manifest):
        raise ValueError("model manifest hash differs from the OSS completion marker")
    data = json.loads(manifest.read_text())
    if data["kind"] != "model" or data["model_revision"] != REVISION or not data["files"]:
        raise ValueError("OSS manifest does not describe the pinned model")
    files = data["files"]
    for name, entry in files.items():
        if not valid_name(name) or not isinstance(entry["size"], int) or entry["size"] < 0 or not SHA256.fullmatch(entry["sha256"]):
            raise ValueError(f"invalid model manifest entry: {name}")
    for name, entry in files.items():
        path = partial / name
        if path.is_symlink() or not path.resolve().is_relative_to(partial_root):
            raise ValueError(f"staging path leaves the local model directory: {path}")
        valid = path.is_file() and path.stat().st_size == entry["size"] and file_sha(path) == entry["sha256"]
        if not valid:
            if shutil.disk_usage(partial).free < entry["size"]:
                raise ValueError(f"insufficient free local space to stage {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            copy_file(f"{prefix}/files/{name}", path, workspace)
            if path.stat().st_size != entry["size"] or file_sha(path) != entry["sha256"]:
                raise ValueError(f"model file verification failed: {name}")
    check_model_files(partial)
    partial.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run-dir", "model-dir", "stage-model", "publish-model", "publish-eval", "publish-ckpt"))
    parser.add_argument("run_id")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        parser.error("run ID must be one directory name")
    run = local_path(Path(setting("CLM_SCRATCH_ROOT")) / args.run_id)
    if args.action == "run-dir":
        print(run)
    elif args.action == "model-dir":
        print(run / "model" / REVISION)
    elif args.action == "stage-model":
        print(stage_model(run))
    else:
        publish(run, args.action.removeprefix("publish-"))


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        sys.exit(str(exc))
