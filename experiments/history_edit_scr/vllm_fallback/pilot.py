#!/usr/bin/env python3
"""Run official vLLM SCR smoke and optional frozen probes with one GPU."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAUNCH = ["bash", str(HERE / "run.sh")]


def wait_ready(process, port, log, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}; see {log}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=4) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"server not ready within {timeout}s; see {log}")


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_arm(mode, root, timeout, manifest, seeds):
    port = int(os.environ.get(f"VLLM_{mode.upper()}_PORT", 32000 if mode == "off" else 32001))
    with socket.socket() as check:
        check.bind(("127.0.0.1", port))
    log = root / mode / "server.log"
    print(f"Starting vLLM SCR {mode} on port {port}; log {log}", flush=True)
    server = subprocess.Popen([*LAUNCH, "serve", mode, str(root)], start_new_session=True)
    try:
        wait_ready(server, port, log, timeout)
        subprocess.run([*LAUNCH, "smoke", mode, str(root)], check=True)
        if manifest:
            subprocess.run([
                sys.executable, str(HERE / "probe.py"), str(manifest),
                str(root / mode / "probe-requests.jsonl"), "--arm", mode,
                "--url", f"http://127.0.0.1:{port}",
                "--model-revision", os.environ["CLM_MODEL_REVISION"],
                "--seeds", *map(str, seeds),
            ], check=True)
    finally:
        stop(server)
    subprocess.run([*LAUNCH, "report", mode, str(root)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="a new result directory")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--manifest", type=Path, help="frozen C probe manifest")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()
    if args.startup_timeout <= 0:
        parser.error("startup timeout must be positive")
    if not os.environ.get("CLM_MODEL_DIR"):
        parser.error("set CLM_MODEL_DIR to the pinned model on local scratch")
    if args.manifest and not os.environ.get("CLM_MODEL_REVISION"):
        parser.error("set CLM_MODEL_REVISION for the frozen probe model")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("seeds must be distinct and non-negative")
    if args.manifest:
        args.manifest = args.manifest.resolve(strict=True)
        sys.path.insert(0, str(HERE.parent))
        from probe import tokenizer_files

        manifest = json.loads(args.manifest.read_text())
        snapshot = Path(os.environ["CLM_MODEL_DIR"]).resolve()
        if (manifest["schema"] != 1 or manifest["model_revision"] != os.environ["CLM_MODEL_REVISION"]
                or Path(manifest["tokenizer_snapshot"]).resolve() != snapshot
                or manifest["tokenizer_files_sha256"] != tokenizer_files(snapshot)
                or manifest["transformers"] != version("transformers")):
            parser.error("probe manifest differs from the staged model/tokenizer or transformers")
    args.directory.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHON"] = sys.executable
    for mode in ("off", "on"):
        run_arm(mode, args.directory, args.startup_timeout, args.manifest, args.seeds)
    if args.manifest:
        subprocess.run([sys.executable, str(HERE / "report.py"), str(args.directory),
                        str(args.manifest), "--seeds", *map(str, args.seeds)], check=True)
    print(f"vLLM SCR smoke complete: {args.directory}")


if __name__ == "__main__":
    main()
