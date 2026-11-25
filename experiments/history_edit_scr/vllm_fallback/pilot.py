#!/usr/bin/env python3
"""Run the official vLLM SCR smoke sequentially with one GPU."""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
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


def run_arm(mode, root, timeout):
    port = int(os.environ.get(f"VLLM_{mode.upper()}_PORT", 32000 if mode == "off" else 32001))
    with socket.socket() as check:
        check.bind(("127.0.0.1", port))
    log = root / mode / "server.log"
    print(f"Starting vLLM SCR {mode} on port {port}; log {log}", flush=True)
    server = subprocess.Popen([*LAUNCH, "serve", mode, str(root)], start_new_session=True)
    try:
        wait_ready(server, port, log, timeout)
        subprocess.run([*LAUNCH, "smoke", mode, str(root)], check=True)
    finally:
        stop(server)
    subprocess.run([*LAUNCH, "report", mode, str(root)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="a new result directory")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.startup_timeout <= 0:
        parser.error("startup timeout must be positive")
    if not os.environ.get("CLM_MODEL_DIR"):
        parser.error("set CLM_MODEL_DIR to the pinned model on local scratch")
    args.directory.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHON"] = sys.executable
    for mode in ("off", "on"):
        run_arm(mode, args.directory, args.startup_timeout)
    print(f"vLLM SCR smoke complete: {args.directory}")


if __name__ == "__main__":
    main()
