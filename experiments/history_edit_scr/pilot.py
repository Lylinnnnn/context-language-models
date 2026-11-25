#!/usr/bin/env python3
"""Run official smoke and an optional native probe on one GPU, one server at a time."""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from probe import HERE

PYTHON = sys.executable
LAUNCH = ["bash", str(HERE / "run.sh")]


def wait_ready(process, url, log, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}; see {log}")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=4) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"server not ready within {timeout}s; see {log}")


def stop(process):
    # The launcher and all SGLang workers are in a process group owned by this run.
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_arm(arm, port, directory, manifest, timeout):
    with socket.socket() as port_check:
        port_check.bind(("127.0.0.1", port))
    args = [arm, str(directory)]
    url = f"http://127.0.0.1:{port}"
    log = directory / arm / "server.log"
    print(f"Starting SCR {arm} on {url}; log {log}", flush=True)
    server = subprocess.Popen([*LAUNCH, "serve", *args], start_new_session=True)
    try:
        wait_ready(server, url, log, timeout)
        subprocess.run([*LAUNCH, "smoke", *args], check=True)
        if manifest:
            subprocess.run(
                [PYTHON, str(HERE / "probe.py"), "run", str(manifest), str(log.parent / "requests.jsonl"),
                 "--arm", arm, "--url", url],
                check=False,  # preserve a failed arm and score incomplete pairs after the other arm
            )
    finally:
        stop(server)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="a new result directory")
    parser.add_argument("--manifest", type=Path, help="add a pre-rendered native probe manifest")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.startup_timeout <= 0:
        parser.error("startup timeout must be positive")
    if args.manifest and not args.manifest.is_file():
        parser.error(f"manifest not found: {args.manifest}")
    args.directory.mkdir(parents=True)
    os.environ["PYTHON"] = PYTHON
    for arm, default in (("off", 31000), ("on", 31001)):
        port = int(os.environ.get(f"{arm.upper()}_PORT", default))
        run_arm(arm, port, args.directory, args.manifest, args.startup_timeout)
    command = [PYTHON, str(HERE / "report.py"), str(args.directory)]
    if args.manifest:
        command += ["--manifest", str(args.manifest), "--require-clean-hit"]
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
