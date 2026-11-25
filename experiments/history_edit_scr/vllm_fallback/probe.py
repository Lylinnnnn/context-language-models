#!/usr/bin/env python3
"""Run frozen history-edit probes through vLLM's token-ID completion endpoint."""

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe import digest, write_row  # noqa: E402


def send(url, payload, timeout):
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def request_probe(url, timeout, common, phase, ids, stream):
    rid = f"{common['cache_key']}-{phase}"
    payload = {
        "model": "qwen36-27b", "prompt": ids, "request_id": rid,
        "cache_salt": common["cache_key"], "add_special_tokens": False,
        "return_token_ids": True, "temperature": 0, "seed": common["seed"],
        "max_tokens": 32 if phase == "warm" else 12,
        "min_tokens": 8 if phase == "warm" else 0,
    }
    row = {
        **common, "phase": phase, "rid": rid,
        "input_sha256": digest(ids), "request": payload,
    }
    try:
        result = send(url, payload, timeout)
        row["response"] = result
        choice = result["choices"][0]
        usage = result["usage"]
        if result["id"] != f"cmpl-{rid}" and not result["id"].startswith(f"cmpl-{rid}-"):
            raise ValueError("response ID differs from request ID")
        if usage["prompt_tokens"] != len(ids) or choice["prompt_token_ids"] != ids:
            raise ValueError("vLLM prompt token IDs differ from frozen input")
        if usage["completion_tokens"] < (8 if phase == "warm" else 1):
            raise ValueError("too few completion tokens")
        if not isinstance(choice["text"], str):
            raise ValueError("invalid completion text")
    except urllib.error.HTTPError as exc:
        row["error"] = f"HTTP {exc.code}: {exc.read().decode(errors='replace')}"
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    write_row(stream, row)
    if "error" in row:
        print(f"{rid}: {row['error']}", file=sys.stderr)
    return "error" not in row


def run(args):
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != 1 or manifest["model_revision"] != args.model_revision:
        raise ValueError("manifest revision differs from the staged model")
    failures = 0
    run_id = uuid.uuid4().hex
    tags = ("case_id", "source_task_id", "fixture_type", "history_material", "ground_truth_label")
    with args.requests.open("x", encoding="utf-8") as stream:
        for case in manifest["cases"]:
            for branch in case["branches"]:
                for seed in args.seeds:
                    key = f"history-{args.arm}-{run_id}-{case['case_id']}-{branch['condition']}-{branch['warm_variant']}-{seed}"
                    common = {
                        **{tag: case[tag] for tag in tags},
                        "condition": branch["condition"], "warm_variant": branch["warm_variant"],
                        "old_label": branch["old_label"], "alternate_old_label": branch["alternate_old_label"],
                        "seed": seed, "scr_enabled": args.arm == "on", "cache_key": key,
                        "manifest_sha256": digest(manifest), "final_input_sha256": digest(branch["final_ids"]),
                    }
                    if branch["warm_ids"] is not None and not request_probe(
                        args.url, args.timeout, common, "warm", branch["warm_ids"], stream
                    ):
                        failures += 1
                        continue
                    if not request_probe(args.url, args.timeout, common, "probe", branch["final_ids"], stream):
                        failures += 1
    print(f"{args.arm}: {failures} failures; transcript {args.requests}", flush=True)
    return int(failures > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("requests", type=Path)
    parser.add_argument("--arm", choices=("off", "on"), required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or args.timeout <= 0:
        parser.error("seeds must be distinct and non-negative; timeout must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
