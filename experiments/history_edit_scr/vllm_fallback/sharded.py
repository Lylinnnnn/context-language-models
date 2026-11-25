#!/usr/bin/env python3
"""Run independent vLLM SCR off/on probe pairs on multiple GPUs and pool by base case."""

import argparse
import json
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from subprocess import DEVNULL, STDOUT, Popen, TimeoutExpired

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe import digest, read_jsonl, write_row  # noqa: E402
from report import effects, paired_metrics  # noqa: E402

HERE = Path(__file__).resolve().parent


def shard_manifest(manifest, index, count):
    return {**manifest, "cases": manifest["cases"][index::count]}


def normalized_config(config):
    return {key: value for key, value in config.items() if key not in {"cuda_visible_devices", "mode"}}


def aggregate(root, manifest, count, seeds):
    original_cases = {case["case_id"] for case in manifest["cases"]}
    seen_cases, paired, joined, shard_summaries, configs = set(), {}, [], [], []
    for index in range(count):
        directory = root / f"shard-{index:02d}"
        part = json.loads((directory / "manifest-input.json").read_text())
        expected = shard_manifest(manifest, index, count)
        if part != expected or digest(part) != digest(expected):
            raise ValueError(f"shard {index}: manifest differs from the frozen partition")
        case_ids = {case["case_id"] for case in part["cases"]}
        if len(case_ids) != len(part["cases"]) or seen_cases & case_ids:
            raise ValueError(f"shard {index}: duplicate base cases")
        seen_cases.update(case_ids)
        summary = json.loads((directory / "run" / "probe-summary.json").read_text())
        branch_count = sum(len(case["branches"]) for case in part["cases"]) * len(seeds)
        if (not summary["passed"] or summary["manifest_sha256"] != digest(part)
                or summary["invalid_pairs"] or summary["expected_pairs"] != branch_count
                or summary["valid_pairs"] != branch_count):
            raise ValueError(f"shard {index}: invalid probe report")
        shard_summaries.append({
            "shard": index, "cases": sorted(case_ids), "expected_pairs": summary["expected_pairs"],
            "valid_pairs": summary["valid_pairs"], "clean_on_hits": summary["clean_on_hits"],
        })
        for arm in ("off", "on"):
            with (directory / "run" / arm / "server.log").open(errors="replace") as stream:
                records = [json.loads(line.partition("] ")[2]) for line in stream
                           if line.startswith("[history-edit-vllm-config] ")]
            if len(records) != 1:
                raise ValueError(f"shard {index} {arm}: missing server configuration")
            configs.append(normalized_config(records[0]))
        rows = read_jsonl(directory / "run" / "probe-joined.jsonl")
        if len(rows) != 2 * summary["expected_pairs"]:
            raise ValueError(f"shard {index}: wrong number of joined rows")
        for row in rows:
            if row["case_id"] not in case_ids or row["seed"] not in seeds or row["errors"]:
                raise ValueError(f"shard {index}: unexpected case, seed or invalid row")
            key = (row["case_id"], row["seed"], row["condition"], row["warm_variant"])
            arm = "on" if row["scr_enabled"] else "off"
            entry = paired.setdefault(key, {})
            if arm in entry:
                raise ValueError(f"shard {index}: duplicate {arm} row for {key}")
            entry[arm] = row
            joined.append({"shard": index, **row})
    if seen_cases != original_cases or len(configs) != 2 * count:
        raise ValueError("shard coverage is incomplete")
    # SCR is the sole service toggle inside a pair; across shards only GPU visibility differs.
    for index in range(count):
        off, on = configs[2 * index:2 * index + 2]
        if off["kvreuse_env"].get("KVREUSE_ENABLED") != "0" or on["kvreuse_env"].get("KVREUSE_ENABLED") != "1":
            raise ValueError(f"shard {index}: SCR enable flags differ from the requested arms")
        off["kvreuse_env"] = {k: v for k, v in off["kvreuse_env"].items() if k != "KVREUSE_ENABLED"}
        on["kvreuse_env"] = {k: v for k, v in on["kvreuse_env"].items() if k != "KVREUSE_ENABLED"}
        if off != on:
            raise ValueError(f"shard {index}: server settings differ beyond SCR enable switch")
    if any(config != configs[0] for config in configs):
        raise ValueError("server settings differ across shards")
    expected_pairs = sum(len(case["branches"]) for case in manifest["cases"]) * len(seeds)
    if len(paired) != expected_pairs or any(set(arms) != {"off", "on"} for arms in paired.values()):
        raise ValueError("pooled off/on branch coverage is incomplete")
    pairs = {key: (arms["off"], arms["on"]) for key, arms in paired.items()}
    result = {
        "passed": True, "manifest_sha256": digest(manifest), "seeds": seeds,
        "n_body_params": 24.3532e9,
        "shards": shard_summaries, "expected_pairs": expected_pairs,
        "valid_pairs": len(pairs),
        "clean_on_hits": sum(on["splice_ok"] for (_, _, condition, _), (_, on) in pairs.items()
                             if condition == "clean"),
        "groups": paired_metrics(pairs), "effects": effects(pairs),
    }
    with (root / "probe-joined.jsonl").open("x") as stream:
        for row in joined:
            write_row(stream, row)
    (root / "probe-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_shards(root, count, gpus, workers, base_port, seeds, startup_timeout):
    pending = deque(range(count))
    active = {}
    failures = []
    try:
        while pending or active:
            while pending and len(active) < workers:
                index = pending.popleft()
                directory = root / f"shard-{index:02d}"
                log = (directory / "driver.log").open("x")
                env = os.environ.copy()
                env.update({
                    "CUDA_VISIBLE_DEVICES": gpus[index],
                    "VLLM_OFF_PORT": str(base_port + 2 * index),
                    "VLLM_ON_PORT": str(base_port + 2 * index + 1),
                })
                command = [
                    sys.executable, str(HERE / "pilot.py"), str(directory / "run"),
                    "--manifest", str(directory / "manifest-input.json"),
                    "--startup-timeout", str(startup_timeout),
                    "--seeds", *map(str, seeds),
                ]
                process = Popen(command, env=env, stdout=log, stderr=STDOUT, start_new_session=True, stdin=DEVNULL)
                active[index] = (process, log)
                print(f"shard {index:02d}: GPU {gpus[index]}, ports {base_port + 2 * index}/{base_port + 2 * index + 1}, log {directory / 'driver.log'}", flush=True)
            time.sleep(2)
            for index, (process, log) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                log.close()
                del active[index]
                if status:
                    failures.append(index)
                print(f"shard {index:02d}: {'PASS' if status == 0 else f'FAIL ({status})'}", flush=True)
    except KeyboardInterrupt:
        print("interrupting running shards; each pilot will stop its own server", file=sys.stderr)
        for process, _ in active.values():
            os.killpg(process.pid, signal.SIGINT)
        for process, log in active.values():
            try:
                process.wait(timeout=60)
            except TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.close()
        raise
    if failures:
        raise RuntimeError(f"failed shards {failures}; inspect their driver.log and server.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="new result directory")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="comma-separated physical GPU indices")
    parser.add_argument("--workers", type=int, help="maximum simultaneous GPU shards; default all listed GPUs")
    parser.add_argument("--base-port", type=int, default=32000)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if not gpus or any(not gpu.isdigit() for gpu in gpus) or len(set(gpus)) != len(gpus):
        parser.error("--gpus must list distinct non-negative GPU indices")
    count = len(gpus)
    workers = args.workers if args.workers is not None else count
    if (workers < 1 or workers > count or args.base_port < 1024
            or args.base_port + 2 * count - 1 > 65535 or args.startup_timeout <= 0):
        parser.error("invalid worker count, port range or startup timeout")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("seeds must be distinct non-negative integers")
    if not os.environ.get("CLM_MODEL_DIR") or not os.environ.get("CLM_MODEL_REVISION"):
        parser.error("set CLM_MODEL_DIR and CLM_MODEL_REVISION before running")
    manifest = json.loads(args.manifest.read_text())
    cases = manifest["cases"]
    if manifest["schema"] != 1 or manifest["model_revision"] != os.environ["CLM_MODEL_REVISION"]:
        parser.error("manifest schema or model revision differs from the staged model")
    if len(cases) < count or len({case["case_id"] for case in cases}) != len(cases):
        parser.error("need at least one distinct base case per GPU shard")
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "manifest-input.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for index in range(count):
        directory = root / f"shard-{index:02d}"
        directory.mkdir()
        (directory / "manifest-input.json").write_text(json.dumps(shard_manifest(manifest, index, count), indent=2) + "\n")
    run_shards(root, count, gpus, workers, args.base_port, args.seeds, args.startup_timeout)
    result = aggregate(root, manifest, count, args.seeds)
    print(f"PASS: {result['valid_pairs']}/{result['expected_pairs']} paired branches; "
          f"clean on hits={result['clean_on_hits']}; see {root / 'probe-summary.json'}")


if __name__ == "__main__":
    main()
