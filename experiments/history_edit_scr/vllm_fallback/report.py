#!/usr/bin/env python3
"""Join vLLM SCR off/on probes against frozen token IDs and server accounting."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe import digest, read_jsonl, write_row  # noqa: E402
from report import ACTION, classify, effects, paired_metrics  # noqa: E402

ARMS = ("off", "on")


def load_arm(root, arm):
    directory = root / arm
    with (directory / "server.log").open(errors="replace") as stream:
        configs = [json.loads(line.partition("] ")[2]) for line in stream
                   if line.startswith("[history-edit-vllm-config] ")]
    if len(configs) != 1:
        raise ValueError(f"{arm}: missing or repeated server config")
    summary = json.loads((directory / "summary.json").read_text())
    if summary["identity_failures"] or summary["requests"] < 4:
        raise ValueError(f"{arm}: official SCR log accounting failed")
    with (directory / "requests.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != summary["requests"] or len({r["rid"] for r in rows}) != len(rows):
        raise ValueError(f"{arm}: duplicate or missing server accounting")
    # vLLM completions append the single prompt's index (-0) to the engine ID.
    stats = {row["rid"][:-2] if row["rid"].startswith("history-") and row["rid"].endswith("-0")
             else row["rid"]: row for row in rows}
    if len(stats) != len(rows):
        raise ValueError(f"{arm}: server request IDs collide after normalization")
    requests = read_jsonl(directory / "probe-requests.jsonl")
    transcript = {row["rid"]: row for row in requests}
    if len(transcript) != len(requests):
        raise ValueError(f"{arm}: repeated request ID in transcript")
    return configs[0], summary, stats, transcript


def checked(row, ids, manifest_hash, stats, arm, phase):
    if row is None:
        return None, [f"missing {phase} request"]
    errors = []
    if row.get("error"):
        errors.append(f"{phase}: {row['error']}")
        return None, errors
    payload = row["request"]
    response = row["response"]
    choice = response["choices"][0]
    account = stats.get(row["rid"])
    if (row["manifest_sha256"] != manifest_hash or row["input_sha256"] != digest(ids)
            or payload["prompt"] != ids or payload["request_id"] != row["rid"]
            or payload["cache_salt"] != row["cache_key"] or payload["add_special_tokens"] is not False
            or choice["prompt_token_ids"] != ids or response["usage"]["prompt_tokens"] != len(ids)
            or row["scr_enabled"] != (arm == "on")):
        errors.append(f"{phase}: frozen input, token IDs or arm differs")
    if account is None:
        errors.append(f"{phase}: missing server accounting")
    elif (int(account["prompt_tokens"]) != len(ids)
          or int(account["decode"]) != response["usage"]["completion_tokens"]
          or sum(int(account[k]) for k in ("radix_matched", "relocated", "forwarded")) != len(ids)):
        errors.append(f"{phase}: client/server token accounting differs")
    return account, errors


def row_for(case, branch, seed, arm, manifest_hash, stats, transcript):
    found = {}
    for phase in ("warm", "probe"):
        matched = [row for row in transcript.values() if
                   (row["case_id"], row["seed"], row["condition"], row["warm_variant"], row["phase"])
                   == (case["case_id"], seed, branch["condition"], branch["warm_variant"], phase)]
        if len(matched) > 1:
            raise ValueError(f"duplicate {phase} for {case['case_id']}")
        found[phase] = matched[0] if matched else None
    warm, probe = found["warm"], found["probe"]
    errors = []
    if branch["warm_ids"] is None:
        if warm:
            errors.append("cold control was warmed")
        warm_stats = None
    else:
        warm_stats, e = checked(warm, branch["warm_ids"], manifest_hash, stats, arm, "warm")
        errors.extend(e)
    probe_stats, e = checked(probe, branch["final_ids"], manifest_hash, stats, arm, "probe")
    errors.extend(e)
    for req in (warm, probe):
        if req and req["final_input_sha256"] != digest(branch["final_ids"]):
            errors.append("final input hash differs")
    if warm and probe and warm["cache_key"] != probe["cache_key"]:
        errors.append("warm and probe cache salts differ")
    if warm_stats and int(warm_stats["relocated"]):
        errors.append("isolated warm request relocated")
    relocated = int(probe_stats["relocated"]) if probe_stats else None
    if relocated:
        if arm == "off" or branch["condition"] == "cold":
            errors.append("unexpected relocation in off or cold control")
        if (not warm_stats or warm_stats["session"] != probe_stats["session"]
                or int(probe_stats["prev_prompt_tokens"]) != len(branch["warm_ids"] or [])):
            errors.append("relocation did not use paired warm session")
    output = probe["response"]["choices"][0]["text"] if probe and "response" in probe else None
    match = ACTION.fullmatch(output.strip()) if isinstance(output, str) else None
    label = match[1].upper() if match else None
    return {
        **{k: case[k] for k in ("case_id", "source_task_id", "fixture_type", "history_material", "ground_truth_label")},
        **{k: branch[k] for k in ("warm_variant", "condition", "old_label", "alternate_old_label")},
        "seed": seed, "scr_enabled": arm == "on",
        "warm_rid": warm["rid"] if warm else None, "probe_rid": probe["rid"] if probe else None,
        "cache_key": probe["cache_key"] if probe else None,
        "final_input_sha256": digest(branch["final_ids"]),
        "prefill_tokens": int(probe_stats["prompt_tokens"]) if probe_stats else None,
        "generated_tokens": int(probe_stats["decode"]) if probe_stats else None,
        **{k: int(probe_stats[k]) if probe_stats else None for k in ("radix_matched", "relocated", "forwarded")},
        "prefix_reuse_pflops": float(probe_stats["prefix_reuse_pflops"]) if probe_stats else None,
        "splice_ok": relocated > 0 if relocated is not None else None,
        "raw_output": output, "parsed_label": label,
        "result_class": classify(label, case["ground_truth_label"], branch),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    data = {arm: load_arm(args.run_directory, arm) for arm in ARMS}
    off_config, on_config = data["off"][0], data["on"][0]
    for key in ("commit", "backend", "model", "model_revision", "model_source", "source_revision", "snapshot", "vllm", "gpu", "torch", "cuda"):
        if off_config[key] != on_config[key]:
            raise ValueError(f"server arms differ in {key}")
    if off_config["mode"] != "off" or on_config["mode"] != "on":
        raise ValueError("server mode labels disagree with result directories")
    for arm in ARMS:
        left = {k: v for k, v in data[arm][0]["kvreuse_env"].items() if k != "KVREUSE_ENABLED"}
        right = {k: v for k, v in on_config["kvreuse_env"].items() if k != "KVREUSE_ENABLED"}
        if left != right:
            raise ValueError("off/on SCR settings differ beyond enable switch")
    if manifest["model_revision"] != on_config["model_revision"]:
        raise ValueError("manifest and serving model revision differ")
    if not all(data[arm][1]["edited_requests_relocated"] > 0 if arm == "on" else data[arm][1]["relocated"] == 0 for arm in ARMS):
        raise ValueError("official smoke relocation check failed")
    pairs, invalid, used = {}, [], {arm: set() for arm in ARMS}
    joined = []
    for case in manifest["cases"]:
        branches = case["branches"]
        clean = [b["final_ids"] for b in branches if b["condition"] == "clean"]
        if len(clean) != 2 or clean[0] != clean[1]:
            raise ValueError(f"{case['case_id']}: A/B clean token IDs differ")
        for branch in branches:
            for seed in args.seeds:
                key = (case["case_id"], seed, branch["condition"], branch["warm_variant"])
                rows = []
                for arm in ARMS:
                    row = row_for(case, branch, seed, arm, digest(manifest), data[arm][2], data[arm][3])
                    rows.append(row)
                    for rid in (row["warm_rid"], row["probe_rid"]):
                        if rid:
                            used[arm].add(rid)
                    joined.append(row)
                for phase in ("warm", "probe"):
                    requests = [data[arm][3].get(rows[i][f"{phase}_rid"]) for i, arm in enumerate(ARMS)]
                    if all(requests):
                        a, b = (r["request"] for r in requests)
                        for field in ("prompt", "temperature", "seed", "max_tokens", "min_tokens"):
                            if a[field] != b[field]:
                                for row in rows:
                                    row["errors"].append(f"{phase}: off/on {field} differs")
                if any(row["errors"] for row in rows):
                    invalid.append({"branch": key, "errors": [row["errors"] for row in rows]})
                else:
                    pairs[key] = rows
    for arm in ARMS:
        if used[arm] != set(data[arm][3]):
            raise ValueError(f"{arm}: unexpected or unpaired transcript requests")
    with (args.run_directory / "probe-joined.jsonl").open("x") as stream:
        for row in joined:
            write_row(stream, row)
    summary = {
        "manifest_sha256": digest(manifest), "expected_pairs": sum(len(c["branches"]) for c in manifest["cases"]) * len(args.seeds),
        "valid_pairs": len(pairs), "invalid_pairs": invalid,
        "groups": paired_metrics(pairs), "effects": effects(pairs),
        "clean_on_hits": sum(on["splice_ok"] for (case, seed, condition, variant), (off, on) in pairs.items()
                             if condition == "clean"),
    }
    summary["passed"] = not invalid and summary["valid_pairs"] == summary["expected_pairs"]
    (args.run_directory / "probe-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{'PASS' if summary['passed'] else 'FAIL'}: {summary['valid_pairs']}/{summary['expected_pairs']} valid pairs; "
          f"clean on hits={summary['clean_on_hits']}; see {args.run_directory / 'probe-summary.json'}")
    return int(not summary["passed"])


if __name__ == "__main__":
    sys.exit(main())
