#!/usr/bin/env python3
"""Validate smoke evidence and join fixed-input probes with official SCR accounting."""

import argparse
import csv
import json
import random
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from probe import HERE, digest, read_jsonl, write_row

OFFICIAL = HERE.parents[1] / "suffix_cache_reuse" / "sglang" / "analysis" / "scr_report.py"
ARMS = ("off", "on")
ACTION = re.compile(r"ACTION:\s*([A-D])", re.IGNORECASE)
RID = re.compile(r"\brid=(\S+)")
FALLBACK = re.compile(
    r"no tail plan|misaligned|normal prefill|aborted|slot stolen|defer|"
    r"snapshot dropped|no .*plan|skip|fallback|failed", re.IGNORECASE,
)
NUMBERS = ("prompt_tokens", "radix_matched", "relocated", "forwarded", "decode", "blocks")


def read_server(directory, arm):
    log = directory / arm / "server.log"
    with log.open(errors="replace") as stream:
        lines = stream.readlines()
    configs = [json.loads(line.partition("] ")[2]) for line in lines if line.startswith("[history-edit-config] ")]
    banner = f"[scr] Suffix Cache Reuse {arm} |"
    if len(configs) != 1 or sum(line.startswith(banner) for line in lines) != 1:
        raise ValueError(f"{log}: expected exactly one config record and an SCR {arm} launcher banner")
    csv_path = log.with_suffix(".csv")
    subprocess.run(
        [sys.executable, str(OFFICIAL), str(log), "--csv", str(csv_path), "--json", str(log.with_suffix(".json"))],
        check=True,
    )
    with csv_path.open(newline="") as stream:
        stats = {
            row["rid"]: {
                **row, **{key: int(row[key]) for key in NUMBERS},
                "prev_prompt_tokens": int(row["prev_prompt_tokens"]) if row["prev_prompt_tokens"] else None,
                "prefix_reuse_pflops": float(row["prefix_reuse_pflops"]),
            }
            for row in csv.DictReader(stream)
        }
    reasons = defaultdict(list)
    for line in lines:
        rid = RID.search(line)
        if rid and "v6 splice OK" not in line and FALLBACK.search(line):
            reasons[rid[1]].append(line.strip())
    return configs[0], stats, reasons


def smoke_report(directory, servers):
    evidence = {}
    for arm in ARMS:
        smoke = (directory / arm / "smoke.log").read_text()
        rids = re.findall(rf"^(smoke-{arm}-[0-9a-f]+-turn[123]): prompt_tokens=", smoke, re.MULTILINE)
        if len(rids) != 3 or {rid[-1] for rid in rids} != set("123") or "smoke: all requests succeeded" not in smoke:
            raise ValueError(f"invalid {arm} smoke transcript")
        rows = [servers[arm][0].get(rid) for rid in rids]
        if any(row is None or row["decode"] <= 0 for row in rows):
            raise ValueError(f"{arm} smoke requests missing completed generation in server log")
        evidence[arm] = rows

    edited_hits = [row["rid"] for row in evidence["on"] if not row["rid"].endswith("turn1") and row["relocated"] > 0]
    regression = (directory / "on" / "regression.log").read_text()
    passed = (
        bool(edited_hits)
        and all(row["relocated"] == 0 for row in evidence["off"])
        and "regression: all passed" in regression
    )
    return {"passed": passed, "edited_hits": edited_hits, "requests": evidence}


def key(row):
    return row["case_id"], row["seed"], row["condition"], row["warm_variant"]


def read_requests(path, arm):
    rows = {}
    rids = set()
    for row in read_jsonl(path):
        request_key = key(row), row["phase"]
        if request_key in rows or row["rid"] in rids or row["scr_enabled"] != (arm == "on"):
            raise ValueError(f"{path}: duplicate request or wrong SCR arm for {row['rid']}")
        rows[request_key] = row
        rids.add(row["rid"])
    return rows, rids


def check_request(row, ids, manifest_hash, phase, stats):
    if not row:
        return f"missing {phase} request"
    if row.get("error"):
        return f"{phase}: {row['error']}"
    request = row["request"]
    if (
        row["manifest_sha256"] != manifest_hash or request["input_ids"] != ids
        or row["input_sha256"] != digest(ids) or request["rid"] != row["rid"]
        or request["extra_key"] != row["cache_key"]
    ):
        return f"{phase}: input IDs, hash or request key differs from the manifest/transcript"
    if stats is None or stats["prompt_tokens"] != len(ids):
        return f"{phase}: server accounting is missing or reports a different prompt length"
    if stats["decode"] != row["response"]["meta_info"]["completion_tokens"]:
        return f"{phase}: server and client completion token counts differ"
    return None


def classify(label, truth, branch):
    if label is None:
        return "unparseable"
    if label == truth:
        return "correct"
    if label == branch["old_label"]:
        return "own_old"
    if label == branch["alternate_old_label"]:
        return "alternate_old"
    return "other_valid"


def join_arm(case, branch, seed, arm, manifest_hash, rows, server):
    entry = (case["case_id"], seed, branch["condition"], branch["warm_variant"])
    warm = rows.get((entry, "warm"))
    probe = rows.get((entry, "probe"))
    stats, reasons = server
    warm_stats = stats.get(warm["rid"]) if warm else None
    probe_stats = stats.get(probe["rid"]) if probe else None
    expected_hash = digest(branch["final_ids"])
    errors = []
    if branch["warm_ids"] is not None:
        errors.append(check_request(warm, branch["warm_ids"], manifest_hash, "warm", warm_stats))
    elif warm is not None:
        errors.append("cold-only control was warmed")
    errors.append(check_request(probe, branch["final_ids"], manifest_hash, "probe", probe_stats))
    for row in (warm, probe):
        if row and row["final_input_sha256"] != expected_hash:
            errors.append("final input hash differs from manifest")
    if warm and probe and warm["cache_key"] != probe["cache_key"]:
        errors.append("warm and probe cache keys differ")
    if warm_stats and warm_stats["relocated"]:
        errors.append("isolated warm request relocated tokens")
    relocated = probe_stats["relocated"] if probe_stats else None
    if relocated:
        if arm == "off" or branch["condition"] == "cold":
            errors.append("unexpected relocation in SCR off or cold-only request")
        if (not warm_stats or not warm_stats["session"] or warm_stats["session"] != probe_stats["session"]
                or warm_stats["prev_prompt_tokens"] != 0
                or probe_stats["prev_prompt_tokens"] != len(branch["warm_ids"] or [])):
            errors.append("splice trace does not identify the paired warm session")
    output = (probe or {}).get("response", {}).get("text")
    match = ACTION.fullmatch(output.strip()) if isinstance(output, str) else None
    label = match[1].upper() if match else None
    probe_rid = probe["rid"] if probe else None
    return {
        **{field: case[field] for field in (
            "case_id", "source_task_id", "fixture_type", "history_material", "ground_truth_label",
        )},
        **{field: branch[field] for field in ("warm_variant", "condition", "old_label", "alternate_old_label")},
        "seed": seed, "scr_enabled": arm == "on",
        "warm_rid": warm["rid"] if warm else None, "probe_rid": probe_rid,
        "cache_key": (probe or warm or {}).get("cache_key"),
        "final_input_sha256": expected_hash,
        "prefill_tokens": probe_stats["prompt_tokens"] if probe_stats else None,
        "generated_tokens": probe_stats["decode"] if probe_stats else None,
        **{field: probe_stats[field] if probe_stats else None for field in
           ("radix_matched", "relocated", "forwarded", "prefix_reuse_pflops")},
        "warm_forwarded": warm_stats["forwarded"] if warm_stats else None,
        "warm_prefix_reuse_pflops": warm_stats["prefix_reuse_pflops"] if warm_stats else None,
        "splice_ok": relocated > 0 if relocated is not None else None,
        "fallback_reason": reasons.get(probe_rid, []) if arm == "on" and not relocated else [],
        "raw_output": output, "parsed_label": label,
        "result_class": classify(label, case["ground_truth_label"], branch),
        "errors": list(filter(None, errors)),
    }


def paired_metrics(pairs):
    by_group = defaultdict(list)
    for off, on in pairs.values():
        group = f"{off['fixture_type']}/{off['condition']}"
        if off["fixture_type"] == "old_still_valid":
            group += "/old_correct" if off["old_label"] == off["ground_truth_label"] else "/old_incorrect"
        by_group[group].append((off, on))
    result = {}
    for name, group in sorted(by_group.items()):
        result[name] = {
            "branches": len(group), "base_cases": len({off["case_id"] for off, _ in group}),
            "label_flips": sum(off["parsed_label"] != on["parsed_label"] for off, on in group),
            "on_hits": sum(on["splice_ok"] for _, on in group),
            "on_misses": sum(not on["splice_ok"] for _, on in group),
            **{
                arm: {
                    "accuracy": sum(row["result_class"] == "correct" for row in rows) / len(rows),
                    "result_classes": dict(Counter(row["result_class"] for row in rows)),
                    "mean_forwarded": statistics.mean(row["forwarded"] for row in rows),
                    "mean_pflops": statistics.mean(row["prefix_reuse_pflops"] for row in rows),
                }
                for arm, rows in zip(ARMS, zip(*group))
            },
        }
    return result


def estimate(values):
    # Average repeated seeds within each base fixture, then bootstrap base fixtures.
    case_means = [statistics.mean(repeats) for repeats in values.values()]
    if not case_means:
        return {"base_cases": 0, "estimate": None, "ci95": None}
    n = len(case_means)
    if n > 1:
        rng = random.Random(42)
        sampled = sorted(statistics.mean(rng.choices(case_means, k=n)) for _ in range(10000))
        ci = [sampled[249], sampled[9749]]
    else:
        ci = None
    return {"base_cases": n, "estimate": statistics.mean(case_means), "ci95": ci}


def effects(pairs):
    def own(row):
        return row["result_class"] == "own_old"

    def contrast(a, b):
        return (
            int(a["parsed_label"] == a["old_label"]) - int(b["parsed_label"] == a["old_label"])
            + int(b["parsed_label"] == b["old_label"]) - int(a["parsed_label"] == b["old_label"])
        ) / 2

    h1, h2, hit_delta = defaultdict(list), defaultdict(list), defaultdict(list)
    contrasts = {arm: defaultdict(list) for arm in ARMS}
    for (case, seed, condition, variant), clean_a in pairs.items():
        if condition != "clean" or variant != "a" or clean_a[0]["fixture_type"] != "refuted":
            continue
        clean_b = pairs.get((case, seed, "clean", "b"))
        if not clean_b:
            continue
        off_a, on_a = clean_a
        off_b, on_b = clean_b
        if off_a["final_input_sha256"] != off_b["final_input_sha256"]:
            raise ValueError(f"{case}: clean A/B inputs differ")
        off_score, on_score = contrast(off_a, off_b), contrast(on_a, on_b)
        contrasts["off"][case].append(off_score)
        contrasts["on"][case].append(on_score)
        h2[case].append(on_score - off_score)
        if on_a["splice_ok"] and on_b["splice_ok"]:
            hit_delta[case].append(on_score - off_score)
        visible_a, visible_b = (pairs.get((case, seed, "visible", v)) for v in ("a", "b"))
        if visible_a and visible_b:
            h1[case].append((int(own(visible_a[0])) - int(own(off_a))
                             + int(own(visible_b[0])) - int(own(off_b))) / 2)
    return {
        "h1_visible_minus_clean_off": estimate(h1),
        "h2_old_label_contrast_off": estimate(contrasts["off"]),
        "h2_old_label_contrast_on": estimate(contrasts["on"]),
        "h2_on_minus_off": estimate(h2),
        "h2_on_minus_off_both_histories_hit": estimate(hit_delta),
    }


def probe_report(directory, manifest, seeds, servers):
    expected_hash = digest(manifest)
    off, off_rids = read_requests(directory / "off" / "requests.jsonl", "off")
    on, on_rids = read_requests(directory / "on" / "requests.jsonl", "on")
    if off_rids & on_rids:
        raise ValueError("request IDs reused across server arms")
    valid_pairs, invalid_pairs, expected = {}, [], set()
    keys = {arm: set() for arm in ARMS}
    with (directory / "joined.jsonl").open("w", encoding="utf-8") as stream:
        for case in manifest["cases"]:
            branches = case["branches"]
            clean_ids = [branch["final_ids"] for branch in branches if branch["condition"] == "clean"]
            if len(clean_ids) != 2 or clean_ids[0] != clean_ids[1]:
                raise ValueError(f"{case['case_id']}: invalid clean A/B manifest inputs")
            for branch in branches:
                for seed in seeds:
                    entry = case["case_id"], seed, branch["condition"], branch["warm_variant"]
                    if entry in expected:
                        raise ValueError(f"duplicate branch {entry}")
                    expected.add(entry)
                    records = []
                    for arm, requests in zip(ARMS, (off, on)):
                        record = join_arm(case, branch, seed, arm, expected_hash, requests, servers[arm])
                        cache_key = record["cache_key"]
                        if cache_key:
                            if cache_key in keys[arm]:
                                record["errors"].append("cache key reused across branches")
                            keys[arm].add(cache_key)
                        records.append(record)
                    for phase in ("warm", "probe"):
                        a, b = off.get((entry, phase)), on.get((entry, phase))
                        if a and b and (
                            a["request"]["input_ids"] != b["request"]["input_ids"]
                            or a["request"]["sampling_params"] != b["request"]["sampling_params"]
                        ):
                            for row in records:
                                row["errors"].append(f"{phase}: SCR on/off IDs or sampling differ")
                    for row in records:
                        if row["cache_key"] in keys["off"] & keys["on"]:
                            row["errors"].append("cache key reused across server arms")
                        write_row(stream, row)
                    if any(row["errors"] for row in records):
                        invalid_pairs.append({"branch": list(entry), **{
                            arm: row["errors"] for arm, row in zip(ARMS, records)
                        }})
                    else:
                        valid_pairs[entry] = records
    if any(entry not in expected or phase not in {"warm", "probe"} for entry, phase in off.keys() | on.keys()):
        raise ValueError("unexpected requests outside the manifest and requested seeds")
    return {
        "manifest_sha256": expected_hash, "seeds": seeds,
        "expected_pairs": len(expected), "valid_pairs": len(valid_pairs), "invalid_pairs": invalid_pairs,
        "groups": paired_metrics(valid_pairs), "effects": effects(valid_pairs),
    }


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def markdown(summary):
    smoke = summary["smoke"]
    lines = [
        "# SCR 配对试跑",
        "",
        f"- 官方 smoke 和回归{'通过' if smoke['passed'] else '未通过'}",
        f"- 官方 smoke 命中的编辑请求 {', '.join(smoke['edited_hits']) or '0'}",
    ]
    if "probes" not in summary:
        return "\n".join(lines) + "\n"
    probes = summary["probes"]
    lines += [
        f"- 输入和服务证据有效的配对 {probes['valid_pairs']}/{probes['expected_pairs']}",
        f"- 无效配对 {len(probes['invalid_pairs'])}",
        "",
        "| 样本 / 终态 | 配对数 | On 命中 / 未命中 | 标签翻转 | Off / On 正确率 | Off / On 转发 tokens | Off / On PFLOPs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, group in probes["groups"].items():
        off, on = group["off"], group["on"]
        lines.append(
            f"| {name} | {group['branches']} | {group['on_hits']} / {group['on_misses']} | {group['label_flips']} | "
            f"{percent(off['accuracy'])} / {percent(on['accuracy'])} | "
            f"{off['mean_forwarded']:.1f} / {on['mean_forwarded']:.1f} | "
            f"{off['mean_pflops']:.4f} / {on['mean_pflops']:.4f} |"
        )
    lines += ["", "| 配对效应 | 基础样本数 | 差值 | 95% CI |", "| --- | ---: | ---: | ---: |"]
    for name, effect in probes["effects"].items():
        ci = "—" if effect["ci95"] is None else " 至 ".join(map(percent, effect["ci95"]))
        lines.append(f"| {name} | {effect['base_cases']} | {percent(effect['estimate'])} | {ci} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--manifest", type=Path, help="include the native /generate probes")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--require-clean-hit", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be distinct")
    if args.require_clean_hit and not args.manifest:
        parser.error("--require-clean-hit requires --manifest")
    base = args.run_directory
    configs, servers = {}, {}
    for arm in ARMS:
        config, stats, reasons = read_server(base, arm)
        configs[arm], servers[arm] = config, (stats, reasons)
    if configs["off"] != configs["on"]:
        differences = [key for key in configs["off"].keys() | configs["on"].keys()
                       if configs["off"].get(key) != configs["on"].get(key)]
        raise ValueError(f"server arms differ in {', '.join(sorted(differences))}")
    summary = {"server": configs["on"], "smoke": smoke_report(base, servers)}
    passed = summary["smoke"]["passed"]
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        for field in ("model_revision", "tokenizer_revision", "tokenizer_files_sha256", "transformers"):
            if manifest[field] != configs["on"][field]:
                raise ValueError(f"probe manifest and serving config disagree on {field}")
        probes = probe_report(base, manifest, args.seeds, servers)
        summary["probes"] = probes
        passed = passed and not probes["invalid_pairs"]
        if args.require_clean_hit:
            passed = passed and probes["effects"]["h2_on_minus_off_both_histories_hit"]["base_cases"] > 0
    summary["passed"] = passed
    (base / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (base / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(f"{'PASS' if passed else 'FAIL'}: see {base / 'summary.md'} and {base / 'summary.json'}")
    return int(not passed)


if __name__ == "__main__":
    sys.exit(main())
