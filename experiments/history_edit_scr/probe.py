#!/usr/bin/env python3
"""Freeze and run ContextBench-style KV Store probes via SGLang /generate."""

import argparse
import hashlib
import itertools
import json
import random
import sys
import urllib.error
import urllib.request
import uuid
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = dict(line.split("=", 1) for line in (HERE / "pins.env").read_text().splitlines())
LABELS = "ABCD"
MIN_SHARED = 1024
TEMPLATE_KWARGS = {"enable_thinking": False}
TOKENIZER_FILES = {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
WARM_TOKENS = {"min_new_tokens": 8, "max_new_tokens": 32}
PROBE_TOKENS = {"max_new_tokens": 12}
SYSTEM = (
    "You dispatch items using a versioned key-value ledger. SET replaces the prior value. "
    "A signed dispatcher update supersedes earlier entries. Work notes may describe plans. "
    "For a dispatch decision, return only ACTION: followed by the station letter."
)


def digest(value):
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def tokenizer_files(snapshot):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(snapshot).iterdir()
        if path.name in TOKENIZER_FILES
    }


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stream.flush()


def ledger(rng, case_id, first):
    objects = (
        "access badge", "calibration kit", "drive enclosure", "pressure gauge",
        "return envelope", "serial scanner", "fiber adapter", "spare filter",
        "shipping seal", "battery module", "relay case", "packing sleeve",
    )
    notes = (
        "pick up from the intake counter after signing",
        "check the return register before dispatch",
        "ring the desk and bring a padded crate",
        "route through the covered loading area",
        "keep the tracking slip for the afternoon handoff",
        "check the tamper seal at the storage desk",
    )
    return "\n".join(
        f"SET {case_id}/asset-{i} = {rng.choice(objects)} at station "
        f"{rng.choice(LABELS)} ({rng.choice(notes)}); owner {rng.choice(('north', 'south', 'east', 'west'))} dispatch."
        for i in range(first, first + 48)
    )


def messages(case_id, earlier, note, later, tail):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Work order {case_id}. Initial inventory:\n{earlier}"},
        {"role": "assistant", "content": note},
        {"role": "user", "content": f"Verified inventory received since that note:\n{later}"},
        {"role": "assistant", "content": "Inventory filed. The site requires a signed transfer receipt."},
        {"role": "user", "content": tail},
    ]


def render(tokenizer, chat):
    text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, **TEMPLATE_KWARGS
    )
    return text, tokenizer.encode(text, add_special_tokens=False)


def shared_prefix(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def find_span(ids, span):
    for index, token in enumerate(ids):
        if token == span[0] and ids[index:index + len(span)] == span:
            return index
    raise ValueError("retained ledger span changed under tokenization")


def fixture(tokenizer, index, roles, rng, stage, control):
    case_id = f"kv-edit-{index:03d}"
    old_a, old_b, correct = roles
    earlier = ledger(rng, case_id, 100)
    later = ledger(rng, case_id, 200)
    warm_tail = "Write a short acknowledgment of the inventory and describe the receipt procedure."
    final_tail = (
        f"Signed dispatcher update: SET {case_id}/target = station {correct}. "
        f"GET {case_id}/target. Which station should receive the next dispatch? "
        "Respond with exactly one of ACTION: A, ACTION: B, ACTION: C, ACTION: D."
    )
    if control:
        old_a = correct
    old_labels = {"a": old_a, "b": old_b}
    warm_chats = {
        name: messages(case_id, earlier, f"Working note: target is at station {label}.", later, warm_tail)
        for name, label in old_labels.items()
    }
    clean = messages(case_id, earlier, "Working note: location to be verified.", later, final_tail)
    finals = {"clean": {"a": clean, "b": clean}}
    if stage == "full":
        finals["visible"] = {}
        for name, label in old_labels.items():
            visible = [dict(message) for message in clean]
            visible[-2]["content"] += f" Pending plan: go to station {label} to collect the target."
            finals["visible"][name] = visible
    warm = {name: render(tokenizer, chat) for name, chat in warm_chats.items()}
    a, b = warm["a"][1], warm["b"][1]
    left = shared_prefix(a, b)
    right = shared_prefix(a[::-1], b[::-1])
    if len(a) != len(b) or left + right != len(a) - 1:
        raise ValueError(f"{case_id}: expected one differing token across warm histories")
    for name, label in old_labels.items():
        if f"target is at station {label}." not in warm[name][0]:
            raise ValueError(f"{case_id}: chat template dropped the historical assistant note")
    span = tokenizer.encode(later, add_special_tokens=False)[8:-8]
    if len(span) < MIN_SHARED:
        raise ValueError(f"{case_id}: retained span only has {len(span)} tokens")
    branches = []
    for condition, chats in finals.items():
        for name, chat in chats.items():
            final_text, ids = render(tokenizer, chat)
            if "Working note: location to be verified." not in final_text or "target is at station" in final_text:
                raise ValueError(f"{case_id}: unexpected edited note in rendered prompt")
            warm_ids = warm[name][1]
            prefix = shared_prefix(warm_ids, ids)
            if prefix < MIN_SHARED or min(find_span(ids, span), find_span(warm_ids, span)) <= prefix:
                raise ValueError(f"{case_id}: missing a long shared prefix followed by an edit and retained span")
            branches.append({
                "warm_variant": name, "condition": condition,
                "old_label": old_labels[name],
                "alternate_old_label": old_labels["b" if name == "a" else "a"],
                "warm_ids": warm_ids, "final_ids": ids,
                "common_prefix_tokens": prefix, "retained_span_tokens": len(span),
            })
    if branches[0]["final_ids"] != branches[1]["final_ids"]:
        raise ValueError(f"{case_id}: clean A/B prompts differ")
    if stage == "full" and not control:
        branches.append({
            "warm_variant": "none", "condition": "cold", "old_label": None,
            "alternate_old_label": None, "warm_ids": None, "final_ids": branches[0]["final_ids"],
        })
    return {
        "case_id": case_id,
        "source_task_id": None,
        "fixture_type": "old_still_valid" if control else "refuted",
        "history_material": "visible_assistant_note",
        "ground_truth_label": correct,
        "warm_messages": warm_chats,
        "final_messages": finals,
        "branches": branches,
    }


def prepare(args):
    from transformers import AutoTokenizer

    snapshot = Path(args.snapshot).resolve()
    if snapshot.name != PINS["MODEL_REVISION"]:
        raise ValueError("use the pinned Hugging Face snapshot printed by run.sh snapshot")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if not isinstance(tokenizer.chat_template, str):
        raise ValueError("expected a single pinned Qwen chat template")
    rng = random.Random(args.fixture_seed)
    roles = list(itertools.permutations(LABELS, 3))
    rng.shuffle(roles)
    cases = [
        fixture(tokenizer, i + 1, roles[i % len(roles)], rng, args.stage, i >= args.cases)
        for i in range(args.cases + args.stable_controls)
    ]
    manifest = {
        "schema": 1,
        "model_id": PINS["MODEL_ID"],
        "model_revision": PINS["MODEL_REVISION"],
        "tokenizer_revision": PINS["MODEL_REVISION"],
        "tokenizer_snapshot": str(snapshot),
        "tokenizer_files_sha256": tokenizer_files(snapshot),
        "transformers": version("transformers"),
        "chat_template": tokenizer.chat_template,
        "chat_template_kwargs": TEMPLATE_KWARGS,
        "fixture_seed": args.fixture_seed,
        "stage": args.stage,
        "cases": cases,
    }
    with args.manifest.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"{len(cases)} base probes, {sum(len(c['branches']) for c in cases)} branches in {args.manifest}")


def generate(url, payload, timeout):
    request = urllib.request.Request(
        f"{url.rstrip('/')}/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def request_probe(url, timeout, common, phase, ids, stream):
    sampling = {"temperature": 0, "top_p": 1, "sampling_seed": common["seed"]}
    sampling.update(WARM_TOKENS if phase == "warm" else PROBE_TOKENS)
    payload = {
        "input_ids": ids, "rid": f"{common['cache_key']}-{phase}",
        "extra_key": common["cache_key"], "sampling_params": sampling,
    }
    row = {
        **common, "phase": phase, "rid": payload["rid"],
        "input_sha256": digest(ids), "request": payload,
    }
    try:
        result = generate(url, payload, timeout)
        row["response"] = result
        meta = result["meta_info"]
        if meta["id"] != payload["rid"] or meta["prompt_tokens"] != len(ids):
            raise ValueError("response RID or prompt token count differs from request")
        minimum = WARM_TOKENS["min_new_tokens"] if phase == "warm" else 1
        if meta["completion_tokens"] < minimum or not isinstance(result["text"], str):
            raise ValueError("invalid completion")
    except urllib.error.HTTPError as exc:
        row["error"] = f"HTTP {exc.code}: {exc.read().decode(errors='replace')}"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    write_row(stream, row)
    if "error" in row:
        print(f"{payload['rid']}: {row['error']}", file=sys.stderr)
    return "error" not in row


def run(args):
    manifest = json.loads(args.manifest.read_text())
    if manifest["schema"] != 1 or manifest["model_revision"] != PINS["MODEL_REVISION"]:
        raise ValueError("manifest schema or model revision differs from this experiment")
    manifest_sha256 = digest(manifest)
    failures = 0
    run_id = uuid.uuid4().hex
    tags = ("case_id", "source_task_id", "fixture_type", "history_material", "ground_truth_label")
    with args.requests.open("x", encoding="utf-8") as stream:
        for case in manifest["cases"]:
            for branch in case["branches"]:
                for seed in args.seeds:
                    key = f"history-{run_id}-{case['case_id']}-{branch['condition']}-{branch['warm_variant']}-{seed}"
                    common = {
                        **{tag: case[tag] for tag in tags},
                        "condition": branch["condition"], "warm_variant": branch["warm_variant"],
                        "old_label": branch["old_label"], "alternate_old_label": branch["alternate_old_label"],
                        "seed": seed, "scr_enabled": args.arm == "on", "cache_key": key,
                        "manifest_sha256": manifest_sha256, "final_input_sha256": digest(branch["final_ids"]),
                    }
                    warm = branch["warm_ids"]
                    if warm and not request_probe(args.url, args.timeout, common, "warm", warm, stream):
                        failures += 1
                        continue
                    if not request_probe(args.url, args.timeout, common, "probe", branch["final_ids"], stream):
                        failures += 1
    print(f"{args.arm}: {failures} request failures; transcript at {args.requests}")
    return int(failures > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("manifest", type=Path)
    build.add_argument("--snapshot", required=True)
    build.add_argument("--cases", type=int, default=36)
    build.add_argument("--stage", choices=("clean", "full"), default="clean")
    build.add_argument("--stable-controls", type=int, default=0)
    build.add_argument("--fixture-seed", type=int, default=42)
    execute = commands.add_parser("run")
    execute.add_argument("manifest", type=Path)
    execute.add_argument("requests", type=Path)
    execute.add_argument("--arm", choices=("off", "on"), required=True)
    execute.add_argument("--url", required=True)
    execute.add_argument("--seeds", type=int, nargs="+", default=[42])
    execute.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.cases < 1 or args.stable_controls < 0:
            parser.error("need at least one base probe and zero or more stable controls")
        return prepare(args)
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or args.timeout <= 0:
        parser.error("seeds must be distinct and non-negative; timeout must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
