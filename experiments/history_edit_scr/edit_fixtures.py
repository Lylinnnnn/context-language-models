#!/usr/bin/env python3
"""Freeze D1 static edit branches and score recorded dispatch events."""

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

from clm_harness.context_utils.context_string import parse_back, render_editable

from probe import LABELS, digest, read_jsonl, write_row

POLICIES = ("keep", "drop_all_notes", "oracle", "random")
SYSTEM = {"role": "system", "content": "Follow the latest signed dispatch ledger for each requested lane."}
TASK = {
    "role": "user",
    "content": "After receiving the update, dispatch each listed lane to its current station, "
               "then hand off using the credential recorded earlier. Each lane may be dispatched only once.",
}


def note(span_id, text, lane=None, station=None):
    return {"span_id": span_id, "text": text, "lane": lane, "station": station}


def scenario(rng, index, count, multi):
    case = f"dispatch-{'sequence' if multi else 'single'}-{index:02d}"
    kind = "refuted" if multi or index < 12 else ("still_valid" if index < 16 else "no_update")
    lanes = ("north", "south")
    stations = {lane: rng.choice(LABELS) for lane in lanes}
    credential = f"permit-{rng.randrange(100000, 999999)}"
    timeline = [{
        "role": "user",
        "content": "Initial signed ledger. " + "; ".join(f"SET {lane} = {station}" for lane, station in stations.items()),
    }, note("handoff", f"Independent work note. Use credential {credential} at the final handoff."),
        note("inventory", "Independent work note. Use an insulated cart for inventory transfer.")]
    for step in range(1, count + 1):
        for lane, station in stations.items():
            timeline.append(note(f"plan-{step}-{lane}", f"Pending plan. Dispatch {lane} to station {station}.",
                                 lane, station))
        latest = dict(stations)
        affected = lanes if (index + step) % 2 else lanes[:1]
        if kind == "refuted":
            for lane in affected:
                latest[lane] = rng.choice([label for label in LABELS if label != stations[lane]])
        receipt = {
            "role": "user",
            "content": ("Signed readback. No ledger values changed." if kind == "no_update" else
                        "Signed dispatcher update. " + "; ".join(f"SET {lane} = {latest[lane]}" for lane in affected))
                       + " Dispatch north and south now, then complete the handoff.",
        }
        timeline.append(receipt)
        candidates = [item for item in timeline if "span_id" in item]
        stale = [item for item in candidates if item["lane"] and latest[item["lane"]] != item["station"]]
        drops = {item["span_id"] for item in stale}
        target_length = sum(len(item["text"]) for item in stale)
        alternatives = [
            combination
            for combination in itertools.combinations(candidates, len(stale))
            if {item["span_id"] for item in combination} != drops
        ]
        rng.shuffle(alternatives)
        random_drop = min(alternatives, key=lambda items: abs(sum(len(item["text"]) for item in items) - target_length),
                          default=[])
        requests = {
            "keep": set(),
            "drop_all_notes": {item["span_id"] for item in candidates},
            "oracle": drops,
            "random": {item["span_id"] for item in random_drop},
        }
        snapshot = {"protected": [SYSTEM, TASK], "timeline": list(timeline)}
        branches = {}
        for policy in POLICIES:
            removed = requests[policy]
            editable = [
                {"role": "assistant", "content": item["text"]} if "span_id" in item else item
                for item in timeline if item.get("span_id") not in removed
            ]
            text = render_editable([SYSTEM, TASK, *editable])
            branch = parse_back(text, [SYSTEM, TASK])
            if branch[:2] != [SYSTEM, TASK] or any(
                item["content"] not in "\n\n".join(message["content"] for message in branch[2:])
                for item in timeline if "span_id" not in item
            ):
                raise ValueError("protected instruction or ledger observation lost in the round-trip")
            branches[policy] = {
                "dropped_span_ids": sorted(removed),
                "removed_chars": sum(len(item["text"]) for item in candidates if item["span_id"] in removed),
                "messages": branch,
            }
        yield {
            "snapshot_id": f"{case}-update-{step}", "scenario_family": case,
            "history_material": "synthetic_static_notes", "fixture_type": kind,
            "update_index": step, "snapshot_sha256": digest(snapshot), **snapshot,
            "branches": branches, "station_by_lane": latest, "handoff_credential": credential,
            "stale_actions": sorted({(item["lane"], item["station"]) for item in stale}),
        }
        stations = latest


def score(case, events):
    """A valid rollout dispatches both lanes exactly once before the final handoff."""
    dispatches = [(event.get("lane"), event.get("station")) for event in events if event.get("action") == "dispatch"]
    stale_actions = {tuple(action) for action in case["stale_actions"]}
    expected = set(case["station_by_lane"].items())
    handoff = {"action": "handoff", "credential": case["handoff_credential"]}
    retained = bool(events and events[-1] == handoff)
    success = retained and len(events) == len(expected) + 1 and len(dispatches) == len(expected)
    return {
        "stale_action": bool(dispatches and dispatches[0] in stale_actions),
        "any_stale_action": any(action in stale_actions for action in dispatches),
        "retained_constraint_used": retained,
        "task_success": success and set(dispatches) == expected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("manifest", type=Path)
    grade = commands.add_parser("score")
    grade.add_argument("manifest", type=Path)
    grade.add_argument("events", type=Path)
    grade.add_argument("results", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        rng = random.Random(42)
        with args.manifest.open("x", encoding="utf-8") as stream:
            for count, multi in ((20, False), (8, True)):
                for i in range(count):
                    for row in scenario(rng, i, 2 + i % 3 if multi else 1, multi):
                        write_row(stream, row)
        return 0
    cases = {row["snapshot_id"]: row for row in read_jsonl(args.manifest)}
    with args.results.open("x", encoding="utf-8") as stream:
        for row in read_jsonl(args.events):
            case = cases[row["snapshot_id"]]
            if row["policy"] not in POLICIES:
                raise ValueError(f"unknown policy {row['policy']}")
            write_row(stream, {**row, **score(case, row["events"])})
    return 0


if __name__ == "__main__":
    sys.exit(main())
