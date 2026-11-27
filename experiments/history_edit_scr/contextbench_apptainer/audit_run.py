#!/usr/bin/env python3
"""Audit archived ContextBench trials without restaging weights or GPU services.

Read the small summary/usage/timing files for every task, then fetch full
ATIF-CTX trajectories only for failed or representative successful cases.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
FAMILIES = ("kv-store", "log-triage", "needle-retention", "sudoku-sketchpad")
EDIT_RECEIPT = re.compile(r"edit applied(?: but it GREW context)?\s*[—-]?\s*context ~(\d+)->(\d+) tokens")


def family(name):
    return next(value for value in FAMILIES if name.startswith(value + "-"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class Archive:
    def __init__(self, files, cache):
        self.files = files.rstrip("/")
        self.cache = cache

    def get(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid artifact path: {relative}")
        target = self.cache / relative
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if self.files.startswith("oss://"):
                source = f"{self.files}/{relative.as_posix()}"
                result = subprocess.run(["ossutil", "cp", source, str(target), "-f"],
                                        capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(f"Cannot fetch {source}: {result.stderr or result.stdout}")
            else:
                source = Path(self.files) / relative
                if not source.is_file():
                    raise FileNotFoundError(source)
                shutil.copyfile(source, target)
        return json.loads(target.read_text(encoding="utf-8"))


def task_row(name, status, usage, timing):
    rewards = status["rewards"]
    delivered = rewards.get("n_ops_delivered")
    total = rewards.get("n_ops_total")
    reason = {0: "submit", 1: "stream_exhausted", 2: "budget_wall",
              3: "timeout", 4: "max_steps"}.get(rewards.get("terminated_by_code"), "unknown")
    synced = sum(event.get("ctx_synced") is True for event in timing)
    if synced != usage["n_ctx_syncs"]:
        raise ValueError(f"{name}: timing edits {synced} != usage edits {usage['n_ctx_syncs']}")
    good = (status["ok"] and rewards.get("grader_valid") == 1
            and float(status["reward"]) == 1.0 and reason == "stream_exhausted"
            and (delivered is None or delivered == total))
    return {
        "task": name, "family": family(name), "gpu": status["gpu"],
        "case": "good" if good else "bad", "reward": float(status["reward"]),
        "terminated_by": reason, "n_ops_delivered": delivered, "n_ops_total": total,
        "n_ctx_syncs": usage["n_ctx_syncs"],
        "n_ctx_syncs_real": usage["n_ctx_syncs_real"],
        "n_ctx_grew": usage["n_ctx_grew"],
        "n_ctx_rejected": usage["n_ctx_rejected"],
        "agent_peak_tokens": rewards.get("agent_peak_tokens"),
        "result": status["result"],
    }


def command_from(step):
    for call in step.get("tool_calls") or []:
        if call.get("function_name") != "bash":
            continue
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return args
        return str(args.get("command", "")) if isinstance(args, dict) else str(args)
    return ""


def observation_from(step):
    observation = step.get("observation") or {}
    return "\n".join(str(item.get("content") or "")
                     for item in observation.get("results") or [])


def edit_events(trajectory):
    events = []
    segments = trajectory["segments"]
    for index, segment in enumerate(segments):
        next_segment = segments[index + 1] if index + 1 < len(segments) else None
        for step in segment.get("steps") or []:
            if step.get("source") != "agent":
                continue
            observation = observation_from(step)
            match = EDIT_RECEIPT.search(observation)
            if match is None:
                continue
            next_context = next_segment.get("input_context") if next_segment else None
            events.append({
                "segment_id": segment["segment_id"], "step_id": step["step_id"],
                "before_tokens": int(match[1]), "after_tokens": int(match[2]),
                "delta_tokens": int(match[2]) - int(match[1]),
                "model_output": step.get("message") or "",
                "reasoning_content": step.get("reasoning_content") or "",
                "command": command_from(step), "observation": observation,
                "next_segment_id": next_segment["segment_id"] if next_segment else None,
                "next_context_tokens": (next_segment.get("input_stats") or {}).get("n_tokens")
                if next_segment else None,
                "next_context_sha256": hashlib.sha256(json.dumps(
                    next_context, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest() if next_context is not None else None,
            })
    return events


def excerpt(value, limit=1600):
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + f"\n... [{len(text) - limit} more characters in events.jsonl]"


def case_report(row, events, out):
    out.mkdir(parents=True)
    with (out / "events.jsonl").open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    by_step = {event["step_id"]: event for event in events}
    selected = [events[0], min(events, key=lambda event: event["delta_tokens"]), events[-1]] if events else []
    selected = [by_step[step] for step in dict.fromkeys(event["step_id"] for event in selected)]
    lines = [f"# {row['task']}", "", f"Case: {row['case']}; reward: {row['reward']:.6f}; "
             f"termination: {row['terminated_by']}; ops: {row['n_ops_delivered']}/{row['n_ops_total']}",
             f"Accepted substantial edits: {row['n_ctx_syncs_real']}; traced edit receipts: {len(events)}.",
             "", "Full model outputs, bash commands and tool observations for every traced edit are in `events.jsonl`.",
             "The edit receipt gives the context size immediately before and after applying the edit.", ""]
    if len(events) != row["n_ctx_syncs"]:
        lines += [f"**Trace incomplete:** {len(events)} receipts versus {row['n_ctx_syncs']} accepted syncs. "
                  "Inspect the original trajectory and timing log before interpreting missing edits.", ""]
    for event in selected:
        lines += [f"## Edit at step {event['step_id']} (segment {event['segment_id']})", "",
                  f"Context: {event['before_tokens']} → {event['after_tokens']} tokens "
                  f"({event['delta_tokens']:+d}). Next model input: segment "
                  f"{event['next_segment_id']}, {event['next_context_tokens']} tokens; "
                  f"SHA-256 {event['next_context_sha256']}.", "", "Model output:", "", "```text",
                  excerpt(event["model_output"] or event["reasoning_content"]), "```", "",
                  "Bash command:", "", "```bash", excerpt(event["command"]), "```", "",
                  "Tool observation and edit receipt:", "", "```text",
                  excerpt(event["observation"]), "```", ""]
    (out / "case.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--mode", choices=("off", "on"), default="off")
    parser.add_argument("--archive-files", type=Path,
                        help="Local archive files directory, for offline inspection")
    parser.add_argument("--output", type=Path,
                        help="Write locally without OSS publication (for offline inspection)")
    args = parser.parse_args()
    if bool(args.archive_files) != bool(args.output):
        parser.error("--archive-files and --output must be supplied together")
    source_subdir = f"contextbench-s42-vllm-{args.mode}"
    report_subdir = f"contextbench-s42-audit-vllm-{args.mode}"
    if args.output:
        output = args.output.resolve()
        files = str(args.archive_files.resolve())
        scratch = None
    else:
        run = Path(subprocess.check_output(
            [sys.executable, str(EXPERIMENT / "storage.py"), "run-dir", args.run_id],
            text=True).strip())
        output = run / "eval" / report_subdir
        scratch = run / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        defaults = dict(line.split("=", 1) for line in
                        (EXPERIMENT / "storage.env").read_text().splitlines())
        prefix = (os.environ.get("CLM_OSS_PREFIX") or defaults["CLM_OSS_PREFIX"]).rstrip("/")
        files = f"{prefix}/eval/{args.run_id}/{source_subdir}/files"
    if output.exists():
        parser.error(f"Report already exists: {output}")
    with tempfile.TemporaryDirectory(prefix="audit-cache-", dir=scratch) as temporary:
        archive = Archive(files, Path(temporary))
        summary = archive.get("summary.json")
        rows = []
        for name, status in sorted(summary["tasks"].items()):
            trial = Path(status["result"]).parent
            usage = archive.get(trial / "agent/usage.json")
            timing = archive.get(trial / "agent/timing.json")
            rows.append(task_row(name, status, usage, timing))
        bad = [row for row in rows if row["case"] == "bad"]
        good = [max((row for row in rows if row["family"] == group and row["case"] == "good"),
                    key=lambda row: row["n_ctx_syncs_real"], default=None) for group in FAMILIES]
        selected = bad + [row for row in good if row]
        output.mkdir(parents=True)
        with (output / "tasks.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[key for key in rows[0] if key != "result"])
            writer.writeheader()
            writer.writerows({key: value for key, value in row.items() if key != "result"} for row in rows)
        for row in selected:
            trial = Path(row["result"]).parent
            trajectory = archive.get(trial / "agent/trajectory.ctx.json")
            case_report(row, edit_events(trajectory), output / "cases" / row["task"])
        report = {
            "run_id": args.run_id, "source": files, "tasks": len(rows),
            "good": len(rows) - len(bad), "bad": len(bad),
            "accepted_edits": sum(row["n_ctx_syncs"] for row in rows),
            "substantial_edits": sum(row["n_ctx_syncs_real"] for row in rows),
            "cases_with_traces": [row["task"] for row in selected],
            "bad_cases": [row["task"] for row in bad],
        }
        write_json(output / "audit.json", report)
        table = ["# ContextBench edit audit", "", f"Run: `{args.run_id}`; mode: `{args.mode}`.",
                 f"Tasks: {len(rows)}; good: {report['good']}; bad: {report['bad']}; "
                 f"substantial accepted edits: {report['substantial_edits']}.", "",
                 "| Case | Task | Reward | Ops | Edits | End |", "|---|---|---:|---:|---:|---|"]
        for row in sorted(rows, key=lambda item: (item["case"] != "bad", item["reward"], item["task"])):
            table.append(f"| {row['case']} | {row['task']} | {row['reward']:.6f} | "
                         f"{row['n_ops_delivered']}/{row['n_ops_total']} | "
                         f"{row['n_ctx_syncs_real']} | {row['terminated_by']} |")
        (output / "summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
        (output / "README.md").write_text(
            "# ContextBench edit audit\n\n"
            "`tasks.csv` lists every task, its reward, stream completion and edit counts.\n"
            "`cases/` contains every bad case and one complete good case per family, "
            "chosen by substantial edit count. Each `events.jsonl` preserves model output, "
            "the full bash command and the tool observation for each traced edit.\n"
            "A good case has reward 1, a valid grader and a fully exhausted stream. "
            "A bad case fails at least one of these checks. The label is a triage rule, "
            "not a causal judgment about the editing strategy.\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not args.output:
        subprocess.run([sys.executable, str(EXPERIMENT / "storage.py"), "publish-eval",
                        args.run_id, "--subdir", report_subdir, "--cleanup"], check=True)


if __name__ == "__main__":
    main()
