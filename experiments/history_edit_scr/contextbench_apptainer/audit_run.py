#!/usr/bin/env python3
"""Write one readable ContextBench edit audit beside the original OSS eval files."""

import argparse
import difflib
import hashlib
import html
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
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "clm"))
from clm_harness.context_utils.context_string import render_editable  # noqa: E402

FAMILIES = ("kv-store", "log-triage", "needle-retention", "sudoku-sketchpad")
REASONS = {0: "submit", 1: "stream_exhausted", 2: "budget_wall",
           3: "timeout", 4: "max_steps"}
RECEIPT = re.compile(r"edit applied(?: but it GREW)?\s*[—-]?\s*context ~(\d+)->(\d+) tokens")


def run(*args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(map(str, args[:2]))} failed: {result.stderr or result.stdout}")
    return result.stdout


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


class Archive:
    def __init__(self, files, cache):
        self.files = files.rstrip("/")
        self.cache = cache

    def path(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid archive path: {relative}")
        target = self.cache / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            if self.files.startswith("oss://"):
                run("ossutil", "cp", f"{self.files}/{relative.as_posix()}", str(target), "-f")
            else:
                shutil.copyfile(Path(self.files) / relative, target)
        return target

    def json(self, relative):
        return json.loads(self.path(relative).read_text(encoding="utf-8"))


def family(name):
    return next(value for value in FAMILIES if name.startswith(value + "-"))


def task_row(name, status, usage, timing):
    rewards = status["rewards"]
    delivered, total = rewards.get("n_ops_delivered"), rewards.get("n_ops_total")
    reason = REASONS.get(rewards.get("terminated_by_code"), "unknown")
    synced = sum(event.get("ctx_synced") is True for event in timing)
    if synced != usage["n_ctx_syncs"]:
        raise ValueError(f"{name}: timing has {synced} edits, usage has {usage['n_ctx_syncs']}")
    good = (status.get("ok") and rewards.get("grader_valid") == 1
            and float(status["reward"]) == 1.0 and reason == "stream_exhausted"
            and delivered == total)
    return {"task": name, "family": family(name), "reward": float(status["reward"]),
            "case": "good" if good else "bad", "gpu": status["gpu"],
            "ops": f"{delivered}/{total}", "reason": reason,
            "edits": usage["n_ctx_syncs_real"], "syncs": usage["n_ctx_syncs"],
            "result": status["result"]}


def tool_command(step):
    for call in step.get("tool_calls") or []:
        if call.get("function_name") != "bash":
            continue
        args = call.get("arguments")
        if isinstance(args, str):
            args = json.loads(args)
        return str(args.get("command", "")) if isinstance(args, dict) else str(args)
    return ""


def tool_return(step):
    return "\n".join(str(result.get("content") or "") for result in
                     (step.get("observation") or {}).get("results") or [])


def materialize(step):
    if step["source"] in ("user", "system"):
        return [{"role": step["source"], "content": step.get("message") or ""}]
    assistant = {"role": "assistant", "content": step.get("message") or ""}
    if step.get("reasoning_content"):
        assistant["reasoning_content"] = step["reasoning_content"]
    calls = step.get("tool_calls") or []
    if calls:
        assistant["tool_calls"] = [{"id": call.get("tool_call_id"),
                                    "function": {"name": call["function_name"],
                                                 "arguments": call.get("arguments")}}
                                   for call in calls]
    messages = [assistant]
    for result in (step.get("observation") or {}).get("results") or []:
        messages.append({"role": "tool", "tool_call_id": result.get("source_call_id"),
                         "content": result.get("content") or ""})
    return messages


def after_edit_messages(next_segment, step):
    messages = next_segment["input_context"]
    ids = {call.get("tool_call_id") for call in step.get("tool_calls") or []}
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "assistant" and any(
                call.get("id") in ids for call in message.get("tool_calls") or []):
            return messages[:index]
    raise ValueError(f"Edit step {step['step_id']} missing from next segment's input_context")


def edits(trajectory):
    found = []
    for number, segment in enumerate(trajectory["segments"][:-1]):
        steps = segment.get("steps") or []
        for index, step in enumerate(steps):
            if step.get("source") != "agent":
                continue
            returned = tool_return(step)
            receipt = RECEIPT.search(returned)
            if not receipt:
                continue
            found.append({"segment_index": number, "step_index": index,
                          "before_tokens": int(receipt[1]), "after_tokens": int(receipt[2])})
    return found


def trace_event(trajectory, event):
    number, index = event["segment_index"], event["step_index"]
    segment = trajectory["segments"][number]
    steps = segment["steps"]
    step = steps[index]
    before = list(segment["input_context"])
    for earlier in steps[:index]:
        before.extend(materialize(earlier))
    after = after_edit_messages(trajectory["segments"][number + 1], step)
    return {**event, "step": step["step_id"], "before": render_editable(before),
            "after": render_editable(after), "model_reply": step.get("message") or "",
            "reasoning": step.get("reasoning_content") or "",
            "command": tool_command(step), "return": tool_return(step)}


def short_lines(lines, limit=4):
    if len(lines) > limit:
        lines = lines[:2] + [f"…省略 {len(lines) - 4} 行…"] + lines[-2:]
    return [line[:220] + ("…" if len(line) > 220 else "") for line in lines]


def cell(lines, start, end, tag):
    prefix = lines[max(start - 1, 0):start]
    changed = short_lines(lines[start:end])
    suffix = lines[end:end + 1]
    rendered = [f"<code>{html.escape(line[:150]).replace('|', '&#124;')}</code>"
                for line in prefix]
    rendered += [f"<mark><{tag}><code>{html.escape(line).replace('|', '&#124;')}</code></{tag}></mark>"
                 for line in changed]
    rendered += [f"<code>{html.escape(line[:150]).replace('|', '&#124;')}</code>"
                 for line in suffix]
    return "<br>".join(rendered) or "（空）"


def comparison(edit):
    # parse_back normalises roles after a rewrite; compare the actual text so
    # renumbered CTX_TURN headers do not hide the model's substantive change.
    def body_lines(value):
        return [line for line in value.splitlines() if not line.startswith("[[CTX_TURN ")]

    before, after = body_lines(edit["before"]), body_lines(edit["after"])
    changes = [(a, b, c, d) for tag, a, b, c, d in
               difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
               if tag != "equal"]
    if not changes:
        return "（文本未变）", "仅消息角色或结构变化", "（文本未变）"
    a, b, c, d = max(changes, key=lambda item: (
        sum(len(line) for line in before[item[0]:item[1]])
        - sum(len(line) for line in after[item[2]:item[3]])))
    before_cell = cell(before, a, b, "del")
    after_cell = cell(after, c, d, "ins")
    change = f"删除 {b-a} 行；新增 {d-c} 行；上下文 {edit['before_tokens']} → {edit['after_tokens']} tokens"
    return before_cell, change, after_cell


def details(title, value):
    return f"<details><summary>{title}</summary>\n\n<pre>{html.escape(value or '(empty)')}</pre>\n\n</details>"


def report_markdown(run_id, mode, rows, cases):
    bad = [row for row in rows if row["case"] == "bad"]
    lines = [f"# ContextBench 历史编辑审计：{run_id}", "",
             f"SCR {mode} · {len(rows)} 题 · {len(rows)-len(bad)} good / {len(bad)} bad · "
             f"实质编辑 {sum(row['edits'] for row in rows)} 次。", "",
             "good = 满分且读完任务流；bad = 未满分或提前结束。标签只用于定位案例，不判断因果。",
             "三列展示同一次 trial 中的一次历史编辑：编辑前、变化最大的片段、编辑后；不是三次独立运行。", "",
             "| 类型 | 任务 | reward | 已送达操作 | 实质编辑 | 结束原因 |",
             "|---|---|---:|---:|---:|---|"]
    for row in cases:
        lines.append(f"| **{row['case']}** | {row['task']} | {row['reward']:.4f} | "
                     f"{row['ops']} | {row['edits']} | {row['reason']} |")
    for row in cases:
        event = row["event"]
        before, change, after = comparison(event)
        lines += ["", f"## {row['task']} · {row['case']} · reward {row['reward']:.4f}", "",
                  f"GPU {row['gpu']}；送达 {row['ops']}；实质编辑 {row['edits']}；"
                  f"结束原因 `{row['reason']}`。展示第 {event['step']} 步的"
                  f"{'最后一次编辑' if row['case'] == 'bad' else '最大幅度缩减'}。", "",
                  "| 编辑前（删除高亮） | 变化 | 编辑后（新增高亮） |",
                  "|---|---|---|", f"| {before} | {change} | {after} |", "",
                  details("Qwen3.6 / ClmAgent 原始回复", event["model_reply"] +
                          ("\n[reasoning_content]\n" + event["reasoning"] if event["reasoning"] else "")),
                  "", details("Qwen3.6 发出的完整 bash 命令", event["command"]),
                  "", details("bash 工具返回与编辑回执", event["return"])]
    lines += ["", "编辑效果取自后续 ATIF-CTX 分段的真实输入；对比时忽略回合编号和角色头，表格仅截取变更最大处，"
              "完整轨迹仍保存在原 trial 的 `trajectory.ctx.json`。工具返回可能受运行时观察长度限制。", ""]
    return "\n".join(lines)


def publish(local, files, temporary):
    url = f"{files}/analysis.md"
    marker = f"{files}/analysis.md._SUCCESS"
    run("ossutil", "cp", str(local), url, "-f")
    copy = temporary / "verified-analysis.md"
    run("ossutil", "cp", url, str(copy), "-f")
    digest = sha256(local)
    if sha256(copy) != digest:
        raise RuntimeError(f"OSS readback differs from local analysis: {url}")
    mark = temporary / "analysis.md._SUCCESS"
    mark.write_text(digest + "\n", encoding="ascii")
    run("ossutil", "cp", str(mark), marker, "-f")
    checked = temporary / "verified-marker"
    run("ossutil", "cp", marker, str(checked), "-f")
    if checked.read_text(encoding="ascii") != mark.read_text(encoding="ascii"):
        raise RuntimeError(f"OSS analysis marker differs: {marker}")
    return url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--mode", choices=("off", "on"), default="off")
    parser.add_argument("--archive-files", type=Path, help="Use local archive files for offline review")
    parser.add_argument("--output", type=Path, help="Local Markdown path; with --archive-files, skip OSS upload")
    args = parser.parse_args()
    if args.archive_files and not args.output:
        parser.error("--archive-files needs --output")
    eval_name = f"contextbench-s42-vllm-{args.mode}"
    if args.archive_files:
        files = str(args.archive_files.resolve())
        scratch = None
    else:
        defaults = dict(line.split("=", 1) for line in
                        (EXPERIMENT / "storage.env").read_text().splitlines())
        prefix = (os.environ.get("CLM_OSS_PREFIX") or defaults["CLM_OSS_PREFIX"]).rstrip("/")
        eval_prefix = f"{prefix}/eval/{args.run_id}/{eval_name}"
        run("ossutil", "stat", f"{eval_prefix}/_SUCCESS")
        files = f"{eval_prefix}/files"
        scratch = Path(subprocess.check_output(
            [sys.executable, str(EXPERIMENT / "storage.py"), "run-dir", args.run_id],
            text=True).strip()) / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
    output = args.output or REPO / "tmp_result" / f"{args.run_id}-{eval_name}-analysis.md"
    output = output.resolve()
    with tempfile.TemporaryDirectory(prefix="audit-cache-", dir=scratch) as temporary:
        cache = Path(temporary)
        archive = Archive(files, cache)
        summary = archive.json("summary.json")
        rows = []
        for name, status in sorted(summary["tasks"].items()):
            trial = Path(status["result"]).parent
            usage = archive.json(trial / "agent/usage.json")
            timing = archive.json(trial / "agent/timing.json")
            rows.append(task_row(name, status, usage, timing))
        bad = [row for row in rows if row["case"] == "bad"]
        good = [max((row for row in rows if row["case"] == "good" and row["family"] == group),
                    key=lambda row: row["edits"], default=None) for group in FAMILIES]
        cases = sorted(bad + [row for row in good if row],
                       key=lambda row: (FAMILIES.index(row["family"]), row["case"] != "bad", row["task"]))
        for row in cases:
            trial = Path(row["result"]).parent
            trajectory_file = archive.path(trial / "agent/trajectory.ctx.json")
            trajectory = json.loads(trajectory_file.read_text(encoding="utf-8"))
            events = edits(trajectory)
            if len(events) != row["syncs"]:
                raise ValueError(f"{row['task']}: trajectory has {len(events)} edit receipts, "
                                 f"usage has {row['syncs']} accepted edits")
            if not events:
                raise ValueError(f"{row['task']}: no edit to compare")
            chosen = events[-1] if row["case"] == "bad" else min(
                events, key=lambda event: event["after_tokens"] - event["before_tokens"])
            row["event"] = trace_event(trajectory, chosen)
            del trajectory
            trajectory_file.unlink()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report_markdown(args.run_id, args.mode, rows, cases), encoding="utf-8")
        print(f"Local report: {output}", flush=True)
        if not args.archive_files:
            print(f"OSS report: {publish(output, files, cache)}", flush=True)


if __name__ == "__main__":
    main()
