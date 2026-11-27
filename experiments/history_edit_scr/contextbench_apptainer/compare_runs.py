#!/usr/bin/env python3
"""Compare archived seed-42 ContextBench CLM and no-edit runs in one Markdown report."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from audit_run import Archive, EXPERIMENT, FAMILIES, REASONS, REPO, publish, run


EDIT_EVAL = "contextbench-s42-vllm-off"
NO_EDIT_EVAL = "contextbench-s42-vllm-off-no-edit"
PINNED_FIELDS = ("seed", "mode", "contextbench_commit", "task_stream_bytes",
                 "model_revision", "max_model_len", "tool_call_parser")


def load(archive):
    summary = archive.json("summary.json")
    if summary["completed"] != 32 or summary["failed"] or summary["missing"]:
        raise ValueError(f"Incomplete archived run: {summary['completed']}/32 completed")
    rows = {}
    for name, status in summary["tasks"].items():
        if not status["ok"]:
            raise ValueError(f"Trial failed: {name}")
        rewards = status["rewards"]
        trial = Path(status["result"]).parent
        usage = archive.json(trial / "agent/usage.json")
        timing = archive.json(trial / "agent/timing.json")
        rows[name] = {
            "reward": float(status["reward"]),
            "matched": int(rewards["matched"]),
            "total": int(rewards["total"]),
            "delivered": int(rewards["n_ops_delivered"]),
            "ops_total": int(rewards["n_ops_total"]),
            "reason": REASONS.get(rewards["terminated_by_code"], "unknown"),
            "syncs": int(usage["n_ctx_syncs"]),
            "real_edits": int(usage["n_ctx_syncs_real"]),
            "lm_calls": int(usage["n_lm_calls"]),
            "prompt_tokens": int(usage["prompt_tokens"]),
            "completion_tokens": int(usage["completion_tokens"]),
            "llm_s": sum(float(event.get("llm_s", 0)) for event in timing),
        }
    return summary, rows


def totals(rows):
    values = list(rows.values())
    return {
        "n": len(values),
        "mean_reward": sum(row["reward"] for row in values) / len(values),
        "perfect": sum(row["reward"] == 1.0 for row in values),
        "stream_complete": sum(row["reason"] == "stream_exhausted" and
                               row["delivered"] == row["ops_total"] for row in values),
        "budget_wall": sum(row["reason"] == "budget_wall" for row in values),
        **{field: sum(row[field] for row in values) for field in
           ("matched", "total", "delivered", "ops_total", "syncs", "real_edits",
            "lm_calls", "prompt_tokens", "completion_tokens", "llm_s")},
    }


def fraction(part, whole):
    return f"{part}/{whole} ({part / whole:.1%})" if whole else "n/a"


def report(edited_id, no_edit_id, edited, no_edit):
    a, b = totals(edited), totals(no_edit)
    changes = sorted(((name, edited[name]["reward"] - no_edit[name]["reward"])
                      for name in edited), key=lambda item: (-abs(item[1]), item[0]))
    improved = sum(delta > 0 for _, delta in changes)
    worsened = sum(delta < 0 for _, delta in changes)
    ties = len(changes) - improved - worsened
    lines = ["# ContextBench：CLM 自编辑 vs 无编辑", "",
             f"32 个相同的原版 seed 42 任务；Qwen3.6-27B；vLLM SCR off。"
             f"有编辑 `{edited_id}`；无编辑 `{no_edit_id}`。", "",
             "| 指标 | 有编辑 | 无编辑 | 有编辑 − 无编辑 |",
             "|---|---:|---:|---:|",
             f"| 平均 reward | {a['mean_reward']:.4f} | {b['mean_reward']:.4f} | "
             f"{a['mean_reward']-b['mean_reward']:+.4f} |",
             f"| 满分题数 | {a['perfect']}/32 | {b['perfect']}/32 | "
             f"{a['perfect']-b['perfect']:+d} |",
             f"| 完整走完任务流 | {a['stream_complete']}/32 | {b['stream_complete']}/32 | "
             f"{a['stream_complete']-b['stream_complete']:+d} |",
             f"| 答对问题 / 全部问题 | {fraction(a['matched'], a['total'])} | "
             f"{fraction(b['matched'], b['total'])} | {a['matched']-b['matched']:+d} 题 |",
             f"| 已送达操作 / 全部操作 | {fraction(a['delivered'], a['ops_total'])} | "
             f"{fraction(b['delivered'], b['ops_total'])} | {a['delivered']-b['delivered']:+d} 步 |",
             f"| 上下文预算墙 | {a['budget_wall']} | {b['budget_wall']} | "
             f"{a['budget_wall']-b['budget_wall']:+d} |",
             f"| 历史编辑（实质 / 接受） | {a['real_edits']} / {a['syncs']} | "
             f"{b['real_edits']} / {b['syncs']} | — |",
             f"| LM 调用 | {a['lm_calls']} | {b['lm_calls']} | {a['lm_calls']-b['lm_calls']:+d} |",
             f"| 输入 tokens | {a['prompt_tokens']:,} | {b['prompt_tokens']:,} | "
             f"{a['prompt_tokens']-b['prompt_tokens']:+,} |",
             f"| 输出 tokens | {a['completion_tokens']:,} | {b['completion_tokens']:,} | "
             f"{a['completion_tokens']-b['completion_tokens']:+,} |",
             f"| LM 时间之和（小时） | {a['llm_s']/3600:.2f} | {b['llm_s']/3600:.2f} | "
             f"{(a['llm_s']-b['llm_s'])/3600:+.2f} |", "",
             f"逐题 reward：有编辑更高 {improved} 题、无编辑更高 {worsened} 题、相同 {ties} 题。", "",
             "## 四类任务", "",
             "| 类别 | 题数 | 有编辑平均 reward | 无编辑平均 reward | 差值 |",
             "|---|---:|---:|---:|---:|"]
    for family in FAMILIES:
        names = [name for name in edited if name.startswith(family + "-")]
        ea = sum(edited[name]["reward"] for name in names) / len(names)
        nb = sum(no_edit[name]["reward"] for name in names) / len(names)
        lines.append(f"| {family} | {len(names)} | {ea:.4f} | {nb:.4f} | {ea-nb:+.4f} |")
    lines += ["", "## 分数变化的任务", "",
              "| 任务 | 有编辑 | 无编辑 | 差值 | 操作送达（编辑 / 无编辑） | 结束原因（编辑 / 无编辑） |",
              "|---|---:|---:|---:|---|---|"]
    for name, delta in changes:
        if delta == 0:
            continue
        x, y = edited[name], no_edit[name]
        lines.append(f"| {name} | {x['reward']:.4f} | {y['reward']:.4f} | {delta:+.4f} | "
                     f"{x['delivered']}/{x['ops_total']} / {y['delivered']}/{y['ops_total']} | "
                     f"{x['reason']} / {y['reason']} |")
    if improved + worsened == 0:
        lines.append("| 全部任务的 reward 相同 | — | — | 0 | — | — |")
    lines += ["", "<details><summary>查看全部 32 题</summary>", "",
              "| 任务 | 有编辑 reward | 无编辑 reward | 差值 | 有编辑 / 无编辑终止原因 |",
              "|---|---:|---:|---:|---|"]
    for name in sorted(edited):
        x, y = edited[name], no_edit[name]
        lines.append(f"| {name} | {x['reward']:.4f} | {y['reward']:.4f} | "
                     f"{x['reward']-y['reward']:+.4f} | {x['reason']} / {y['reason']} |")
    lines += ["", "</details>", "",
              "说明：reward 是官方任务评分；答对率用 `matched/total` 计算，操作覆盖率用 "
              "`n_ops_delivered/n_ops_total` 计算。LM 时间来自逐轮 `timing.json` 的近似求和，"
              "不是八卡运行的墙钟时间。一次 seed 42、随机采样不能给出显著性结论；"
              "无编辑组同时移除了编辑提示和方法 skill，因此这是方法对照，不是纯文本编辑因果消融。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edited_run_id")
    parser.add_argument("no_edit_run_id")
    args = parser.parse_args()
    defaults = dict(line.split("=", 1) for line in
                    (EXPERIMENT / "storage.env").read_text().splitlines())
    prefix = (os.environ.get("CLM_OSS_PREFIX") or defaults["CLM_OSS_PREFIX"]).rstrip("/")
    edited_prefix = f"{prefix}/eval/{args.edited_run_id}/{EDIT_EVAL}"
    no_edit_prefix = f"{prefix}/eval/{args.no_edit_run_id}/{NO_EDIT_EVAL}"
    run("ossutil", "stat", f"{edited_prefix}/_SUCCESS")
    run("ossutil", "stat", f"{no_edit_prefix}/_SUCCESS")
    scratch = Path(subprocess.check_output(
        [sys.executable, str(EXPERIMENT / "storage.py"), "run-dir", args.no_edit_run_id],
        text=True).strip()) / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compare-cache-", dir=scratch) as temporary:
        temporary = Path(temporary)
        edited = Archive(f"{edited_prefix}/files", temporary / "edited")
        no_edit = Archive(f"{no_edit_prefix}/files", temporary / "no-edit")
        a_manifest, b_manifest = edited.json("manifest.json"), no_edit.json("manifest.json")
        for field in PINNED_FIELDS:
            if a_manifest[field] != b_manifest[field]:
                raise ValueError(f"Different {field}: {a_manifest[field]} vs {b_manifest[field]}")
        if b_manifest.get("agent") != "no-edit":
            raise ValueError("Second run is not marked as no-edit")
        _, a_rows = load(edited)
        _, b_rows = load(no_edit)
        if a_rows.keys() != b_rows.keys():
            raise ValueError("Task lists differ between the two runs")
        if any(row["syncs"] for row in b_rows.values()):
            raise ValueError("No-edit run contains accepted context edits")
        output = REPO / "tmp_result" / f"{args.no_edit_run_id}-vs-{args.edited_run_id}.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report(args.edited_run_id, args.no_edit_run_id, a_rows, b_rows),
                          encoding="utf-8")
        print(f"Local report: {output}", flush=True)
        print(f"OSS report: {publish(output, f'{no_edit_prefix}/files', temporary, 'comparison.md')}",
              flush=True)


if __name__ == "__main__":
    main()
