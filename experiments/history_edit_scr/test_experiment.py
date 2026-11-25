"""Model-free checks of fixture, request, and report invariants."""

import argparse
import hashlib
import io
import json
import random
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import edit_fixtures
import probe
import report


class TestTokenizer:
    def apply_chat_template(self, chat, **kwargs):
        return "\n".join(f"<{m['role']}>\n{m['content']}" for m in chat) + "\n<assistant>"

    def encode(self, text, **kwargs):
        return [int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
                for token in re.findall(r"\w+|[^\w\s]", text)]


def finish(rid, prompt, decode, cached):
    return f"Finish: {{'id': '{rid}', 'prompt_tokens': {prompt}, 'cached_tokens': {cached}, 'completion_tokens': {decode}}}"


class ExperimentTests(unittest.TestCase):
    def test_paired_requests_and_official_log_accounting(self):
        case = probe.fixture(TestTokenizer(), 1, ("A", "B", "C"), random.Random(42), "full", False)
        manifest = {"schema": 1, "model_revision": probe.PINS["MODEL_REVISION"], "cases": [case]}
        clean_a, clean_b, *_, cold = case["branches"]
        self.assertEqual(clean_a["final_ids"], clean_b["final_ids"])
        self.assertEqual(clean_a["final_ids"], cold["final_ids"])
        self.assertGreaterEqual(clean_a["common_prefix_tokens"], 1024)

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            manifest_path = base / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            servers = {}
            for arm in report.ARMS:
                directory = base / arm
                directory.mkdir()

                def generate(url, request, timeout):
                    rid = request["rid"]
                    label = "C"
                    if "-visible-" in rid or (arm == "on" and "-clean-" in rid):
                        label = "A" if "-a-42-" in rid else "B"
                    return {
                        "text": f"ACTION: {label}",
                        "meta_info": {"id": rid, "prompt_tokens": len(request["input_ids"]),
                                      "completion_tokens": 12},
                    }

                path = directory / "requests.jsonl"
                args = argparse.Namespace(manifest=manifest_path, requests=path, arm=arm,
                                          url="http://unused", seeds=[42], timeout=10)
                with patch.object(probe, "generate", generate), redirect_stdout(io.StringIO()):
                    self.assertEqual(probe.run(args), 0)
                rows = probe.read_jsonl(path)
                lines = ["[history-edit-config] {}", f"[scr] Suffix Cache Reuse {arm} | sglang 0.5.16"]
                rids = [f"smoke-{arm}-aaaa-turn{turn}" for turn in range(1, 4)]
                (directory / "smoke.log").write_text(
                    "\n".join(f"{rid}: prompt_tokens=500" for rid in rids) + "\nsmoke: all requests succeeded\n"
                )
                (directory / "regression.log").write_text("regression: all passed\n")
                if arm == "on":
                    lines += ["[kvreuse-site] max_blocks=6 splice_retry=8",
                              f"v6 splice OK: rid={rids[1]} sid=smoke slot=0 A+B'=100 C'=200 extend=100"]
                lines += [finish(rid, 500, 12, 400 if arm == "on" and rid == rids[1] else 0) for rid in rids]
                warm_lengths = {}
                for row in rows:
                    rid = row["rid"]
                    size = len(row["request"]["input_ids"])
                    cache_key = row["cache_key"]
                    old_length = warm_lengths.get(cache_key, 0)
                    sid = f"session-{cache_key}"
                    splice = arm == "on" and old_length > 0
                    if arm == "on":
                        lines.append(f"kv6trace ev=req rid={rid} sid={sid} slot=0 n_new={size} n_old={old_length}")
                    if splice:
                        lines.append(f"v6 splice OK: rid={rid} sid={sid} slot=0 A+B'=1024 C'=256 extend=100")
                    lines.append(finish(rid, size, 12, 1536 if splice else 0))
                    if row["phase"] == "warm":
                        warm_lengths[cache_key] = size
                (directory / "server.log").write_text("\n".join(lines) + "\n")
                _, stats, reasons = report.read_server(base, arm)
                servers[arm] = stats, reasons

            smoke = report.smoke_report(base, servers)
            self.assertTrue(smoke["passed"])
            result = report.probe_report(base, manifest, [42], servers)
            self.assertEqual(result["valid_pairs"], 5)
            self.assertEqual(result["effects"]["h1_visible_minus_clean_off"]["estimate"], 1)
            self.assertEqual(result["effects"]["h2_on_minus_off_both_histories_hit"]["estimate"], 1)
            self.assertEqual(result["groups"]["refuted/cold"]["on_hits"], 0)

            rows[1]["request"]["input_ids"][0] += 1
            with path.open("w") as stream:
                for row in rows:
                    probe.write_row(stream, row)
            bad = report.probe_report(base, manifest, [42], servers)
            self.assertEqual(bad["valid_pairs"], 4)
            self.assertIn("input IDs", bad["invalid_pairs"][0]["on"][0])

    def test_edit_oracle_retains_observations_and_credential(self):
        cases = list(edit_fixtures.scenario(random.Random(42), 0, 4, True))
        self.assertEqual(len(cases), 4)
        for case in cases:
            oracle = case["branches"]["oracle"]
            self.assertNotIn("handoff", oracle["dropped_span_ids"])
            self.assertNotIn("inventory", oracle["dropped_span_ids"])
            self.assertGreaterEqual(len(oracle["dropped_span_ids"]), 1)
            actions = [
                {"action": "dispatch", "lane": lane, "station": station}
                for lane, station in case["station_by_lane"].items()
            ]
            self.assertFalse(edit_fixtures.score(case, actions)["task_success"])
            actions.append({"action": "handoff", "credential": case["handoff_credential"]})
            self.assertTrue(edit_fixtures.score(case, actions)["task_success"])
            lane, station = case["stale_actions"][0]
            actions[0] = {"action": "dispatch", "lane": lane, "station": station}
            self.assertTrue(edit_fixtures.score(case, actions)["stale_action"])
            self.assertFalse(edit_fixtures.score(case, actions)["task_success"])


if __name__ == "__main__":
    unittest.main()
