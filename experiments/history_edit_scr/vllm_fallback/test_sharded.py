"""Verify that multi-GPU reports pool complete off/on pairs by base case."""

import json
import tempfile
import unittest
from pathlib import Path

from sharded import aggregate, shard_manifest


class ShardedReportTests(unittest.TestCase):
    def test_partition_keeps_case_branches_together_and_aggregate_checks_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = []
            for index in range(2):
                cases.append({
                    "case_id": f"kv-edit-{index + 1:03d}", "source_task_id": None,
                    "fixture_type": "refuted", "history_material": "visible_assistant_note",
                    "ground_truth_label": "A",
                    "branches": [{
                        "condition": "clean", "warm_variant": variant,
                        "old_label": old, "alternate_old_label": other,
                        "final_ids": [1, 2, 3], "warm_ids": [1, 4 if variant == "a" else 5, 3],
                    } for variant, old, other in (("a", "B", "C"), ("b", "C", "B"))],
                })
            manifest = {"schema": 1, "model_revision": "modelscope-master-v1", "cases": cases}
            from probe import digest

            for index in range(2):
                directory = root / f"shard-{index:02d}"
                run = directory / "run"
                for arm in ("off", "on"):
                    (run / arm).mkdir(parents=True, exist_ok=True)
                    config = {
                        "mode": arm, "cuda_visible_devices": str(index), "model_revision": "modelscope-master-v1",
                        "kvreuse_env": {"KVREUSE_ENABLED": "1" if arm == "on" else "0", "KVREUSE_MAX_BLOCKS": "6"},
                    }
                    (run / arm / "server.log").write_text("[history-edit-vllm-config] " + json.dumps(config) + "\n")
                part = shard_manifest(manifest, index, 2)
                (directory / "manifest-input.json").write_text(json.dumps(part))
                (run / "probe-summary.json").write_text(json.dumps({
                    "passed": True, "manifest_sha256": digest(part),
                    "invalid_pairs": [], "expected_pairs": 2, "valid_pairs": 2,
                    "clean_on_hits": 1,
                }))
                with (run / "probe-joined.jsonl").open("w") as stream:
                    for branch in part["cases"][0]["branches"]:
                        for arm in ("off", "on"):
                            row = {
                                "case_id": part["cases"][0]["case_id"], "seed": 42,
                                "condition": "clean", "warm_variant": branch["warm_variant"],
                                "fixture_type": "refuted", "ground_truth_label": "A",
                                "old_label": branch["old_label"], "alternate_old_label": branch["alternate_old_label"],
                                "scr_enabled": arm == "on", "errors": [], "splice_ok": arm == "on",
                                "parsed_label": "A", "result_class": "correct", "forwarded": 2 if arm == "on" else 3,
                                "prefix_reuse_pflops": 0.01, "final_input_sha256": digest(branch["final_ids"]),
                            }
                            stream.write(json.dumps(row) + "\n")
            result = aggregate(root, manifest, 2, [42])
            self.assertEqual((result["valid_pairs"], result["expected_pairs"], result["clean_on_hits"]), (4, 4, 4))
            self.assertEqual(len(result["shards"]), 2)
            cases[0]["case_id"] = "different"
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                aggregate(root, manifest, 2, [42])


if __name__ == "__main__":
    unittest.main()
