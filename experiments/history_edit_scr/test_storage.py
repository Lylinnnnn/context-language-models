"""Exercise artifact publication and model staging with a fake OSS service."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import storage


class StorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name) / "run-test"
        self.objects = {}
        self.bad_size = False
        mocked = patch.object(storage, "ossutil", side_effect=self.ossutil)
        mocked.start()
        self.addCleanup(mocked.stop)

    def ossutil(self, action, source, *args, capture=False, check=True):
        if action == "cp":
            target = str(args[0])
            if str(source).startswith("oss://"):
                Path(target).write_bytes(self.objects[source])
            else:
                self.objects[target] = Path(source).read_bytes()
            return
        if source not in self.objects:
            return subprocess.CompletedProcess([], 1, b"", b"Error Code: NoSuchKey")
        size = len(self.objects[source]) + self.bad_size
        return subprocess.CompletedProcess([], 0, f"Content-Length : {size}\n".encode(), b"")

    def test_size_verification_precedes_completion(self):
        root = self.run / "eval"
        root.mkdir(parents=True)
        (root / "result.json").write_text("{}")
        prefix = storage.oss_url(f"eval/{self.run.name}")
        self.bad_size = True
        with self.assertRaisesRegex(ValueError, "size verification failed"):
            storage.publish(self.run, "eval")
        self.assertNotIn(f"{prefix}/_SUCCESS", self.objects)
        self.assertTrue((root / "result.json").exists())
        self.bad_size = False
        storage.publish(self.run, "eval")
        self.assertIn(f"{prefix}/_SUCCESS", self.objects)
        with self.assertRaisesRegex(ValueError, "already published"):
            storage.publish(self.run, "eval")

    def test_model_staging_checks_hash_and_withholds_final_path(self):
        source = self.run / "model" / storage.REVISION
        source.mkdir(parents=True)
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            (source / name).write_text("{}")
        (source / ".gitattributes").write_text("metadata")
        (source / "weights.safetensors").write_bytes(b"weight")
        (source / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"sample": "weights.safetensors"},
        }))
        storage.publish(self.run, "model")

        new_run = self.run.with_name("second")
        target = new_run / "model" / storage.REVISION
        key = storage.oss_url(f"{storage.MODEL_PREFIX}/files/weights.safetensors")
        self.objects[key] = b"broken"
        with self.assertRaisesRegex(ValueError, "file verification failed"):
            storage.stage_model(new_run)
        self.assertFalse(target.exists())
        self.objects[key] = b"weight"
        self.assertEqual(storage.stage_model(new_run), target)
        self.assertEqual((target / "weights.safetensors").read_bytes(), b"weight")
        self.assertFalse((target / ".gitattributes").exists())

    def test_eval_subdir_has_independent_completion_marker(self):
        root = self.run / "eval"
        (root / "vllm-smoke").mkdir(parents=True)
        (root / "vllm-c2").mkdir()
        (root / "vllm-smoke" / "result.json").write_text("{}")
        (root / "vllm-c2" / "result.json").write_text('{"pairs": 4}')
        storage.publish(self.run, "eval")
        storage.publish(self.run, "eval", "vllm-c2")
        prefix = storage.oss_url(f"eval/{self.run.name}/vllm-c2")
        self.assertIn(f"{prefix}/_SUCCESS", self.objects)
        self.assertEqual(self.objects[f"{prefix}/files/result.json"], b'{"pairs": 4}')
        self.assertNotIn(f"{prefix}/files/vllm-smoke/result.json", self.objects)

    def test_eval_cleanup_requires_verified_oss_copy(self):
        root = self.run / "eval" / "full-s42"
        root.mkdir(parents=True)
        (root / "summary.json").write_text('{"completed": 32}')
        prefix = storage.oss_url(f"eval/{self.run.name}/full-s42")

        self.bad_size = True
        with self.assertRaisesRegex(ValueError, "size verification failed"):
            storage.publish_eval_and_cleanup(self.run, "full-s42")
        self.assertTrue(root.exists())
        self.assertNotIn(f"{prefix}/_SUCCESS", self.objects)

        self.bad_size = False
        storage.publish_eval_and_cleanup(self.run, "full-s42")
        self.assertFalse(root.exists())
        self.assertFalse((self.run / "transfer" / "publish-eval-full-s42").exists())
        self.assertIn(f"{prefix}/_SUCCESS", self.objects)

    def test_eval_cleanup_rejects_changed_local_result_after_publication(self):
        root = self.run / "eval" / "full-s42"
        root.mkdir(parents=True)
        result = root / "summary.json"
        result.write_text('{"completed": 32}')
        storage.publish(self.run, "eval", "full-s42")
        result.write_text('{"completed": 31}')
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            storage.publish_eval_and_cleanup(self.run, "full-s42")
        self.assertTrue(root.exists())
        result.write_text('{"completed": 32}')
        storage.publish_eval_and_cleanup(self.run, "full-s42")
        self.assertFalse(root.exists())

    def test_eval_cleanup_rejects_invalid_remote_completion_marker(self):
        root = self.run / "eval" / "full-s42"
        root.mkdir(parents=True)
        (root / "summary.json").write_text('{"completed": 32}')
        storage.publish(self.run, "eval", "full-s42")
        prefix = storage.oss_url(f"eval/{self.run.name}/full-s42")
        self.objects[f"{prefix}/_SUCCESS"] = b"0" * 64 + b"\n"
        with self.assertRaisesRegex(ValueError, "completion marker does not match"):
            storage.publish_eval_and_cleanup(self.run, "full-s42")
        self.assertTrue(root.exists())


if __name__ == "__main__":
    unittest.main()
