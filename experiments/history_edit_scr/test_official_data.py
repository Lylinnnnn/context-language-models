"""Offline verification of the already-published data retry path."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import storage
from storage import collect_files, file_sha


class PublishedDataTest(unittest.TestCase):
    def test_remote_manifest_and_marker_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "bcp-data-test"
            data = run / "data" / "queries"
            data.mkdir(parents=True)
            (data / "sample.parquet").write_bytes(b"fixture")
            remote = root / "remote"
            remote.mkdir()
            manifest = remote / "manifest.json"
            manifest.write_text(json.dumps({"kind": "data", "files": collect_files(run / "data")}) + "\n")
            marker = remote / "_SUCCESS"
            marker.write_text(file_sha(manifest) + "\n")

            def download(source, destination, _workspace):
                shutil.copyfile(remote / source.rsplit("/", 1)[-1], destination)

            with patch.dict("os.environ", {"CLM_OSS_PREFIX": "oss://test"}), \
                 patch.object(storage, "copy_file", side_effect=download), \
                 patch.object(storage, "verify_size") as verify_size:
                storage.verify_published(run, "data")
                verify_size.assert_any_call(
                    "oss://test/data/bcp-data-test/files/queries/sample.parquet", 7
                )
                marker.write_text("invalid\n")
                with self.assertRaisesRegex(ValueError, "completion marker"):
                    storage.verify_published(run, "data")


if __name__ == "__main__":
    unittest.main()
