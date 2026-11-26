"""Offline verification of the already-published data retry path."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import official_data
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

            with patch.object(official_data, "copy_file", side_effect=download), \
                 patch.object(official_data, "verify_size") as verify_size:
                official_data.verify_published(run, "oss://test/data/bcp-data-test")
                verify_size.assert_called_once_with(
                    "oss://test/data/bcp-data-test/files/queries/sample.parquet", 7
                )
                marker.write_text("invalid\n")
                with self.assertRaisesRegex(ValueError, "completion marker"):
                    official_data.verify_published(run, "oss://test/data/bcp-data-test")


if __name__ == "__main__":
    unittest.main()
