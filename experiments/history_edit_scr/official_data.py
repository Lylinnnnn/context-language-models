#!/usr/bin/env python3
"""Download pinned BrowseComp-Plus source data, verify OSS upload, then clear scratch."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from probe import PINS
from storage import local_path, oss_url, publish, published_marker_exists, setting, verify_published


DATASETS = {
    "queries": "Tevatron/browsecomp-plus",
    "corpus": "Tevatron/browsecomp-plus-corpus",
    "indexes": "Tevatron/browsecomp-plus-indexes",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="new bcp-data-... directory below CLM_SCRATCH_ROOT")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--verify-only", action="store_true", help="verify an existing OSS archive against scratch without network downloads")
    parser.add_argument("--keep-scratch", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"bcp-data-[A-Za-z0-9._-]+", args.run_id):
        parser.error("run ID must start with bcp-data- and be one safe directory name")

    run = local_path(Path(setting("CLM_SCRATCH_ROOT")) / args.run_id)
    sources_file = run / "data" / "sources.json"
    marker_file = run / ".official-data-run"
    if run.exists():
        if not marker_file.is_file() or marker_file.read_text() != "bcp-data\n":
            parser.error(f"scratch directory was not created by this downloader: {run}")
    else:
        if args.verify_only:
            parser.error(f"scratch directory is absent: {run}")
        run.mkdir(parents=True)
        marker_file.write_text("bcp-data\n")
    prefix = oss_url(f"data/{args.run_id}")
    if args.verify_only:
        if not sources_file.is_file():
            raise ValueError(f"saved source revisions are missing: {sources_file}")
        verify_published(run, "data")
        print(f"VERIFIED {prefix}/_SUCCESS", flush=True)
        if not args.keep_scratch:
            shutil.rmtree(run)
            print(f"REMOVED SCRATCH {run}", flush=True)
        return
    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ["HF_HOME"] = str(run / "hf-cache")
    from huggingface_hub import HfApi, snapshot_download

    if sources_file.exists():
        sources = json.loads(sources_file.read_text(encoding="utf-8"))
        if (sources.get("bcp_code_revision") != PINS["BCP_REVISION"]
                or {name: value.get("repo") for name, value in sources.get("datasets", {}).items()} != DATASETS):
            raise ValueError("saved source revisions do not match this experiment")
    else:
        api = HfApi(endpoint=args.endpoint)
        sources = {"bcp_code_revision": PINS["BCP_REVISION"], "datasets": {}}
        for name, repo in DATASETS.items():
            sources["datasets"][name] = {"repo": repo, "revision": api.dataset_info(repo).sha}
        sources_file.parent.mkdir(parents=True, exist_ok=True)
        sources_file.write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for name, repo in DATASETS.items():
        revision = sources["datasets"][name]["revision"]
        print(f"Downloading {name}: {repo}@{revision}", flush=True)
        snapshot_download(
            repo_id=repo, repo_type="dataset", revision=revision,
            local_dir=run / "data" / name, max_workers=4,
        )

    for name in DATASETS:
        directory = run / "data" / name
        if not any(path.is_file() and not path.name.startswith(".")
                   for path in directory.rglob("*")):
            raise ValueError(f"no dataset files downloaded in {directory}")
    # storage.publish verifies every uploaded size and writes manifest.json then _SUCCESS.
    if not published_marker_exists(prefix):
        publish(run, "data")
    verify_published(run, "data")
    print(f"VERIFIED {prefix}/_SUCCESS", flush=True)
    if not args.keep_scratch:
        shutil.rmtree(run)
        print(f"REMOVED SCRATCH {run}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        sys.exit(str(exc))
