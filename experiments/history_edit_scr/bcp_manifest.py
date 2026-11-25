"""Freeze the prepared BrowseComp-Plus Harbor tasks and their external dependencies."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from probe import HERE, PINS

TASK_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
PROVENANCE_FIELDS = (
    "corpus_revision", "queries_revision", "retriever", "index_revision", "judge_revision",
)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(root, task_list, provenance):
    names = task_list.read_text().splitlines()
    if not 8 <= len(names) <= 16 or len(set(names)) != len(names):
        raise ValueError("freeze 8–16 distinct original BCP task directory names")
    if any(not TASK_NAME.fullmatch(name) for name in names):
        raise ValueError("use one bare task directory name per line with no blank lines")
    metadata = json.loads(provenance.read_text())
    for key in PROVENANCE_FIELDS:
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"missing provenance field {key}")
    tasks = []
    for name in names:
        task = root / name
        if not all((task / file).is_file() for file in ("task.toml", "instruction.md")):
            raise ValueError(f"{task}: missing task.toml or instruction.md")
        if not all((task / directory).is_dir() for directory in ("environment", "tests")):
            raise ValueError(f"{task}: missing environment/ or tests/")
        files = sorted(path for path in task.rglob("*") if path.is_file())
        tasks.append({
            "name": name,
            "files_sha256": {str(path.relative_to(task)): file_hash(path) for path in files},
        })
    config = HERE.parents[1] / "clm" / "clm_harness" / "configs" / "bcp.yaml"
    return {
        "benchmark": "BrowseComp-Plus",
        "source_revision": PINS["BCP_REVISION"],
        "task_resources": metadata,
        "clm_bcp_config_sha256": file_hash(config),
        "clm_bcp_config": config.read_text(),
        "tasks": tasks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("task_list", type=Path)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    result = freeze(args.root, args.task_list, args.provenance)
    with args.manifest.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, sort_keys=True)
        out.write("\n")


if __name__ == "__main__":
    main()
