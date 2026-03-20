"""Merge split evalplus sample files from parallel GPU runs.

When codegen is split across multiple GPUs (each generating n/k samples),
this script concatenates the JSONL files and validates the result.

Usage:
    uv run python scripts/eval/merge_split_samples.py \
        outputs/run_a/humaneval/samples.jsonl \
        outputs/run_b/humaneval/samples.jsonl \
        --output outputs/merged/humaneval/samples.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def merge_samples(input_paths: list[Path], output_path: Path) -> None:
    """Merge multiple evalplus sample JSONL files into one.

    Validates that all inputs have the same task_ids and consistent
    sample counts per task.
    """
    all_lines: list[str] = []
    per_file_tasks: list[set[str]] = []

    for path in input_paths:
        tasks_in_file: set[str] = set()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                task_id = rec["task_id"]
                tasks_in_file.add(task_id)
                all_lines.append(line)
        per_file_tasks.append(tasks_in_file)
        print(f"  {path}: {len(tasks_in_file)} tasks")

    # Validate: all files have the same task_ids
    all_task_ids = per_file_tasks[0]
    for i, tasks in enumerate(per_file_tasks[1:], 1):
        missing = all_task_ids - tasks
        extra = tasks - all_task_ids
        if missing:
            raise ValueError(
                f"File {input_paths[i]} is missing tasks present in "
                f"file {input_paths[0]}: {sorted(missing)[:5]}..."
            )
        if extra:
            raise ValueError(
                f"File {input_paths[i]} has extra tasks not in "
                f"file {input_paths[0]}: {sorted(extra)[:5]}..."
            )

    # Count samples per task in merged result
    task_counts: Counter[str] = Counter()
    for line in all_lines:
        rec = json.loads(line)
        task_counts[rec["task_id"]] += 1

    counts = set(task_counts.values())
    if len(counts) != 1:
        min_count = min(task_counts.values())
        max_count = max(task_counts.values())
        raise ValueError(
            f"Inconsistent sample counts after merge: "
            f"min={min_count}, max={max_count}. "
            f"All tasks should have the same number of samples."
        )

    n_tasks = len(task_counts)
    samples_per_task = counts.pop()

    # Write merged output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for line in all_lines:
            f.write(line + "\n")

    print(f"\nMerged: {n_tasks} tasks × {samples_per_task} samples = "
          f"{len(all_lines)} total lines → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge split evalplus sample JSONL files"
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="Paths to evalplus sample JSONL files to merge",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Path for merged output JSONL",
    )
    args = parser.parse_args()

    for path in args.inputs:
        if not path.exists():
            print(f"Error: {path} does not exist", file=sys.stderr)
            sys.exit(1)

    merge_samples(args.inputs, args.output)


if __name__ == "__main__":
    main()
