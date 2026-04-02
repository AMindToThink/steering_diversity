#!/usr/bin/env python3
"""One-time migration: wrap hand-written tables in the paper with GENERATED TABLE markers.

After running this, `generate_paper_tables.py --update-paper` will keep them in sync.

Usage:
    uv run scripts/wire_paper_tables.py
    uv run scripts/generate_paper_tables.py --update-paper
"""

from __future__ import annotations

import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "docs" / "steering_diversity_paper.md"

# Each entry: (table_name, unique_string_before_table, table_header_start, lines_after_header)
# We find tables by matching the header row and wrapping from there to the last | row.
TABLES: list[tuple[str, str]] = [
    # (marker name, a unique substring from the table's header row)
    ("happy_full_stats", "| Mixed-effects model (primary)             | beta = -0.0027"),
    ("happy_full_metrics", "| 0.0   | 29       | 0.444"),
    ("happy_recon_within_vs_pooled", "| 0.0   |       0.555 +/- 0.037"),
    ("happy_recon_stats", "| Mixed-effects (primary)                | beta = -0.0002"),
    ("style_full_metrics", "| 0.0   | 7        |        0.703"),
    ("style_full_stats", "| Mixed-effects (primary)                 | beta = -0.004"),
    ("creativity_full_metrics", "| 0.0   | 27       |        0.682"),
    ("creativity_full_stats", "| Mixed-effects (primary)                       | beta = +0.017"),
    ("cross_experiment_comparison", "| happy_recon     | Qwen2.5-1.5B | happy"),
    ("eval_awareness_qwen3_32b", "| All     | 0.64 | 0.36  | 0.36"),
    ("ms_triggers_unsteered", "| Model awareness | 0 (0%) | 50 (100%)"),
    ("ms_triggers_steered", "| Model awareness | 0 (0%) | 36 (100%)"),
    ("ms_triggers_comparison", "| Awareness (Yes+Maybe) | 0% | 0%"),
    ("passk_preliminary", "| **pass@1 (base)** | 0.432 | 0.407"),
    ("power_analysis", "| 10 | −0.020 ± 0.023 | −0.058 ± 0.040"),
    ("coverage_gain_n100", "| 2 | −0.020 | 0.005 | 0.0001"),
    ("qwq_passk", "| pass@1 | 0.413 | 0.391 | −0.022"),
    ("qwq_coverage_gain", "| 2 | +0.0013 | 0.0076"),
]


def find_table_bounds(lines: list[str], anchor: str) -> tuple[int, int] | None:
    """Find the first and last line index of the markdown table containing the anchor."""
    anchor_idx = None
    for i, line in enumerate(lines):
        if anchor in line:
            anchor_idx = i
            break
    if anchor_idx is None:
        return None

    # Walk backwards to find the header row (first | row)
    start = anchor_idx
    while start > 0 and lines[start - 1].strip().startswith("|"):
        start -= 1

    # Walk forwards to find the last | row
    end = anchor_idx
    while end < len(lines) - 1 and lines[end + 1].strip().startswith("|"):
        end += 1

    return start, end


def main() -> None:
    text = PAPER.read_text()
    lines = text.splitlines()

    # Process tables in reverse order of position so indices don't shift
    insertions: list[tuple[int, int, str]] = []  # (start, end, marker_name)

    for name, anchor in TABLES:
        bounds = find_table_bounds(lines, anchor)
        if bounds is None:
            print(f"  SKIP {name}: anchor not found")
            continue
        start, end = bounds
        # Check if already wrapped
        if start > 0 and "BEGIN GENERATED TABLE" in lines[start - 1]:
            print(f"  SKIP {name}: already wrapped")
            continue
        insertions.append((start, end, name))
        print(f"  FOUND {name}: lines {start+1}–{end+1}")

    # Sort by start line descending so we can insert without shifting earlier indices
    insertions.sort(key=lambda x: x[0], reverse=True)

    for start, end, name in insertions:
        lines.insert(end + 1, f"<!-- END GENERATED TABLE: {name} -->")
        lines.insert(start, f"<!-- BEGIN GENERATED TABLE: {name} -->")

    PAPER.write_text("\n".join(lines))
    print(f"\nWrapped {len(insertions)} tables with markers.")
    print("Now run: uv run scripts/generate_paper_tables.py --update-paper")


if __name__ == "__main__":
    main()
