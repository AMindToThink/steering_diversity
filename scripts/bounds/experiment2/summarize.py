"""Phase 7 of experiment 2 — summarize results as a markdown table.

Reads every ``configs/bounds/experiment2/*.yaml``, loads the corresponding
``bounds_metrics.json``, and prints a one-line-per-run table of the
Claim 8 scaling slopes and Claim 7 pass/fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _describe_config(cfg: dict, run_name: str) -> str:
    steering = cfg["steering"]
    target_layers = steering["target_layers"]
    n_layers = len(target_layers)
    if n_layers == 1:
        layer_str = f"L{target_layers[0]}"
    else:
        layer_str = f"L{min(target_layers)}-{max(target_layers)}"

    if steering.get("vector_path"):
        src = "REAL"
    elif steering.get("random_reference_path"):
        match = steering.get("random_match", "per_layer")
        src = f"RAND({match})"
    else:
        src = "?"
    return f"{src} {layer_str}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    for cfg_path in sorted(args.configs_dir.glob("*.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text())
        run_name = cfg["run_name"]
        description = _describe_config(cfg, run_name)
        metrics_path = Path("outputs/bounds") / run_name / "bounds_metrics.json"
        if not metrics_path.exists():
            rows.append({
                "run": run_name,
                "config": cfg_path.name,
                "description": description,
                "status": "MISSING",
                "slope_V": None,
                "slope_trZ": None,
                "claim7_all": None,
            })
            continue
        metrics = json.loads(metrics_path.read_text())
        c8 = metrics.get("claim8_scaling_fit", {})
        slope_v = c8.get("detail", {}).get("slope_spherical_variance")
        slope_tz = c8.get("detail", {}).get("slope_trace_cov_z")

        # Claim 7 pass across all scales?
        claim7_passes = []
        for key, entry in metrics.items():
            try:
                scale = float(key)
            except (TypeError, ValueError):
                continue
            c7 = (entry or {}).get("claim7_reduction_condition")
            if c7 and c7.get("passed") is not None:
                claim7_passes.append(bool(c7["passed"]))
        if claim7_passes:
            all_pass = all(claim7_passes)
            all_fail = not any(claim7_passes)
            claim7_str = "all pass" if all_pass else ("all fail" if all_fail else "mixed")
        else:
            claim7_str = "n/a"

        rows.append({
            "run": run_name,
            "config": cfg_path.name,
            "description": description,
            "status": "OK",
            "slope_V": slope_v,
            "slope_trZ": slope_tz,
            "claim7_all": claim7_str,
        })

    # Build markdown.
    lines = [
        "# Experiment 2 summary\n",
        "## Runs\n",
        "| run | description | slope V | slope tr(Σ) | Claim 7 |",
        "|---|---|---:|---:|:---:|",
    ]
    for r in rows:
        sv = f"{r['slope_V']:+.3f}" if r.get("slope_V") is not None else "—"
        stz = f"{r['slope_trZ']:+.3f}" if r.get("slope_trZ") is not None else "—"
        c7 = r.get("claim7_all", "—") or "—"
        lines.append(
            f"| `{r['run']}` | {r['description']} | {sv} | {stz} | {c7} |"
        )
    lines.append("")
    lines.append("Expected slopes per paper: slope_V = −1, slope_trZ = −2.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
