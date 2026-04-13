"""Phase 2 of experiment 2 — pick 3 layers per model from the diagnostic.

Policy: edge-middle-edge of the trained target range for the steering
vector. The layer set stays constant regardless of what the diagnostic
shows (those layers are picked by the vector's original trainers as
where the steering is most effective). The diagnostic is reported for
the record so we know what we're committing to, but it doesn't override
the pick unless something flags as clearly broken (e.g. a layer with
``tr(Σ_x) == 0``, which would mean no data at that position).

Writes ``outputs/bounds/experiment2_layer_picks.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def pick_edge_middle_edge(target_layers: list[int]) -> list[int]:
    """Return the first, middle, and last layers from the target range."""
    if len(target_layers) < 3:
        raise ValueError(
            f"need at least 3 target layers, got {len(target_layers)}"
        )
    low = min(target_layers)
    high = max(target_layers)
    mid = (low + high) // 2
    # Snap mid to the nearest actual layer in target_layers.
    if mid not in target_layers:
        mid = min(target_layers, key=lambda x: abs(x - mid))
    return sorted([low, mid, high])


def print_report(diagnostic: dict, chosen_layers: list[int]) -> None:
    print(f"\n=== {diagnostic['model']} ===")
    print(f"{'site':>5} {'layer':>6} {'E[|x|]':>10} {'|mu|':>10} "
          f"{'tr(Sigma)':>12} {'cos(h,mu_hat)':>14}")
    print("-" * 66)
    for row in diagnostic["per_site"]:
        layer = row["layer_idx"]
        label = f"L{layer}" if layer >= 0 else "emb"
        mark = "  ← PICKED" if layer in chosen_layers else ""
        cos_str = f"{row['cos_ref_mu']:+.4f}" if row['cos_ref_mu'] is not None else "   n/a"
        print(
            f"{row['site_i']:>5} {label:>6} "
            f"{row['e_norm']:>10.2f} {row['mu_norm']:>10.2f} "
            f"{row['tr_sigma']:>12.2f} {cos_str:>14}{mark}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-diagnostic", type=Path, required=True)
    parser.add_argument("--qwen-target-layers", type=int, nargs="+", required=True)
    parser.add_argument("--llama-diagnostic", type=Path, required=True)
    parser.add_argument("--llama-target-layers", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    qwen_diag = json.loads(args.qwen_diagnostic.read_text())
    llama_diag = json.loads(args.llama_diagnostic.read_text())

    qwen_picks = pick_edge_middle_edge(args.qwen_target_layers)
    llama_picks = pick_edge_middle_edge(args.llama_target_layers)

    print_report(qwen_diag, qwen_picks)
    print(f"Qwen happy — edge/middle/edge of {args.qwen_target_layers}: {qwen_picks}")

    print_report(llama_diag, llama_picks)
    print(f"Llama creativity — edge/middle/edge of {args.llama_target_layers}: {llama_picks}")

    # Sanity check: warn loudly if any picked layer has pathological stats.
    def _warn_if_pathological(diag: dict, picks: list[int], label: str) -> None:
        for row in diag["per_site"]:
            if row["layer_idx"] in picks:
                if row["tr_sigma"] == 0:
                    print(
                        f"⚠ WARNING: {label} layer {row['layer_idx']} has tr(Sigma)=0",
                        file=sys.stderr,
                    )
                if row["e_norm"] == 0:
                    print(
                        f"⚠ WARNING: {label} layer {row['layer_idx']} has E[|x|]=0",
                        file=sys.stderr,
                    )

    _warn_if_pathological(qwen_diag, qwen_picks, "qwen")
    _warn_if_pathological(llama_diag, llama_picks, "llama")

    picks = {
        "qwen": {
            "model": qwen_diag["model"],
            "target_layers": args.qwen_target_layers,
            "single_layer_picks": qwen_picks,
            "policy": "edge_middle_edge",
        },
        "llama": {
            "model": llama_diag["model"],
            "target_layers": args.llama_target_layers,
            "single_layer_picks": llama_picks,
            "policy": "edge_middle_edge",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(picks, indent=2))
    print(f"\nLayer picks written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
