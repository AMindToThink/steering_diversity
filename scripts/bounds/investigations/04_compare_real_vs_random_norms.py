"""Investigation: is our random-vector control effectively shorter than
the real vector at matched scales, and if so, by how much?

## Question we were answering

Matthew raised a sharp design question: when we generate a "random
steering vector" as a control for ``happy_diffmean``, each per-layer
random vector has L2 norm equal to the reference (both = 1.0, since
``happy_diffmean`` is per-layer normalized). But the **aggregate**
``Σ_i random[i]`` over target layers is suspiciously small because
random directions don't stack coherently in high dimensions. Concretely,
for ``n`` independent unit vectors in ``R^d``, ``E[‖Σ r_i‖] ≈ √n``,
while a learned steering vector trained to point at a single semantic
pole has correlated per-layer directions that stack closer to ``n``.

So: how much smaller is the aggregate random steering compared to the
aggregate real steering? And does that gap show up in the empirical
``‖s_eff‖`` at the final RMSNorm?

## What we found

**Per-layer** — perfect match. Every happy layer has ``‖h_i‖ = 1.0``
(``normalize: true`` at training time), and every random layer has
``‖r_i‖ = 1.0`` by construction in ``generate_random_steering_vector``.
No shortening per-layer.

**Aggregate** — ~3× gap, confirming the theory:
- ``‖Σ happy[10..25]‖ = 11.27`` (16 correlated unit vectors)
- ``‖Σ random_qwen[10..25]‖ = 4.05`` ≈ ``√16 = 4.00``
- ``‖Σ create[16..29]‖ = 11.86`` (14 correlated unit vectors)
- ``‖Σ random_llama[16..29]‖ = 3.74`` ≈ ``√14 = 3.74``

**Effective ``‖s_eff‖`` at the final site** — measured from the 5
``stats.pt`` files, the ratio ``real/random`` is consistently ~3.2–3.8×
across scales. E.g., at scale 8:
- ``qwen_happy``: ``‖s_eff‖ = 124.22``
- ``qwen_random``: ``‖s_eff‖ = 38.38``
- ratio = 3.24

This is the full arithmetic chain: coherent per-layer directions stack
to ~``n``, incoherent directions stack to ~``√n``, and that ~3× gap
propagates through the model's intermediate attention+MLP blocks to the
final-site mean shift in a proportional way.

## Why it still matters

This investigation is the direct motivation for
``src/bounds/random_vectors.py::generate_random_steering_vector_aggregate_matched``
(and the 4 ``test_aggregate_matched_*`` tests in
``tests/bounds/test_random_vectors.py``). Matthew's follow-up: "matched
per-layer is a valid control but answers a different question than
matched aggregate — can we add a control that isolates 'direction
coherence' from 'direction semantics' by matching the aggregate?" That
control became Experiment 2's ``qwen_random_agg_matched`` and
``llama_random_agg_matched`` runs.

Re-running this probe after any change to either the GGUF vector
extraction OR the random vector generator is the fastest way to confirm
the aggregate-matched control is still behaving as designed.

## Requirements

Depends on the 5 post-fix ``outputs/bounds/bounds_*/stats.pt`` files,
all gitignored. Regenerate via the experiment 1 pipeline (``01_verify``
→ ``02_record``). CPU-only otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# 4 levels up: investigations/ → bounds/ → scripts/ → PROJECT_ROOT
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.bounds.gguf_loader import load_steering_vector_gguf  # noqa: E402
from src.bounds.random_vectors import generate_random_steering_vector  # noqa: E402


def _nominal(vecs: dict[int, torch.Tensor], target_layers: list[int]) -> float:
    total = torch.zeros_like(next(iter(vecs.values())), dtype=torch.float32)
    for i in target_layers:
        total += vecs[i].to(torch.float32)
    return float(total.norm().item())


def main() -> None:
    print("=== Nominal direction norms (sum over target_layers, unit per-layer) ===\n")

    happy = load_steering_vector_gguf(_PROJECT_ROOT / "EasySteer/vectors/happy_diffmean.gguf")
    create = load_steering_vector_gguf(
        _PROJECT_ROOT / "EasySteer/replications/creative_writing/create.gguf"
    )

    happy_layers = list(range(10, 26))
    create_layers = list(range(16, 30))

    happy_norms = [happy[i].norm().item() for i in happy_layers]
    create_norms = [create[i].norm().item() for i in create_layers]
    print(f"happy_diffmean per-layer norms: min={min(happy_norms):.4f} max={max(happy_norms):.4f}")
    print(f"create per-layer norms:         min={min(create_norms):.4f} max={max(create_norms):.4f}")
    print()

    qwen_rand = generate_random_steering_vector({i: happy[i] for i in happy_layers}, seed=0)
    llama_rand = generate_random_steering_vector({i: create[i] for i in create_layers}, seed=0)
    qwen_rand_norms = [qwen_rand[i].norm().item() for i in happy_layers]
    llama_rand_norms = [llama_rand[i].norm().item() for i in create_layers]
    print(f"qwen_random per-layer norms:    min={min(qwen_rand_norms):.4f} max={max(qwen_rand_norms):.4f}")
    print(f"llama_random per-layer norms:   min={min(llama_rand_norms):.4f} max={max(llama_rand_norms):.4f}")
    print()

    print(f"||sum(happy[10..25])||       = {_nominal(happy, happy_layers):.4f}   (sqrt(16)={4.0:.4f} if random)")
    print(f"||sum(qwen_random[10..25])|| = {_nominal(qwen_rand, happy_layers):.4f}")
    print()
    print(f"||sum(create[16..29])||      = {_nominal(create, create_layers):.4f}   (sqrt(14)={14**0.5:.4f} if random)")
    print(f"||sum(llama_random[16..29])||= {_nominal(llama_rand, create_layers):.4f}")
    print()

    # Now compare ACTUAL effective ||s_eff|| at the final site from stats.pt.
    print("=== Effective ||s_eff|| at the final RMSNorm (from stats.pt) ===\n")
    print(f"{'run':<25} {'scale':>6} {'||s_eff||':>12}")
    print("-" * 50)
    for run in ["bounds_qwen_happy", "bounds_qwen_random", "bounds_qwen_style",
                "bounds_llama_creativity", "bounds_llama_random"]:
        path = _PROJECT_ROOT / "outputs" / "bounds" / run / "stats.pt"
        if not path.exists():
            print(f"{run}: NO stats.pt (skipping — regenerate via 02_record_stats.py)")
            continue
        raw = torch.load(path, weights_only=False, map_location="cpu")
        per_scale = raw["per_scale"]
        zero_mean = per_scale[0.0]["mean"]
        for scale in [0.5, 1.0, 2.0, 4.0, 8.0]:
            if scale not in per_scale:
                continue
            mu_steered = per_scale[scale]["mean"]
            s_eff = (mu_steered - zero_mean).float()
            s_norm = float(s_eff.norm().item())
            print(f"{run:<25} {scale:>6.1f} {s_norm:>12.3f}")
        print()


if __name__ == "__main__":
    main()
