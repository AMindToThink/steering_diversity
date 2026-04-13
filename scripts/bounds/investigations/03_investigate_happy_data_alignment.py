"""Investigation: why does Claim 7 fail for ``qwen_happy`` specifically?

## Question we were answering

Claim 7 of the paper states that steering reduces residual-stream
diversity when ``‖μ + s‖ > ‖μ‖``, equivalently when ``2 μ·s + ‖s‖² > 0``.
After fixing the batch-indexing bug, the re-run of experiment 1 showed
Claim 7 **failing at every scale** for ``qwen_happy`` (i.e., adding the
happy steering vector to Qwen's unsteered residual mean actually
*decreases* ‖μ‖), while it **passes at every scale** for ``qwen_style``,
``qwen_random``, ``llama_creativity``, and ``llama_random``.

The hypothesis Matthew proposed: maybe the FineWeb-Edu training data is
"sad" from Qwen's perspective, so the mean residual direction is tilted
away from the happy pole and adding happy pushes μ sideways rather than
elongating it.

This script tests that hypothesis two ways:

1. **Direct cosine**: compute ``cos(μ_unsteered, Σ happy_layers)`` — if
   negative, μ is literally anti-aligned with the happy direction.
2. **Token-level projection**: push μ_unsteered through Qwen's final
   RMSNorm and ``lm_head``, then look at the top tokens it projects to.
   Do the same for the happy direction. If the data is "sad" the top
   tokens for μ should include sad-valence words; if it's tone-neutral,
   they should be function words (punctuation, articles, etc.).

## What we found

- ``cos(μ_unsteered, Σ happy[10..25]) = −0.108`` (≈ 4σ from random, since
  two random unit vectors in 1536 dimensions have ``E[cos] ≈ 1/√1536 ≈
  0.026``). Small in absolute value but statistically meaningful and
  **systematic**: 12 of 16 per-layer cosines are negative, with the
  rest only slightly positive.
- **The data is NOT sad — it is tone-neutral and frequency-dominated.**
  μ's top 15 lm_head projections are all function words and punctuation
  (``,``, ``and``, ``in``, ``to``, ``of``, ``the``, ``a``, ``\\n``, etc.),
  with zero emotional valence.
- The happy direction's top upweighted tokens are cleanly positive-valence
  (``delighted``, ``exciting``, ``欢呼``, ``高兴``), and its bottom-
  downweighted tokens are sad (``grief``, ``tragedy``, ``depression``),
  confirming the vector is aimed correctly.

**Conclusion:** FineWeb-Edu residuals at Qwen's final layer don't carry
strong sentiment either way — they're dominated by token-frequency
structure. The happy direction is *mildly* anti-aligned with them
(because educational prose trends serious-to-neutral, occasionally
discussing difficult topics), just enough that adding the happy vector
at small scales decreases ``‖μ + s_eff‖`` instead of increasing it. At
very large scales (``‖s_eff‖² ≫ |2 μ·s_eff|``) Claim 7 eventually flips
to passing, which we observed in the extended-scale pilot at scale 128.

## Why it still matters

This is the canonical evidence that:

1. Matthew's "is the data sad?" hypothesis was directionally correct
   (cosine *is* negative) but the cosmetic interpretation is wrong —
   calling the data "sad" is misleading; the real story is that sentiment
   is near-orthogonal to what dominates the mean residual.
2. A happy-specific failure of Claim 7 does not mean the steering vector
   is broken; it's an interaction between the vector's axis and the data
   distribution.

The ``−0.108`` number is the one quoted in ``docs/bounds/README.md``'s
Claim 7 paragraph. Re-running this probe is how future readers (or future
me) should reproduce that number if questioned.

## Requirements

Depends on ``outputs/bounds/bounds_qwen_happy/stats.pt`` — which is
gitignored. Regenerate via
``uv run python scripts/bounds/02_record_stats.py --config configs/bounds/qwen_happy.yaml``
(requires GPU). Also loads Qwen2.5-1.5B-Instruct in float32 on CPU for the
``lm_head`` projection, which takes ~30s to load and ~5s to run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 4 levels up: investigations/ → bounds/ → scripts/ → PROJECT_ROOT
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.bounds.gguf_loader import load_steering_vector_gguf  # noqa: E402

HAPPY_PATH = _PROJECT_ROOT / "EasySteer/vectors/happy_diffmean.gguf"
STATS_PATH = _PROJECT_ROOT / "outputs/bounds/bounds_qwen_happy/stats.pt"
TARGET_LAYERS = list(range(10, 26))
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    if not STATS_PATH.exists():
        print(
            f"ERROR: {STATS_PATH} does not exist.\n"
            "This script depends on the post-batch-fix bounds_qwen_happy stats.pt,\n"
            "which is gitignored. Regenerate by running:\n"
            "  uv run python scripts/bounds/02_record_stats.py --config configs/bounds/qwen_happy.yaml\n"
            "(requires GPU).",
            file=sys.stderr,
        )
        sys.exit(2)

    print("Loading stats.pt ...")
    raw = torch.load(STATS_PATH, weights_only=False, map_location="cpu")
    mu_unsteered = raw["per_scale"][0.0]["mean"].float()  # [d]
    d = mu_unsteered.shape[0]
    print(f"d = {d}, ‖mu_unsteered‖ = {mu_unsteered.norm().item():.2f}")

    print("\nLoading happy vectors ...")
    happy = load_steering_vector_gguf(HAPPY_PATH)
    nominal_happy = torch.zeros(d, dtype=torch.float32)
    for layer in TARGET_LAYERS:
        nominal_happy += happy[layer]
    print(f"‖sum(happy[10..25])‖ = {nominal_happy.norm().item():.4f}")
    print(f"per-layer happy norms min/max: "
          f"{min(happy[i].norm().item() for i in TARGET_LAYERS):.4g} / "
          f"{max(happy[i].norm().item() for i in TARGET_LAYERS):.4g}")

    mu_hat = mu_unsteered / mu_unsteered.norm().clamp_min(1e-12)
    happy_hat = nominal_happy / nominal_happy.norm().clamp_min(1e-12)
    cos_mu_happy = float(torch.dot(mu_hat, happy_hat).item())
    print(f"\ncos(mu_unsteered, nominal_happy_sum) = {cos_mu_happy:+.4f}")
    if cos_mu_happy < -0.2:
        print("  → STRONGLY NEGATIVE: mean residual is anti-correlated with happy direction.")
        print("    Adding happy pushes the mean AWAY from its natural direction,")
        print("    decreasing ‖mu‖ at small scales (consistent with our Claim 7 failure).")
    elif cos_mu_happy < 0:
        print("  → mildly negative")
    else:
        print("  → positive (Claim 7 failure must come from ‖s_eff‖ direction, not raw happy)")

    # Also check per-layer cosines to see if it's a systematic or
    # accidental-cancellation effect.
    print("\nper-layer cos(mu, happy[k]):")
    for layer in TARGET_LAYERS:
        h = happy[layer].float()
        c = float(torch.dot(
            mu_unsteered / mu_unsteered.norm(),
            h / h.norm().clamp_min(1e-12),
        ).item())
        print(f"  layer {layer:>2d}: cos = {c:+.4f}, ‖h‖ = {h.norm().item():.3f}")

    print("\nLoading Qwen2.5-1.5B-Instruct for lm_head projection ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    final_norm = model.model.norm
    lm_head = model.lm_head
    with torch.no_grad():
        mu_post_norm = final_norm(mu_unsteered.unsqueeze(0)).squeeze(0)
        logits_mu = lm_head(mu_post_norm.unsqueeze(0)).squeeze(0)  # [vocab]

        # For happy, the "as if it were the residual" projection.
        # This asks: if you push the residual in the +happy direction,
        # which tokens get upweighted by lm_head?
        happy_unit_residual = nominal_happy / nominal_happy.norm().clamp_min(1e-12)
        happy_post_norm = final_norm(
            (happy_unit_residual * mu_unsteered.norm()).unsqueeze(0)
        ).squeeze(0)
        logits_happy = lm_head(happy_post_norm.unsqueeze(0)).squeeze(0)

    def _show(logits: torch.Tensor, label: str, n: int = 15) -> None:
        top = logits.topk(n)
        bot = logits.topk(n, largest=False)
        top_tokens = [tok.decode([int(i.item())]).replace("\n", "\\n") for i in top.indices]
        bot_tokens = [tok.decode([int(i.item())]).replace("\n", "\\n") for i in bot.indices]
        print(f"\n{label}")
        print(f"  TOP  {n}: " + " | ".join(f"{t!r}({v:+.2f})" for t, v in zip(top_tokens, top.values.tolist())))
        print(f"  BOT  {n}: " + " | ".join(f"{t!r}({v:+.2f})" for t, v in zip(bot_tokens, bot.values.tolist())))

    _show(logits_mu, "Tokens corresponding to unsteered μ (avg residual direction):")
    _show(logits_happy, "Tokens corresponding to +happy direction (projected at ‖μ‖ magnitude):")

    diff_logits = logits_happy - logits_mu
    _show(diff_logits, "Tokens that happy UPWEIGHTS vs the natural direction (diff):")


if __name__ == "__main__":
    main()
