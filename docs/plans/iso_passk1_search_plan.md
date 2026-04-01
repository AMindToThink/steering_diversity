# Plan: Finding the Iso-pass@1 Scale (Deferred)

## Goal

Find α* such that pass@1(steered at α*) ≈ pass@1(unsteered). This is the "maximum steering that doesn't reduce capabilities" that practitioners target. We want to show that even at this matched pass@1, diversity (pass@k for k >> 1) still collapses.

## Approach: Coarse Grid Then Interpolation (Preferred Over Binary Search)

Binary search at n=10 is unreliable because SE ≈ 0.016, so pass@1 values within ~3pp are indistinguishable — the search direction is essentially a coin flip for small effects.

**Better approach:**
1. Run a coarse grid at n=10: α ∈ {0.5, 1.0, 1.5, 2.0, 3.0}
2. Plot pass@1 vs α — should show a monotonic decline
3. Fit a line or isotonic regression to find α* where pass@1(α*) = pass@1(0)
4. Validate at n=20 before committing to the full n=100/200 run

## What We Know from Existing Data

| Scale | pass@1 (base) | Δ from unsteered | Significant? |
|-------|--------------|-----------------|-------------|
| α=0 | 0.432 | — | — |
| α=2 | 0.407 | −0.025 | p=0.118, no |

If effect is roughly linear: Δpass@1 ≈ −0.0125 × α

Predicted iso-pass@1 scale: depends on precision target
- Within ±0.01 at n=100 (SE ≈ 0.007): α ≲ 0.8
- Within ±0.02 at n=100: α ≲ 1.5

## Noise Considerations

- At n=10: SE ≈ 0.016 → can distinguish Δ > 0.032 (α > ~2.5)
- At n=100: SE ≈ 0.007 → can distinguish Δ > 0.014 (α > ~1.1)
- At n=200: SE ≈ 0.005 → can distinguish Δ > 0.010 (α > ~0.8)

## Implementation

### Script: `scripts/eval/find_iso_passk1_scale.py`

Grid-based approach:
1. Accept a list of scales to evaluate
2. For each scale: set via `POST /v1/steering`, run EvalPlus codegen+evaluate, compute pass@1
3. Save all outputs (samples + eval_results per scale)
4. Fit pass@1 vs α, report interpolated α*
5. Output: grid_results.json with all scales/pass@1 values, plus a recommended α*

### Using Both GPUs

Run pairs of scales simultaneously (one per GPU). With 5 grid points + baseline, that's 3 parallel pairs ≈ 3 × 11 min = 33 min total.

## Status: Deferred

For the immediate experiment, we're using α=0.5 (small enough that pass@1 should be very close to unsteered). The grid search can be run later if a more precise α* is needed.
