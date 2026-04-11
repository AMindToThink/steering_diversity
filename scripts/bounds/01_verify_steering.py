"""Steering-effect verification — hard prerequisite for bounds recording.

This script is the answer to Matthew's "trust no silent steering" policy.
Before running any bounds recording job, we verify end-to-end that the
nnsight + gguf steering path actually changes model behavior — by
**sampling from the steered model through the same code path the recording
script will use**, comparing side-by-side with unsteered samples, and
running cheap automated sanity checks. If the effect is ambiguous we
auto-escalate the scale up a geometric sweep until it's unmistakable.

On success, writes ``outputs/bounds/<run_name>/verification/PASSED``.
``scripts/bounds/02_record_stats.py`` refuses to run without that file.

Usage
-----
``uv run python scripts/bounds/01_verify_steering.py --config CONFIG
[--scale FLOAT] [--auto-escalate] [--max-new-tokens N]``

Exit codes:
  0 — verification passed, PASSED sentinel written
  1 — verification failed, FAILED sentinel written
  2 — usage / config error

Automated checks
----------------
1. **KL divergence** between steered and unsteered next-token distributions,
   averaged over the last real token of each verification prompt. Must
   exceed ``steering_verification.kl_threshold`` (default 0.05 nats). A
   vector that produces zero KL is silently inactive.

2. **Residual-delta magnitude ratio**: for the difference between steered
   and unsteered pre-RMSNorm residuals at the final site, computes
   ``mean(‖delta‖) / mean(‖pre_unsteered‖)`` over last real tokens. Must
   exceed ``steering_verification.magnitude_ratio_threshold`` (default
   0.02). A vector whose intervention makes zero change to the final-site
   residual is silently inactive — this check catches that.

   NB: the check is deliberately NOT a cosine against the raw sum of
   ``scale * s_layer``. Steering fires at ``target_layers`` and then has
   to propagate through attention + MLP sublayers BETWEEN those layers
   and the capture site, which reshapes the residual significantly — the
   final-site delta can be nearly orthogonal to the raw layer-sum direction
   even when the intervention is landing perfectly. The cosine is printed
   as an informational line but not gated on.

Both automated checks sit on top of visual inspection of the printed
samples — that's deliberate. Numbers can pass while the generated text is
gibberish, or fail on an effect that's obvious to a reader. Both checks
are required.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch
from nnsight import LanguageModel

# Make ``src.bounds`` importable when this script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.bounds.config import BoundsExperimentConfig  # noqa: E402
from src.bounds.gguf_loader import load_steering_vector_gguf  # noqa: E402
from src.bounds.nnsight_runner import (  # noqa: E402
    run_verification_forward_pass,
    sample_from_steered_model,
)
from src.bounds.random_vectors import generate_random_steering_vector  # noqa: E402


SYNTHETIC_SENTINEL = "__synthetic__"


# ---------------------------------------------------------------------------
# Steering loading (real, random, or synthetic-for-smoke-tests)
# ---------------------------------------------------------------------------


def load_or_build_steering(
    cfg: BoundsExperimentConfig, lm: LanguageModel
) -> dict[int, torch.Tensor]:
    """Resolve the steering vector for this config.

    - ``vector_path == "__synthetic__"``: build a synthetic vector from the
      model architecture (used by ``configs/bounds/smoke.yaml`` with
      tiny-gpt2 since we don't have a real .gguf for it).
    - ``vector_path == "*.gguf"``: load from GGUF directly.
    - ``random_reference_path``: load the reference .gguf, then draw a
      norm-matched random vector seeded by ``random_seed``.
    """
    if cfg.steering.vector_path == SYNTHETIC_SENTINEL:
        d = lm.config.hidden_size if hasattr(lm.config, "hidden_size") else lm.config.n_embd
        n_layers = lm.config.num_hidden_layers if hasattr(lm.config, "num_hidden_layers") else lm.config.n_layer
        # Distinct, non-degenerate vectors per layer.
        return {
            i: (torch.arange(d, dtype=torch.float32) + float(i) * 10.0)
            * (1.0 / max(d, 1))
            for i in range(n_layers)
        }

    if cfg.steering.vector_path:
        return load_steering_vector_gguf(cfg.steering.vector_path)

    if cfg.steering.random_reference_path:
        ref = load_steering_vector_gguf(cfg.steering.random_reference_path)
        return generate_random_steering_vector(ref, seed=cfg.steering.random_seed or 0)

    raise ValueError("BoundsSteeringConfig has no vector source")


# ---------------------------------------------------------------------------
# Automated sanity checks
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CheckResult:
    name: str
    value: float
    threshold: float
    passed: bool
    detail: str = ""


def _last_real_token_index(attention_mask_row: torch.Tensor) -> int:
    """Return the index of the last 1 in the attention mask row.

    Handles both left-padding (GPT-2 style) and right-padding (most Llama /
    Qwen HF models) since we just scan from the end.
    """
    nonzero = (attention_mask_row > 0).nonzero(as_tuple=True)[0]
    if nonzero.numel() == 0:
        raise ValueError("attention mask row has no real tokens")
    return int(nonzero[-1].item())


def kl_last_token(
    logits_a: torch.Tensor, logits_b: torch.Tensor, attention_mask: torch.Tensor
) -> float:
    """Mean KL(P_a || P_b) over the last real token of each row.

    Returns KL in nats.
    """
    B = logits_a.shape[0]
    kls: list[float] = []
    for i in range(B):
        t = _last_real_token_index(attention_mask[i])
        log_p = torch.log_softmax(logits_a[i, t].float(), dim=-1)
        log_q = torch.log_softmax(logits_b[i, t].float(), dim=-1)
        p = log_p.exp()
        kls.append(float((p * (log_p - log_q)).sum().item()))
    return sum(kls) / len(kls)


def residual_delta_magnitude_ratio(
    pre_steered: torch.Tensor,
    pre_unsteered: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    """Average ``‖delta‖ / ‖pre_unsteered‖`` over last real tokens.

    This is the 'did anything happen?' signal. A no-op intervention gives
    ratio ≈ 0. A visible intervention gives ratio ≥ 0.01 or so (scales
    with ``scale * ‖steering‖``).
    """
    delta = (pre_steered - pre_unsteered).float()
    unsteered = pre_unsteered.float()
    ratios: list[float] = []
    B = delta.shape[0]
    for i in range(B):
        t = _last_real_token_index(attention_mask[i])
        d_norm = float(delta[i, t].norm().item())
        u_norm = float(unsteered[i, t].norm().clamp_min(1e-12).item())
        ratios.append(d_norm / u_norm)
    return sum(ratios) / len(ratios)


def residual_delta_cosine_informational(
    pre_steered: torch.Tensor,
    pre_unsteered: torch.Tensor,
    steering_direction: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    """Average cosine of ``(pre_steered - pre_unsteered)`` with ``steering_direction``.

    NOT gated on — printed as an informational line. The final-site residual
    delta can be nearly orthogonal to the raw ``sum(scale*s_layer)`` direction
    because of propagation through intermediate sublayers, even when the
    intervention is landing perfectly; see module docstring for why.
    """
    delta = (pre_steered - pre_unsteered).float()
    direction = steering_direction.float()
    direction_hat = direction / direction.norm().clamp_min(1e-12)
    cosines: list[float] = []
    B = delta.shape[0]
    for i in range(B):
        t = _last_real_token_index(attention_mask[i])
        d_norm = delta[i, t].norm().clamp_min(1e-12)
        cosines.append(
            float(torch.dot(delta[i, t] / d_norm, direction_hat).item())
        )
    return sum(cosines) / len(cosines)


def _nominal_accumulated_direction(
    steering: dict[int, torch.Tensor], target_layers: list[int], scale: float
) -> torch.Tensor:
    """Sum of scale * steering[layer] over target layers, normalized.

    This is the 'nominal' direction — what the user nominally asked for.
    The real accumulated residual shift at the final site also passes
    through intermediate attention/MLP sublayers and may not point in
    exactly this direction, but it should be positively correlated.
    """
    total = None
    for i in target_layers:
        s = scale * steering[i]
        total = s.clone() if total is None else total + s
    if total is None:
        raise ValueError("No target layers for nominal direction")
    return total


# ---------------------------------------------------------------------------
# Per-scale verification run
# ---------------------------------------------------------------------------


def run_at_scale(
    lm: LanguageModel,
    cfg: BoundsExperimentConfig,
    steering: dict[int, torch.Tensor],
    scale: float,
) -> dict:
    """Run all verification steps at one scale and return a structured result."""
    prompts = cfg.steering_verification.sample_prompts
    target_layers = cfg.steering.target_layers
    max_new_tokens = cfg.steering_verification.max_new_tokens
    max_seq_len = cfg.dataset.max_seq_len

    # 1. Side-by-side decoded samples. Sampling (not greedy) reveals the
    # distributional shift better — greedy decode hides steering effects
    # when the argmax next token is so confident even large logit shifts
    # don't flip it.
    torch.manual_seed(cfg.seed)  # determinism for side-by-side comparison
    samples_unsteered = sample_from_steered_model(
        lm, prompts, steering=None, scale=0.0, target_layers=[],
        max_new_tokens=max_new_tokens,
        do_sample=cfg.steering_verification.do_sample,
    )
    torch.manual_seed(cfg.seed)  # re-seed so the decode noise is identical
    samples_steered = sample_from_steered_model(
        lm, prompts, steering=steering, scale=scale, target_layers=target_layers,
        max_new_tokens=max_new_tokens,
        do_sample=cfg.steering_verification.do_sample,
    )

    # 2. Forward-pass captures for the numeric checks.
    fp_unsteered = run_verification_forward_pass(
        lm, prompts, steering=None, scale=0.0, target_layers=[], max_seq_len=max_seq_len
    )
    fp_steered = run_verification_forward_pass(
        lm, prompts, steering=steering, scale=scale,
        target_layers=target_layers, max_seq_len=max_seq_len,
    )

    kl = kl_last_token(
        fp_unsteered["logits"], fp_steered["logits"], fp_unsteered["attention_mask"]
    )
    mag_ratio = residual_delta_magnitude_ratio(
        fp_steered["pre"], fp_unsteered["pre"], fp_unsteered["attention_mask"]
    )
    cos_info = residual_delta_cosine_informational(
        fp_steered["pre"],
        fp_unsteered["pre"],
        _nominal_accumulated_direction(steering, target_layers, scale),
        fp_unsteered["attention_mask"],
    )

    kl_check = CheckResult(
        name="kl_next_token",
        value=kl,
        threshold=cfg.steering_verification.kl_threshold,
        passed=kl >= cfg.steering_verification.kl_threshold,
        detail=f"mean KL over last real token of {len(prompts)} prompts",
    )
    mag_check = CheckResult(
        name="residual_delta_magnitude_ratio",
        value=mag_ratio,
        threshold=cfg.steering_verification.magnitude_ratio_threshold,
        passed=mag_ratio >= cfg.steering_verification.magnitude_ratio_threshold,
        detail="mean(‖delta‖ / ‖pre_unsteered‖) over last real tokens",
    )

    return {
        "scale": scale,
        "prompts": prompts,
        "samples_unsteered": samples_unsteered,
        "samples_steered": samples_steered,
        "checks": [dataclasses.asdict(kl_check), dataclasses.asdict(mag_check)],
        "info": {"residual_delta_cosine_vs_layer_sum": cos_info},
        "passed": kl_check.passed and mag_check.passed,
    }


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


def _print_result(result: dict) -> None:
    scale = result["scale"]
    header = f"===  VERIFICATION @ scale={scale}  ".ljust(78, "=")
    print(header)
    for check in result["checks"]:
        icon = "✅" if check["passed"] else "❌"
        print(
            f"  {icon} {check['name']:35s} value={check['value']:.4g} "
            f"threshold={check['threshold']:.4g}"
        )
        if check["detail"]:
            print(f"     {check['detail']}")
    for k, v in result.get("info", {}).items():
        print(f"  ℹ  {k:35s} value={v:.4g}  (informational, not gated)")
    print()
    print(f"  Side-by-side samples (first {len(result['prompts'])}):")
    for i, prompt in enumerate(result["prompts"]):
        print(f"    [prompt {i}] {prompt!r}")
        print(f"      unsteered: {result['samples_unsteered'][i]!r}")
        print(f"      steered:   {result['samples_steered'][i]!r}")
    print()
    icon = "✅  PASSED" if result["passed"] else "❌  FAILED"
    print(f"  → {icon} at scale={scale}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--scale", type=float, default=None,
        help="Override verification_scale from the config.",
    )
    parser.add_argument(
        "--auto-escalate", action="store_true",
        help="If the initial scale fails, retry at each scale in "
        "steering_verification.auto_escalate_scales until one passes or all fail.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="Override steering_verification.max_new_tokens.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 2

    cfg = BoundsExperimentConfig.from_yaml(args.config)
    if args.max_new_tokens is not None:
        cfg.steering_verification.max_new_tokens = args.max_new_tokens

    verification_dir = cfg.output_dir / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    # Clean any old sentinels so a pass/fail is always fresh.
    for sentinel in ("PASSED", "FAILED"):
        (verification_dir / sentinel).unlink(missing_ok=True)

    print(f"Loading model {cfg.model.name!r} ...")
    lm = LanguageModel(cfg.model.name, device_map="auto")
    print(f"Loading steering from {cfg.steering.vector_path or cfg.steering.random_reference_path!r} ...")
    steering = load_or_build_steering(cfg, lm)

    # Build the scale sweep.
    initial_scale = args.scale if args.scale is not None else cfg.steering_verification.verification_scale
    scales_to_try: list[float] = [initial_scale]
    if args.auto_escalate:
        for s in cfg.steering_verification.auto_escalate_scales:
            if s not in scales_to_try:
                scales_to_try.append(s)

    all_results: list[dict] = []
    final_passed = False
    for scale in scales_to_try:
        result = run_at_scale(lm, cfg, steering, scale)
        _print_result(result)
        all_results.append(result)
        out_path = verification_dir / f"scale_{scale}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  → wrote {out_path}")
        print()
        if result["passed"]:
            final_passed = True
            break

    sentinel_name = "PASSED" if final_passed else "FAILED"
    sentinel_path = verification_dir / sentinel_name
    sentinel_path.write_text(
        json.dumps(
            {
                "run_name": cfg.run_name,
                "model": cfg.model.name,
                "vector_path": cfg.steering.vector_path,
                "random_reference_path": cfg.steering.random_reference_path,
                "target_layers": cfg.steering.target_layers,
                "passed_at_scale": next(
                    (r["scale"] for r in all_results if r["passed"]), None
                ),
                "tried_scales": [r["scale"] for r in all_results],
            },
            indent=2,
        )
    )

    icon = "✅" if final_passed else "❌"
    print(f"{icon} Verification {sentinel_name} — sentinel at {sentinel_path}")
    return 0 if final_passed else 1


if __name__ == "__main__":
    sys.exit(main())
