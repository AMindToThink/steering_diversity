"""Visualize pass@k curves across steering scales with bootstrap CIs.

Reads the aggregated pass_at_k_curves.json (which includes per_problem data)
and produces:
  1. pass@k curves with bootstrap confidence bands
  2. Δ pass@k bar chart colored by paired t-test significance

Usage:
    uv run python scripts/eval/plot_pass_at_k.py --config configs/eval_code.yaml
    uv run python scripts/eval/plot_pass_at_k.py --input outputs/passk_main_v1/code/pass_at_k_curves.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

from src.pass_at_k import pass_at_k
from src.utils import ensure_dir

logger = logging.getLogger(__name__)


def load_curves(path: Path) -> list[dict]:
    """Load aggregated pass@k curves JSON."""
    with open(path) as f:
        return json.load(f)


def _per_problem_passk(per_problem: list[dict], k: int) -> np.ndarray:
    """Compute per-problem pass@k scores, returning one value per problem."""
    return np.array([pass_at_k(p["n"], p["c"], k) for p in per_problem])


def _bootstrap_ci(
    per_problem: list[dict],
    k: int,
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap the mean pass@k across problems.

    Returns (mean, ci_low, ci_high).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    scores = _per_problem_passk(per_problem, k)
    n = len(scores)
    boot_means = np.array([
        rng.choice(scores, size=n, replace=True).mean() for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(scores.mean()), float(lo), float(hi)


def plot_pass_at_k_with_ci(
    curves: list[dict],
    output_dir: Path,
    use_plus: bool = True,
    title_prefix: str = "",
) -> list[Path]:
    """pass@k curves with bootstrap confidence bands.

    One plot per (dataset, temperature). Each scale gets a line with a
    shaded CI band.
    """
    sns.set_theme(style="whitegrid")
    saved: list[Path] = []
    suffix = "plus" if use_plus else "base"
    per_problem_key = f"per_problem_{suffix}"

    groups: dict[tuple[str, float], list[dict]] = {}
    for entry in curves:
        key = (entry["dataset"], entry["temperature"])
        groups.setdefault(key, []).append(entry)

    for (dataset, temp), entries in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        entries.sort(key=lambda e: e["scale"])

        # Use distinct colors for unsteered vs steered
        scale_colors: dict[float, tuple[str, str]] = {}
        if len(entries) == 2 and entries[0]["scale"] == 0.0:
            scale_colors[entries[0]["scale"]] = ("tab:blue", "Unsteered (α=0)")
            scale_colors[entries[1]["scale"]] = ("tab:red", f"Happy steered (α={entries[1]['scale']})")
        else:
            cmap = plt.get_cmap("viridis")
            for i, e in enumerate(entries):
                c = cmap(i / max(len(entries) - 1, 1))
                scale_colors[e["scale"]] = (c, f"scale={e['scale']}")

        for entry in entries:
            per_problem = entry.get(per_problem_key)
            if per_problem is None:
                logger.warning("No %s in entry scale=%.1f, skipping CI", per_problem_key, entry["scale"])
                continue

            k_values = sorted(int(k) for k in entry[f"pass_at_k_{suffix}"].keys())
            means, lows, highs = [], [], []
            for k in k_values:
                m, lo, hi = _bootstrap_ci(per_problem, k)
                means.append(m)
                lows.append(lo)
                highs.append(hi)

            color, label = scale_colors[entry["scale"]]
            marker = "o" if entry["scale"] == 0.0 else "s"
            ax.plot(k_values, means, marker=marker, markersize=5,
                    label=label, color=color, linewidth=2)
            ax.fill_between(k_values, lows, highs, alpha=0.2, color=color)

        ax.set_xscale("log")
        ax.set_xlabel("k (number of attempts)", fontsize=12)
        ax.set_ylabel("pass@k", fontsize=12)
        dataset_label = "HumanEval+" if (dataset == "humaneval" and use_plus) else dataset
        ax.set_title(f"{title_prefix}{dataset_label} pass@k", fontsize=13)
        ax.legend(loc="lower right", fontsize=10)

        plot_name = f"pass_at_k_{dataset}_{suffix}_temp{temp}.png"
        plot_path = output_dir / plot_name
        ensure_dir(output_dir)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(plot_path)
        logger.info("Saved %s", plot_path)

    return saved


def plot_delta_significance(
    curves: list[dict],
    output_dir: Path,
    use_plus: bool = True,
) -> list[Path]:
    """Bar chart of Δ pass@k (steered − baseline) colored by significance.

    Uses paired t-test across problems. Bars are red if p < 0.05, gray otherwise.
    Error bars show 95% CI of the difference.
    """
    sns.set_theme(style="whitegrid")
    saved: list[Path] = []
    suffix = "plus" if use_plus else "base"
    per_problem_key = f"per_problem_{suffix}"

    groups: dict[tuple[str, float], list[dict]] = {}
    for entry in curves:
        key = (entry["dataset"], entry["temperature"])
        groups.setdefault(key, []).append(entry)

    for (dataset, temp), entries in sorted(groups.items()):
        baseline = next((e for e in entries if e["scale"] == 0.0), None)
        if baseline is None:
            continue
        steered_entries = [e for e in entries if e["scale"] != 0.0]
        if not steered_entries:
            continue

        for steered in steered_entries:
            per_problem_base = baseline.get(per_problem_key)
            per_problem_steer = steered.get(per_problem_key)
            if per_problem_base is None or per_problem_steer is None:
                continue

            k_values = sorted(int(k) for k in baseline[f"pass_at_k_{suffix}"].keys())
            deltas, ci_lows, ci_highs, p_values = [], [], [], []

            for k in k_values:
                scores_0 = _per_problem_passk(per_problem_base, k)
                scores_s = _per_problem_passk(per_problem_steer, k)
                diff = scores_s - scores_0
                mean_diff = diff.mean()
                se_diff = diff.std(ddof=1) / np.sqrt(len(diff))
                _, p_val = stats.ttest_rel(scores_s, scores_0)

                deltas.append(mean_diff)
                ci_lows.append(mean_diff - 1.96 * se_diff)
                ci_highs.append(mean_diff + 1.96 * se_diff)
                p_values.append(p_val)

            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(k_values))
            colors = ["tab:red" if p < 0.05 else "0.6" for p in p_values]
            yerr_lo = [d - lo for d, lo in zip(deltas, ci_lows)]
            yerr_hi = [hi - d for d, hi in zip(deltas, ci_highs)]

            ax.bar(x, deltas, color=colors, width=0.7)
            ax.errorbar(x, deltas, yerr=[yerr_lo, yerr_hi],
                        fmt="none", ecolor="black", capsize=4, linewidth=1.5)
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels([str(k) for k in k_values])
            ax.set_xlabel("k (number of attempts)", fontsize=12)
            ax.set_ylabel("Δ pass@k (steered − unsteered)", fontsize=12)

            dataset_label = "HumanEval+" if (dataset == "humaneval" and use_plus) else dataset
            ax.set_title(
                f"Effect of happy steering (α={steered['scale']}) on pass@k",
                fontsize=13,
            )

            # Legend for significance colors
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="tab:red", label="p < 0.05"),
                Patch(facecolor="0.6", label="p ≥ 0.05"),
            ]
            ax.legend(handles=legend_elements, loc="lower left", fontsize=10)

            plot_name = f"delta_{dataset}_{suffix}_scale{steered['scale']}_temp{temp}.png"
            plot_path = output_dir / plot_name
            ensure_dir(output_dir)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(plot_path)
            logger.info("Saved %s", plot_path)

    return saved


def plot_combined(
    curves: list[dict],
    output_dir: Path,
    use_plus: bool = True,
) -> list[Path]:
    """Side-by-side: pass@k curves (left) + Δ significance bars (right).

    Matches the style of the preliminary figure.
    """
    sns.set_theme(style="whitegrid")
    saved: list[Path] = []
    suffix = "plus" if use_plus else "base"
    per_problem_key = f"per_problem_{suffix}"

    groups: dict[tuple[str, float], list[dict]] = {}
    for entry in curves:
        key = (entry["dataset"], entry["temperature"])
        groups.setdefault(key, []).append(entry)

    for (dataset, temp), entries in sorted(groups.items()):
        baseline = next((e for e in entries if e["scale"] == 0.0), None)
        steered_entries = sorted(
            [e for e in entries if e["scale"] != 0.0], key=lambda e: e["scale"]
        )
        if baseline is None or not steered_entries:
            continue

        per_problem_baseline = baseline.get(per_problem_key)
        if per_problem_baseline is None:
            continue

        # Use first steered entry for the combined plot
        steered = steered_entries[0]
        per_problem_steer = steered.get(per_problem_key)
        if per_problem_steer is None:
            continue

        k_values = sorted(int(k) for k in baseline[f"pass_at_k_{suffix}"].keys())

        fig, (ax_curve, ax_delta) = plt.subplots(1, 2, figsize=(16, 5.5))

        # --- Left panel: curves with CI ---
        for entry, color, marker, label in [
            (baseline, "tab:blue", "o", "Unsteered (α=0)"),
            (steered, "tab:red", "s", f"Happy steered (α={steered['scale']})"),
        ]:
            pp = entry.get(per_problem_key)
            if pp is None:
                continue
            means, lows, highs = [], [], []
            for k in k_values:
                m, lo, hi = _bootstrap_ci(pp, k)
                means.append(m)
                lows.append(lo)
                highs.append(hi)
            ax_curve.plot(k_values, means, marker=marker, markersize=5,
                          label=label, color=color, linewidth=2)
            ax_curve.fill_between(k_values, lows, highs, alpha=0.2, color=color)

        ax_curve.set_xscale("log")
        ax_curve.set_xlabel("k (number of attempts)", fontsize=12)
        ax_curve.set_ylabel("pass@k", fontsize=12)
        dataset_label = "HumanEval+" if (dataset == "humaneval" and use_plus) else dataset
        ax_curve.set_title(f"{dataset_label} pass@k", fontsize=13)
        ax_curve.legend(loc="lower right", fontsize=10)

        # --- Right panel: Δ bars with significance ---
        deltas, ci_lows, ci_highs, p_values = [], [], [], []
        for k in k_values:
            scores_0 = _per_problem_passk(per_problem_baseline, k)
            scores_s = _per_problem_passk(per_problem_steer, k)
            diff = scores_s - scores_0
            mean_diff = diff.mean()
            se_diff = diff.std(ddof=1) / np.sqrt(len(diff))
            _, p_val = stats.ttest_rel(scores_s, scores_0)
            deltas.append(mean_diff)
            ci_lows.append(mean_diff - 1.96 * se_diff)
            ci_highs.append(mean_diff + 1.96 * se_diff)
            p_values.append(p_val)

        x = np.arange(len(k_values))
        colors = ["tab:red" if p < 0.05 else "0.6" for p in p_values]
        yerr_lo = [d - lo for d, lo in zip(deltas, ci_lows)]
        yerr_hi = [hi - d for d, hi in zip(deltas, ci_highs)]

        ax_delta.bar(x, deltas, color=colors, width=0.7)
        ax_delta.errorbar(x, deltas, yerr=[yerr_lo, yerr_hi],
                          fmt="none", ecolor="black", capsize=4, linewidth=1.5)
        ax_delta.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax_delta.set_xticks(x)
        ax_delta.set_xticklabels([str(k) for k in k_values])
        ax_delta.set_xlabel("k (number of attempts)", fontsize=12)
        ax_delta.set_ylabel("Δ pass@k (steered − unsteered)", fontsize=12)
        ax_delta.set_title(
            f"Effect of happy steering (α={steered['scale']}) on pass@k",
            fontsize=13,
        )
        from matplotlib.patches import Patch
        ax_delta.legend(
            handles=[
                Patch(facecolor="tab:red", label="p < 0.05"),
                Patch(facecolor="0.6", label="p ≥ 0.05"),
            ],
            loc="lower left", fontsize=10,
        )

        fig.tight_layout()
        plot_name = f"passk_{dataset}_{suffix}_combined_temp{temp}.png"
        plot_path = output_dir / plot_name
        ensure_dir(output_dir)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(plot_path)
        logger.info("Saved %s", plot_path)

    return saved


def plot_coverage_gain(
    curves: list[dict],
    output_dir: Path,
    use_plus: bool = True,
) -> list[Path]:
    """Side-by-side: coverage gain curves (left) + Δ coverage gain bars (right).

    Coverage gain = pass@k - pass@1, measuring the benefit of additional attempts.
    If steering collapses diversity, the steered model's coverage gain should be
    smaller than the unsteered model's, even after accounting for pass@1 differences.
    """
    sns.set_theme(style="whitegrid")
    saved: list[Path] = []
    suffix = "plus" if use_plus else "base"
    per_problem_key = f"per_problem_{suffix}"

    groups: dict[tuple[str, float], list[dict]] = {}
    for entry in curves:
        key = (entry["dataset"], entry["temperature"])
        groups.setdefault(key, []).append(entry)

    for (dataset, temp), entries in sorted(groups.items()):
        baseline = next((e for e in entries if e["scale"] == 0.0), None)
        steered_entries = sorted(
            [e for e in entries if e["scale"] != 0.0], key=lambda e: e["scale"]
        )
        if baseline is None or not steered_entries:
            continue

        per_problem_baseline = baseline.get(per_problem_key)
        if per_problem_baseline is None:
            continue

        steered = steered_entries[0]
        per_problem_steer = steered.get(per_problem_key)
        if per_problem_steer is None:
            continue

        k_values = sorted(int(k) for k in baseline[f"pass_at_k_{suffix}"].keys())
        if not k_values:
            continue

        fig, (ax_curve, ax_delta) = plt.subplots(1, 2, figsize=(16, 5.5))

        # --- Left panel: coverage gain curves with CI ---
        for entry, color, marker, label in [
            (baseline, "tab:blue", "o", "Unsteered (α=0)"),
            (steered, "tab:red", "s", f"Happy steered (α={steered['scale']})"),
        ]:
            pp = entry.get(per_problem_key)
            if pp is None:
                continue
            # pass@1 per problem (fixed baseline for coverage gain)
            passk1 = _per_problem_passk(pp, 1)
            means, lows, highs = [], [], []
            rng = np.random.default_rng(42)
            for k in k_values:
                passk_scores = _per_problem_passk(pp, k)
                cg = passk_scores - passk1  # coverage gain per problem
                n = len(cg)
                boot_means = np.array([
                    rng.choice(cg, size=n, replace=True).mean()
                    for _ in range(10_000)
                ])
                means.append(float(cg.mean()))
                lo, hi = np.quantile(boot_means, [0.025, 0.975])
                lows.append(float(lo))
                highs.append(float(hi))

            ax_curve.plot(k_values, means, marker=marker, markersize=5,
                          label=label, color=color, linewidth=2)
            ax_curve.fill_between(k_values, lows, highs, alpha=0.2, color=color)

        ax_curve.set_xscale("log")
        ax_curve.set_xlabel("k (number of attempts)", fontsize=12)
        ax_curve.set_ylabel("Coverage gain (pass@k − pass@1)", fontsize=12)
        dataset_label = "HumanEval+" if (dataset == "humaneval" and use_plus) else dataset
        ax_curve.set_title(f"{dataset_label} coverage gain", fontsize=13)
        ax_curve.legend(loc="upper left", fontsize=10)

        # --- Right panel: Δ coverage gain bars with significance ---
        passk1_base = _per_problem_passk(per_problem_baseline, 1)
        passk1_steer = _per_problem_passk(per_problem_steer, 1)

        deltas, ci_lows, ci_highs, p_values = [], [], [], []
        for k in k_values:
            cg_base = _per_problem_passk(per_problem_baseline, k) - passk1_base
            cg_steer = _per_problem_passk(per_problem_steer, k) - passk1_steer
            diff = cg_steer - cg_base
            mean_diff = float(diff.mean())
            se_diff = float(diff.std(ddof=1) / np.sqrt(len(diff)))
            # k=1: coverage gain is identically 0 for both, no test needed
            if k == 1:
                p_val = 1.0
            else:
                _, p_val = stats.ttest_rel(cg_steer, cg_base)

            deltas.append(mean_diff)
            ci_lows.append(mean_diff - 1.96 * se_diff)
            ci_highs.append(mean_diff + 1.96 * se_diff)
            p_values.append(p_val)

        x = np.arange(len(k_values))
        colors = ["tab:red" if p < 0.05 else "0.6" for p in p_values]
        yerr_lo = [d - lo for d, lo in zip(deltas, ci_lows)]
        yerr_hi = [hi - d for d, hi in zip(deltas, ci_highs)]

        ax_delta.bar(x, deltas, color=colors, width=0.7)
        ax_delta.errorbar(x, deltas, yerr=[yerr_lo, yerr_hi],
                          fmt="none", ecolor="black", capsize=4, linewidth=1.5)
        ax_delta.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax_delta.set_xticks(x)
        ax_delta.set_xticklabels([str(k) for k in k_values])
        ax_delta.set_xlabel("k (number of attempts)", fontsize=12)
        ax_delta.set_ylabel("Δ coverage gain (steered − unsteered)", fontsize=12)
        ax_delta.set_title(
            f"Diversity collapse test (α={steered['scale']})",
            fontsize=13,
        )
        from matplotlib.patches import Patch
        ax_delta.legend(
            handles=[
                Patch(facecolor="tab:red", label="p < 0.05"),
                Patch(facecolor="0.6", label="p ≥ 0.05"),
            ],
            loc="lower left", fontsize=10,
        )

        fig.tight_layout()
        plot_name = f"coverage_gain_{dataset}_{suffix}_scale{steered['scale']}_temp{temp}.png"
        plot_path = output_dir / plot_name
        ensure_dir(output_dir)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(plot_path)
        logger.info("Saved %s", plot_path)

    return saved


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Plot pass@k curves")
    parser.add_argument("--config", help="Path to eval config YAML (derives input/output paths)")
    parser.add_argument("--input", help="Direct path to pass_at_k_curves.json")
    parser.add_argument("--output", help="Output directory for plots")
    parser.add_argument(
        "--metric", choices=["plus", "base", "both"], default="both",
        help="Which EvalPlus metric to plot (default: both)",
    )
    args = parser.parse_args()

    if args.input:
        curves_path = Path(args.input)
        output_dir = Path(args.output) if args.output else curves_path.parent / "plots"
    elif args.config:
        from src.eval_config import CodeEvalConfig

        cfg = CodeEvalConfig.from_yaml(args.config)
        curves_path = cfg.output_dir / "code" / "pass_at_k_curves.json"
        output_dir = Path(args.output) if args.output else cfg.output_dir / "code" / "plots"
    else:
        parser.error("Either --config or --input is required")

    curves = load_curves(curves_path)
    logger.info("Loaded %d conditions from %s", len(curves), curves_path)

    use_plus_options = []
    if args.metric in ("plus", "both"):
        use_plus_options.append(True)
    if args.metric in ("base", "both"):
        use_plus_options.append(False)

    all_plots: list[Path] = []
    for use_plus in use_plus_options:
        all_plots.extend(plot_pass_at_k_with_ci(curves, output_dir, use_plus=use_plus))
        all_plots.extend(plot_delta_significance(curves, output_dir, use_plus=use_plus))
        all_plots.extend(plot_combined(curves, output_dir, use_plus=use_plus))
        all_plots.extend(plot_coverage_gain(curves, output_dir, use_plus=use_plus))

    logger.info("Generated %d plots in %s", len(all_plots), output_dir)


if __name__ == "__main__":
    main()
