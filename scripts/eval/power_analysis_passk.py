"""Power analysis: how many samples per problem (n) do we need for pass@k
effect sizes at different k values to have non-overlapping CIs?

Uses the observed per-problem pass rates from the n=10 preliminary run as
ground truth, then simulates binomial draws at various n to estimate CI widths.

Usage:
    uv run python scripts/eval/power_analysis_passk.py
"""

from __future__ import annotations

import json

import numpy as np

from src.pass_at_k import pass_at_k


def load_observed_rates() -> tuple[np.ndarray, np.ndarray]:
    """Load per-problem pass rates from the preliminary n=10 runs."""
    with open("outputs/passk_test_steered/code/pass_at_k_curves.json") as f:
        steered = json.load(f)[0]
    with open(
        "outputs/test_unsteered_full/humaneval/"
        "Qwen--Qwen2.5-1.5B-Instruct_openai_temp_0.8_eval_results.json"
    ) as f:
        unsteered_eval = json.load(f)

    un_rates = []
    for task_id in sorted(unsteered_eval["eval"].keys()):
        samples = unsteered_eval["eval"][task_id]
        un_rates.append(
            sum(1 for s in samples if s["plus_status"] == "pass") / len(samples)
        )
    st_rates = [d["c"] / d["n"] for d in steered["per_problem_plus"]]

    return np.array(un_rates), np.array(st_rates)


def simulate_ci_widths(
    un_rates: np.ndarray,
    st_rates: np.ndarray,
    n_samples_list: list[int],
    k_values: list[int],
    n_boot: int = 1000,
    seed: int = 42,
) -> None:
    """Simulate pass@k at various n and report CI widths for each k."""
    rng = np.random.default_rng(seed)

    print(f"Observed mean pass rate: unsteered={un_rates.mean():.3f}, steered={st_rates.mean():.3f}")
    print(f"Per-problem rate diff: mean={np.mean(st_rates - un_rates):.4f}")
    print()

    header = f"{'n':>5s}"
    for k in k_values:
        header += f"  |  Δpass@{k} (mean ± 95%CI)"
    print(header)
    print("-" * len(header))

    for n in n_samples_list:
        line = f"{n:>5d}"
        deltas_by_k: dict[int, np.ndarray] = {}

        for k in k_values:
            k_eff = min(k, n)
            boot_deltas = []
            for _ in range(n_boot):
                un_c = rng.binomial(n, un_rates)
                st_c = rng.binomial(n, st_rates)
                un_pk = np.array([pass_at_k(n, c, k_eff) for c in un_c])
                st_pk = np.array([pass_at_k(n, c, k_eff) for c in st_c])
                boot_deltas.append(np.mean(st_pk - un_pk))

            boot_deltas = np.array(boot_deltas)
            deltas_by_k[k] = boot_deltas
            mean_d = np.mean(boot_deltas)
            ci = 1.96 * np.std(boot_deltas)
            line += f"  |  {mean_d:+.4f} ± {ci:.4f}"

        # Check overlap between k=1 and k=10 (or first and last k)
        k_first, k_last = k_values[0], k_values[-1]
        d1 = deltas_by_k[k_first]
        d2 = deltas_by_k[k_last]
        m1, ci1 = np.mean(d1), 1.96 * np.std(d1)
        m2, ci2 = np.mean(d2), 1.96 * np.std(d2)
        overlap = (m1 - ci1) < (m2 + ci2) and (m2 - ci2) < (m1 + ci1)
        line += f"  | k={k_first} vs k={k_last}: {'OVERLAP' if overlap else 'SEPARATED'}"

        print(line)


def main() -> None:
    un_rates, st_rates = load_observed_rates()
    simulate_ci_widths(
        un_rates,
        st_rates,
        n_samples_list=[10, 20, 30, 50, 75, 100, 150, 200],
        k_values=[1, 10],
        n_boot=1000,
    )


if __name__ == "__main__":
    main()
