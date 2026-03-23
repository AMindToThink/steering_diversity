"""Orchestrate BigCodeBench pass@k evaluation across steering scales.

Same architecture as eval_code.py but uses BigCodeBench instead of EvalPlus.
BigCodeBench has 1140 harder problems (full subset) with a single test suite
(no plus/base distinction). We store results in both per_problem_plus and
per_problem_base fields for compatibility with coverage_gain_test().

Supports the same steering modes as eval_code.py: server, proxy, none.

Usage:
    uv run python scripts/eval/eval_bigcodebench.py --config configs/eval_bigcodebench.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

import uvicorn
from tqdm import tqdm

from src.eval_config import CodeEvalConfig
from src.pass_at_k import pass_at_k_curve
from src.steering_proxy import app as proxy_app
from src.steering_proxy import configure as configure_proxy
from src.steering_proxy import verify_upstream_supports_steering
from src.utils import ensure_dir, save_provenance, seed_everything

# Reuse steering infrastructure from eval_code
from scripts.eval.eval_code import (
    set_steering_scale,
    start_proxy,
    stop_proxy,
    verify_server_steering,
)

logger = logging.getLogger(__name__)


def extract_per_problem_results_bcb(
    eval_results: dict,
) -> list[tuple[int, int]]:
    """Extract (n, c) pairs from BigCodeBench eval_results.

    BigCodeBench format: eval[task_id] is a list of per-sample dicts, each with
    a ``status`` field ("pass" or failure string). No plus/base distinction.

    Returns
    -------
    List of (n_total, n_correct) tuples, one per problem, sorted by task_id.
    """
    results: list[tuple[int, int]] = []
    eval_data = eval_results.get("eval", {})
    for task_id in sorted(eval_data.keys()):
        samples = eval_data[task_id]
        n = len(samples)
        c = sum(1 for s in samples if s.get("status") == "pass")
        results.append((n, c))
    return results


def run_bigcodebench_generate(
    model_name: str,
    base_url: str,
    split: str,
    subset: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    output_dir: Path,
    log_dir: Path,
    batch_size: int = 10,
    id_range: str | None = None,
) -> Path:
    """Run BigCodeBench code generation and return the samples file path."""
    ensure_dir(output_dir)
    ensure_dir(log_dir)

    cmd = [
        sys.executable, "-m", "bigcodebench.generate",
        "--model", model_name,
        "--split", split,
        "--subset", subset,
        "--backend", "openai",
        "--base_url", base_url,
        "--n_samples", str(n_samples),
        "--temperature", str(temperature),
        "--max_new_tokens", str(max_tokens),
        "--root", str(output_dir),
        "--resume", "True",
        "--bs", str(batch_size),
    ]
    if id_range is not None:
        cmd.extend(["--id_range", id_range])
    logger.info("Running BigCodeBench generate: %s", " ".join(cmd))

    t0 = time.monotonic()
    with open(log_dir / "generate_stdout.log", "w") as stdout_log:
        result = subprocess.run(cmd, stdout=stdout_log, stderr=None, text=True)
    elapsed = time.monotonic() - t0

    (log_dir / "generate_timing.txt").write_text(f"{elapsed:.1f}s\n")
    logger.info("Generate completed in %.1fs (exit=%d)", elapsed, result.returncode)

    if result.returncode != 0:
        stdout_content = (log_dir / "generate_stdout.log").read_text()
        raise RuntimeError(
            f"bigcodebench.generate failed (exit {result.returncode}):\n"
            f"stdout: {stdout_content[-500:]}"
        )

    # BigCodeBench writes: <root>/<model>--bigcodebench-<split>--openai-<temp>-<n>-sanitized_calibrated.jsonl
    samples_files = list(output_dir.rglob("*.jsonl"))
    if not samples_files:
        raise FileNotFoundError(f"No samples file found in {output_dir}")
    # Pick the most recently modified one
    samples_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return samples_files[0]


def run_bigcodebench_evaluate(
    split: str,
    subset: str,
    samples_path: Path,
    log_dir: Path,
    pass_k: str = "1,2,5,10,25,50,100",
) -> Path:
    """Run BigCodeBench evaluation and return path to eval_results.json.

    Uses local execution to avoid external service dependencies.
    """
    ensure_dir(log_dir)

    cmd = [
        sys.executable, "-m", "bigcodebench.evaluate",
        "--split", split,
        "--subset", subset,
        "--samples", str(samples_path),
        "--execution", "local",
        "--pass_k", pass_k,
        "--no_gt",  # skip ground-truth timing check (saves time)
    ]
    logger.info("Running BigCodeBench evaluate: %s", " ".join(cmd))

    t0 = time.monotonic()
    with open(log_dir / "evaluate_stdout.log", "w") as stdout_log:
        result = subprocess.run(cmd, stdout=stdout_log, stderr=None, text=True)
    elapsed = time.monotonic() - t0

    (log_dir / "evaluate_timing.txt").write_text(f"{elapsed:.1f}s\n")
    logger.info("Evaluate completed in %.1fs (exit=%d)", elapsed, result.returncode)

    if result.returncode != 0:
        stdout_content = (log_dir / "evaluate_stdout.log").read_text()
        raise RuntimeError(
            f"bigcodebench.evaluate failed (exit {result.returncode}):\n"
            f"stdout: {stdout_content[-500:]}"
        )

    # BigCodeBench writes: <samples_stem>_eval_results.json
    results_path = samples_path.with_name(
        samples_path.stem + "_eval_results.json"
    )
    if not results_path.exists():
        results_files = list(samples_path.parent.rglob("*eval_results.json"))
        if not results_files:
            raise FileNotFoundError(
                f"No eval_results.json found near {samples_path}"
            )
        results_path = results_files[0]

    return results_path


def run_single_condition(
    cfg: CodeEvalConfig,
    scale: float,
    temperature: float,
    dataset: str,
) -> dict:
    """Run evaluation for a single (scale, temperature) condition on BigCodeBench."""
    # Parse BigCodeBench split/subset from dataset string
    # Format: "bigcodebench" or "bigcodebench-hard"
    if dataset == "bigcodebench-hard":
        split = "instruct"
        subset = "hard"
    else:
        split = "instruct"
        subset = "full"

    condition_name = f"scale_{scale}_temp_{temperature}"
    output_dir = cfg.output_dir / "code" / dataset / condition_name
    log_dir = output_dir / "logs"

    mode = cfg.endpoint.steering_mode
    proc = None

    if mode == "none":
        codegen_url = cfg.endpoint.base_url
        if not codegen_url.endswith("/v1"):
            codegen_url = codegen_url.rstrip("/") + "/v1"
    elif mode == "server":
        set_steering_scale(cfg.endpoint.base_url, scale)
        codegen_url = cfg.endpoint.base_url
        if not codegen_url.endswith("/v1"):
            codegen_url = codegen_url.rstrip("/") + "/v1"
    elif mode == "proxy":
        proxy_url = f"http://localhost:{cfg.endpoint.proxy_port}/v1"
        upstream = cfg.endpoint.base_url.rstrip("/")
        if upstream.endswith("/v1"):
            upstream = upstream[:-3]
        assert cfg.steering is not None, "proxy mode requires steering config"
        assert cfg.vector_path is not None, "proxy mode requires vector_path"
        proc = start_proxy(
            upstream=upstream,
            vector_path=cfg.vector_path,
            scale=scale,
            target_layers=cfg.steering.target_layers,
            algorithm=cfg.steering.algorithm,
            normalize=cfg.steering.normalize,
            port=cfg.endpoint.proxy_port,
        )
        codegen_url = proxy_url
    else:
        raise ValueError(f"Unknown steering_mode: {mode!r}. Use 'server', 'proxy', or 'none'.")

    condition_t0 = time.monotonic()
    try:
        # Generate
        samples_path = run_bigcodebench_generate(
            model_name=cfg.model.name,
            base_url=codegen_url,
            split=split,
            subset=subset,
            n_samples=cfg.pass_at_k.n_samples,
            temperature=temperature,
            max_tokens=cfg.pass_at_k.max_tokens,
            output_dir=output_dir,
            log_dir=log_dir,
            batch_size=cfg.pass_at_k.batch_size,
            id_range=cfg.pass_at_k.id_range,
        )

        # Evaluate
        pass_k_str = ",".join(str(k) for k in cfg.pass_at_k.k_values)
        eval_results_path = run_bigcodebench_evaluate(
            split, subset, samples_path, log_dir, pass_k=pass_k_str,
        )
        with open(eval_results_path) as f:
            eval_results = json.load(f)

        # Compute pass@k — BigCodeBench has single test suite
        per_problem = extract_per_problem_results_bcb(eval_results)

        valid_k = [k for k in cfg.pass_at_k.k_values if k <= cfg.pass_at_k.n_samples]
        curve = pass_at_k_curve(per_problem, valid_k)

        condition_elapsed = time.monotonic() - condition_t0

        # Store in both plus and base fields for coverage_gain_test() compatibility
        per_problem_dicts = [{"n": n, "c": c} for n, c in per_problem]
        condition_results = {
            "scale": scale,
            "temperature": temperature,
            "dataset": dataset,
            "steering_mode": mode,
            "benchmark": "bigcodebench",
            "split": split,
            "subset": subset,
            "pass_at_k_plus": {str(k): v for k, v in curve.items()},
            "pass_at_k_base": {str(k): v for k, v in curve.items()},
            "n_problems": len(per_problem),
            "n_samples_per_problem": cfg.pass_at_k.n_samples,
            "per_problem_plus": per_problem_dicts,
            "per_problem_base": per_problem_dicts,
            "elapsed_seconds": round(condition_elapsed, 1),
            "vector_path": cfg.vector_path or "none",
        }
        results_path = output_dir / "pass_at_k.json"
        ensure_dir(results_path.parent)
        with open(results_path, "w") as f:
            json.dump(condition_results, f, indent=2)

        (log_dir / "total_timing.txt").write_text(f"{condition_elapsed:.1f}s\n")

        save_provenance(
            step="eval_bigcodebench",
            config_path=cfg.steering_config_path,
            cfg=cfg,
            inputs={"samples": str(samples_path), "eval_results": str(eval_results_path)},
            outputs=[str(results_path)],
        )

        logger.info(
            "Completed %s/%s in %.1fs: pass@1=%.3f",
            dataset,
            condition_name,
            condition_elapsed,
            curve[valid_k[0]],
        )
        return condition_results

    finally:
        if proc is not None:
            stop_proxy(proc)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="BigCodeBench pass@k evaluation")
    parser.add_argument("--config", required=True, help="Path to eval config YAML")
    args = parser.parse_args()

    cfg = CodeEvalConfig.from_yaml(args.config)
    seed_everything(cfg.seed)

    mode = cfg.endpoint.steering_mode
    logger.info(
        "Starting BigCodeBench eval: run=%s, mode=%s, scales=%s, datasets=%s, n=%d",
        cfg.run_name,
        mode,
        cfg.scales,
        cfg.datasets,
        cfg.pass_at_k.n_samples,
    )

    if mode == "none":
        logger.info("Steering mode 'none': skipping steering verification.")
    elif mode == "server":
        logger.info("Verifying server-level steering at %s ...", cfg.endpoint.base_url)
        verify_server_steering(cfg.endpoint.base_url)
        logger.info("Server steering verification passed.")
    elif mode == "proxy":
        upstream = cfg.endpoint.base_url.rstrip("/")
        if upstream.endswith("/v1"):
            upstream = upstream[:-3]
        assert cfg.vector_path is not None, "proxy mode requires vector_path"
        assert cfg.steering is not None, "proxy mode requires steering config"
        logger.info("Verifying upstream steering support at %s ...", upstream)
        verify_upstream_supports_steering(
            upstream, cfg.vector_path, cfg.steering.target_layers
        )
        logger.info("Upstream steering verification passed.")
    else:
        raise ValueError(f"Unknown steering_mode: {mode!r}. Use 'server', 'proxy', or 'none'.")

    total_t0 = time.monotonic()
    all_results: list[dict] = []

    conditions = [
        (temperature, dataset, scale)
        for temperature in cfg.pass_at_k.temperatures
        for dataset in cfg.datasets
        for scale in cfg.scales
    ]
    pbar = tqdm(conditions, desc="Conditions", unit="cond")
    for temperature, dataset, scale in pbar:
        pbar.set_postfix(scale=scale, temp=temperature, dataset=dataset)
        result = run_single_condition(cfg, scale, temperature, dataset)
        all_results.append(result)

    total_elapsed = time.monotonic() - total_t0

    curves_path = cfg.output_dir / "code" / "pass_at_k_curves.json"
    ensure_dir(curves_path.parent)
    with open(curves_path, "w") as f:
        json.dump(all_results, f, indent=2)

    timing_path = cfg.output_dir / "code" / "total_timing.txt"
    with open(timing_path, "w") as f:
        f.write(f"Total: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)\n")
        f.write(f"Conditions: {len(all_results)}\n")
        for r in all_results:
            f.write(
                f"  scale={r['scale']} temp={r['temperature']} "
                f"dataset={r['dataset']}: {r['elapsed_seconds']}s\n"
            )

    logger.info(
        "All results saved to %s (total: %.1fs / %.1fmin)",
        curves_path,
        total_elapsed,
        total_elapsed / 60,
    )


if __name__ == "__main__":
    main()
