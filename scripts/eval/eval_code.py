"""Orchestrate code-domain pass@k evaluation across steering scales.

Supports two steering modes:
  - **server** (default, 2.3x faster): vLLM started with --steer-vector-path.
    Scale is changed at runtime via POST /v1/steering. No proxy needed.
  - **proxy** (legacy): Per-request steering via a local steering proxy.

For each (temperature, scale) combination:
  1. Set steering scale (server mode) or start proxy (proxy mode)
  2. Run EvalPlus codegen
  3. Run EvalPlus evaluate
  4. Parse per-problem pass/fail → compute pass@k at all k values
  5. Save results + provenance

Usage:
    uv run python scripts/eval/eval_code.py --config configs/eval_code.yaml
    uv run python scripts/eval/eval_code.py --config configs/eval_code_dev.yaml
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

logger = logging.getLogger(__name__)


def _ensure_port_free(port: int) -> None:
    """Raise if the port is already in use. Fail loud, don't silently reuse a stale proxy."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(("localhost", port))
        if result == 0:
            raise RuntimeError(
                f"Port {port} is already in use. A stale proxy or other process "
                f"is occupying it. Kill it before running eval_code.py."
            )


def start_proxy(
    upstream: str,
    vector_path: str,
    scale: float,
    target_layers: list[int],
    algorithm: str,
    normalize: bool,
    port: int,
    host: str = "0.0.0.0",
) -> multiprocessing.Process:
    """Start the steering proxy in a subprocess."""
    _ensure_port_free(port)

    def _run() -> None:
        configure_proxy(
            upstream=upstream,
            vector_path=vector_path,
            scale=scale,
            target_layers=target_layers,
            algorithm=algorithm,
            normalize=normalize,
        )
        uvicorn.run(proxy_app, host=host, port=port, log_level="warning")

    proc = multiprocessing.Process(target=_run, daemon=True)
    proc.start()
    logger.info("Started proxy (pid=%d) on port %d with scale=%.2f", proc.pid, port, scale)

    # Wait for proxy to be ready
    _wait_for_proxy(f"http://localhost:{port}", timeout=15.0)

    # Verify this is actually OUR proxy (not a stale one) by checking it's alive
    if not proc.is_alive():
        raise RuntimeError(
            f"Proxy process (pid={proc.pid}) died immediately after start. "
            f"Check if port {port} is available."
        )

    return proc


def _wait_for_proxy(base_url: str, timeout: float = 15.0) -> None:
    """Poll the proxy health endpoint until it responds."""
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=2.0)
            if resp.status_code == 200:
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.3)
    raise TimeoutError(f"Proxy at {base_url} did not become ready within {timeout}s")


def stop_proxy(proc: multiprocessing.Process) -> None:
    """Terminate the proxy subprocess."""
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
    logger.info("Stopped proxy (pid=%d)", proc.pid)


def set_steering_scale(base_url: str, scale: float) -> None:
    """Set the steering scale on a server-level steered vLLM instance.

    Calls POST {base_url}/v1/steering with the new scale value.
    """
    import httpx

    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    resp = httpx.post(f"{url}/v1/steering", json={"scale": scale}, timeout=10.0)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to set steering scale to {scale}: "
            f"status={resp.status_code}, body={resp.text}"
        )
    logger.info("Set server steering scale to %.2f", scale)


def verify_server_steering(base_url: str) -> dict:
    """Verify that the vLLM server has server-level steering enabled.

    Calls GET {base_url}/v1/steering and confirms a vector is loaded.
    Returns the steering status dict.
    """
    import httpx

    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    resp = httpx.get(f"{url}/v1/steering", timeout=10.0)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Server steering endpoint not available: "
            f"status={resp.status_code}, body={resp.text}"
        )
    status = resp.json()
    if not status.get("active", False):
        raise RuntimeError(
            f"Server reports steering is not active: {status}. "
            f"Start vLLM with --steer-vector-path."
        )
    logger.info("Server steering verified: %s", status)
    return status


def run_evalplus_codegen(
    model_name: str,
    dataset: str,
    base_url: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    output_dir: Path,
    log_dir: Path,
) -> Path:
    """Run EvalPlus code generation and return the samples file path."""
    ensure_dir(output_dir)
    ensure_dir(log_dir)

    # EvalPlus uses fire: positional args for model and dataset
    cmd = [
        sys.executable, "-m", "evalplus.codegen",
        model_name, dataset,
        "--backend", "openai",
        "--base-url", base_url,
        "--n-samples", str(n_samples),
        "--temperature", str(temperature),
        "--root", str(output_dir),
    ]
    logger.info("Running codegen: %s", " ".join(cmd))

    t0 = time.monotonic()
    # Show stderr live (evalplus progress) while saving stdout and stderr to logs
    with open(log_dir / "codegen_stdout.log", "w") as stdout_log:
        result = subprocess.run(cmd, stdout=stdout_log, stderr=None, text=True)
    elapsed = time.monotonic() - t0

    # Save timing
    (log_dir / "codegen_timing.txt").write_text(f"{elapsed:.1f}s\n")
    logger.info("Codegen completed in %.1fs (exit=%d)", elapsed, result.returncode)

    if result.returncode != 0:
        stdout_content = (log_dir / "codegen_stdout.log").read_text()
        raise RuntimeError(
            f"evalplus.codegen failed (exit {result.returncode}):\n"
            f"stdout: {stdout_content[-500:]}"
        )

    # EvalPlus writes: <root>/<dataset>/<Model--Name>_<backend>_temp_<T>.jsonl
    samples_files = list(output_dir.rglob("*.jsonl"))
    # Filter to sanitized (non-raw) files
    samples_files = [f for f in samples_files if ".raw." not in f.name]
    if not samples_files:
        raise FileNotFoundError(f"No samples file found in {output_dir}")
    return samples_files[0]


def run_evalplus_evaluate(
    dataset: str,
    samples_path: Path,
    log_dir: Path,
) -> Path:
    """Run EvalPlus evaluation and return path to eval_results.json."""
    cmd = [
        sys.executable, "-m", "evalplus.evaluate",
        "--dataset", dataset,
        "--samples", str(samples_path),
    ]
    logger.info("Running evaluation: %s", " ".join(cmd))

    t0 = time.monotonic()
    # Show stderr live (evalplus progress) while saving stdout to log
    with open(log_dir / "evaluate_stdout.log", "w") as stdout_log:
        result = subprocess.run(cmd, stdout=stdout_log, stderr=None, text=True)
    elapsed = time.monotonic() - t0

    # Save timing
    (log_dir / "evaluate_timing.txt").write_text(f"{elapsed:.1f}s\n")
    logger.info("Evaluate completed in %.1fs (exit=%d)", elapsed, result.returncode)

    if result.returncode != 0:
        stdout_content = (log_dir / "evaluate_stdout.log").read_text()
        raise RuntimeError(
            f"evalplus.evaluate failed (exit {result.returncode}):\n"
            f"stdout: {stdout_content[-500:]}"
        )

    # EvalPlus writes eval_results.json next to the samples file with a suffix
    # Pattern: <samples_stem>_eval_results.json
    results_path = samples_path.with_name(
        samples_path.stem + "_eval_results.json"
    )
    if not results_path.exists():
        # Fallback: search nearby
        results_files = list(samples_path.parent.rglob("*eval_results.json"))
        if not results_files:
            raise FileNotFoundError(
                f"No eval_results.json found near {samples_path}"
            )
        results_path = results_files[0]

    return results_path


def extract_per_problem_results(
    eval_results: dict,
    use_plus: bool = True,
) -> list[tuple[int, int]]:
    """Extract (n, c) pairs from EvalPlus eval_results.

    EvalPlus format: eval[task_id] is a list of per-sample dicts, each with
    ``base_status`` and ``plus_status`` fields ("pass" or "fail").

    Parameters
    ----------
    eval_results:
        Loaded eval_results.json dict.
    use_plus:
        If True, use plus_status (stricter). If False, use base_status.

    Returns
    -------
    List of (n_total, n_correct) tuples, one per problem.
    """
    status_key = "plus_status" if use_plus else "base_status"
    results: list[tuple[int, int]] = []
    eval_data = eval_results.get("eval", {})
    for task_id in sorted(eval_data.keys()):
        samples = eval_data[task_id]
        n = len(samples)
        c = sum(1 for s in samples if s.get(status_key) == "pass")
        results.append((n, c))
    return results


def run_single_condition(
    cfg: CodeEvalConfig,
    scale: float,
    temperature: float,
    dataset: str,
) -> dict:
    """Run evaluation for a single (scale, temperature, dataset) condition."""
    condition_name = f"scale_{scale}_temp_{temperature}"
    output_dir = cfg.output_dir / "code" / dataset / condition_name
    log_dir = output_dir / "logs"

    mode = cfg.endpoint.steering_mode
    proc = None

    if mode == "none":
        # No runtime steering — model weights already contain any modifications
        codegen_url = cfg.endpoint.base_url
        if not codegen_url.endswith("/v1"):
            codegen_url = codegen_url.rstrip("/") + "/v1"
    elif mode == "server":
        # Server-level steering: set scale directly on vLLM
        set_steering_scale(cfg.endpoint.base_url, scale)
        codegen_url = cfg.endpoint.base_url
        if not codegen_url.endswith("/v1"):
            codegen_url = codegen_url.rstrip("/") + "/v1"
    elif mode == "proxy":
        # Legacy per-request proxy mode
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
        samples_path = run_evalplus_codegen(
            model_name=cfg.model.name,
            dataset=dataset,
            base_url=codegen_url,
            n_samples=cfg.pass_at_k.n_samples,
            temperature=temperature,
            max_tokens=cfg.pass_at_k.max_tokens,
            output_dir=output_dir,
            log_dir=log_dir,
        )

        # Evaluate
        eval_results_path = run_evalplus_evaluate(dataset, samples_path, log_dir)
        with open(eval_results_path) as f:
            eval_results = json.load(f)

        # Compute pass@k for both base and plus
        per_problem_plus = extract_per_problem_results(eval_results, use_plus=True)
        per_problem_base = extract_per_problem_results(eval_results, use_plus=False)

        # Filter k_values to those <= n_samples
        valid_k = [k for k in cfg.pass_at_k.k_values if k <= cfg.pass_at_k.n_samples]
        curve_plus = pass_at_k_curve(per_problem_plus, valid_k)
        curve_base = pass_at_k_curve(per_problem_base, valid_k)

        condition_elapsed = time.monotonic() - condition_t0

        # Save results
        condition_results = {
            "scale": scale,
            "temperature": temperature,
            "dataset": dataset,
            "steering_mode": mode,
            "pass_at_k_plus": {str(k): v for k, v in curve_plus.items()},
            "pass_at_k_base": {str(k): v for k, v in curve_base.items()},
            "n_problems": len(per_problem_plus),
            "n_samples_per_problem": cfg.pass_at_k.n_samples,
            "per_problem_plus": [{"n": n, "c": c} for n, c in per_problem_plus],
            "per_problem_base": [{"n": n, "c": c} for n, c in per_problem_base],
            "elapsed_seconds": round(condition_elapsed, 1),
            "vector_path": cfg.vector_path or "none",
        }
        results_path = output_dir / "pass_at_k.json"
        ensure_dir(results_path.parent)
        with open(results_path, "w") as f:
            json.dump(condition_results, f, indent=2)

        # Timing log
        (log_dir / "total_timing.txt").write_text(f"{condition_elapsed:.1f}s\n")

        # Provenance
        save_provenance(
            step="eval_code",
            config_path=cfg.steering_config_path,
            cfg=cfg,
            inputs={"samples": str(samples_path), "eval_results": str(eval_results_path)},
            outputs=[str(results_path)],
        )

        logger.info(
            "Completed %s/%s in %.1fs: pass@1_plus=%.3f, pass@1_base=%.3f",
            dataset,
            condition_name,
            condition_elapsed,
            curve_plus[valid_k[0]],
            curve_base[valid_k[0]],
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

    parser = argparse.ArgumentParser(description="Code-domain pass@k evaluation")
    parser.add_argument("--config", required=True, help="Path to eval config YAML")
    args = parser.parse_args()

    cfg = CodeEvalConfig.from_yaml(args.config)
    seed_everything(cfg.seed)

    mode = cfg.endpoint.steering_mode
    logger.info(
        "Starting code eval: run=%s, mode=%s, scales=%s, datasets=%s, n=%d",
        cfg.run_name,
        mode,
        cfg.scales,
        cfg.datasets,
        cfg.pass_at_k.n_samples,
    )

    if mode == "none":
        # No runtime steering — just verify the server is reachable
        logger.info("Steering mode 'none': skipping steering verification.")
    elif mode == "server":
        # Server-level steering: verify the server has a vector loaded
        logger.info("Verifying server-level steering at %s ...", cfg.endpoint.base_url)
        verify_server_steering(cfg.endpoint.base_url)
        logger.info("Server steering verification passed.")
    elif mode == "proxy":
        # Legacy proxy mode: verify upstream supports per-request steering
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

    # Save aggregated curves
    curves_path = cfg.output_dir / "code" / "pass_at_k_curves.json"
    ensure_dir(curves_path.parent)
    with open(curves_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save total timing
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
