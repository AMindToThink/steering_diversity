"""Phase 1 of experiment 2 — per-layer norm diagnostic.

Captures ``E[‖x_i‖]``, ``‖μ_i‖``, ``tr(Σ_x_i)``, and
``cos(h_i, mean_i_hat)`` at every decoder layer of a model at scale=0
(no steering), where ``h_i`` is the reference steering vector's direction
at layer i. Used to inform the layer-picking decision for the
single-layer sweep.

Runs on GPU; caller is responsible for setting ``CUDA_VISIBLE_DEVICES``
and confirming the GPU is idle.

Output: ``outputs/bounds/diagnostic_per_layer_<model_tag>.json``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from nnsight import LanguageModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.bounds.activation_streams import CheapMoments  # noqa: E402
from src.bounds.dataset import load_fineweb_prompts  # noqa: E402
from src.bounds.gguf_loader import load_steering_vector_gguf  # noqa: E402
from src.bounds.nnsight_runner import (  # noqa: E402
    _get_layer_list,
    _is_tuple_output_architecture,
    _tokenize_for_trace,
)


def run_diagnostic(
    model_name: str,
    vector_path: Path,
    num_prompts: int = 100,
    max_seq_len: int = 256,
    batch_size: int = 8,
    seed: int = 0,
    out_path: Path | None = None,
) -> dict:
    """Capture per-layer residual-stream stats at scale=0."""
    print(f"[diagnose] Loading {model_name!r} ...")
    t0 = time.time()
    lm = LanguageModel(model_name, device_map="auto")
    n_layers = len(lm.model.layers)
    print(f"[diagnose]   loaded in {time.time() - t0:.1f}s, n_layers={n_layers}")

    tuple_output = _is_tuple_output_architecture(lm)
    layer_list = _get_layer_list(lm)

    d = getattr(lm.config, "hidden_size", None) or lm.config.n_embd
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Per-layer CheapMoments — index 0 = embedding output, 1..n_layers = layer i output.
    moments = [CheapMoments(d=d, device=device) for _ in range(n_layers + 1)]

    print(f"[diagnose] Loading steering reference from {vector_path} ...")
    ref_vecs = load_steering_vector_gguf(vector_path)

    print(f"[diagnose] Loading {num_prompts} fineweb-edu prompts ...")
    prompts = load_fineweb_prompts(num_prompts, seed=seed)

    t0 = time.time()
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = _tokenize_for_trace(lm, batch, max_seq_len)

        embed = None
        layer_outs: list = []
        with lm.trace(enc):
            embed = lm.model.embed_tokens.output.save()
            for i in range(n_layers):
                if tuple_output:
                    layer_outs.append(lm.model.layers[i].output[0].save())
                else:
                    layer_outs.append(lm.model.layers[i].output.save())

        mask = enc["attention_mask"]
        B, T = mask.shape
        bool_mask = mask.to(torch.bool)

        sites = [embed.detach().cpu()] + [lo.detach().cpu() for lo in layer_outs]
        for site_i, tensor in enumerate(sites):
            assert tensor.ndim == 3, (
                f"site {site_i}: expected [B,T,d] got {tuple(tensor.shape)}"
            )
            assert tensor.shape[0] == B, (
                f"site {site_i}: batch dim {tensor.shape[0]} != {B}"
            )
            flat_real = tensor[bool_mask]  # [N_real, d]
            moments[site_i].update(flat_real)

    elapsed = time.time() - t0
    print(f"[diagnose]   forward-passes done in {elapsed:.1f}s")

    # Finalize and gather per-site stats.
    per_site: list[dict] = []
    for site_i, mom in enumerate(moments):
        stats = mom.finalize()
        mu = stats["mean"]
        mu_norm = float(mu.norm().item())
        tr_sigma = float(stats["trace_cov"])
        e_norm = float(stats["E_norm"])
        r_max = float(stats["R"])

        # Cosine of the reference steering vector at this layer with the
        # mean direction. Only defined for layers that are in the
        # reference (embedding output has no reference direction).
        ref_cos: float | None = None
        # Site index i: i=0 is embedding, i=1..n_layers is layer i-1's output.
        # Reference steering is applied at the OUTPUT of layer k, i.e.
        # the residual going INTO layer k+1. So ref_vecs[k] matches site (k+1).
        layer_idx = site_i - 1  # -1 for embedding, 0..n_layers-1 for layers
        if layer_idx in ref_vecs and mu_norm > 0:
            h = ref_vecs[layer_idx].to(torch.float32)
            h_hat = h / h.norm().clamp_min(1e-12)
            mu_hat = mu / mu_norm
            ref_cos = float(torch.dot(h_hat, mu_hat).item())

        per_site.append({
            "site_i": site_i,
            "layer_idx": layer_idx,  # -1 for embedding
            "mu_norm": mu_norm,
            "tr_sigma": tr_sigma,
            "e_norm": e_norm,
            "max_norm": r_max,
            "cos_ref_mu": ref_cos,
        })

    result = {
        "model": model_name,
        "n_layers": n_layers,
        "num_prompts": num_prompts,
        "vector_path": str(vector_path),
        "per_site": per_site,
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[diagnose] wrote {out_path}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    run_diagnostic(
        model_name=args.model,
        vector_path=args.vector,
        num_prompts=args.num_prompts,
        batch_size=args.batch_size,
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
