"""Per-layer residual-stream variance trajectory under steering.

Captures the residual stream at every decoder layer's OUTPUT across a
scale sweep and accumulates streaming variance statistics at each layer.
Produces a plot showing how variance evolves through the network as a
function of depth, with steered-layer range marked.

This answers the diagnostic question: "why doesn't the final-site variance
drop with steering — is the variance actually being injected by the
intermediate attention/MLP blocks that sit between the last steered layer
and the final RMSNorm?"

Two variance trajectories are plotted side-by-side:

- **All real tokens.** Every non-padding token across every prompt is
  folded into one accumulator per (layer, scale). This is the total
  residual variance including both content variance and across-position
  variance.
- **Last real token per prompt.** One sample per prompt per (layer, scale).
  This strips out the across-position component — if the shape still
  matches the all-tokens trajectory, position structure isn't the source
  of the amplification.

RoPE applies to Q/K inside attention but does NOT add anything to the
residual stream (confirmed via ``AutoConfig`` — Qwen/Llama use RoPE and
have no ``wpe`` / ``embed_positions`` module). So the per-position control
needed here is "variance across prompts at a fixed token position," not
"residual minus positional embedding."
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from nnsight import LanguageModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.bounds.activation_streams import CheapMoments  # noqa: E402
from src.bounds.config import BoundsExperimentConfig  # noqa: E402
from src.bounds.dataset import load_fineweb_prompts  # noqa: E402
from src.bounds.gguf_loader import load_steering_vector_gguf  # noqa: E402
from src.bounds.nnsight_runner import (  # noqa: E402
    _add_steering_at_layers,
    _get_layer_list,
    _is_tuple_output_architecture,
    _model_dtype_device,
    _tokenize_for_trace,
)
from src.bounds.random_vectors import generate_random_steering_vector  # noqa: E402


def _load_steering(cfg: BoundsExperimentConfig) -> dict[int, torch.Tensor]:
    if cfg.steering.vector_path and cfg.steering.vector_path != "__synthetic__":
        return load_steering_vector_gguf(cfg.steering.vector_path)
    if cfg.steering.random_reference_path:
        ref = load_steering_vector_gguf(cfg.steering.random_reference_path)
        return generate_random_steering_vector(ref, seed=cfg.steering.random_seed or 0)
    raise ValueError("config has no real steering vector; refusing to run")


def _last_real_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the index of the last real (non-padding) token for each row.

    Works for both left- and right-padding conventions since we scan from
    the right: argmax of reversed mask.
    """
    B, T = attention_mask.shape
    # Flip along seq dim, argmax returns first 1 in the reversed order.
    reversed_mask = attention_mask.flip(-1)
    first_one_in_reversed = reversed_mask.argmax(dim=-1)
    return (T - 1) - first_one_in_reversed


def run_one_scale(
    lm: LanguageModel,
    prompts: list[str],
    steering: dict[int, torch.Tensor] | None,
    scale: float,
    target_layers: list[int],
    max_seq_len: int,
    batch_size: int,
    n_layers: int,
) -> dict:
    """Stream all prompts through one scale. Returns per-layer finalized stats."""
    d = getattr(lm.config, "hidden_size", None) or lm.config.n_embd
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # All-tokens accumulators: one per (layer, position-0..n_layers).
    # Includes "layer 0 input" (the embedding output) plus every decoder
    # layer's output. So we index 0..n_layers, giving n_layers+1 sites.
    moments_all = [CheapMoments(d=d, device=device) for _ in range(n_layers + 1)]
    moments_last = [CheapMoments(d=d, device=device) for _ in range(n_layers + 1)]

    dtype, device_t = _model_dtype_device(lm)
    layer_list = _get_layer_list(lm)
    tuple_output = _is_tuple_output_architecture(lm)

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = _tokenize_for_trace(lm, batch, max_seq_len)

        # Pre-declare so the references survive the lm.trace() block exit.
        embed = None
        layer_outs: list = []
        with lm.trace(enc):
            _add_steering_at_layers(
                layer_list, steering, scale, target_layers, dtype, device_t,
                tuple_output=tuple_output,
            )
            # IMPORTANT: saves must come AFTER the in-place steering writes,
            # so they snapshot the post-intervention values.
            embed = lm.model.embed_tokens.output.save()
            for i in range(n_layers):
                # For Qwen2/Llama3, layer.output is already the [B,T,d]
                # tensor; .output[0] would select batch 0. For GPT-2
                # tuple_output=True and layer.output[0] unpacks the tuple.
                if tuple_output:
                    layer_outs.append(lm.model.layers[i].output[0].save())
                else:
                    layer_outs.append(lm.model.layers[i].output.save())

        mask = enc["attention_mask"]  # [B, T]
        B, T = mask.shape
        bool_mask = mask.to(torch.bool)
        last_idx = _last_real_indices(mask)  # [B]

        # Site 0: embedding output (no steering possible here).
        sites = [embed.detach().cpu()] + [lo.detach().cpu() for lo in layer_outs]
        for site_i, tensor in enumerate(sites):
            # Shape assertion — catches the .output[0] / tuple_output bug class.
            assert tensor.ndim == 3, (
                f"site {site_i}: expected [B,T,d] got {tuple(tensor.shape)}"
            )
            assert tensor.shape[0] == B, (
                f"site {site_i}: batch dim {tensor.shape[0]} != expected {B}"
            )
            assert tensor.shape[1] == T, (
                f"site {site_i}: seq dim {tensor.shape[1]} != expected {T}"
            )
            flat_real = tensor[bool_mask]  # [N_real, d]
            moments_all[site_i].update(flat_real)

            # Gather last-real-token: advanced indexing by (row, last_idx).
            last_tokens = tensor[torch.arange(tensor.shape[0]), last_idx]  # [B, d]
            moments_last[site_i].update(last_tokens)

    all_stats = [m.finalize() for m in moments_all]
    last_stats = [m.finalize() for m in moments_last]
    return {"all_tokens": all_stats, "last_token": last_stats}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = BoundsExperimentConfig.from_yaml(args.config)
    cfg.dataset.num_prompts = args.num_prompts
    output_dir = args.output_dir or (cfg.output_dir.parent / f"{cfg.run_name}_variance_trajectory")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diagnose] Loading model {cfg.model.name!r} ...")
    t0 = time.time()
    lm = LanguageModel(cfg.model.name, device_map="auto")
    n_layers = len(lm.model.layers)
    print(f"[diagnose]   loaded in {time.time() - t0:.1f}s, n_layers = {n_layers}")

    print("[diagnose] Loading steering vector ...")
    steering = _load_steering(cfg)

    print(f"[diagnose] Loading {args.num_prompts} prompts from {cfg.dataset.name} ...")
    prompts = load_fineweb_prompts(args.num_prompts, seed=cfg.seed, dataset_name=cfg.dataset.name)

    per_scale: dict[float, dict] = {}
    for scale in cfg.steering.scale_sweep:
        print(f"[diagnose] scale={scale} ...")
        t0 = time.time()
        per_scale[scale] = run_one_scale(
            lm=lm,
            prompts=prompts,
            steering=steering if scale > 0 else None,
            scale=scale,
            target_layers=cfg.steering.target_layers if scale > 0 else [],
            max_seq_len=cfg.dataset.max_seq_len,
            batch_size=cfg.dataset.batch_size,
            n_layers=n_layers,
        )
        dt = time.time() - t0
        final_all = per_scale[scale]["all_tokens"][-1]["trace_cov"]
        final_last = per_scale[scale]["last_token"][-1]["trace_cov"]
        print(
            f"[diagnose]   done in {dt:.1f}s  "
            f"tr(Σ)@last_layer  all_tokens={final_all:.4g}  "
            f"last_token={final_last:.4g}"
        )

    # Save numerical trajectories.
    traj_all = {
        str(scale): [s["trace_cov"] for s in per_scale[scale]["all_tokens"]]
        for scale in cfg.steering.scale_sweep
    }
    traj_last = {
        str(scale): [s["trace_cov"] for s in per_scale[scale]["last_token"]]
        for scale in cfg.steering.scale_sweep
    }
    trajectory_json = {
        "run_name": cfg.run_name,
        "model": cfg.model.name,
        "n_layers": n_layers,
        "target_layers": cfg.steering.target_layers,
        "scale_sweep": cfg.steering.scale_sweep,
        "num_prompts": args.num_prompts,
        "trace_cov_all_tokens": traj_all,
        "trace_cov_last_token": traj_last,
    }
    (output_dir / "variance_trajectory.json").write_text(
        json.dumps(trajectory_json, indent=2)
    )
    print(f"[diagnose] wrote {output_dir / 'variance_trajectory.json'}")

    # Plot.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cmap = plt.get_cmap("viridis")
    scales = cfg.steering.scale_sweep
    x = np.arange(n_layers + 1)
    site_labels = ["emb"] + [f"L{i}" for i in range(n_layers)]

    for i, scale in enumerate(scales):
        color = cmap(i / max(len(scales) - 1, 1))
        ax1.plot(x, traj_all[str(scale)], "-o", color=color, label=f"scale={scale}", markersize=4)
        ax2.plot(x, traj_last[str(scale)], "-o", color=color, label=f"scale={scale}", markersize=4)

    first_steered = min(cfg.steering.target_layers) + 1  # +1 because site 0 is embedding
    last_steered = max(cfg.steering.target_layers) + 1
    for ax in (ax1, ax2):
        ax.axvspan(first_steered - 0.4, last_steered + 0.4, alpha=0.15, color="red",
                   label=f"steered layers [{min(cfg.steering.target_layers)}..{max(cfg.steering.target_layers)}]")
        ax.set_xlabel("site  (0 = embedding output, L_k = layer k output)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    ax1.set_ylabel("tr(Σ)  (sum of per-coord variance)")
    ax1.set_title(f"{cfg.run_name} — all real tokens")
    ax2.set_title(f"{cfg.run_name} — last real token only")

    fig.tight_layout()
    plot_path = output_dir / "variance_trajectory.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # Caption sidecar.
    caption_lines = [
        f"# Residual-stream variance trajectory — `{cfg.run_name}`",
        "",
        "Per-layer `tr(Σ)` (sum of per-coord variance) across the full",
        f"{n_layers}-layer stack, streaming over {args.num_prompts} "
        f"FineWeb-Edu prompts at every scale in the sweep. The red band",
        f"marks the steered-layer range "
        f"[{min(cfg.steering.target_layers)}..{max(cfg.steering.target_layers)}]",
        "— steering is added to the output of each layer in that range.",
        "",
        "**Left panel (all real tokens):** every non-padding token across",
        "every prompt folded into one `CheapMoments` per (layer, scale).",
        "Includes both content variance and across-position variance.",
        "",
        "**Right panel (last real token only):** one sample per prompt",
        "(the last real token). Strips out across-position structure.",
        "If the two panels agree in shape, position dependence isn't the",
        "source of the trajectory.",
        "",
        "**How to read the trajectory.** If the paper's theory were in",
        "force, `tr(Σ)` should DROP inside the steered band "
        "(each steered layer contracts variance via normalization in",
        "the next block) and stay dropped until the end. If subsequent",
        "attention + MLP blocks amplify variance, we'd see the trajectory",
        "REBOUND after the last steered layer, growing with steering scale.",
        "",
        "**Observed values — tr(Σ) at final layer:**",
        "",
        "| scale | all tokens | last token |",
        "|------:|-----------:|-----------:|",
    ]
    for scale in cfg.steering.scale_sweep:
        caption_lines.append(
            f"| {scale:g} | {traj_all[str(scale)][-1]:.4g} | {traj_last[str(scale)][-1]:.4g} |"
        )

    (plot_path.with_suffix(".md")).write_text("\n".join(caption_lines) + "\n")
    print(f"[diagnose] wrote {plot_path} + caption")
    return 0


if __name__ == "__main__":
    sys.exit(main())
