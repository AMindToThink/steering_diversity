"""Phase 3 of experiment 2 — generate config YAMLs for all 14 runs.

Reads ``experiment2_layer_picks.json`` and writes:

- 2 aggregate-matched random configs (one per model)
- 12 single-layer configs (3 layers × 2 vectors × 2 models)

All configs go to ``configs/bounds/experiment2/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Static per-model settings — pulled from the original qwen_happy.yaml /
# llama_creativity.yaml so we stay consistent with prior results.
# ---------------------------------------------------------------------------

QWEN = {
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "model_type": "qwen2.5",
    "happy_vector": "EasySteer/vectors/happy_diffmean.gguf",
    "happy_target_layers": list(range(10, 26)),  # 10..25
    "batch_size": 8,
    "max_seq_len": 256,
    "num_prompts": 1000,
    "verification_prompts": [
        "Today I am feeling",
        "Describe in a few words how you feel about your life:",
        "My mood right now is",
        "Write a short diary entry about your day:",
    ],
    "verification_scale": 4.0,
    "auto_escalate_scales": [4.0, 8.0, 16.0],
}

LLAMA = {
    "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
    "model_type": "llama3",
    "creativity_vector": "EasySteer/replications/creative_writing/create.gguf",
    "creativity_target_layers": list(range(16, 30)),  # 16..29
    "batch_size": 4,
    "max_seq_len": 256,
    "num_prompts": 1000,
    "verification_prompts": [
        "Once upon a time, in a forest deep,",
        "Write the opening of a mystery novel:",
        "Describe an alien landscape:",
        "Tell me a story about a clockmaker:",
    ],
    "verification_scale": 4.0,
    "auto_escalate_scales": [4.0, 8.0],
}


# ---------------------------------------------------------------------------
# Config templates
# ---------------------------------------------------------------------------


def _base(run_name: str, model_meta: dict, target_layers: list[int],
          verification_scale: float) -> dict:
    return {
        "run_name": run_name,
        "seed": 0,
        "model": {
            "name": model_meta["model_name"],
            "model_type": model_meta["model_type"],
            "gpu_memory_utilization": 0.9,
        },
        "steering": {
            "target_layers": target_layers,
            "scale_sweep": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
            # filled in by specific builders:
            # vector_path  OR  random_reference_path + random_seed + random_match
        },
        "dataset": {
            "name": "HuggingFaceFW/fineweb-edu",
            "num_prompts": model_meta["num_prompts"],
            "max_seq_len": model_meta["max_seq_len"],
            "batch_size": model_meta["batch_size"],
        },
        "capture_specs": [{"site": "final", "tier": "full"}],
        "steering_verification": {
            "enabled": True,
            "sample_prompts": model_meta["verification_prompts"],
            "verification_scale": verification_scale,
            "kl_threshold": 0.05,
            "magnitude_ratio_threshold": 0.02,
            "cosine_threshold": 0.1,
            "auto_escalate_scales": model_meta["auto_escalate_scales"],
            "max_new_tokens": 32,
            "do_sample": True,
            "sample_temperature": 1.0,
        },
        "reservoir_size": 1024,
        "dtype": "bfloat16",
    }


def build_agg_matched_config(model_meta: dict, vector_path: str,
                             target_layers: list[int], run_name: str) -> dict:
    cfg = _base(run_name, model_meta, target_layers, verification_scale=4.0)
    cfg["steering"]["random_reference_path"] = vector_path
    cfg["steering"]["random_seed"] = 0
    cfg["steering"]["random_match"] = "aggregate"
    return cfg


def build_single_layer_real_config(
    model_meta: dict, vector_path: str, layer: int, run_name: str
) -> dict:
    cfg = _base(run_name, model_meta, target_layers=[layer], verification_scale=4.0)
    cfg["steering"]["vector_path"] = vector_path
    return cfg


def build_single_layer_random_config(
    model_meta: dict, vector_path: str, layer: int, run_name: str
) -> dict:
    cfg = _base(run_name, model_meta, target_layers=[layer], verification_scale=4.0)
    cfg["steering"]["random_reference_path"] = vector_path
    cfg["steering"]["random_seed"] = 0
    cfg["steering"]["random_match"] = "per_layer"  # trivially == aggregate for 1 layer
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picks", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    picks = json.loads(args.picks.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    # --- Aggregate-matched (2 configs) -------------------------------------
    cfg_qr = build_agg_matched_config(
        QWEN, QWEN["happy_vector"],
        target_layers=QWEN["happy_target_layers"],
        run_name="bounds_qwen_random_agg_matched",
    )
    p = args.out_dir / "qwen_random_agg_matched.yaml"
    p.write_text(yaml.safe_dump(cfg_qr, sort_keys=False))
    written.append(p)

    cfg_lr = build_agg_matched_config(
        LLAMA, LLAMA["creativity_vector"],
        target_layers=LLAMA["creativity_target_layers"],
        run_name="bounds_llama_random_agg_matched",
    )
    p = args.out_dir / "llama_random_agg_matched.yaml"
    p.write_text(yaml.safe_dump(cfg_lr, sort_keys=False))
    written.append(p)

    # --- Single-layer sweep (12 configs) ------------------------------------
    for layer in picks["qwen"]["single_layer_picks"]:
        real = build_single_layer_real_config(
            QWEN, QWEN["happy_vector"], layer,
            run_name=f"bounds_qwen_happy_single_L{layer}",
        )
        p = args.out_dir / f"qwen_happy_single_L{layer}.yaml"
        p.write_text(yaml.safe_dump(real, sort_keys=False))
        written.append(p)

        rnd = build_single_layer_random_config(
            QWEN, QWEN["happy_vector"], layer,
            run_name=f"bounds_qwen_random_single_L{layer}",
        )
        p = args.out_dir / f"qwen_random_single_L{layer}.yaml"
        p.write_text(yaml.safe_dump(rnd, sort_keys=False))
        written.append(p)

    for layer in picks["llama"]["single_layer_picks"]:
        real = build_single_layer_real_config(
            LLAMA, LLAMA["creativity_vector"], layer,
            run_name=f"bounds_llama_creativity_single_L{layer}",
        )
        p = args.out_dir / f"llama_creativity_single_L{layer}.yaml"
        p.write_text(yaml.safe_dump(real, sort_keys=False))
        written.append(p)

        rnd = build_single_layer_random_config(
            LLAMA, LLAMA["creativity_vector"], layer,
            run_name=f"bounds_llama_random_single_L{layer}",
        )
        p = args.out_dir / f"llama_random_single_L{layer}.yaml"
        p.write_text(yaml.safe_dump(rnd, sort_keys=False))
        written.append(p)

    print(f"Wrote {len(written)} configs to {args.out_dir}:")
    for p in written:
        print(f"  {p.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
