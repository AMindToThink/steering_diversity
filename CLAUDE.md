# Steering Diversity — Project Instructions

## Overview

This project measures whether activation steering collapses the diversity of LLM outputs. It uses a 5-step pipeline: steering vector extraction → steered generation → embedding → clustering/metrics → visualization.

## Project Layout

- `src/` — Core library: config, generation, embedding, clustering, utils (provenance tracking)
- `scripts/01-05_*.py` — Pipeline steps, each a standalone CLI tool with `--config`, `--output`, and input override flags
- `configs/` — YAML experiment configs (`experiment1.yaml` = full, `experiment1_dev.yaml` = smoke test)
- `tests/` — Pytest suite (config, embedding, clustering, utils/provenance)
- `EasySteer/` — Git submodule (ZJU-REAL/EasySteer)

## Key Conventions

- **Package manager**: uv only. `uv run pytest`, `uv run python scripts/...`, `uv add <pkg>`.
- **Config-driven**: All experiment parameters live in YAML. Scripts derive defaults from `ExperimentConfig.output_dir` but accept explicit `--input`/`--output` overrides.
- **Provenance**: Every script calls `save_provenance()` after writing outputs, producing `.provenance.json` sidecar files.
- **Type hints**: Always use type hints.
- **Tests**: Run with `uv run pytest tests/ -v`. Tests must pass before committing.

## Pipeline Steps

| Step | Script | Requires GPU | Key I/O |
|------|--------|-------------|---------|
| 1 | `01_compute_steering_vector.py` | Yes | contrastive pairs → `.gguf` vector |
| 2 | `02_generate_responses.py` | Yes | vector + prompts → `responses.jsonl` |
| 3 | `03_embed_responses.py` | No | responses → `embeddings.npz` |
| 4 | `04_compute_metrics.py` | No | embeddings → `metrics.json` |
| 5 | `05_visualize.py` | No | embeddings + metrics → `plots/` |

## Steering: Mandatory vLLM Settings

When steering is enabled (via `--enable-steer-vector` or `--steer-vector-path`), you **MUST** set:

- `enable_chunked_prefill=False` (or `--no-enable-chunked-prefill`)
- `enable_prefix_caching=False` (or `--no-enable-prefix-caching`)

**Chunked prefill** is not supported by EasySteer's steering wrappers — it causes silent numerical errors.

**Prefix caching** keys on `steer_vector_name` but NOT scale — reusing KV states across different scales silently disables steering. This caused a real data-invalidation bug (commit `9b999cb`).

If you see logprob diffs > 1e-4 between modes that should be equivalent, check these flags FIRST.

## Synthetic Data Rules

Any plots generated from synthetic/fixture data **must** have a "DEMO" watermark. Never present synthetic outputs as real experiment results.
