"""Seeding, I/O helpers, and shared utilities."""

from __future__ import annotations

import dataclasses
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of dicts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def load_contrastive_pairs(path: str | Path) -> list[dict[str, str]]:
    """Load contrastive pairs JSON (list of {positive, negative})."""
    with open(path) as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _git_head_sha() -> str:
    """Return current HEAD sha, or empty string if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def save_provenance(
    step: str,
    config_path: str,
    cfg: Any,
    inputs: dict[str, str],
    outputs: list[str],
) -> None:
    """Write a .provenance.json sidecar next to each output file.

    Parameters
    ----------
    step:
        Script name, e.g. ``"01_compute_steering_vector"``.
    config_path:
        Path to the YAML config used for this run.
    cfg:
        An ``ExperimentConfig`` dataclass instance (serialized via
        ``dataclasses.asdict``).
    inputs:
        Named input paths, e.g. ``{"steering_vector": "outputs/.../x.gguf"}``.
    outputs:
        List of output paths produced by this step.
    """
    record = {
        "step": step,
        "config_path": str(config_path),
        "config_snapshot": dataclasses.asdict(cfg),
        "inputs": inputs,
        "outputs": [str(o) for o in outputs],
        "git_commit": _git_head_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    for output_path in outputs:
        sidecar = Path(str(output_path) + ".provenance.json")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with open(sidecar, "w") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
