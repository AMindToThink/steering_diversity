"""Broad-text prompt loader for the bounds-verification pipeline.

Loads ``HuggingFaceFW/fineweb-edu`` in streaming mode so we don't have to
download the full 1.3T-token dataset — we only take the first ``num_prompts``
non-empty documents, which is enough for the 1000-prompt × 256-token sweeps
the bounds experiments need.

Fails loudly (per the project's error-handling policy) if the stream yields
fewer documents than requested, rather than silently returning a shorter list.
"""

from __future__ import annotations

from datasets import load_dataset


DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"


def load_fineweb_prompts(
    num_prompts: int,
    seed: int = 0,
    dataset_name: str = DEFAULT_DATASET,
    max_chars: int = 1024,
) -> list[str]:
    """Stream ``num_prompts`` non-empty documents from a HuggingFace dataset.

    Parameters
    ----------
    num_prompts : int
        Number of prompts to return. The function raises if fewer are
        available (e.g. if the stream errors out early).
    seed : int
        Currently unused — streaming fineweb-edu is deterministic in its
        iteration order, so ``num_prompts=1000`` always returns the same
        1000 documents. The parameter is kept for API consistency with the
        other bounds modules in case we later switch to shuffled sampling.
    dataset_name : str
        Override for the HuggingFace dataset path. Defaults to
        ``HuggingFaceFW/fineweb-edu``.
    max_chars : int
        Per-document character truncation applied before returning. Keeps
        the tokenizer from choking on very long documents and matches the
        typical residual-stream capture window of ~256 tokens.

    Returns
    -------
    list[str]
        Exactly ``num_prompts`` non-empty strings, each truncated to
        ``max_chars`` characters.
    """
    if num_prompts <= 0:
        raise ValueError(f"num_prompts must be positive, got {num_prompts}")

    ds = load_dataset(dataset_name, split="train", streaming=True)

    prompts: list[str] = []
    for ex in ds:
        text = (ex.get("text") or "").strip()
        if not text:
            continue
        prompts.append(text[:max_chars])
        if len(prompts) >= num_prompts:
            break

    if len(prompts) < num_prompts:
        raise RuntimeError(
            f"{dataset_name} yielded only {len(prompts)} / {num_prompts} "
            "non-empty documents. This is usually a network/availability "
            "issue, not a logic bug — try again or pick a different dataset."
        )

    return prompts
