# Bounds-pipeline investigations

Archive of the one-off probes and data investigations that led to
specific pieces of production code in the bounds-verification pipeline.
Each script is runnable and each finding is cross-linked to the
production file it drove.

**Reproducibility is the point here.** These are kept around not as
historical curiosities but because they're the fastest way to
re-validate a finding if a dependency (`transformers`, `nnsight`, etc.)
changes, or if a future reader sees a number quoted in
`docs/bounds/README.md` and wonders where it came from.

## The chronological trail

### 01 — probe nnsight 0.6's decoder-layer API

**File:** [`01_probe_nnsight_api.py`](./01_probe_nnsight_api.py)

**Question:** When building `src/bounds/nnsight_runner.py`, what
do `layer.output`, `layer_norm.input`, `layer_norm.output`, and `.save()`
actually return on a decoder layer? Do in-place writes propagate? How do
we batch prompts in a single trace?

**Method:** Load `sshleifer/tiny-gpt2` on CPU, run four small probes
inside `lm.trace(...)` contexts, print shapes and diffs.

**Finding:** On GPT-2, `layer.output` is a tuple `(hidden_states, ...)`
and `.output[0]` unpacks the hidden state correctly. In-place
modifications at `.output[0][:]` propagate to the final layernorm. External
tokenization (dict input to `trace()`) is required to avoid nnsight's
`padding=True` kwarg collision.

**Downstream impact:** Directly shaped `src/bounds/nnsight_runner.py`'s
intervention helper. **Also the probe that missed a bug** — see the
docstring of `01_probe_nnsight_api.py`: the same `.output[0]` pattern is
a silent batch-indexing bug on Qwen2/Llama3 (which return a bare tensor,
not a tuple), and testing only on tiny-gpt2 gave false confidence. The
production fix `_is_tuple_output_architecture` in `nnsight_runner.py` now
dispatches on model architecture.

### 02 — probe nnsight 0.6's `generate` + intervention pattern

**File:** [`02_probe_nnsight_generate.py`](./02_probe_nnsight_generate.py)

**Question:** An `lm.trace()` context fires the intervention once
per forward pass. But `lm.generate()` runs many forward passes (one per
new token). How do we make the steering fire on every decode step, not
just the prompt-processing forward?

**Method:** On tiny-gpt2, generate 6 tokens from "The quick brown fox"
twice — once without intervention, once with a +1000 perturbation at the
last block inside a `tracer.all()` context. Compare decoded outputs.

**Finding:** `with tracer.all(): ...` is the correct pattern — the
intervention fires on every forward pass during generation. Baseline
decodes to `"...factors factors factors..."`, perturbed decodes to
`"...stairs stairs stairs..."`. Different, as expected.

**Downstream impact:** This pattern became the core of
`src/bounds/nnsight_runner.py::sample_from_steered_model`, which in turn
powers `scripts/bounds/01_verify_steering.py`'s side-by-side verification.
The "trust no silent steering" policy depends on this pattern being
correct.

### 03 — is FineWeb-Edu "sad" from Qwen's perspective? (Claim 7 diagnosis)

**File:** [`03_investigate_happy_data_alignment.py`](./03_investigate_happy_data_alignment.py)

**Question:** After the batch-indexing fix, the re-run of Experiment 1
showed Claim 7 (`‖μ+s‖ > ‖μ‖`) **failing at every scale** for
`qwen_happy`, while passing everywhere else. Matthew's hypothesis: maybe
FineWeb-Edu is "sad" from Qwen's perspective, so adding the happy vector
pushes μ sideways rather than elongating it.

**Method:**
1. Compute `cos(μ_unsteered, Σ happy_layers)` directly from
   `bounds_qwen_happy/stats.pt`.
2. Project `μ_unsteered` through Qwen's final RMSNorm + `lm_head` and
   look at the top 15 tokens the projection upweights. Do the same for
   the happy direction. If the data is "sad" we should see sad-valence
   tokens for μ.

**Finding:**
- `cos(μ_unsteered, Σ happy[10..25]) = −0.108`, ≈ 4σ from random
  (random unit vectors in 1536-d have `E[cos] ≈ 0.026`). Mildly negative
  and **systematic** — 12 of 16 per-layer cosines are negative.
- **The data is NOT sad — it is tone-neutral and frequency-dominated.**
  μ's top 15 tokens are all function words and punctuation (`,`, `and`,
  `in`, `to`, `of`, …) with zero emotional valence.
- The happy direction cleanly upweights positive-valence tokens
  (`delighted`, `exciting`, …) and downweights sad ones (`grief`,
  `tragedy`, `depression`), so the vector itself is aimed correctly.

**Conclusion:** Sentiment is near-orthogonal to what dominates the mean
residual. The happy direction is *mildly* anti-aligned with the
mean residual for Qwen + FineWeb-Edu specifically (educational prose
trending serious-to-neutral), just enough that adding it at small scales
decreases `‖μ + s_eff‖`. At very large scales (extended pilot, scale
128) Claim 7 eventually flips to passing as `‖s_eff‖²` dominates
`|2 μ·s_eff|`.

**Downstream impact:** The `−0.108` number is the one quoted in
[`docs/bounds/README.md`](../../../docs/bounds/README.md) in the Claim 7
paragraph. Re-run this probe to re-produce it. The finding is also
saved as a memory note (`project_bounds_hold_but_loose.md`).

### 04 — real vs. random aggregate steering magnitude

**File:** [`04_compare_real_vs_random_norms.py`](./04_compare_real_vs_random_norms.py)

**Question:** Matthew asked: is the random-vector control effectively
*shorter* than the real vector at matched scales? Each per-layer random
vector has `‖r_i‖ = 1` (same as the real per-layer norm), but if the
aggregate `‖Σ r_i‖` is smaller than `‖Σ h_i‖` because random directions
don't stack coherently, then the random control is applying *less total
push* at the same nominal scale — which would confound "direction
matters" with "magnitude matters."

**Method:** For both Qwen+happy and Llama+creativity, load the real
GGUF vectors, generate the random equivalents, compute per-layer norms
and aggregate norms, and cross-check with the empirical `‖s_eff‖` at
the final RMSNorm from `stats.pt`.

**Finding:**
- Per-layer norms match exactly (all 1.0).
- Aggregate norms differ by ~3×: `‖Σ happy‖ = 11.27` vs `‖Σ random‖ =
  4.05` (close to `√16 = 4.00`, as predicted by iid unit vectors in
  high dimension). `‖Σ create‖ = 11.86` vs `‖Σ random‖ = 3.74` (close
  to `√14 = 3.74`).
- Effective `‖s_eff‖` at the final site tracks the aggregate ratio:
  at scale 8, qwen_happy has 124.22 while qwen_random has 38.38 —
  ratio ~3.24.

**Downstream impact:** Direct motivation for
`src/bounds/random_vectors.py::generate_random_steering_vector_aggregate_matched`
and the 4 `test_aggregate_matched_*` tests in
`tests/bounds/test_random_vectors.py`. That function rescales per-layer
random vectors uniformly so `‖Σ r_i‖` matches `‖Σ h_i‖` — the
aggregate-matched control used by Experiment 2's
`qwen_random_agg_matched` and `llama_random_agg_matched` runs.

## How to run

All four scripts are standalone:

```bash
# Scripts 01 and 02 — CPU only, tiny-gpt2, <5s each
uv run python scripts/bounds/investigations/01_probe_nnsight_api.py
uv run python scripts/bounds/investigations/02_probe_nnsight_generate.py

# Scripts 03 and 04 — require post-fix bounds_*/stats.pt (gitignored)
# Regenerate stats.pt first via scripts/bounds/02_record_stats.py (GPU)
uv run python scripts/bounds/investigations/03_investigate_happy_data_alignment.py
uv run python scripts/bounds/investigations/04_compare_real_vs_random_norms.py
```

Scripts 03 and 04 will print a clear error and exit 2 if the required
`stats.pt` files are missing locally.
