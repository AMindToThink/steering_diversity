# Plan: Incremental/Resumable pass@k Runs

## Context

When running pass@k evaluations, we want to start with a small n (e.g., n=50) and later increase to a larger n (e.g., n=200) without re-generating the first 50 samples. For example, going from n=50 to n=200 only requires generating 150 new samples instead of 200 from scratch, saving 25% of GPU time.

**Good news:** EvalPlus codegen already has built-in resume support (on by default). If the `.jsonl` file exists, it counts existing samples per task and only generates the missing ones. So the codegen layer works out of the box.

**The problems are in our orchestrator (`eval_code.py`) and evaluation layer:**

1. **Stale eval_results.json**: EvalPlus evaluate caches results. If `_eval_results.json` exists from a previous n=10 run, it skips re-evaluation and returns the old (n=10) results. We need to force re-evaluation after appending samples.
2. **Stale pass_at_k.json**: Our `pass_at_k.json` and `pass_at_k_curves.json` from the old run would be overwritten, which is fine, but we should log that this is a resumed run.
3. **Sample count validation**: After codegen resumes, we should verify the `.jsonl` actually has `n_samples × n_problems` lines, not silently accept a partial file.
4. **k_values may change**: If we go from n=10 (k_values: [1,2,5,10]) to n=50 (k_values: [1,2,5,10,25,50]), the config changes. This is fine — pass_at_k.json just gets recomputed.

## Changes Required

### File: `scripts/eval/eval_code.py`

#### A. `run_evalplus_codegen` — add sample count logging

After codegen completes, count lines in the `.jsonl` to log how many were generated vs already existed. This makes resumed runs transparent.

```python
# After finding samples_path:
n_total = sum(1 for line in open(samples_path) if line.strip())
n_expected = n_samples * n_problems  # n_problems = 164 for humaneval
logger.info("Samples file has %d lines (expected %d)", n_total, n_expected)
```

We don't know `n_problems` at this point without loading the dataset, so just log the total count — we can validate later when eval_results come back.

#### B. `run_evalplus_evaluate` — delete stale eval_results before re-running

Before calling evalplus.evaluate, delete any existing `_eval_results.json` so it doesn't short-circuit with cached (old-n) results.

```python
# Before running evaluate, remove stale cached results
stale_results = samples_path.with_name(samples_path.stem + "_eval_results.json")
if stale_results.exists():
    logger.info("Removing stale eval results (will re-evaluate): %s", stale_results)
    stale_results.unlink()
```

**Why delete instead of `--i-just-wanna-run`?** The `i_just_wanna_run` flag prompts interactively ("Press Y/N to overwrite"), which doesn't work in our subprocess. Deleting the file is cleaner.

#### C. `run_single_condition` — log resume state

At the start, check if samples already exist and log the resume state:

```python
# Check for existing samples
existing_jsonl = list(output_dir.rglob("*.jsonl"))
existing_jsonl = [f for f in existing_jsonl if ".raw." not in f.name]
if existing_jsonl:
    n_existing = sum(1 for line in open(existing_jsonl[0]) if line.strip())
    logger.info("Found %d existing samples in %s — will resume", n_existing, existing_jsonl[0])
```

#### D. `run_single_condition` — validate final sample count

After evaluation, verify that the number of samples per problem matches `n_samples`:

```python
# After extracting per-problem results:
for n, c in per_problem_plus:
    if n != cfg.pass_at_k.n_samples:
        raise RuntimeError(
            f"Expected {cfg.pass_at_k.n_samples} samples per problem but got {n}. "
            f"The samples file may be corrupt or from an interrupted run."
        )
```

### File: `configs/eval_code.yaml` (and similar configs)

No changes needed — `n_samples` in the config represents the *target* total, not the number to generate this run. EvalPlus handles the math.

### No new files needed

All changes are in `eval_code.py`. The resume behavior is already in EvalPlus.

## Files to Modify

| File | Change |
|------|--------|
| `scripts/eval/eval_code.py` | A–D above: resume logging, stale eval deletion, sample validation |

## Verification

1. **Unit test not needed** — this is orchestration logic, tested by running the pipeline
2. **Manual test**:
   - Existing n=10 run is at `outputs/passk_test_steered/code/humaneval/scale_2.0_temp_0.8/`
   - Change config to `n_samples: 20` and re-run — should generate only 10 new samples per problem
   - Verify logs show "Found 1640 existing samples — will resume"
   - Verify eval_results.json has 20 samples per problem, not 10
   - Verify pass_at_k.json is recomputed with the new n=20
3. **Edge case**: Run with same n_samples as existing — should be a no-op for codegen, re-evaluate only
