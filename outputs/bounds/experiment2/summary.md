# Experiment 2 summary

## Runs

| run | description | slope V | slope tr(Σ) | Claim 7 |
|---|---|---:|---:|:---:|
| `bounds_llama_creativity_single_L16` | REAL L16 | -0.060 | -0.036 | mixed |
| `bounds_llama_creativity_single_L22` | REAL L22 | -0.055 | -0.033 | all pass |
| `bounds_llama_creativity_single_L29` | REAL L29 | -0.029 | -0.017 | all pass |
| `bounds_llama_random_agg_matched` | RAND(aggregate) L16-29 | -1.851 | -1.710 | all pass |
| `bounds_llama_random_single_L16` | RAND(per_layer) L16 | -0.039 | -0.023 | all pass |
| `bounds_llama_random_single_L22` | RAND(per_layer) L22 | -0.043 | -0.026 | all pass |
| `bounds_llama_random_single_L29` | RAND(per_layer) L29 | -0.031 | -0.018 | all pass |
| `bounds_qwen_happy_single_L10` | REAL L10 | -0.004 | -0.004 | all pass |
| `bounds_qwen_happy_single_L17` | REAL L17 | +0.005 | +0.004 | all fail |
| `bounds_qwen_happy_single_L25` | REAL L25 | +0.003 | +0.002 | all fail |
| `bounds_qwen_random_agg_matched` | RAND(aggregate) L10-25 | -0.196 | -0.172 | all pass |
| `bounds_qwen_random_single_L10` | RAND(per_layer) L10 | -0.004 | -0.003 | all pass |
| `bounds_qwen_random_single_L17` | RAND(per_layer) L17 | +0.003 | +0.002 | all fail |
| `bounds_qwen_random_single_L25` | RAND(per_layer) L25 | +0.000 | +0.000 | mixed |

Expected slopes per paper: slope_V = −1, slope_trZ = −2.
