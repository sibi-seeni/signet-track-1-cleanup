# Four numbers: live-run

| Model | Condition | Arm | Runs | Detection ↑ | False flags ↓ | Coverage ↑ | Cost / 100 Qs |
|---|---|---|---|---|---|---|---|
| deepseek/deepseek-v4.1-flash | normal | plain | 240 | 0% (n=120) | 0% (n=120) | 0% (n=240) | $0.165 |
| deepseek/deepseek-v4.1-flash | normal | trust-layer | 240 | 100% (n=120) | 0% (n=120) | 100% (n=240) | $0.039 |
| google/gemma-4-31b-it | normal | plain | 240 | 0% (n=120) | 0% (n=120) | 0% (n=240) | $0.070 |
| google/gemma-4-31b-it | normal | trust-layer | 240 | 100% (n=120) | 0% (n=120) | 100% (n=240) | $0.058 |

Judge cost (not in the cost column): $0.0000
