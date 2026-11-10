# Four numbers: live-smoke

| Model | Condition | Arm | Runs | Detection ↑ | False flags ↓ | Coverage ↑ | Cost / 100 Qs |
|---|---|---|---|---|---|---|---|
| deepseek/deepseek-v4.1-flash | normal | plain | 2 | 0% (n=1) | 0% (n=1) | 0% (n=2) | $0.057 |
| deepseek/deepseek-v4.1-flash | normal | trust-layer | 2 | 100% (n=1) | 0% (n=1) | 100% (n=2) | $0.044 |
| google/gemma-4-31b-it | normal | plain | 2 | 0% (n=1) | 0% (n=1) | 0% (n=2) | $0.085 |
| google/gemma-4-31b-it | normal | trust-layer | 2 | 100% (n=1) | 0% (n=1) | 100% (n=2) | $0.066 |

Judge cost (not in the cost column): $0.0000
