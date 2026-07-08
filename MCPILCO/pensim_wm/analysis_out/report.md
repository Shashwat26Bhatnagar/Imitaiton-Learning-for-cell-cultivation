# PenSim sub-optimal trajectory analysis

- Episodes: **2**
- Transitions: **2300**
- Observation dim: 9, Action dim: 6

## Yield summary

- Yield-per-step: mean 3.155, std 1.426, range [0, 6.567]
- Total yield per episode: mean 3628, std 155.2, range [3473, 3783]
- Documented baseline batch yield: ~3640 kg (reference for suboptimality)

## Per-observation-channel statistics (raw units)

| channel | min | max | mean | std | median |
|---|---|---|---|---|---|
| obs_0 | 0.2 | 230 | 115.1 | 66.4 | 115.1 |
| obs_1 | 6.124 | 6.638 | 6.498 | 0.03048 | 6.501 |
| obs_2 | 297.8 | 299.8 | 298 | 0.1961 | 298 |
| obs_3 | 0 | 4.679 | 0.02573 | 0.2293 | 0 |
| obs_4 | 0 | 225 | 106.1 | 40.21 | 103.2 |
| obs_5 | 0.0001 | 719.1 | 90.26 | 127.9 | 35.55 |
| obs_6 | 0.0001 | 472.6 | 20.68 | 43.39 | 0.6473 |
| obs_7 | 6.251e+04 | 1.015e+05 | 8.737e+04 | 1.236e+04 | 9.342e+04 |
| obs_8 | 9.172 | 19.39 | 14.79 | 1.576 | 14.97 |

> Channel ranges span ~0.514 to ~3.9e+04 (ratio ~7.59e+04x) -- motivates per-dimension standardization before GP fitting.

## Per-action statistics (raw units)

| action | min | max | mean | std | median |
|---|---|---|---|---|---|
| act_0 | 0 | 4095 | 199.9 | 852.7 | 0 |
| act_1 | 7.361 | 146.9 | 76.31 | 23.61 | 81.31 |
| act_2 | 21.68 | 35.8 | 26.11 | 4.709 | 24.38 |
| act_3 | 29.85 | 73.81 | 64.25 | 10.23 | 69.77 |
| act_4 | 0.587 | 1.125 | 0.932 | 0.1327 | 0.915 |
| act_5 | 0 | 503.8 | 154.2 | 150.1 | 107.9 |

## Figures

- `yield_per_step.png`
- `cumulative_yield.png`
- `obs_trajectories.png`
- `obs_correlation.png`
- `action_ranges.png`
- `yield_hist.png`
