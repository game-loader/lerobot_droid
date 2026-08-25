# PPO Stability Smoke Evidence

## Historical run with fixed old policy and actor LR 1e-5

Source: `outputs/rl100/offline_moya_il_online_20260818-141706/metrics.jsonl`

| Actor update | Ratio mean | Approx KL | Clip fraction |
| ---: | ---: | ---: | ---: |
| 1 | 1.000000 | 0.000000 | 0.000000 |
| 2 | 1.558771 | 2.342053 | 0.753125 |
| 10 | 0.824997 | 24.820961 | 0.900000 |
| 12 | 0.916150 | 27.037396 | 0.915625 |

## CUDA diagnostic with per-update synchronization and actor LR 1e-6

Source: `/tmp/rl100-diag-10step-l3zIng/metrics.jsonl`

Across actor updates 1 through 3, ratio mean/q05/q50/q95/max remained 1,
approx KL and clip fraction remained 0, snapshot sync age remained 0, and
old-policy replay maximum absolute error remained 0.

## CUDA diagnostic with fixed old policy and actor LR 1e-6

Source: `/tmp/rl100-diag-fixed-old-wIoCYh/metrics.jsonl`

| Actor update | Ratio q05 | Ratio q50 | Ratio q95 | Ratio max | Approx KL | Clip fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| 2 | 0.773236 | 1.004542 | 1.230827 | 1.567948 | 0.027420 | 0.150000 |
| 3 | 0.543480 | 0.986441 | 1.430237 | 2.499312 | 0.092152 | 0.225000 |

At update 3, the first eight denoising transitions used effective sigma 0.1.
The penultimate transition used 0.0248936 and ratio q95 1.78894. The final
transition used 0.0067 and ratio q95 2.30691, the widest upper tail in the
schedule.

## Matched CUDA A/B with separate probability sigma

Sources:

- Shared sampling/probability sigma: `/tmp/rl100-diag-fixed-old-wIoCYh/metrics.jsonl`
- Probability sigma floored at 0.1: `/tmp/rl100-prob-sigma-0qZWnR/metrics.jsonl`

Both runs used seed 17, batch size 4, actor LR 1e-6, 10 DDIM transitions,
a fixed old-policy reference, and three actor updates. Only the probability
sigma floor differed.

| Update 3 metric | Shared sigma | Probability sigma >= 0.1 |
| --- | ---: | ---: |
| Approx KL | 0.092152 | 0.005444 |
| Ratio q95 | 1.430237 | 1.119735 |
| Ratio max | 2.499312 | 1.151274 |
| Clip fraction | 0.225000 | 0.075000 |
| Penultimate-step ratio q95 | 1.788941 | 1.032729 |
| Final-step ratio q95 | 2.306914 | 1.000485 |

The final transition retained sampling sigma 0.0067 while its likelihood used
probability sigma 0.1. The deterministic regression in
`tests/rl100/test_ddim_trace.py::test_probability_sigma_does_not_change_sampled_trace`
verified bitwise-identical latents, next latents, and final actions when only
the probability floor changed.

## Immediate-sync production metric audit

Source: `outputs/rl100/offline-moya-il-dualsigma-50k-20260818-195725/metrics.jsonl`

The interrupted run completed 100,000 IQL updates and 30,165 actor updates.
Across every actor row:

- ratio mean/q05/q95/max were exactly 1;
- approximate KL, clip fraction, delta log-probability, and snapshot age were exactly 0;
- old and new joint log-probability means were equal within each row;
- actor loss ranged from -1.3411045e-7 to 1.7136335e-7, with 13,211 negative,
  13,288 positive, and 3,666 exactly-zero rows.

Code inspection established that the recorded likelihoods are both computed
before `optimizer.step()`. The policy is then updated and immediately copied
into the old-policy snapshot before the pre-update tensors are aggregated.
Consequently these ratio diagnostics cannot observe the update they accompany.
