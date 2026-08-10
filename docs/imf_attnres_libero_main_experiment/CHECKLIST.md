# IMF-AttnRes LIBERO Checklist

- [x] Confirm Table-2 style protocol: four LIBERO suites, 10 tasks each, success rate.
- [x] Materialize physical subset datasets to avoid EpisodeAwareSampler global-index issue.
- [x] Verify smoke run succeeds.
- [x] Launch local `spatial -> object` driver on RTX 5090.
- [x] Launch remote `goal` and `long` drivers on dual RTX 5880.
- [x] Confirm `spatial` checkpoint 005000 and periodic rollout10-per-task eval succeed.
- [x] Complete `spatial` training/eval/final eval.
- [x] Complete `object` training/eval/final eval.
- [x] Complete `goal` training/eval/final eval.
- [x] Complete `long` training/eval/final eval.
- [x] Copy remote results back locally.
- [x] Produce final LIBERO success-rate table.
