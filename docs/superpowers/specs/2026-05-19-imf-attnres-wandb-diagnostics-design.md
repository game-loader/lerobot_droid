# iMF-AttnRes WandB Diagnostics Design

## Goal
Add lightweight per-training-step diagnostics for IMF velocity terms, spike/non-spike buckets, AttnRes depth attention behavior, and module gradient norms so variance spikes can be inspected in WandB.

## Spike Definition
A training step is a spike when the scalar training loss for that step is greater than `0.2`. Non-spike is `loss <= 0.2`.

The threshold is configurable with `policy.imf_diagnostics_spike_loss_threshold`; the default remains `0.2`.

## Enable / Disable
Diagnostics are explicit opt-in through `policy.enable_imf_diagnostics`. The default is `false` to avoid noisy every-step WandB logs and extra scalar computation during normal training.

Enable with:

```bash
--policy.enable_imf_diagnostics=true \
--policy.imf_diagnostics_spike_loss_threshold=0.2
```

## Architecture
The policy computes scalar diagnostics inside `IMFAttnResModel.compute_loss()` where `v*`, `uθ`, `Dt uθ`, `t`, and `r` are already available. `IMFAttnResPolicy.forward()` returns these diagnostics in `output_dict`, preserving existing scalar loss behavior. The AttnRes operator caches detached depth-attention weights during forward; the model summarizes entropy and max depth attention after loss construction.

`lerobot_train.update_policy()` adds post-backward gradient norm diagnostics before the optimizer step. These values are merged into the same `output_dict` that the existing WandB logging path already emits.

## Training Loss
IMF-AttnRes uses pseudo-Huber velocity loss by default:

```python
delta**2 * (sqrt(1 + ((compound_velocity - target) / delta)**2) - 1)
```

The default `policy.pseudo_huber_delta` is `1.0`. The previous MSE objective remains available for ablations with `--policy.loss_type=mse`.

## Metrics
For IMF quantities, log all/spike/non_spike aggregate scalar stats using keys under `imf_diagnostics/`, including mean/std/max/count where appropriate for:
- `target_norm` = `||v*|| = ||e - x||`
- `u_norm` = `||uθ||`
- `du_dt_norm` = `||Dt uθ||`
- `delta_du_dt_norm` = `||(t-r)Dt uθ||`
- `delta`, `t`, `r`

For AttnRes depth attention, log entropy and max source weight summaries under `imf_diagnostics/attnres/`.

For gradients, log:
- `grad_norm/total`
- `grad_norm/attnres`
- `grad_norm/main_dit`

## Constraints
Only scalar values are returned because the repository WandB wrapper ignores non-scalar values. The implementation avoids changing model outputs, optimizer behavior, loss semantics, or inference.
