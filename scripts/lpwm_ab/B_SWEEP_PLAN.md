# B-only world-loss strength and reconstruction/dynamics balance sweep

User-authorized experiment plan, 2026-09-18. New runs only; existing A/B weights and
local FR3 SmolVLA/IMF jobs are not modified. All candidate effects are hypotheses.

## Question and estimand

How does factual world-model supervision of the shared DLP visual encoder change
LIBERO Spatial closed-loop success? First vary its overall strength; subsequently
vary the reconstruction/dynamics coefficient balance at the selected strength.

`loss = FM + scheduled_world_weight * (rec_weight * normalized_reconstruction
                                      + 0.001 * normalized_particle_prior
                                      + dyn_weight * normalized_dynamic_KL)`

This is a coefficient-controlled experiment, not a claim that equal coefficients
produce equal loss values or equal encoder gradients. Log each loss and the shared
encoder's FM-versus-weighted-world gradient norm ratio/cosine to inspect that distinction.
Keep native reduction/normalization fixed; no arbitrary new loss rescaling.

## Common controls

- Variant B, native trainable DLP, ordinary FM expert, GT clean dynamics actions only.
- Three repeated action condition tokens; no AdaLN, IMF/JVP, action-latent alignment,
  policy-conditioned future targets or free-running world-rollout loss.
- All runs start from the same seed42 random model initialization (NOT fine-tuned
  from the old20k models), frozen cached language features, exact same episode split.
- Full30000 optimizer updates, microbatch8 x accumulation4 = effective32.
- AdamW: peak LR1e-4, linear warmup500 updates, cosine decay to1e-5 at update30000;
  weight decay1e-6, gradient clipping10, original architecture and float32/TF32.
- World weight ramps from zero to target over1000 updates. Prior coefficient0.001.
- Image128, two cameras, history2, prediction horizon16, execution queue8, Euler10.
- Width256, scene2/expert4/world4 layers. One-step factual dynamics.
- Train-only state normalizer; native action identity; unchanged view/control mapping.
- Offline heldout loss every500; online SwanLab metrics every10; no telemetry secrets.

## Phase 1: world strength (four fresh runs, two rounds)

| Round | GPU | Run | world | rec | dyn | prior |
|---|---|---|---:|---:|---:|---:|
| 1 | 0 | w003 | 0.03 | 1 | 1 | 0.001 |
| 1 | 1 | w030 | 0.30 | 1 | 1 | 0.001 |
| 2 | 0 | w010 | 0.10 | 1 | 1 | 0.001 |
| 2 | 1 | w100 | 1.00 | 1 | 1 | 0.001 |

0.10 is a REQUIRED contemporaneous cosine/30k control. The old constant-LR20k B
run is historical context, not a matched control. 1.00 tests strong world supervision
rather than presuming stronger is better. Start round2 only after both round1 runs
and all scheduled rollouts complete successfully.

Selection: after all four30k runs and all six scheduled checkpoint rollouts complete,
maximize the mean validation success at20k,25k,30k. Break exact ties by30k success,
then by the smaller world weight. Best individual checkpoint is also reported but
must not replace the preregistered late-training selection score.

## Phase 2: rec/dyn coefficient balance (two fresh runs)

With the selected phase1 world weight, train:
- GPU0: rec0.5 / dyn1.5 / prior0.001.
- GPU1: rec1.5 / dyn0.5 / prior0.001.

Keep `rec+dyn=2` to avoid accidentally doubling total nominal coefficient strength.
Use the selected phase1 rec1/dyn1 run as the matched balanced control. Each new run
starts from scratch with the same seed/model/split and full30k cosine schedule.
This controls coefficients, NOT the empirical gradient share of rec versus dyn.

## Checkpoints and actual simulation evaluation

At5k,10k,15k,20k,25k,30k:
1. Write an immutable complete checkpoint (weights, config, normalizer, language
   cache metadata, optimizer/RNG and completion manifest). Retain ALL six/run.
2. Pause this trainer while its fresh child process runs actual LIBERO Spatial
   simulation on the same GPU; do not contend two training jobs on one GPU.
3. Ten tasks x ten episodes =100 per checkpoint, seed42, existing `validation`
   initial-state pool, paired across every candidate; max280 control steps, original
   evaluator camera/state/action transformations. Success denominator is episodes.
4. Record overall/per-task/per-episode success, protocol and model hashes, and
   SwanLab evaluation run. No selection from offline loss alone.
5. Fail closed if checkpoint integrity, rollout completion or required online
   recording fails. Never silently label an unevaluated run successful.

No final-pool initial states are used for hyperparameter selection. Independent final
pool evaluation and additional training seeds are follow-up validation, not automatic
claims from this one-seed tuning sweep. All36 planned checkpoint evaluations yield
3600 validation episodes; repeated use is for tuning, not independent generalization.

## Resource and provenance boundaries

Two remote L40S GPUs. Store all new runs/checkpoints/evaluations/logs under the user data disk
`/root/gpufree-data/lpwm_b_sweeps/20260918-102103-b-world-balance` (49GiB filesystem,
about49GiB free at preparation), not the30GiB system overlay. Existing old experiments
and environments stay unchanged. Historical B took about13.1h/20k including its prior overhead,
so a first-order30k estimate is about20h/run plus evaluation: the full three-round
sweep may span roughly2.5–3days, not a promised completion time.
Historical full checkpoint is about216MiB:36 retained checkpoints need about7.6GiB,
plus videos/logs/code and safety margin. Verify free space before launch and each
round; stop rather than deleting old checkpoints. Snapshot code/config/environment
and use an exclusive queue lock. No publishing/private asset uploads.

Progress and selection files are evidence of actual state. The plan alone does not
mean later phases have run, nor that any candidate improves success.
