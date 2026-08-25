# Claims

## C01: A stale PPO reference converts local policy updates into cumulative diffusion-ratio drift
- **Statement**: When a diffusion-policy behavior snapshot remains fixed across successive actor updates, its likelihood ratio measures cumulative policy drift rather than one update's local change, so clipping does not keep later updates close to the current behavior policy.
- **Conditions**: Offline state-conditioned action-chunk PPO using old-policy sampled denoising traces and one joint likelihood ratio per denoising transition; OPE-gated behavior-policy promotion remains untested in this implementation.
- **Sources**: []
- **Status**: testing
- **Provenance**: ai-suggested
- **Falsification**: Under matched data, optimizer, and stochastic seeds, a fixed-reference run maintains the same bounded ratio and KL behavior as a per-update synchronized reference over an extended actor run.
- **Proof**: [N02, N04, N05, `ara/evidence/tables/ppo-stability-smoke.md`]
- **Dependencies**: []
- **Tags**: diffusion-ppo, snapshot, kl, action-chunk

## C02: Decoupled likelihood variance controls tail-ratio sensitivity without changing rollout samples
- **Statement**: Decoupling DDIM rollout sampling variance from surrogate likelihood variance preserves the sampled transition while reducing the sensitivity of diffusion-PPO ratios to near-deterministic denoising steps.
- **Conditions**: Old and new policies replay the same stored transition with the same probability variance; current evidence covers a deterministic trace regression and a short matched fixed-reference CUDA smoke, not long-horizon policy quality.
- **Sources**: []
- **Status**: supported
- **Provenance**: user
- **Falsification**: Under a fixed model, observation, seed, and sampling schedule, changing only the probability floor changes sampled latents or actions, or a matched fixed-reference run shows no reduction in tail-ratio and KL sensitivity.
- **Proof**: [N08, `ara/evidence/tables/ppo-stability-smoke.md`, `tests/rl100/test_ddim_trace.py`]
- **Dependencies**: []
- **Tags**: diffusion-ppo, ddim, likelihood, sigma, ratio

## C03: Immediate snapshot synchronization makes pre-update PPO diagnostics structurally trivial
- **Statement**: When the behavior snapshot equals the current policy during the only likelihood evaluation and is synchronized immediately after the optimizer step, pre-update likelihood ratios cannot measure that step and clipping is inactive for the update.
- **Conditions**: Offline diffusion-policy training with a fresh behavior trace per batch, one optimizer evaluation per snapshot, and metrics derived from likelihood tensors captured before the optimizer step.
- **Sources**: []
- **Status**: testing
- **Provenance**: ai-suggested
- **Falsification**: Under the stated ordering, a recorded pre-update row contains a non-unit old/new ratio caused by the optimizer step it accompanies, or clipping changes that step's objective despite identical old and current policies at evaluation time.
- **Proof**: [N12, `ara/evidence/tables/ppo-stability-smoke.md`, `RL/trainers/offline.py`, `RL/algorithms/ppo.py`]
- **Dependencies**: [C01]
- **Tags**: diffusion-ppo, snapshot, observability, clipping, kl

## C04: IQL advantage and AM-Q policy promotion are distinct parts of RL-100
- **Statement**: RL-100 uses an IQL Q-minus-V estimate to weight actor updates, while AM-Q separately evaluates candidate and behavior policies through learned dynamics and gates behavior-snapshot promotion; the actor advantage alone does not implement AM-Q.
- **Conditions**: The iterative offline policy-improvement stage described by the RL-100 paper and represented by its offline BPPO and dynamics rollout code.
- **Sources**: []
- **Status**: testing
- **Provenance**: ai-suggested
- **Falsification**: The paper or executable original training path uses AM-Q directly as the per-sample actor advantage, or promotes behavior policy without a model-Q policy-evaluation comparison.
- **Proof**: [N14, `ara/evidence/tables/rl100-offline-parity-audit.md`, `arXiv:2510.14830`]
- **Dependencies**: []
- **Tags**: rl100, iql, am-q, ope, behavior-policy

## C05: Real-environment checkpoint selection must remain outcome-driven
- **Statement**: When model-based policy improvement and environment performance are not guaranteed to be monotonic, subsequent data collection should use the highest measured real-environment checkpoint rather than the final training state.
- **Conditions**: Iterative offline learning with periodic real-environment evaluations and a cumulative rollout dataset; the current evidence covers one completed promotion run and its first iterative collection.
- **Sources**: []
- **Status**: testing
- **Provenance**: ai-suggested
- **Falsification**: Across repeated promotion runs, the final checkpoint consistently matches or exceeds every earlier evaluated checkpoint and outcome-driven selection does not change the collected-data success rate.
- **Proof**: [N33, N42, `ara/evidence/tables/iterative-il2-launch.md`]
- **Dependencies**: [C04]
- **Tags**: rl100, checkpoint-selection, evaluation, dataset-expansion
