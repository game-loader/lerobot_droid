# RL-100 Diffusion RL Migration Design

## Objective

Create an isolated `RL/` package that adapts the offline and online
reinforcement-learning stages from RL-100 to this repository's LeRobot
Diffusion Policy and Moya Newton task. The first complete path uses the
existing 39-dimensional state observation, 14-dimensional action, LeRobot v3
dataset, and CUDA vector environment. The data contracts also preserve
optional 2D image observations so a later image feature encoder can be added
without changing the algorithms or trainers.

The initial policy bundle is:

```text
outputs/train/moya_diffusion_300k_20260815-093537/train/checkpoints/080000/pretrained_model
```

It has `n_obs_steps=2`, `horizon=64`, `n_action_steps=32`, a DDPM-trained
noise predictor, and MIN_MAX state/action processors. RL training must load the
entire bundle, including both processors and their saved statistics.

## Source And Fidelity

The algorithm reference is RL-100, arXiv:2510.14830v4, and source commit
`7c5df9a5d3111e5fa8d2fe814c4fdcb3acfa3f26` under
`/home/droid/project/RL-100`. The migration retains these method-level ideas:

1. train an offline IQL critic and use `Q(s, a) - V(s)` as the advantage;
2. treat stochastic diffusion denoising transitions as sub-policies;
3. apply a clipped PPO likelihood-ratio objective to each denoising step;
4. use environment-level GAE and a value critic for online updates;
5. optionally train state dynamics and apply an offline policy-promotion gate;
6. keep old-policy traces fixed while recomputing differentiable current-policy
   likelihoods on the same denoising transitions.

The migration does not copy the RL-100 workspace wholesale. Its DP3/PointNet
policies, point-cloud schema, hard-coded `agent_pos` buffers, Hydra launchers,
MuJoCo runners, Python 3.8 environment, vendored packages, flow policy, and
one-step distillation are incompatible with this checkpoint and remain out of
scope. `RL/NOTICE` and the migration notes will retain upstream attribution and
identify rewritten and adapted behavior.

## Package Boundary

```text
RL/
  adapters/
    checkpoint.py
    lerobot_v3.py
    moya_newton.py
  algorithms/
    dynamics.py
    gae.py
    iql.py
    ppo.py
  policy/
    ddim.py
    diffusion_adapter.py
    observation_encoder.py
  trainers/
    offline.py
    online.py
  cli/
    inspect_dataset.py
    train_offline.py
    train_online.py
  checkpointing.py
  config.py
  types.py
  MIGRATION.md
  NOTICE
  README.md
```

The package stays outside `src/lerobot`. It uses public LeRobot dataset,
processor, policy, and environment entry points and does not change existing
BC training or evaluation behavior. Commands run as `uv run python -m
RL.cli.<command>`.

## Observation Contract

An observation batch contains:

- `observation.state`: required for the first complete implementation, shaped
  `[batch, n_obs_steps, state_dim]`;
- zero or more `observation.images.*` tensors, retained under their original
  keys;
- validity metadata needed for episode and action-chunk boundaries.

Replay buffers and trainers operate on observation dictionaries rather than
RL-100's fixed `point_cloud`, `image`, and `agent_pos` fields.

`StateFeatureEncoder` completely implements the critic/dynamics feature path
for state input. `ObservationFeatureEncoder` defines the extension contract for
image features. Image tensors can pass through the data, replay, and actor
adapter paths, but selecting image-conditioned critics without registering an
image encoder raises a specific configuration error. The initial migration
does not silently discard images and does not claim image RL training is
implemented.

For a future image Diffusion Policy checkpoint, the actor adapter may use the
checkpoint's own LeRobot RGB encoder. The remaining work will be a concrete
image feature encoder for IQL, value, and dynamics; their interfaces and
checkpoint slots are fixed in this design.

## Dataset And Sparse Reward

The current LeRobot v3 dataset contains state, action, and index fields but no
reward or done fields. The adapter groups rows by episode, validates monotonic
frame indices, and constructs decision-level transitions with the policy's
32-step execution chunk:

```text
(two-frame observation history,
 32-step action chunk,
 valid-action-step mask,
 discounted chunk reward,
 next two-frame observation history,
 done,
 discount)
```

The final partial chunk is padded for fixed tensor shapes and carries a mask so
unused actions never contribute to critic inputs or PPO likelihood ratios.
`discount` is `gamma ** valid_steps` for non-terminal transitions.

Episode labels come from the sibling `collection_summary.json` and use the
agreed terminal criterion:

```text
success = true_grasp_ever
          and clear_table_ever
          and final_lift_height_m >= 0.015
          and final_table_contacts == 0
          and final_hand_contacts > 0
```

Only the last decision in an episode receives reward `1.0` for success or
`0.0` for failure; every earlier reward is `0.0`. The adapter fails if summary
episode indices do not match the dataset. For the supplied data, inspection
must report 100 episodes, 94 positive labels, 6 negative labels, and 93,000
frames.

The action statistics show that rotation dimensions 3 through 11 are constant.
An active-action mask is derived from checkpoint normalization ranges. IQL and
PPO use the five variable dimensions (translation and hand closure) while the
actor and environment retain the full 14-dimensional action contract.

## Diffusion Policy Adapter

The adapter loads `DiffusionPolicy`, the preprocessor, and the postprocessor
from one checkpoint directory. It provides four operations:

1. normalize observation histories and behavior action chunks with the saved
   LeRobot normalizer;
2. sample a complete stochastic DDIM denoising trace with an old policy;
3. recompute current-policy log probabilities on a detached stored trace;
4. unnormalize the executable action slice with the saved postprocessor.

RL-100 requires a stochastic Gaussian denoising transition, whereas the
existing checkpoint was trained with a DDPM noise predictor and the standard
LeRobot sampler returns only final actions. The weights are compatible with a
DDIM inference schedule, so the RL adapter constructs a private stochastic
DDIM schedule from the checkpoint beta configuration. It does not modify the
saved policy config or standard evaluator.

Each transition records `x_t`, `x_prev`, timestep, mean, standard deviation,
and per-event log probability. The standard deviation has explicit positive
minimum and maximum bounds. Log probabilities remain shaped by action step and
action dimension until the trainer applies the valid-step and active-dimension
masks. Fixed-noise tests must show that recomputed old-policy ratios are one,
all values are finite, and PPO loss gradients reach the LeRobot U-Net.

The default RL schedule uses 10 denoising steps, `eta=1.0`,
`sigma_min=0.0067`, and `sigma_max=0.1`, matching RL-100's chunk-oriented
stochastic setting. Smoke tests may use fewer steps. Existing deterministic
evaluation remains the authority for success-rate comparison.

## Offline Training

The offline trainer performs:

1. dataset inspection and transition construction;
2. state normalization with checkpoint statistics;
3. double-Q IQL and expectile value training on masked action chunks;
4. old-policy stochastic trace sampling on offline observations;
5. normalized `Q - V` advantages on sampled executable chunks;
6. clipped denoising PPO updates to the current Diffusion Policy;
7. optional state-dynamics training and model-based policy-promotion checks;
8. checkpoint and metric persistence.

IQL uses the valid-step mask when flattening a partial action chunk. The target
is `r + discount * (1 - done) * V(next_state)`. The actor accumulates gradients
from all denoising sub-policy losses before one optimizer step, matching the
paper's summed objective while avoiding retention of all U-Net graphs at once.

The state-dynamics ensemble predicts the next state history, sparse reward,
and termination at decision boundaries. Its OPE result is reported with model
disagreement. Promotion is disabled when the model has not passed its held-out
validation threshold. The basic scalar-IQL actor update does not depend on a
learned dynamics model.

## Online Training

The online trainer starts from the base or offline RL checkpoint and creates
the existing fused Moya CUDA vector environment. It samples one stochastic
action chunk per environment decision and executes actions sequentially without
recording video.

For each environment it stores observation history, denoising trace, old log
probability, actually executed action mask, accumulated reward, next history,
done/truncated flags, and success. Gymnasium SAME_STEP autoreset is handled by
using Moya's terminal `final_obs` for the terminal transition rather than the
next episode's reset observation.

GAE is calculated separately for each vector world. The per-decision discount
is `gamma ** executed_steps`, so the final partial chunk is correct. Online PPO
uses the same denoising likelihood-ratio implementation as offline training,
with a value loss and gradient clipping. Evaluation uses the existing
15-millimeter success contract and records success rate without video.

## Checkpoints And Resumption

Every saved stage contains:

```text
pretrained_model/        # standard LeRobot config, model, processors, stats
rl_state.pt              # IQL/value/dynamics and optimizer states
rl_config.json           # resolved immutable training contract
metrics.jsonl            # append-only scalar metrics
provenance.json           # base checkpoint hash and dataset/summary paths
```

The standard model bundle remains loadable by existing LeRobot evaluation.
Resume validates architecture fields, observation keys, chunk sizes,
normalization fingerprints, action masks, and source checkpoint hash before
loading optimizer state. Mismatches fail before training.

## Error Handling And Safety

- Missing or incompatible processor statistics are fatal.
- Dataset/summary episode mismatches are fatal.
- Non-finite state, action, reward, log probability, ratio, advantage, or loss
  values stop the update and identify the tensor.
- Image-conditioned critic use without an image encoder is fatal and names the
  required extension point.
- Online actions are unnormalized and checked against the Moya 14D space before
  stepping.
- The Newton environment contract hash and success schema remain enforced by
  the existing wrapper.
- Online smoke tests are short, headless, CUDA-backed, and never start a long
  training job.

## Verification

Unit tests cover:

- terminal sparse reward relabeling and the 15-millimeter threshold;
- episode boundaries, state histories, full and partial chunks, and masks;
- image-key preservation and image-encoder fail-fast behavior;
- checkpoint normalization round trips and active-action mask discovery;
- IQL targets, expectile loss, finite updates, and `Q - V` shape;
- stochastic DDIM sampling/replay, ratio-one identity, masking, and U-Net
  gradients;
- vector GAE isolation and variable decision discounts;
- Moya SAME_STEP terminal observation selection;
- checkpoint save/resume contract validation.

Integration acceptance consists of:

1. load the real 80k checkpoint and processors;
2. inspect the real v3 dataset and obtain the required 94/6 label split;
3. complete at least one CUDA offline IQL update and one denoising actor update;
4. collect a short 16-world CUDA Newton rollout into the online buffer;
5. complete one online GAE/PPO update on smoke-sized data;
6. save a resulting LeRobot-compatible checkpoint and reload it;
7. run the existing Moya and state-only Diffusion Policy regression tests.

Smoke completion proves wiring, finite gradients, device placement, and
serialization. It does not claim that a smoke-sized update improves success
rate; full training and comparative evaluation are separate experiments.

