# LPWM-FM: shared real DLP representation, ordinary conditional Flow Matching

This module implements the robot **action-policy half** of the A/B comparison.
The complete native LPWM reference and robot GT-action world model live in
`world_model.py` / `_vendor` (see their provenance/license). No pretrained weights
or language model are downloaded automatically.

## Controlled variants

- **A:** DLP → frame-causal scene Transformer → language-conditioned FM expert.
  The world architecture is instantiated for identical initialization, but its
  reconstruction/dynamics loss is never run (`variant="A"`, `world_weight=0`).
- **B:** same architecture, plus `world_loss` on **clean recorded GT actions**.
  FM and world gradients both reach the *same* unfrozen DLP encoder. There is no
  predicted-action world input, IMF/JVP, action-latent alignment, or EMA teacher.

Using the same seed before each construction produces identical parameter tensors.
Changing variant/weight after constructing the model is also supported. B with
`world_weight=0` skips world loss entirely. Each run can use one independent GPU;
this code does not launch/distribute training or assume a batch size fits.

## Model

```text
RGB history [B,H,V,3,128,128]
  → shared real DLP (per-frame/per-view independent encoding)
  → foreground [B,H,V,N,10], background [B,H,V,4]
  → per-type projection + view/time embeddings + state token each frame
  → frame-causal Scene Transformer
  + frozen cached language projected to scene width
  + sinusoidal flow-time token
  + noisy action tokens (learned action-position embeddings)
  → plain pre-LayerNorm attention + FFN expert → action velocity [B,K,7]
```

Scene tokens can see all cameras/particles/state within the same frame and all
previous frames, never later frames. Action tokens attend each other and all
valid conditions; condition-prefix queries cannot read noisy action tokens.
There is no AdaLN or FiLM in the robot expert. Dynamics injects the clean GT
action as three identical tokens; it does not feed those tokens back to the
scene/expert. Separate camera coordinates are not treated as common 3-D points.

Ordinary FM, with `tau=0` clean and `tau=1` Gaussian noise:

```python
x_tau = (1 - tau) * actions + tau * noise
velocity_target = noise - actions
loss_fm = mean_valid((expert(x_tau, tau, condition) - velocity_target)**2)
```

The mean is over non-padding action positions **and dimensions**. Padded targets
are zeroed before attention, so arbitrary padded label values cannot leak into
valid predictions. Euler inference runs from 1 to 0 with 10 steps by default:
`x = x - velocity / num_inference_steps`. There is no DCT or clipping.
`predict_action_chunk` returns all 16 actions starting **at the current time**;
`select_action` executes 8, updates observation history every tick, then replans.
Call `reset()` on episode changes and `eval()` before inference.

## Configuration and batches

Registered configuration: `LPWMFMConfig`, tag **`lpwm-fm`**. Global factory imports
are owned by the integration task. Real DLP structural defaults come from
`LPWMIMFConfig`; encoder weights are trainable by default and freezing is rejected
for these A/B runs.

Key fields:

| Field | Default |
|---|---:|
| `n_obs_steps`, `horizon`, `n_action_steps` | 2, 16, 8 |
| `hidden_dim`, `n_heads` | 256, 8 |
| `scene_n_layers`, `expert_n_layers`, `world_n_layers` | 2, 4, 4 |
| `world_hidden_dim`, `world_n_heads` | 256, 8 |
| `language_dim` | 512 (set **960** for cached SmolVLM2-500M text states) |
| `action_dim`, `state_dim` | 7, 8 |
| `dropout`, `action_token_repeat` | 0, 3 |
| `optimizer_lr` | 1e-4 |
| `world_weight` | 1 |
| `reconstruction_weight`, `prior_weight`, `dynamics_weight` | 1, 0.001, 1 |
| `world_warmup_steps`, `world_ramp_steps` | 0, 0 |

World compatibility aliases: `world_layers`, `rec_weight`, `dyn_weight`. The
world module applies its component weights and normalizes its terms; the policy
applies `world_weight` exactly once. Choose/calibrate weights from logged gradient
and loss scales; these defaults are not evidence of optimal balancing. The outer
schedule is zero before warmup and `(step-warmup+1)/ramp` during its ramp; `current_step`
must be supplied if a schedule is enabled. Metrics are detached.

`forward(batch, current_step=None)` receives **already processed** tensors:

```python
batch = {
    # Cameras in config.input_features insertion order; images remain [0,1].
    "observation.images.agent": images_agent,    # [B,2,3,h,w]
    "observation.images.wrist": images_wrist,    # [B,2,3,h,w]
    "observation.state": state,                 # [B,2,8], train-only mean/std
    "observation.language.embedding": language, # [B,L,D], real frozen embeddings REQUIRED
    "observation.language.attention_mask": mask,# [B,L], True/1 = valid (optional)
    "action": action,                           # [B,16,7], native LIBERO units
    "action_is_pad": padding,                   # [B,16], True = padded (optional)
    # Required only for B with positive world_weight:
    "world.images": world_images,               # [B,T,V,3,h,w], raw [0,1]
    "world.actions": world_actions,             # [B,T-1,7], native GT controls
}
loss, metrics = policy(batch, current_step=step)
```

For history 2/world 1, world images are `[t-1,t,t+1]` and actions are
`[a_(t-1),a_t]`. They train one-step teacher-forced transitions; neither future
images nor GT actions enter the deployable action condition. Construct valid
within-episode windows; padded world transitions are rejected, not silently
trained. `world.states` may be present but the current world-loss interface does
not consume it. Cached language is detached; only its projection is learned.

`make_lpwm_fm_pre_post_processors` is for **raw-input export/evaluation**, not a
second pass over the standalone trainer's already normalized tensors. Defaults
are STATE MEAN_STD, VISUAL IDENTITY, ACTION IDENTITY. It preserves `world.*`, moves
them to the device, and leaves already-processed world-action units unchanged.
When restoring a serialized pipeline for world batches, supply
`lpwm_batch_to_transition` as `to_transition`; generic LeRobot conversion discards
unknown `world.*` fields. Language is never replaced with learned task IDs or
empty/pseudo embeddings.

## Scoped verification

```bash
uv run --no-sync pytest tests/policies/lpwm_fm/test_policy.py -q
```

Tests exercise the small **real** LPWM model: FM formula/no JVP, shared encoder
backprop from both losses, clean-GT routing, identical A/B initialization,
frame causality, future-label isolation, multi-camera ordering/shared weights,
Euler sign, action queue/history semantics, config/model save-load, and processor
normalization. Tests do not train a production model or launch a robot/service.
