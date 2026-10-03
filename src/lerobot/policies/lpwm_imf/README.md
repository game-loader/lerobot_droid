# LPWM-IMF — visual encoder stage

**Status: encoder only. This is not yet a trainable IMF action policy.**

Registered LeRobot name: `lpwm-imf`. The implementation imports the real DLPv3 visual
encoder from [taldatech/lpwm](https://github.com/taldatech/lpwm), pinned at
`4cf53c403433e64c01652ac2adbec66231a46dea`, stopping at its visual particle latents.
It does **not** substitute a ResNet, DINO, pooled MLP, or mock for the upstream architecture.
See [`_vendor/UPSTREAM.md`](_vendor/UPSTREAM.md) and [`_vendor/LICENSE`](_vendor/LICENSE)
for the exact source subset and MIT attribution. Only PyTorch/NumPy operations are
needed; no extra package install, runtime source checkout or automatic weight download.

## Boundary and architecture

```text
RGB uint8 [0,255] or floating [0,1]
  └─ resize to configured image_size (128 by default)
     └─ optional RGB conversion to [-1,1] (off by default)
        ├─ patch CNN + spatial softmax → keypoint proposals / patch anchors
        ├─ particle attribute encoder → offsets, scale, presence, depth
        ├─ differentiable image glimpses + foreground CNN → appearance posterior
        ├─ foreground masking + background CNN → background posterior
        └─ particle interaction Transformer → refined appearance/depth/background
           └─ structured visual latents Z + background
                    STOP HERE
```

No image-reconstruction decoder, latent-action/context encoder (`DLPContext`),
world-model dynamics (`DLPDynamics`), IMF head or loss is constructed.
`DLPEncoder.context_dim=0` and `ctx_enc=None` explicitly stop before latent context.
An upstream helper called `ParticleAttributeDecoder` **is** retained: it decodes
particle attributes *inside* the visual encoder, not images or future states.

Default visual architecture follows upstream `configs/sketchy.json`: 128×128 RGB,
16×16 proposal patches, one keypoint per patch, 64 visual particles, four-dimensional
Gaussian foreground and background appearance, 256-wide projections, and one
particle-interaction Transformer layer/head. The object CNN uses channels
`32 × (1,4,8)`; the background CNN uses `32 × (1,1,1,2,4)`.

The LPWM video model encodes **64 particles**, even though the upstream config
specifies `n_kp_enc=30`: its top-level `DLP` constructor changes the encoder count to
`n_kp_prior=64` and keeps 30 as `n_kp_dec`. Here those effective counts are explicit.
The `n_kp_dec=30` setting is retained because it also affects the foreground mask
used by the background encoder, despite no image decoder being constructed.

## Output contract

`LPWMVisualLatents` preserves batch, time, camera and particle axes:

| Field | Shape | Meaning |
|---|---|---|
| `position` | `[B,T,V,N,2]` | Upstream `z`, coordinates ordered **(y,x)** |
| `scale` | `[B,T,V,N,2]` | Upstream pre-sigmoid scale latent, not pixel box sizes |
| `depth` | `[B,T,V,N,1]` | Learned depth-order attribute, **not calibrated metric depth** |
| `presence` | `[B,T,V,N,1]` | Upstream Beta transparency / object-on value |
| `features` | `[B,T,V,N,F]` | Foreground appearance latent |
| `background` | `[B,T,V,F_bg]` | Separate background appearance latent |
| `z` | `[B,T,V,N,6+F]` | Adapter packing `[position, scale, depth, presence, features]` |

Uppercase **Z** denotes this structured visual representation. Upstream's dictionary
key **`z` alone is only the two-dimensional particle position**; it must not be
mistaken for the complete descriptor. We add the descriptor packing property without
changing any upstream field. No pooling, projection to IMF width, particle pruning,
or foreground/background fusion is added.

With defaults: foreground `z.shape == [B,T,V,64,10]`, background shape `[B,T,V,4]`.
Cameras share encoder weights but are encoded independently, not concatenated into
input channels. Camera order follows `config.input_features`. There is no
cross-camera calibration or fusion, and no learned temporal correspondence/tracking.
Callers supply their history; this stage owns no observation/action queue.

## Use

```python
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import make_policy_config, get_policy_class

config = make_policy_config(
    "lpwm-imf",
    device="cpu",  # Explicitly move the policy to config.device when using it directly.
    n_obs_steps=2,
    push_to_hub=False,
    input_features={
        "observation.images.front": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 128, 128)
        ),
    },
)
policy = get_policy_class("lpwm-imf")(config).to(config.device).eval()

# Architecture only: random initialization until weights are loaded explicitly.
# policy.model.load_lpwm_encoder("/path/to/upstream_lpwm_checkpoint.pth")

with torch.no_grad():
    result = policy.encode_observation({
        "observation.images.front": torch.rand(1, 2, 3, 128, 128),
    })
print(result.z.shape)           # torch.Size([1, 2, 1, 64, 10])
print(result.background.shape)  # torch.Size([1, 2, 1, 4])
```

The standalone `LPWMVisualEncoder` also accepts `[B,3,H,W]`, `[B,T,3,H,W]`, or
`[B,T,V,3,H,W]`; singleton time/view axes are retained in outputs. The policy adapter
supports distinct spatial resolutions per camera. `VISUAL` preprocessing must remain
`IDENTITY`: do not pass ImageNet-normalized or floating 0–255 images. Resize is explicit
bilinear/antialiased preprocessing; the source's fixed-resolution encoder itself is
unchanged. For checkpoint parity supply the same pre-resized RGB as upstream.

Posterior means / Beta means are the default (`deterministic=True`); use `.eval()`
for inference. `deterministic=False` enables upstream posterior sampling. Autograd
is preserved when unfrozen; `freeze_encoder=True` disables visual parameter gradients
and keeps the encoder in eval mode. Freezing random initialization does **not**
produce pretrained representations.

## Checkpoints and exclusions

- `policy.model.load_lpwm_encoder(path_or_state_dict)` accepts a bare encoder state
  dict, an upstream `encoder_module.*` state dict, or a `state_dict` wrapper; uniform
  `module.` prefixes are handled. Local `.pth` loading uses `weights_only=True`;
  `.safetensors` is also supported.
- Only visual encoder keys are loaded. Full-model decoder/dynamics keys and the
  encoder's optional `ctx_enc.*` latent-context keys are not part of this stage.
- All remaining keys and tensor shapes must match. Missing/mismatched weights raise
  an error **before** loading; there is no silent partial load or random fallback.
- The default topology is for the Gaussian Sketchy visual encoder. Other checkpoint
  configurations must be matched explicitly; arbitrary categorical/context variants
  are not claimed to be supported by this adapter.
- LeRobot `save_pretrained` / `from_pretrained` round-trip config and encoder weights.
- `policy.forward`, `select_action`, and `predict_action_chunk` deliberately raise
  `NotImplementedError`. Do not launch `lerobot-train` or robot evaluation with this
  policy yet. The future IMF head/objective is a separate implementation step.

## Verification

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync pytest tests/policies/lpwm_imf -q

# Optional: compare all retained class/function ASTs and real 128x128 outputs
# with a trusted local checkout at the pinned upstream revision.
LPWM_UPSTREAM_DIR=/path/to/lpwm \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync pytest tests/policies/lpwm_imf/test_upstream_parity.py -q
```

Tests run actual CNNs, spatial transforms and particle attention. They cover shape
and field preservation, multi-camera ordering, input validation, deterministic vs.
stochastic encoding, gradients, freezing, registry/processors, config/policy round
trips, strict upstream checkpoint extraction, and explicit action-path rejection.
No task-success or pretrained-weight quality claim follows from these software tests.
