# LPWM encoder provenance

Upstream: https://github.com/taldatech/lpwm

Pinned revision: `4cf53c403433e64c01652ac2adbec66231a46dea`. License: MIT (see `LICENSE` and source headers).

Only the transitive encoder class/function definitions are retained, with local imports. No image decoder, DLPContext, dynamics, training loss, plotting, checkpoint download or top-level `models.py` is imported. Definitions retain upstream names and state-dict layout.

## `particle_encoder.py` ← `modules/modules.py`

`AlternativeSpatialSoftmaxKP`, `ImagePatcher`, `ParticleNorm`, `RMSNorm`, `SimpleRelativePositionalBias`, `ParticleSelfAttention`, `MLP`, `SelfBlock`, `ParticleSelfAttTransformer`, `ParticleAttributesProjection`, `ParticleAttributeDecoder`, `BgEncoder`, `ParticleAttributeEncoder`, `ParticleFeaturesEncoder`, `DLPPrior`, `ParticleInteractionEncoder`, `ParticleEncoder`, `DLPEncoder`.

## `vision_encoder.py` ← `modules/vision_modules.py`

`nonlinearity`, `norm_layer`, `Downsample`, `ConvBlock`, `ResnetBlock`, `AttnBlock`, `Encoder`.

## `utils.py` ← `utils/util_func.py`

`reparameterize`, `create_masks_fast`, `create_masks_with_scale`, `spatial_transform`, `modulate`, `affine_grid_sample`.
