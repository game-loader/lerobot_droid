# Heuristics

## H01: Keep rollout sigma and likelihood sigma separate
- **Rationale**: Use the schedule-clamped sampling sigma for the DDIM transition mean and exploration noise, then floor a separate probability sigma for both old and new Gaussian likelihoods on the stored transition. This removes the inverse-variance amplification of nearly deterministic tail steps without perturbing the behavior trajectory.
- **Sources**: []
- **Status**: active
- **Provenance**: user
- **Sensitivity**: high
- **Code ref**: [`RL/config.py`, `RL/policy/ddim.py`, `RL/policy/diffusion_adapter.py`]

## H02: Validate processor semantics, not unused metadata representation
- **Rationale**: Determine the statistics required by the configured normalization mode, enforce strict preprocessor/postprocessor equality for those tensors, and do not reject a checkpoint solely because unused auxiliary metadata has an equivalent scalar versus length-one representation.
- **Sources**: []
- **Status**: active
- **Provenance**: ai-suggested
- **Sensitivity**: medium
- **Code ref**: [`RL/adapters/checkpoint.py`, `tests/rl100/test_checkpoint_adapter.py`]
