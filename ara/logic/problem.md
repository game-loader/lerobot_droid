# Problem

The migrated diffusion-PPO actor became numerically unstable during offline
training: KL estimates grew rapidly, while the existing scalar logs did not
identify whether the cause was action-chunk joint probability reduction,
individual denoising transitions, stochastic DDIM scale, or snapshot replay.
