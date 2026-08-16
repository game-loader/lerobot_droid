# Moya Diffusion Successful Rollout Video Design

## Goal

Record one visually inspectable successful rollout from the selected 80k
Diffusion Policy checkpoint in the Moya Newton simulator. Produce a 1280x720,
60 FPS H.264 MP4 without debug text or video recording changes to the normal
training and evaluation paths.

## Source Policy And Trajectory

Use the checkpoint selected by the completed 100-episode evaluation:

```text
outputs/train/moya_diffusion_300k_20260815-093537/
  train/checkpoints/080000/pretrained_model
```

The model SHA-256 is:

```text
15b5babed040c260281fd95cb1b624e5da533e847dd926345c02d70b4c16cfec
```

The recorder must reproduce the first batch from the 100-episode evaluation:

```text
batch size: 16 fused Newton worlds
seed list: 1000 through 1015
recorded world: 0
episode length: 930 outer steps
simulation substeps: 8
policy inference steps: 100
policy device: cuda
simulation device: cuda:0
```

World 0 was successful in the persisted 100-episode result. Keeping the full
seed list and batch size is required because Moya deterministically folds the
whole seed list; changing to one world would change the randomized initial
state.

## Recording Architecture

Use a run-local standalone recorder under `outputs/eval/`; do not change the
normal Moya environment config, `lerobot-eval`, or training behavior.

The recorder will:

1. create a headless Newton `ViewerGL` at 1280x720;
2. construct the pinned Moya backend under the approved
   `randomized_grasp_v1` preset with 16 CUDA worlds and the viewer attached;
3. wrap the backend in the existing `MoyaNewtonVectorEnv`, preserving the
   accepted success criterion;
4. load the 80k Diffusion Policy and its saved preprocessor/postprocessor;
5. execute the same LeRobot action path as standalone evaluation;
6. render after reset and after every outer simulation step;
7. capture the existing front camera focused on world 0; and
8. encode frames as H.264/yuv420p at 60 FPS.

The camera uses the established Moya rollout defaults:

```text
position: (1.172, -0.055, 1.314)
pitch: -9.4 degrees
yaw: 179.8 degrees
```

No reward HUD, labels, point markers, or other text overlays are added.

## Success And Failure Handling

The rollout is accepted only if world 0 satisfies the same terminal success
definition used by evaluation:

```text
true_grasp_ever
and clear_table_ever
and final_lift_height >= 0.015 m
and final_table_contacts == 0
and final_hand_contacts > 0
```

The script records to a temporary frame directory and encodes the MP4 only
after the rollout. If world 0 is not successful, the artifact is rejected and
reported instead of presenting a failed trajectory as successful.

## Output And Verification

The run-local output contains:

```text
record_success_video.py
record.log
summary.json
successful_rollout_world0.mp4
```

Verification requires all of the following:

- `summary.json` reports world 0 success under the accepted criterion;
- the MP4 exists, is non-empty, and is H.264/yuv420p;
- resolution is 1280x720 and frame rate is 60 FPS;
- duration is approximately 15.5 seconds for 930 steps plus the initial frame;
- no video is blank or nearly constant, checked from decoded start, middle,
  and final frames; and
- visual inspection confirms the camera contains the hand, charger, and table
  throughout the grasp and lift.

## Scope

This is a one-off recording artifact. It does not enable video in periodic
training evaluation, alter the checkpoint, change Moya physics, or modify the
success threshold.
