# Moya Diffusion Successful Rollout Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the known-successful world 0 rollout from the 80k Diffusion Policy checkpoint and save a verified 1280x720, 60 FPS H.264 video.

**Architecture:** Keep all executable code and artifacts in one timestamped `outputs/eval/` run directory. Build the standard 16-world Moya environment and LeRobot policy pipeline, attach a headless Newton `ViewerGL` to the validated backend at runtime, stream the initial state plus all 930 post-action world-0 states through PyAV, and retain the temporary MP4 only when LeRobot's accepted success signal is true for world 0.

**Tech Stack:** Python 3.12, uv, PyTorch/CUDA, LeRobot evaluation pipeline, Newton/Warp ViewerGL, PyAV/FFmpeg, jq, ffprobe.

---

### Task 1: Create The Run-Local Recorder

**Files:**
- Create: `outputs/eval/moya_diffusion_best_080000_success_video_20260816-130143/record_success_video.py`

- [x] **Step 1: Add deterministic constants and startup validation**

Define absolute repository/checkpoint/output paths, `SEEDS = list(range(1000, 1016))`, `WORLD_INDEX = 0`, `EPISODE_LENGTH = 930`, `FRAME_COUNT = 931`, `FPS = 60`, 1280x720 dimensions, front-camera pose, and expected model SHA-256. Before allocating CUDA state, hash every checkpoint model shard selected by the saved index or the single model file and verify the known aggregate checkpoint hash used by this run.

- [x] **Step 2: Add a streaming H.264 recorder**

Implement a small context-managed recorder around PyAV:

```python
container = av.open(str(temp_video_path), mode="w")
stream = container.add_stream("libx264", rate=FPS)
stream.width = WIDTH
stream.height = HEIGHT
stream.pix_fmt = "yuv420p"
frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
for packet in stream.encode(frame):
    container.mux(packet)
```

Flush the encoder with `stream.encode(None)`, close the container on every path, and delete the temporary video on rollout failure.

- [x] **Step 3: Construct the exact evaluation environment and policy**

Use `set_seed(1000)`, `MoyaNewtonEnvConfig()`, and `make_env(..., n_envs=16, use_async_envs=False)` before importing `newton.viewer`. Extract the sole vector environment, load `PreTrainedConfig.from_pretrained(CHECKPOINT)`, set `pretrained_path`, `device="cuda"`, and `num_inference_steps=100`, then create the policy plus saved preprocessor/postprocessor and environment processors using the same factories as `lerobot_eval.eval_main`.

- [x] **Step 4: Attach ViewerGL without changing normal environment behavior**

Create `ViewerGL(width=1280, height=720, headless=True, paused=False)`, bind its model to `env.unwrapped._batched_sim.model`, call `viewer.set_visible_worlds([0])`, set the approved camera, and assign the viewer to both `env.unwrapped._viewer` and `env.unwrapped._batched_sim.viewer`. The render callback must call `env.unwrapped.render()` and then `viewer.get_frame(render_ui=False)`; it must never call `MoyaNewtonVectorEnv.render()`.

- [x] **Step 5: Reuse LeRobot rollout and gate the artifact on success**

Call `lerobot.scripts.lerobot_eval.rollout()` with all four processor pipelines and `seeds=SEEDS`. The callback encodes the initial state and post-step states 1 through 929. A run-local wrapper around the backend's `SAME_STEP` reset helper captures post-step state 930 immediately before reset, giving 931 frames without recording the next episode's reset state. Compute the 16-world success vector from `rollout_data["success"].any(dim=1)`, require world 0 to be true, atomically rename the temporary video to `successful_rollout_world0.mp4`, and write `summary.json` containing paths, hashes, devices, seeds, folded seed `747872231`, environment contract, success criterion, success vector, frame count, and elapsed time.

- [x] **Step 6: Guarantee cleanup and useful failure output**

Close encoder, viewer, and environment in `finally` blocks. Write `summary.json` with `success=false` and the exception message before re-raising, so `record.log` and diagnostics remain available even if rendering or encoding fails.

### Task 2: Check The Recorder Before The Full GPU Rollout

**Files:**
- Verify: `outputs/eval/moya_diffusion_best_080000_success_video_20260816-130143/record_success_video.py`

- [x] **Step 1: Run syntax and import checks**

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m py_compile outputs/eval/moya_diffusion_best_080000_success_video_20260816-130143/record_success_video.py
UV_CACHE_DIR=.uv-cache uv run python -c 'import av; print(av.__version__)'
```

Expected: both commands exit 0 and PyAV reports an installed version.

- [x] **Step 2: Verify immutable inputs**

Run a script-level validation mode that checks checkpoint existence and SHA-256 without constructing Newton. Expected: checkpoint validation exits 0, seed count is 16, folded seed is `747872231`, and requested frame count is 931 for the initial state plus 930 action results.

### Task 3: Record And Verify The Successful Rollout

**Files:**
- Create: `outputs/eval/moya_diffusion_best_080000_success_video_20260816-130143/record.log`
- Create: `outputs/eval/moya_diffusion_best_080000_success_video_20260816-130143/summary.json`
- Create: `outputs/eval/moya_diffusion_best_080000_success_video_20260816-130143/successful_rollout_world0.mp4`
- Create: `outputs/eval/moya_diffusion_best_080000_success_video_20260816-130143/verification_frames/frame_start.png`
- Create: `outputs/eval/moya_diffusion_best_080000_success_video_20260816-130143/verification_frames/frame_middle.png`
- Create: `outputs/eval/moya_diffusion_best_080000_success_video_20260816-130143/verification_frames/frame_end.png`

- [x] **Step 1: Run the recorder outside the restricted sandbox**

Run with writable local caches and log capture:

```bash
UV_CACHE_DIR=.uv-cache \
WARP_CACHE_PATH=outputs/train/moya_diffusion_300k_20260815-093537/cache/warp/1.14.0 \
uv run python outputs/eval/moya_diffusion_best_080000_success_video_20260816-130143/record_success_video.py \
  2>&1 | tee outputs/eval/moya_diffusion_best_080000_success_video_20260816-130143/record.log
```

Expected: exit 0, world 0 success is true, and the final MP4 exists.

- [x] **Step 2: Verify metadata and success provenance**

Run:

```bash
jq -e '.success == true and .world_index == 0 and .folded_seed == 747872231 and .frame_count == 931 and .terminal_frame_captured == true and .inference_steps == 100' outputs/eval/moya_diffusion_best_080000_success_video_20260816-130143/summary.json
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames:format=duration -of json outputs/eval/moya_diffusion_best_080000_success_video_20260816-130143/successful_rollout_world0.mp4
```

Expected: H.264, yuv420p, 1280x720, `60/1`, 931 frames, and approximately 15.52 seconds.

- [x] **Step 3: Decode and inspect representative frames**

Use ffmpeg to save frame 0, frame 465, and frame 930. Compute mean, standard deviation, and pairwise absolute differences; each image must be non-black with visible variation, and start-to-middle/end differences must be non-zero. Inspect all three images to confirm the hand, charger, and table are in frame, only world 0 is visible, and no text/debug overlay is present.

- [x] **Step 4: Review the final implementation and artifacts**

Have separate agents perform specification and code-quality reviews, then rerun the metadata and decoded-frame checks after any fixes.
