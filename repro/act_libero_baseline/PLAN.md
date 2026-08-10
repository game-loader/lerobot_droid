# PLAN

## User requirements
1. 说明 SmolVLA 论文工作在当前 LeRobot 中如何快速复现。
2. 优先使用 `uv` 管理环境。
3. 解决本机环境问题。
4. 在 LIBERO 数据集上复现一条可信的 ACT 基线结果（至少打通训练/评测路径，若算力/时间不足则完成 smoke/partial reproduction 并明确阻塞）。

## Chosen route
Fast-path baseline reproduction for `ACT @ LIBERO` in the current `lerobot` repo.

## Source identity
- Repo: `/home/droid/project/lerobot`
- Branch/commit: `main` @ `ba27aab7`
- Primary benchmark dataset: `HuggingFaceVLA/libero`
- Benchmark env: `env.type=libero`
- Policy: `act`

## Command path
1. Inspect CLI/config/docs for ACT + LIBERO.
2. Prepare `uv` env with training + evaluation + libero deps.
3. Smoke-check imports and CLI help.
4. Run a minimal ACT-on-LIBERO train/eval command.
5. If feasible, scale toward benchmark-like settings or clearly report remaining blockers.

## Expected outputs
- Working `uv` environment command set.
- Concrete ACT train/eval commands for LIBERO.
- Smoke or real run artifacts under `outputs/` or `tests/outputs/`.
- Verified notes on what is and is not reproduced.

## Acceptance condition
- Preferred: ACT can train/eval on LIBERO in this repo with a verified command path.
- Minimum acceptable for this session: environment is fixed, required deps are verified, and a smoke/partial ACT@LIBERO run is executed with exact blockers to full benchmark reproduction documented.

## Main risks
- Missing system deps for MuJoCo / rendering / ffmpeg.
- `hf-libero` asset/config initialization issues (`~/.libero/config.yaml`).
- Full published ACT benchmark may be too expensive to finish within one session.

## Fallback
- Use a smoke or 1-task/1-episode/very-short-step run to validate the pipeline, then provide the exact scale-up command for full reproduction.
