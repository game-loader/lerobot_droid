# Environment

Target worktree: `/home/droid/project/lerobot_droid/.worktrees/upstream-0.6-custom`.
Source checkout: `/home/droid/project/lerobot_droid`.
Version: `0.6.2+droid.1`; migration code commit: `ebd5836b`.

The target `.venv` and `.venv-base` are independent of the source environment.
Pinned dependencies are in `../../uv.lock`; detailed validation environment and
commands are in `../../docs/CUSTOM_VERSION_VALIDATION.md` and
`../../scripts/test_custom_version.sh`. Model weights, datasets, simulator assets,
live services and long-running training jobs were not migrated or modified.

The checkpoint smoke harness is archived under `execution/`. Its absolute paths
refer to pre-existing local source artifacts, which are not shipped in this repo.
