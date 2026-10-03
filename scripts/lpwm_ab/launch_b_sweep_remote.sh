#!/usr/bin/env bash
# Invoke only on the authorized remote machine; never changes old runs or environments.
set -euo pipefail
SUITE=${LPWM_SWEEP_ROOT:?Set LPWM_SWEEP_ROOT to the prepared data-disk suite directory}
BASE=${LPWM_RUN_ROOT:-/root/lpwm_ab}
[[ "$SUITE" == /root/gpufree-data/* ]] || { echo 'Expected user data-disk suite' >&2; exit 1; }
[[ -d "$SUITE/repo" && ! -e "$SUITE/queue_state.json" ]] || { echo 'Missing snapshot or queue already exists' >&2; exit 1; }
cd "$SUITE/repo"
export PYTHONPATH="$SUITE/repo/src" LIBERO_CONFIG_PATH="$BASE/libero_config"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
"$BASE/.venv/bin/python" - "$SUITE" <<'PY'
import hashlib,json,sys
from pathlib import Path
s=Path(sys.argv[1]);experiments=[]
for gpu in (0,1):
 p=s/f'preflight/verified-gpu{gpu}'
 status=json.loads((p/'status.json').read_text())
 assert status['status']=='completed' and status['step']==2 and status['preflight'] is True
 e=json.loads((p/'experiment.json').read_text());experiments.append(e)
 m=json.loads((p/'eval/step_000002.swanlab.json').read_text())
 assert m['status']=='complete' and m['verified_upload'] is True and m['formal_result_eligible'] is False
 c=p/'checkpoints/step_000002';manifest=json.loads((c/'complete.json').read_text())
 for name,digest in manifest['sha256'].items():
  assert hashlib.sha256((c/name).read_bytes()).hexdigest()==digest
for key in ('initial_weights_sha256','split_sha256','dataset_manifest_sha256'):
 assert experiments[0][key]==experiments[1][key]
print('Verified both-GPU real train/gradient/checkpoint/EGL/online-eval preflights.',flush=True)
PY
nohup "$BASE/.venv/bin/python" -u "$SUITE/repo/scripts/lpwm_ab/run_b_sweep.py" \
 --repo "$SUITE/repo" --python "$BASE/.venv/bin/python" --eval-python "$BASE/.venv-eval/bin/python" \
 --root "$SUITE" --data "$BASE/data/libero_spatial" --credential-file "$BASE/.swanlab_api_key" \
 --project lpwm-fm-b-world-balance --uv /opt/conda/bin/uv --min-free-gib 3 --estimated-run-gib 2 \
 > "$SUITE/queue.stdout.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$SUITE/queue.pid"
printf 'SWEEP_QUEUE_STARTED pid=%s root=%s\n' "$pid" "$SUITE"
