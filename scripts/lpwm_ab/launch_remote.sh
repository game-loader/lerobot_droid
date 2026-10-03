#!/usr/bin/env bash
# Launch two independent, one-GPU experiments; credentials stay outside the source tree.
set -euo pipefail
ROOT=${LPWM_RUN_ROOT:-/root/lpwm_ab}
PYTHON="$ROOT/.venv/bin/python"
UV=${UV_BIN:-/opt/conda/bin/uv}
STAMP=${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}
cd "$ROOT/repo"
export PYTHONPATH="$ROOT/repo/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

"$PYTHON" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
results=[]
for variant in ('A','B'):
    p=root/'runs'/f'preflight-{variant}'
    assert json.loads((p/'status.json').read_text())['status']=='completed'
    assert (p/'checkpoints/latest/model.safetensors').is_file()
    results.append(json.loads((p/'experiment.json').read_text()))
assert results[0]['initial_weights_sha256']==results[1]['initial_weights_sha256']
assert results[0]['split_sha256']==results[1]['split_sha256']
print('Paired CUDA training/validation/checkpoint gates passed.')
PY

for GPU in 0 1; do
    if [[ "$GPU" == 0 ]]; then VARIANT=A; WORLD_WEIGHT=0; else VARIANT=B; WORLD_WEIGHT=0.1; fi
    OUTPUT="$ROOT/runs/${STAMP}-${VARIANT}"
    LOG="$ROOT/runs/${STAMP}-${VARIANT}.log"
    [[ ! -e "$OUTPUT" ]] || { echo "Refusing existing output: $OUTPUT" >&2; exit 1; }
    nohup env CUDA_VISIBLE_DEVICES="$GPU" \
        "$UV" run --no-project --no-config -- "$PYTHON" -u scripts/lpwm_ab/train.py \
        --data "$ROOT/data/libero_spatial" --output "$OUTPUT" --variant "$VARIANT" \
        --steps 20000 --batch-size 8 --grad-accumulation 4 --workers 4 --seed 42 \
        --world-weight "$WORLD_WEIGHT" --world-ramp-steps 1000 \
        --validate-every 500 --validation-batches 8 --save-every 2000 --log-every 10 \
        --swanlab-mode online --swanlab-project lpwm-fm-libero-spatial-ab \
        --run-name "LPWM-FM-${VARIANT}-Spatial-seed42-${STAMP}" \
        --credential-file "$ROOT/.swanlab_api_key" > "$LOG" 2>&1 < /dev/null &
    PID=$!
    printf '%s\n' "$PID" > "$ROOT/runs/${STAMP}-${VARIANT}.pid"
    printf 'STARTED variant=%s gpu=%s supervisor_pid=%s output=%s log=%s\n' "$VARIANT" "$GPU" "$PID" "$OUTPUT" "$LOG"
done
printf '%s\n' "$STAMP" > "$ROOT/runs/current_ab_stamp"
