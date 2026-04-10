#!/bin/bash
# Wait for the corrlabels probe to be retrained, then launch GRPO on GPUs 4,5.
set -euo pipefail

source /data3/ishaplan/medvpq-venv/bin/activate

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="${REPO_ROOT}/final_version/src:/data3/ishaplan/faithscan_vqarad/src:${PYTHONPATH:-}"
export PYTHONPATH

CKPT_BASE="/data3/ishaplan/slake_reproduction/checkpoints"
RESULTS_DIR="${REPRO_DIR}/results"
mkdir -p "${RESULTS_DIR}"

PROBE_PKL="${CKPT_BASE}/spatial_probe_corrlabels/spatial_classifier.pkl"

# Delete old probe so we wait for the new one
rm -f "${PROBE_PKL}"
echo "Deleted old probe. Waiting for new corrlabels probe..."

while [ ! -f "${PROBE_PKL}" ]; do
    echo "  Waiting... ($(date))"
    sleep 60
done
echo "New probe found! Launching corrlabels GRPO on GPUs 4 and 5..."

CUDA_VISIBLE_DEVICES=4 python -m faithscan.train_grpo \
    --config "${REPRO_DIR}/configs/spatial_grpo_a07_corrlabels_seed42.yaml" \
    2>&1 | tee "${RESULTS_DIR}/train_spatial_corr_s42.log" &
PID1=$!

CUDA_VISIBLE_DEVICES=5 python -m faithscan.train_grpo \
    --config "${REPRO_DIR}/configs/spatial_grpo_a07_corrlabels_seed123.yaml" \
    2>&1 | tee "${RESULTS_DIR}/train_spatial_corr_s123.log" &
PID2=$!

echo "  GPU 4: Spatial-corr s42  (PID ${PID1})"
echo "  GPU 5: Spatial-corr s123 (PID ${PID2})"

wait ${PID1} && echo "[OK] Spatial-corr s42" || echo "[FAIL] Spatial-corr s42"
wait ${PID2} && echo "[OK] Spatial-corr s123" || echo "[FAIL] Spatial-corr s123"

echo "Both corrlabels GRPO runs done!"
