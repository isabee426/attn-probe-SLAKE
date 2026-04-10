#!/bin/bash
# Launch all GRPO training after BOTH probes are done.
# 3 conditions x 2 seeds = 6 runs on 6 GPUs, all parallel.
#
# Conditions:
#   1. Correctness-only (α=1.0) — baseline
#   2. Spatial GRPO (α=0.7) with bbox_overlap probe — non-circular
#   3. Spatial GRPO (α=0.7) with correctness-labeled probe — original setup
set -euo pipefail

source /data3/ishaplan/medvpq-venv/bin/activate

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="${REPO_ROOT}/final_version/src:/data3/ishaplan/faithscan_vqarad/src:${PYTHONPATH:-}"
export PYTHONPATH

CKPT_BASE="/data3/ishaplan/slake_reproduction/checkpoints"
RESULTS_DIR="${REPRO_DIR}/results"
mkdir -p "${RESULTS_DIR}"

# Wait for BOTH probes to exist
echo "Waiting for both spatial probes..."
while [ ! -f "${CKPT_BASE}/spatial_probe/spatial_classifier.pkl" ]; do
    echo "  Waiting for bbox_overlap probe... ($(date))"
    sleep 60
done
echo "  bbox_overlap probe found."

while [ ! -f "${CKPT_BASE}/spatial_probe_corrlabels/spatial_classifier.pkl" ]; do
    echo "  Waiting for correctness-labeled probe... ($(date))"
    sleep 60
done
echo "  correctness-labeled probe found."
echo ""

# ---- All 6 runs in parallel on GPUs 0-5 ----
echo "========================================"
echo "Launching all 6 GRPO runs in parallel"
echo "========================================"

CUDA_VISIBLE_DEVICES=0 python -m faithscan.train_grpo \
    --config "${REPRO_DIR}/configs/correctness_only_seed42.yaml" \
    2>&1 | tee "${RESULTS_DIR}/train_corr_s42.log" &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python -m faithscan.train_grpo \
    --config "${REPRO_DIR}/configs/correctness_only_seed123.yaml" \
    2>&1 | tee "${RESULTS_DIR}/train_corr_s123.log" &
PID2=$!

CUDA_VISIBLE_DEVICES=2 python -m faithscan.train_grpo \
    --config "${REPRO_DIR}/configs/spatial_grpo_a07_seed42.yaml" \
    2>&1 | tee "${RESULTS_DIR}/train_spatial_bbox_s42.log" &
PID3=$!

CUDA_VISIBLE_DEVICES=3 python -m faithscan.train_grpo \
    --config "${REPRO_DIR}/configs/spatial_grpo_a07_seed123.yaml" \
    2>&1 | tee "${RESULTS_DIR}/train_spatial_bbox_s123.log" &
PID4=$!

CUDA_VISIBLE_DEVICES=4 python -m faithscan.train_grpo \
    --config "${REPRO_DIR}/configs/spatial_grpo_a07_corrlabels_seed42.yaml" \
    2>&1 | tee "${RESULTS_DIR}/train_spatial_corr_s42.log" &
PID5=$!

CUDA_VISIBLE_DEVICES=5 python -m faithscan.train_grpo \
    --config "${REPRO_DIR}/configs/spatial_grpo_a07_corrlabels_seed123.yaml" \
    2>&1 | tee "${RESULTS_DIR}/train_spatial_corr_s123.log" &
PID6=$!

echo "  GPU 0: Corr-only s42        (PID ${PID1})"
echo "  GPU 1: Corr-only s123       (PID ${PID2})"
echo "  GPU 2: Spatial-bbox s42     (PID ${PID3})"
echo "  GPU 3: Spatial-bbox s123    (PID ${PID4})"
echo "  GPU 4: Spatial-corr s42     (PID ${PID5})"
echo "  GPU 5: Spatial-corr s123    (PID ${PID6})"
echo ""
echo "Waiting for all 6 runs..."

wait ${PID1} && echo "[OK] Corr-only s42" || echo "[FAIL] Corr-only s42"
wait ${PID2} && echo "[OK] Corr-only s123" || echo "[FAIL] Corr-only s123"
wait ${PID3} && echo "[OK] Spatial-bbox s42" || echo "[FAIL] Spatial-bbox s42"
wait ${PID4} && echo "[OK] Spatial-bbox s123" || echo "[FAIL] Spatial-bbox s123"
wait ${PID5} && echo "[OK] Spatial-corr s42" || echo "[FAIL] Spatial-corr s42"
wait ${PID6} && echo "[OK] Spatial-corr s123" || echo "[FAIL] Spatial-corr s123"

echo ""
echo "========================================"
echo "All 6 GRPO runs complete!"
echo "Run: bash run_all.sh eval"
echo "========================================"
