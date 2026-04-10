#!/bin/bash
# =============================================================================
# SLAKE Spatial GRPO Reproduction — Full Pipeline
# =============================================================================
# Reproduces the April 4-5, 2026 SLAKE organ-only experiment:
#   - Correctness-only GRPO (alpha=1.0) baseline
#   - Spatial GRPO (alpha=0.7) with spatial grounding probe
#   - Two seeds (42, 123) for reproducibility
#
# Server: ssh ishaplan@vlaa-01.be.ucsc.edu
# Setup:  TMUX_TMPDIR=/data3/ishaplan/tmp tmux new -s slake_repro
#
# Expected results (from original run):
#   Corr-only:  peaked at ~0.697 (step 80)
#   Spatial:    peaked at ~0.723 (step 40), +3.7% relative
#
# GPU allocation (using GPUs 3 and 6 only):
#   GPU 3: Probe training (step 0), then correctness-only seed 42, then seed 123
#   GPU 6: Spatial GRPO seed 42, then seed 123
#   Eval: GPU 3 and 6 after training completes
# =============================================================================

set -euo pipefail

# Activate the venv
source /data3/ishaplan/medvpq-venv/bin/activate

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="${REPO_ROOT}/final_version/src:/data3/ishaplan/faithscan_vqarad/src:${PYTHONPATH:-}"
export PYTHONPATH

CKPT_BASE="/data3/ishaplan/slake_reproduction/checkpoints"
RESULTS_DIR="${REPRO_DIR}/results"
mkdir -p "${RESULTS_DIR}"

GPU_A=3
GPU_B=6

echo "============================================"
echo "SLAKE Spatial GRPO Reproduction Pipeline"
echo "============================================"
echo "Repo root:   ${REPO_ROOT}"
echo "PYTHONPATH:   ${PYTHONPATH}"
echo "Checkpoints:  ${CKPT_BASE}"
echo "Results:      ${RESULTS_DIR}"
echo "GPUs:         ${GPU_A}, ${GPU_B}"
echo ""

# ----- Step 0: Train BOTH spatial grounding probes (parallel on 2 GPUs) -----
step0_probe() {
    echo "[Step 0] Training both spatial probes in parallel..."

    # Probe A: bbox_overlap labels (non-circular) on GPU_A
    CUDA_VISIBLE_DEVICES=${GPU_A} python "${REPRO_DIR}/scripts/train_spatial_probe.py" \
        --slake-dir /data3/ishaplan/slake_full/Slake1.0 \
        --output "${CKPT_BASE}/spatial_probe/" \
        --organ-only \
        --labels bbox_overlap \
        --max-examples 1000 \
        --max-new-tokens 256 &
    PID_PA=$!

    # Probe B: correctness labels (original setup) on GPU_B
    CUDA_VISIBLE_DEVICES=${GPU_B} python "${REPRO_DIR}/scripts/train_spatial_probe.py" \
        --slake-dir /data3/ishaplan/slake_full/Slake1.0 \
        --output "${CKPT_BASE}/spatial_probe_corrlabels/" \
        --organ-only \
        --labels correctness \
        --max-examples 1000 \
        --max-new-tokens 256 &
    PID_PB=$!

    echo "  bbox_overlap probe (GPU ${GPU_A}): PID ${PID_PA}"
    echo "  correctness probe  (GPU ${GPU_B}): PID ${PID_PB}"

    wait ${PID_PA} && echo "  [OK] bbox_overlap probe" || echo "  [FAIL] bbox_overlap probe"
    wait ${PID_PB} && echo "  [OK] correctness probe" || echo "  [FAIL] correctness probe"
    echo "[Step 0] Both probes complete."
}

# ----- Step 1: GRPO training (2 runs in parallel per seed) -----
step1_grpo() {
    echo "[Step 1] Launching GRPO training (seed 42)..."

    # Seed 42: corr-only on GPU_A, spatial on GPU_B (parallel)
    CUDA_VISIBLE_DEVICES=${GPU_A} python -m faithscan.train_grpo \
        --config "${REPRO_DIR}/configs/correctness_only_seed42.yaml" \
        > "${RESULTS_DIR}/train_corr_s42.log" 2>&1 &
    PID_CO42=$!

    CUDA_VISIBLE_DEVICES=${GPU_B} python -m faithscan.train_grpo \
        --config "${REPRO_DIR}/configs/spatial_grpo_a07_seed42.yaml" \
        > "${RESULTS_DIR}/train_spatial_s42.log" 2>&1 &
    PID_SP42=$!

    echo "  Corr-only seed42  (GPU ${GPU_A}): PID ${PID_CO42}"
    echo "  Spatial seed42    (GPU ${GPU_B}): PID ${PID_SP42}"
    echo "Waiting for seed 42 runs..."

    wait ${PID_CO42} && echo "  [OK] Corr-only seed42" || echo "  [FAIL] Corr-only seed42"
    wait ${PID_SP42} && echo "  [OK] Spatial seed42" || echo "  [FAIL] Spatial seed42"

    echo ""
    echo "[Step 1] Launching GRPO training (seed 123)..."

    # Seed 123: corr-only on GPU_A, spatial on GPU_B (parallel)
    CUDA_VISIBLE_DEVICES=${GPU_A} python -m faithscan.train_grpo \
        --config "${REPRO_DIR}/configs/correctness_only_seed123.yaml" \
        > "${RESULTS_DIR}/train_corr_s123.log" 2>&1 &
    PID_CO123=$!

    CUDA_VISIBLE_DEVICES=${GPU_B} python -m faithscan.train_grpo \
        --config "${REPRO_DIR}/configs/spatial_grpo_a07_seed123.yaml" \
        > "${RESULTS_DIR}/train_spatial_s123.log" 2>&1 &
    PID_SP123=$!

    echo "  Corr-only seed123 (GPU ${GPU_A}): PID ${PID_CO123}"
    echo "  Spatial seed123   (GPU ${GPU_B}): PID ${PID_SP123}"
    echo "Waiting for seed 123 runs..."

    wait ${PID_CO123} && echo "  [OK] Corr-only seed123" || echo "  [FAIL] Corr-only seed123"
    wait ${PID_SP123} && echo "  [OK] Spatial seed123" || echo "  [FAIL] Spatial seed123"

    echo "[Step 1] All GRPO training complete."
}

# ----- Step 2: Evaluation -----
step2_eval() {
    echo "[Step 2] Running evaluations..."

    # Checkpoint comparisons — seed 42 on GPU_A, seed 123 on GPU_B (parallel)
    CUDA_VISIBLE_DEVICES=${GPU_A} python "${REPRO_DIR}/scripts/compare_checkpoints.py" \
        --corr-ckpt "${CKPT_BASE}/correctness_only/seed42/best" \
        --spatial-ckpt "${CKPT_BASE}/spatial_grpo/a07_seed42/best" \
        --spatial-corr-ckpt "${CKPT_BASE}/spatial_grpo_corrlabels/a07_seed42/best" \
        --n 188 --seed 42 \
        --output "${RESULTS_DIR}/comparison_seed42.json" \
        > "${RESULTS_DIR}/eval_seed42.log" 2>&1 &
    PID_E42=$!

    CUDA_VISIBLE_DEVICES=${GPU_B} python "${REPRO_DIR}/scripts/compare_checkpoints.py" \
        --corr-ckpt "${CKPT_BASE}/correctness_only/seed123/best" \
        --spatial-ckpt "${CKPT_BASE}/spatial_grpo/a07_seed123/best" \
        --spatial-corr-ckpt "${CKPT_BASE}/spatial_grpo_corrlabels/a07_seed123/best" \
        --n 188 --seed 123 \
        --output "${RESULTS_DIR}/comparison_seed123.json" \
        > "${RESULTS_DIR}/eval_seed123.log" 2>&1 &
    PID_E123=$!

    echo "  Eval seed42  (GPU ${GPU_A}): PID ${PID_E42}"
    echo "  Eval seed123 (GPU ${GPU_B}): PID ${PID_E123}"

    wait ${PID_E42} && echo "  [OK] Eval seed42" || echo "  [FAIL] Eval seed42"
    wait ${PID_E123} && echo "  [OK] Eval seed123" || echo "  [FAIL] Eval seed123"

    # Rollout analysis on seed 42 (sequential, uses GPU_A)
    echo "  Running rollout analysis..."
    CUDA_VISIBLE_DEVICES=${GPU_A} python "${REPRO_DIR}/scripts/rollout_analysis.py" \
        --corr-ckpt "${CKPT_BASE}/correctness_only/seed42/best" \
        --spatial-ckpt "${CKPT_BASE}/spatial_grpo/a07_seed42/best" \
        --n 30 --seed 42 \
        --output "${RESULTS_DIR}/rollout_analysis_seed42.json" \
        > "${RESULTS_DIR}/rollout_analysis.log" 2>&1
    echo "  [OK] Rollout analysis"

    echo "[Step 2] All evaluations complete."
}

# ----- Main -----
case "${1:-all}" in
    probe|step0)
        step0_probe
        ;;
    train|step1)
        step1_grpo
        ;;
    eval|step2)
        step2_eval
        ;;
    all)
        step0_probe
        step1_grpo
        step2_eval
        echo ""
        echo "============================================"
        echo "All done! Results in: ${RESULTS_DIR}/"
        echo "============================================"
        ;;
    *)
        echo "Usage: $0 {probe|train|eval|all}"
        exit 1
        ;;
esac
