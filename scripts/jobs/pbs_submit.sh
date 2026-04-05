#!/bin/bash
#PBS -N gradnorm_sweep
#PBS -q normal
#PBS -P 32000010
#PBS -l select=1:ngpus=1:ncpus=14:mem=200gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /data/projects/32000010/thanhdo/pbsoutput

set -euo pipefail

BASE="/data/projects/32000010/thanhdo"
PROJECT_DIR="${BASE}/grad-norm"
LOG_DIR="${BASE}/logs"
PBS_OUT_DIR="${BASE}/pbsoutput"

mkdir -p "${LOG_DIR}" "${PBS_OUT_DIR}"
cd "${PROJECT_DIR}"

LOG_DIR_TASK="${LOG_DIR}/gradnorm_sweep_${PBS_JOBID:-NA}"
mkdir -p "${LOG_DIR_TASK}"
LOG_FILE="${LOG_DIR_TASK}/sweep.log"
MON_LOG="${LOG_DIR_TASK}/monitor.log"

(
  echo "[START] time=$(date)" >> "${LOG_FILE}"
  # ── GPU monitor ────────────────────────────────────────────────────
  (
    while true; do
      {
        echo "[NVSMI] $(date)"
        nvidia-smi 2>&1
      } > "${MON_LOG}"
      sleep 1
    done
  ) &
  MON_PID=$!

  # ── Run sweep ──────────────────────────────────────────────────────
  source .venv/bin/activate
  CUDA_VISIBLE_DEVICES=0 \
    bash "${PROJECT_DIR}/scripts/gradnorm/sweep_large.sh" >> "${LOG_FILE}" 2>&1

  # ── Cleanup monitor ────────────────────────────────────────────────
  kill "${MON_PID}" >/dev/null 2>&1 || true
  wait "${MON_PID}" >/dev/null 2>&1 || true
  echo "[DONE] time=$(date)" >> "${LOG_FILE}"
  echo "[DONE] time=$(date)" >> "${MON_LOG}"
)
