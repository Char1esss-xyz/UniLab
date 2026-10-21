#!/usr/bin/env bash
set -euo pipefail

# Run from any working directory while keeping Hydra paths rooted at UniLab.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${DEVICE:=cuda}"
: "${NUM_ENVS:=4096}"
: "${MAX_ITERATIONS:=20000}"
: "${SAVE_INTERVAL:=500}"
: "${LOG_ROOT:=}"
: "${RESUME_PATH:=}"
: "${MJWARP_NCONMAX:=}"
: "${MJWARP_NJMAX:=}"

if [[ "${DEVICE}" != cuda && "${DEVICE}" != cuda:* ]]; then
    echo "MJWarp requires a CUDA learner device; got DEVICE=${DEVICE}" >&2
    exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found; CUDA/MJWarp may be unavailable" >&2
fi

args=(
    --sim mjwarp
    --device "${DEVICE}"
    --num-envs "${NUM_ENVS}"
    --max-iterations "${MAX_ITERATIONS}"
    "algo.save_interval=${SAVE_INTERVAL}"
)

if [[ -n "${LOG_ROOT}" ]]; then
    args+=(--log-root "${LOG_ROOT}")
fi
if [[ -n "${RESUME_PATH}" ]]; then
    args+=(--resume-path "${RESUME_PATH}")
fi
if [[ -n "${MJWARP_NCONMAX}" ]]; then
    args+=("env.mjwarp_nconmax=${MJWARP_NCONMAX}")
fi
if [[ -n "${MJWARP_NJMAX}" ]]; then
    args+=("env.mjwarp_njmax=${MJWARP_NJMAX}")
fi

echo "Starting Wheelbipe V14 PPO with MJWarp: device=${DEVICE}, envs=${NUM_ENVS}, iterations=${MAX_ITERATIONS}"
exec uv run train "${args[@]}" "$@"
