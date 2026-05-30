#!/usr/bin/env bash
# Straylight Phase 1 dev launcher.
#
# Usage:
#   bash scripts/dev.sh                  (reads .env, wake word mode)
#   bash scripts/dev.sh --listen         (skip wake word, always-on STT)
#   bash scripts/dev.sh --no-validate    (skip startup asset checks)
#   bash scripts/dev.sh --listen --no-validate
#   LLAMA_MODEL=... bash scripts/dev.sh
#
# Starts llama-server with performance-tuned flags, waits for /health,
# then starts the voice service. Ctrl-C tears down everything.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${REPO_ROOT}/.venv/bin/python"

# Load .env from repo root if present.
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -o allexport
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.env"
  set +o allexport
fi

# --------------------------------------------------------------------------
# Argument parsing — flags forwarded verbatim to voice.main
# --------------------------------------------------------------------------
VOICE_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --listen)        VOICE_ARGS+=("--listen") ;;
    --no-validate)   VOICE_ARGS+=("--no-validate") ;;
    --help|-h)
      echo "Usage: $0 [--listen] [--no-validate]"
      echo "  --listen       Skip wake word; STT always active"
      echo "  --no-validate  Skip startup asset/service checks"
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: ${arg}" >&2
      echo "Run '$0 --help' for usage." >&2
      exit 1
      ;;
  esac
done

# --------------------------------------------------------------------------
# Configuration — all overridable via .env
# --------------------------------------------------------------------------
: "${LLAMA_SERVER:=llama-server}"

# Network
: "${LLAMA_PORT:=8080}"
: "${LLAMA_ALIAS:=cass}"

# Performance flags (proven config: ~25 t/s on RTX + hybrid MoE)
: "${LLAMA_GPU_LAYERS:=99}"   # layers to offload to GPU
: "${LLAMA_CPU_MOE:=31}"      # MoE expert layers kept on CPU
: "${LLAMA_CTX_SIZE:=80128}"  # context window
: "${LLAMA_THREADS:=3}"       # CPU inference threads
: "${LLAMA_BATCH:=512}"       # logical batch size

# Health-check
: "${LLAMA_HEALTH_TIMEOUT:=300}"  # seconds; 35B+ needs ~2-3 min to warm up
HEALTH_URL="http://127.0.0.1:${LLAMA_PORT}/health"
HEALTH_INTERVAL=2

# Expose base URL so VoiceConfig picks up the right port automatically.
export CASS_LLM_BASE_URL="http://127.0.0.1:${LLAMA_PORT}"

# --------------------------------------------------------------------------
# Validate environment
# --------------------------------------------------------------------------
if [[ -z "${LLAMA_MODEL:-}" ]]; then
  echo "ERROR: LLAMA_MODEL env var must point to a .gguf model file." >&2
  exit 1
fi

if [[ ! -x "${LLAMA_SERVER}" ]]; then
  echo "ERROR: llama-server binary not found or not executable: ${LLAMA_SERVER}" >&2
  echo "  Set LLAMA_SERVER=/path/to/llama-server and retry." >&2
  exit 1
fi

if [[ ! -f "${LLAMA_MODEL}" ]]; then
  echo "ERROR: LLAMA_MODEL not found: ${LLAMA_MODEL}" >&2
  exit 1
fi

if [[ ! -x "${VENV}" ]]; then
  echo "ERROR: venv Python not found at ${VENV}" >&2
  echo "Run: python -m venv .venv && .venv/bin/pip install -r services/voice/requirements.txt" >&2
  exit 1
fi

# --------------------------------------------------------------------------
# Start llama-server (skip if already healthy on the target port)
# --------------------------------------------------------------------------
LLAMA_PID=""
if curl -sf "${HEALTH_URL}" > /dev/null 2>&1; then
  echo "[dev.sh] llama-server already healthy on ${HEALTH_URL} — skipping launch"
else
  echo "[dev.sh] starting llama-server"
  echo "  model:   ${LLAMA_MODEL}"
  echo "  port:    ${LLAMA_PORT}  gpu-layers: ${LLAMA_GPU_LAYERS}  cpu-moe: ${LLAMA_CPU_MOE}"
  echo "  ctx:     ${LLAMA_CTX_SIZE}  threads: ${LLAMA_THREADS}  batch: ${LLAMA_BATCH}"

  "${LLAMA_SERVER}" \
    --model "${LLAMA_MODEL}" \
    --host 127.0.0.1 \
    --port "${LLAMA_PORT}" \
    --alias "${LLAMA_ALIAS}" \
    --main-gpu 0 \
    --n-gpu-layers "${LLAMA_GPU_LAYERS}" \
    --n-cpu-moe "${LLAMA_CPU_MOE}" \
    --ctx-size "${LLAMA_CTX_SIZE}" \
    --flash-attn on \
    --no-mmap \
    --jinja \
    --threads "${LLAMA_THREADS}" \
    --metrics \
    --batch-size "${LLAMA_BATCH}" \
    --no-mmproj-offload \
    &
  LLAMA_PID=$!
  echo "[dev.sh] llama-server PID=${LLAMA_PID}"
fi

# Only kill llama-server on exit if we were the ones who started it.
VOICE_PID=""
CLEANED_UP=0

cleanup() {
  # Avoid double cleanup when a signal handler exits and then triggers EXIT.
  if (( CLEANED_UP == 1 )); then
    return
  fi
  CLEANED_UP=1

  if [[ -n "${VOICE_PID}" ]]; then
    echo "[dev.sh] stopping voice service (PID=${VOICE_PID})"
    kill "${VOICE_PID}" 2>/dev/null || true
    wait "${VOICE_PID}" 2>/dev/null || true
  fi

  if [[ -n "${LLAMA_PID}" ]]; then
    echo "[dev.sh] stopping llama-server (PID=${LLAMA_PID})"
    kill "${LLAMA_PID}" 2>/dev/null || true
    wait "${LLAMA_PID}" 2>/dev/null || true
  fi
}

on_signal() {
  local sig="$1"
  echo "[dev.sh] received ${sig}; shutting down"
  cleanup
  exit 130
}

trap cleanup EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

# --------------------------------------------------------------------------
# Wait for llama-server /health
# --------------------------------------------------------------------------
echo "[dev.sh] waiting for ${HEALTH_URL} (max ${LLAMA_HEALTH_TIMEOUT}s — large models take a while)..."
elapsed=0
until curl -sf "${HEALTH_URL}" > /dev/null 2>&1; do
  if (( elapsed >= LLAMA_HEALTH_TIMEOUT )); then
    echo "ERROR: llama-server did not become healthy after ${LLAMA_HEALTH_TIMEOUT}s" >&2
    exit 1
  fi
  sleep "${HEALTH_INTERVAL}"
  (( elapsed += HEALTH_INTERVAL ))
  echo "[dev.sh] still loading... ${elapsed}s / ${LLAMA_HEALTH_TIMEOUT}s"
done
echo "[dev.sh] llama-server healthy after ${elapsed}s"

# --------------------------------------------------------------------------
# Start the voice service
# --------------------------------------------------------------------------
echo "[dev.sh] starting voice service (args: ${VOICE_ARGS[*]:-none})"
cd "${REPO_ROOT}/services"
"${VENV}" -m voice.main "${VOICE_ARGS[@]}" &
VOICE_PID=$!

set +e
wait "${VOICE_PID}"
VOICE_EXIT=$?
set -e
exit "${VOICE_EXIT}"
