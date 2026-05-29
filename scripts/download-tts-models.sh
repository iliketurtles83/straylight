#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/models/tts"

mkdir -p "$TARGET_DIR"

echo "Phase 1 uses Piper, not Kokoro."
echo "Place a local Piper voice model at: ${TTS_PIPER_MODEL:-$TARGET_DIR/piper-model.onnx}"
echo "Optional acknowledgement sounds can live in: $ROOT_DIR/assets/audio/"
