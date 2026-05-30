# Straylight — Copilot Onboarding

> Trust these instructions first. Search only if something here is incomplete or appears wrong.

## What this repo is

Straylight is a **local-first voice AI agent bus**. A personal assistant called "Cass" that listens for a wake word, transcribes speech, runs an agent loop, and speaks back via TTS. No cloud on the hot path. Single user, localhost. The voice pipeline runs natively on the host; all other services (future phases) run in Docker Compose.

**Style note:** Write response text in a cyberpunk tone — terse, street-level, technical jargon welcome. But prioritize correctness, clarity, and concise technical guidance. For code, commands, diffs, and risk notes, be plain and precise

## Languages, runtimes, frameworks

- **Python 3.12.3** (system `python3`; do not assume a newer version)
- **Pipecat 1.2.1** — frame-processing audio pipeline
- **OpenWakeWord 0.4.0** — pinned; 0.6.0 has no tflite wheels for Python 3.12
- **faster-whisper 1.2.1** — STT
- **sounddevice 0.5.5**, **loguru 0.7.3**, **httpx 0.28.1**
- **llama.cpp / llama-server** (external binary; not in repo) — OpenAI-compatible HTTP server

## Bootstrap / install

```bash
# From repo root — only needs to be done once
python3 -m venv .venv
.venv/bin/pip install -r services/voice/requirements.txt
```

Always run commands from the **repo root** — the `services/` namespace package resolves relative to it.

## Running tests — ALWAYS do this to validate changes

```bash
# From repo root — 15 unit tests, ~1.6s, no hardware required
.venv/bin/python -m pytest tests/ -v
```

`tests/test_voice_core.py` is the only pytest suite. The other files in `tests/` require real audio hardware; do not run them in CI. Tests import as `from services.voice.core import …` — `services/` is a PEP 420 namespace package (no `__init__.py`).

There is **no pyproject.toml, pytest.ini, or setup.cfg**. No linting is enforced. No GitHub Actions CI exists yet.

## Running the voice service

```bash
# Requires: LLAMA_MODEL pointing to a .gguf file; LLAMA_SERVER binary
LLAMA_MODEL=/path/to/model.gguf bash scripts/dev.sh [--listen] [--no-validate]
```

`scripts/dev.sh` starts llama-server (port 8080, alias `cass`), polls `/health` up to 300 s (large models need 2–4 min), then starts the voice service as `python -m voice.main` from `services/`. `--listen` skips wake word (always-on STT). `Ctrl-C` tears down everything cleanly.

## Project layout

```
.env / .env.example          # All runtime config; copy .env.example to .env
services/voice/              # NATIVE host process — the only active service
  main.py                    # Pipeline entry point (Phase 1; LLM wiring is scaffolding)
  wake.py                    # WakeWordProcessor (custom FrameProcessor)
  core.py                    # VoiceConfig.from_env(), ConversationWindow
  clients.py                 # OpenWakeWordDetector, VoiceDependencyError
  cass_prompt.txt            # Cass persona system prompt
  requirements.txt           # pip deps for the voice service
tests/test_voice_core.py     # Only automated tests; run with pytest from repo root
scripts/dev.sh               # Dev launcher (llama-server + voice service)
scripts/download-models.sh   # Downloads wake-word ONNX files to models/wake/
models/wake/                 # .gitignored; computer_v2.onnx + melspectrogram.onnx + embedding_model.onnx
models/tts/                  # .gitignored; en_US-amy-medium.onnx + ack.mp3
PLAN.md                      # Authoritative roadmap; read before adding new features
```

Services under `services/memory/`, `services/gateway/`, `services/tools/` are **planned but not yet implemented**. The `shared/` directory is currently empty.

## Key gotchas — read before touching the pipeline

- **`openwakeword==0.4.0` is pinned.** Do not upgrade; 0.6.0 has no tflite wheels for Python 3.12.
- **`PiperTTSService`**: use `Settings(voice=model_path.stem)` and `download_dir=model_path.parent`. Passing a full path creates a double `.onnx` extension and triggers a re-download 404.
- **`SileroVADAnalyzer`** is a pipeline component only. Never call `.voice_confidence()` standalone — its internal state is uninitialized outside a `VADProcessor`.
- **`VADProcessor` must be an explicit pipeline stage** between `WakeWordProcessor` and `WhisperSTTService`. `WhisperSTTService` only emits `TranscriptionFrame` after `VADUserStartedSpeakingFrame` / `VADUserStoppedSpeakingFrame`.
- **Audio device auto-selection**: `_select_audio_devices()` in `main.py` skips virtual backends (PulseAudio/PipeWire/JACK) which cause 30 s+ startup hangs on Linux.
- **Wake-word prediction key** resolves to `computer_v2` (not `computer_v2.onnx`). `OpenWakeWordDetector._normalize_label()` handles this.
- **`WakeWordFrame` does not propagate past `PiperTTSService`**. Use the `t_wake` sentinel comparison in `LatencyObserver` instead.
- **Phase 1 LLM wiring (`OpenAILLMService`) is scaffolding.** Phase 2 replaces it with `AgentProcessor`. Do not build on top of it.
- **`InputAudioRawFrame(audio, sample_rate, num_channels)`** — no `num_frames` kwarg in Pipecat 1.2.1.
- **`LLMContextAggregatorPair(context).user()` / `.assistant()`** for pipeline placement.
- **llama-server health timeout is 300 s** — large models (35B+) need 2–4 min to warm.
- **`WhisperSTTService`** import path: `from pipecat.services.whisper.stt import WhisperSTTService`.

## Environment variables (key subset; see `.env.example` for full list)

| Variable | Default | Notes |
|---|---|---|
| `LLAMA_MODEL` | — | **Required.** Absolute path to `.gguf` model file |
| `LLAMA_SERVER` | `~/Projects/llama.cpp/build/bin/llama-server` | Absolute path to binary |
| `LLAMA_PORT` | `8080` | `CASS_LLM_BASE_URL` is set automatically from this |
| `CASS_LLM_MODEL` | `cass` | Must match `--alias` passed to llama-server |
| `WHISPER_MODEL` | `base.en` | faster-whisper auto-downloads on first run |
| `CASS_WAKE_THRESHOLD` | `0.5` | OWW confidence threshold |

## Model downloads

```bash
bash scripts/download-models.sh   # Wake-word ONNX models → models/wake/
# TTS model: place en_US-amy-medium.onnx + .json + ack.mp3 in models/tts/ manually
# STT: faster-whisper auto-downloads base.en on first pipeline run
```