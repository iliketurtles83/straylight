# Straylight — Agent Notes

Local-first voice AI agent bus. Cass is the resident agent. Pipecat pipeline on host; future services in Docker Compose.

## Commands

```bash
# Install (from repo root)
python3 -m venv .venv
.venv/bin/pip install -r services/voice/requirements.txt

# Run tests (no hardware needed, ~1.6s)
.venv/bin/python -m pytest tests/ -v

# Start voice service (requires LLAMA_MODEL env + llama-server binary)
LLAMA_MODEL=/path/to/model.gguf bash scripts/dev_gemma.sh [--listen] [--no-validate]

# Download wake-word ONNX models
bash scripts/download-models.sh

# Score router corpus (Phase 2 accuracy gate)
.venv/bin/python scripts/score_corpus.py \
  --exemplars exemplars.jsonl --corpus corpus_labeled.jsonl \
  --embed-model models/embed/nomic-embed-text-v1.5.f16.gguf \
  --threshold 0.80 --min-gap 0.05 --required-accuracy 0.90
```

**Always run from repo root.** `services/` is a PEP 420 namespace package (no `__init__.py` in `services/` itself) — imports resolve relative to the workspace root.

## Pipeline

```
LocalAudioTransport.input()
→ [WakeWordProcessor]  (omitted with --listen)
→ VADProcessor
→ WhisperSTTService
→ AgentProcessor
→ PiperTTSService
→ WakeResetRelay
→ LatencyObserver
→ LocalAudioTransport.output()
```

Key source files: `services/voice/main.py` (build_pipeline, entry point), `agent.py` (AgentProcessor), `wake.py` (WakeWordProcessor), `core.py` (VoiceConfig, ConversationWindow), `clients.py` (OpenWakeWordDetector).

## Architecture

- **Voice service** runs natively on host (owns mic, speaker, wake word, STT, TTS, interrupts)
- **AgentProcessor** replaces Phase 1 `OpenAILLMService` scaffolding. Do not reintroduce `LLMContextAggregatorPair` — AgentProcessor owns context via `ConversationWindow` internally.
- **Fast path**: nomic-embed classifier → skill.entities() → skill.execute() → Gemma 4 single-shot format
- **Slow path**: ConversationWindow → Gemma 4 streaming ReAct loop
- nomic-embed runs **in-process** via `llama-cpp-python` (`embedding=True`), called through `asyncio.to_thread()`. No separate embed server.
- **Publisher** (`services/voice/publisher.py`) is a stub logging at DEBUG. Phase 4 wires it to Redis.
- Event schemas: `shared/straylight_shared/events.py` (TranscriptEvent, IntentEvent, StateEvent, SpeakingEvent, TurnDiagnosticsEvent, ToolCallEvent, ToolResultEvent)

## Gotchas

- **`openwakeword==0.4.0` is pinned.** Do not upgrade; 0.6.0 lacks tflite wheels for Python 3.12.
- **PiperTTSService**: use `Settings(voice=model_path.stem)` and `download_dir=model_path.parent`. Full path creates double `.onnx` extension → re-download 404.
- **VADProcessor must be an explicit pipeline stage** between WakeWordProcessor and WhisperSTTService. Whisper only emits TranscriptionFrame after VAD start/stop frames.
- **SileroVADAnalyzer** is a pipeline component only. Never call `.voice_confidence()` standalone — state uninitialized outside VADProcessor.
- **WakeWordFrame does not reach LatencyObserver.** LatencyObserver detects new wake events by comparing `WakeWordProcessor.t_wake` sentinel instead.
- **Wake-word prediction key** is `computer_v2` (not `computer_v2.onnx`). `OpenWakeWordDetector._normalize_label()` handles stripping.
- **Audio device selection** (`_select_audio_devices`) skips PulseAudio/PipeWire/JACK virtual backends — they cause 30s+ startup hangs on Linux.
- **llama-server health timeout is 300s** — large models (35B+) need 2–4 min to warm.
- **Token counting uses llama.cpp `/tokenize`**, not tiktoken. Different models tokenize differently; tiktoken would silently miscount.
- **`dev.sh`** is archived (tuned for Qwen MoE). Use **`dev_gemma.sh`** as the active launcher.
- **`dev_gemma.sh` sets `PYTHONPATH` to repo root** and runs `python -m services.voice.main`. `dev.sh` used `cd services/ && python -m voice.main` — that was Phase 1 convention.
- **Classifier diagnostics must distinguish embedding vs heuristic routing.** Heuristic fallback confidence is `-1.0`, never `1.0`.
- **System prompt loading** uses `{assistant_name}` template variable. No global string replacement.
- **Docker deployment** uses Docker Compose with host network mode and direct audio device access. Requires external llama.cpp server.