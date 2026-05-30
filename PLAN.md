# Straylight — Project Plan

> Local-first personal agent bus. Cass is the first resident agent. Voice is a first-class interface, not the whole product.

---

## Overview

Straylight is a personal agent bus: a local-first runtime where agents, tools, memory, models, and user interfaces communicate through shared events and stable service contracts. Cass is the first/default agent on the bus. Voice is a high-value interface for Cass, but the product is bigger than voice: the browser UI is the primary chat, control, and diagnostics surface, and future adapters can enter through the same bus without calling models directly.

**Three-layer architecture:**
- **Surfaces** — Native voice, chat-first browser UI, and future adapters such as Signal, Discord, Telegram, or CLI. Surfaces submit user turns and render events; they do not own reasoning or call llama-server directly.
- **Agent bus** — Sessions, routing, turn lifecycle, diagnostics, event schemas, agent/task dispatch, and tool/memory orchestration. Cass is the first runtime on the bus; additional agents come later only after Cass is stable.
- **Services** — Voice I/O, local LLM servers, optional downstream cloud providers, MCP tool servers, memory stores, gateway, Redis, and telemetry collectors.

**Infrastructure split:**
- Voice service runs **natively on the host** — it owns microphone, speaker, wake word, STT, TTS, and low-latency interruption.
- Browser gateway, MCP tool servers, memory, Redis, diagnostics, and future agent services run in **Docker Compose** where practical.
- The boundary is **Redis pub/sub plus HTTP/MCP service calls**: surfaces publish input events, runtimes publish turn events, and services subscribe or expose explicit APIs.

**Product direction:**
- Agent bus first; Cass-first appliance second; open-source dev kit third.
- Chat-first browser UI served by the gateway, accessible from a phone or second screen on the local network. No desktop app.
- Voice remains a flagship interface: Hearth-grade wake/listen/transcribe/speak behavior is mandatory.
- llama-server-style transparency is mandatory: context counter, token counts, tokens/sec, latency, model/provider labels, tool timeline, memory injection details, and trace/reasoning summaries where available.
- Predictable setup over lowest latency — clean architecture over micro-optimisations.
- Local-first means local preferred/default, not local-only. Gemma 4 is the primary local model; nomic-embed is used where embeddings clearly help; Qwen3.6 is reserved as local heavy artillery for long-context or difficult work.
- External APIs such as Claude or ChatGPT are deferred until the local bus is stable. When added, they are explicit user-selected escalation paths per query or session, never hidden default dependencies.
- All input surfaces talk to Cass through the same agent/event bus. No surface calls llama-server, cloud APIs, or tools directly.
- llama-server's own UI remains useful as a low-level model console; Straylight's UI becomes the normal chat/control/diagnostics surface.

**Saved design principles:**
- Keep the real-time audio path separate from async orchestration. Pipecat owns microphone, speaker, VAD, wake word, STT, TTS, and interrupts; the agent/event bus owns routing, tools, memory, and UI fan-out.
- Build the parts that teach Straylight's core lessons: local voice UX, embedding-based routing, event-driven service boundaries, MCP tool integration, and memory consolidation. Use proven systems for everything else.
- Prefer boring, measurable reliability over research-demo cleverness. Every new layer must state the latency it adds and how it fails out loud.
- Treat local-first as a priority stack, not a cage. Local models and local services get first claim; explicit downstream cloud escalation can exist after local reliability, observability, and cost/privacy controls are real.

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Browser UI | **Chat-first web app** | Cass chat, sessions, diagnostics, voice controls, tool/memory timeline |
| Gateway | **FastAPI** SSE + HTTP | Chat input, event stream, diagnostics cache, settings, future adapter bridge |
| Voice I/O | **Pipecat** | Frame-processing pipeline; owns audio only |
| Audio I/O | Pipecat `LocalTransport` (sounddevice) | Mic + speaker directly on host |
| VAD | Pipecat `SileroVAD` | Built-in; gates wake word |
| Wake word | **OpenWakeWord** | Custom `FrameProcessor`; models in `models/wake/` |
| STT | Pipecat `WhisperSTTService` (faster-whisper) | Models in `models/stt/` |
| TTS | Pipecat `PiperTTSService` | Streaming, sentence-by-sentence; models in `models/tts/` |
| Agent | **AgentProcessor** | Custom `FrameProcessor`; bridges Pipecat I/O to intelligence layer |
| Fast-path classifier | **nomic-embed** via llama-cpp-python | Embedding similarity; in-process inside AgentProcessor; no HTTP overhead |
| Fast-path NER | **spaCy** (`en_core_web_sm`) | Entity extraction (location, time, quantity) for skills; ~20ms |
| Primary local LLM | **llama-server** (Gemma 4, gemma-4-E4B) | Default Cass formatter + reasoning loop; port 8080 |
| Heavy local LLM | **llama-server** (Qwen3.6) | Optional high-context/deep-reasoning escalation; separate port when enabled |
| Optional cloud LLMs | Claude / ChatGPT APIs | Deferred until local stack is stable; explicit user-selected escalation only |
| Tool protocol | **MCP** (Model Context Protocol) | Standard tool interface; all tool services expose MCP endpoints |
| Event bus | **Redis** pub/sub | Introduced Phase 4; voice → gateway IPC bridge |
| Diagnostics | Redis cache + event stream | Context tokens, token/sec, latency waterfall, model/provider, tools, memory |
| Memory: episodic | **SQLite** | Full conversation turn log + extracted facts table; never truncated |
| Memory: graph | **Kuzu** | Personal knowledge graph; entities + relationships; embedded, no server |
| Orchestration | **Docker Compose** | All services except voice pipeline |

---

## Project Structure

```
straylight/
├── services/
│   ├── voice/                  # Native host process — Pipecat pipeline
│   │   ├── main.py             # Pipeline entry point; AgentProcessor wired here
│   │   ├── wake.py             # WakeWordProcessor (custom FrameProcessor)
│   │   ├── agent.py            # AgentProcessor: fast path + slow path ReAct loop
│   │   ├── skills/             # Phase 2 in-process fast-path skill bundles
│   │   │   ├── __init__.py     # Skill base class + registry
│   │   │   └── weather.py      # TEMP: WeatherSkill lives here until Phase 3 MCP extraction
│   │   ├── publisher.py        # Redis event publisher (Phase 4)
│   │   ├── cass_prompt.txt     # System prompt (Cass persona)
│   │   └── requirements.txt
│   ├── memory/
│   │   ├── api/                # MCP server: retrieve facts from Kuzu
│   │   ├── sleep.py            # Consolidation worker: idle → LLM extract → Kuzu merge
│   │   ├── episodic.py         # SQLite: turns + facts tables
│   │   └── graph.py            # Kuzu graph interface: upsert entities + relationships
│   ├── gateway/
│   │   ├── main.py             # FastAPI: chat input, SSE events, diagnostics cache
│   │   └── requirements.txt
│   └── tools/
│       └── weather/            # Phase 3 target home for weather tool service
│           ├── main.py         # MCP server wrapping Open-Meteo adapter
│           ├── adapter.py      # Open-Meteo geocode + fetch (from imported_code)
│           └── requirements.txt
├── models/
│   ├── wake/                   # OpenWakeWord .onnx files + embedding model
│   ├── stt/                    # Whisper model
│   ├── tts/                    # Piper model + ack.mp3
│   └── embed/                  # nomic-embed .gguf (fast-path classifier + memory retrieval)
├── shared/
│   └── straylight_shared/
│       ├── events.py           # Canonical event schemas
│       ├── redis_client.py
│       └── logging.py
├── frontend/                   # Chat-first Cass UI + diagnostics surface
│   ├── index.html
│   ├── style.css
│   └── app.js
├── infra/
│   ├── redis/redis.conf
│   └── kuzu/                   # Kuzu data directory (mounted volume)
├── scripts/
│   ├── download-models.sh
│   └── dev_gemma.sh            # Start Gemma 4 llama-server + docker compose + voice
├── tests/
│   ├── test_voice_core.py
│   └── ...
└── docker-compose.yml
```

---

## Cass Runtime & Agent Bus Design

`AgentProcessor` is the first Cass runtime, not the final abstraction for every future agent. In the short term, Pipecat sees it as an opaque `FrameProcessor`: `TranscriptionFrame` in, `TextFrame`s out. In the bus model, it is the component that turns a user turn into routed work, tool calls, diagnostics, memory events, and response text. Browser input and voice transcripts should converge on the same turn handler as soon as Phase 4 lands.

Internally Cass runs a two-path local decision:

```
TranscriptionFrame arrives
│
├─ optional nomic-embed classifier (~50ms) — is this a known skill with high confidence?
│
├─ FAST PATH (high-confidence skill match)
│   ├─ spaCy NER extracts entities from transcript (~20ms)
│   ├─ Phase 2: Skill returns local stub/temporary adapter output
│   ├─ Phase 3+: Skill calls MCP tool server(s) directly
│   ├─ Gemma 4 (port 8080, Cass-prompted) formats response from structured tool result
│   └─ emit TextFrames → TTS
│
└─ SLOW PATH (conversational / ambiguous / multi-step)
    ├─ Gemma 4 (port 8080) with MCP tool registry
    ├─ ReAct loop: reason → call tool → observe → reason → ... → done
    └─ emit TextFrames as they stream → TTS
```

Qwen3.6 is not part of the normal turn path. It is reserved for explicit local heavy-artillery escalation after Gemma 4 proves insufficient for a high-context or difficult request. External APIs are out of scope until the local bus is stable.

**Skills** are named fast-path bundles. In Phase 2 they live under `services/voice/skills/` because `AgentProcessor` is still in the native voice process. Phase 3 must split durable tools out to MCP services under `services/tools/`; `services/voice/skills/` then becomes a thin fast-path client layer, not the home for tool adapters.

Each `Skill` owns:
- Embedding exemplars for the classifier
- spaCy entity patterns
- Phase 2 stub/temporary adapter logic, then Phase 3 MCP tool call(s)
- Format prompt for Cass's voice (Gemma 4 formats all responses)

Adding a skill never touches the classifier thresholds — it registers exemplars into the shared embedding index at startup.

**Tools** are MCP servers once Phase 3 lands. Both the fast path and the slow path use the same MCP servers. The slow-path large LLM discovers available tools via `tools/list`.

**When to use which path:**
- Fast path: clear tool intent + extractable entities + structured response (weather, time, timer)
- Slow path: conversational, ambiguous, multi-step, or any query the classifier scores below threshold
- Heavy local path: explicit user or runtime escalation to Qwen3.6 for long-context or high-difficulty work after local Gemma 4 limits are visible

**Assistant state lifecycle:**

`StateEvent` is the canonical state contract for the gateway UI, memory worker, and future adapters. The expected lifecycle is:

```
idle → listening → transcribing → thinking → tool_calling → speaking → idle
```

Not every turn visits every state: pure conversation may skip `tool_calling`; browser text input skips wake/listen/STT but still enters `thinking`; interruption returns the active turn to `idle` before the next transcript is processed. Do not add a separate dialogue service just to track state — publish better events from `AgentProcessor` and the voice pipeline.

---


## Features

---

### Phase 1 — Voice I/O Validation ✓ COMPLETE

**Goal:** Validate the full audio stack. Wake word triggers, ack sound plays, STT transcribes, TTS speaks. Prove Pipecat handles the hardware correctly.

**Note:** Phase 1 went slightly further than the original intent — it wired `OpenAILLMService` directly into the pipeline and produced a working conversational loop. This was useful for validating end-to-end latency but the `OpenAILLMService` wiring is **scaffolding**. Phase 2 replaces it with `AgentProcessor`. The Phase 1 pipeline does not survive into Phase 3.

**What was built:**
- `services/voice/` with `wake.py`, `core.py`, `clients.py`, `main.py`
- `WakeWordProcessor` — custom `FrameProcessor` wrapping OpenWakeWord; plays `ack.mp3` on trigger
- `LatencyObserver` — logs `wake_to_stt_ms`, `stt_to_ttfb_ms` per turn
- `validate_startup()` — checks model files, audio device, llama-server health
- `VoiceConfig.from_env()` — all config from environment variables
- `scripts/dev.sh` — starts llama-server, polls `/health`, starts voice service

**Exit criteria met:**
- [x] Wake word detected reliably from 1–3m in a quiet room; **< 2 false triggers per hour** over 30 minutes of continuous ambient noise
- [x] STT transcript arrives within 500ms of speech end
- [x] First audio word of Cass's response plays within 2s of transcript
- [x] Interrupt handling works: speaking while Cass is talking stops TTS and begins new STT cycle
- [x] Pipeline runs 30 consecutive turns without crash, memory growth, or audio dropout
- [x] Per-turn latency (`wake_to_stt_ms`, `stt_to_ttfb_ms`) logged to stdout in structured form

**Baseline to preserve:**
- Phase 1 is the hardware baseline. Before adding a new agent, tool, gateway, or memory layer, re-check wake reliability, STT latency, first-audio latency, interrupt handling, audio dropouts, memory growth, model warmup time, and CPU/GPU headroom.
- Wake acknowledgement playback is part of that baseline. `WakeWordProcessor` currently shells out to `ffplay`; startup validation should either verify `ffplay` is available and log non-zero playback exits loudly, or replace it with a Python/audio-stack fallback. A missing or broken ack player must not look like failed wake detection.
- Do not chase kernel real-time rewrites until measurements prove Pipecat is the bottleneck. The archived hard-real-time plan is a diagnostic checklist, not the default architecture.

---

### Phase 2 — Cass Runtime & Routing ⚠ HARDENING PENDING

**Goal:** Replace the Phase 1 `OpenAILLMService` scaffolding with the first Cass runtime. Establish canonical turn handling, local fast/slow routing, structured event emission, and diagnostics metadata for the future chat UI. Implement `WeatherSkill` as the first in-process skill client (stub or temporary local adapter at end of this phase — durable weather tool moves to MCP in Phase 3).

**Current status (reviewed May 30, 2026):** The main AgentProcessor wiring is in place, but Phase 2 is not fully closed. Token-budget conversation trimming and `/tokenize` startup validation are implemented; remaining hardening: prove the router corpus/accuracy gate with real Whisper transcripts, separate heuristic routing from real embedding confidence in diagnostics, clean up prompt loading, make wake ack playback dependency explicit, and unwind the temporary direct Open-Meteo weather adapter into the Phase 3 MCP shape.

**Depends on:** Phase 1 complete and latency numbers acceptable (TTFB < 2s). Phase 2 may add ~70ms to the hot path when nomic-embed + spaCy are enabled. Measure Phase 1 baseline before starting.

**Tricky requirements upfront:**
- `AgentProcessor` replaces `OpenAILLMService` in the pipeline. The pipeline becomes: `LocalTransport → WakeWord → VAD → STT → AgentProcessor → PiperTTS → LocalTransport`. The `LLMContextAggregatorPair` used in Phase 1 is replaced — `AgentProcessor` owns context management internally. The `VADProcessor` from the Phase 1 fix remains as an explicit pipeline stage between `WakeWordProcessor` and `WhisperSTTService`; it is not embedded inside an aggregator pair.
- nomic-embed runs in-process via `llama-cpp-python` (`embedding=True`, model in `models/embed/`) only if fast-path routing needs embedding similarity. No port 8081 server. The `Llama` object is held by `AgentProcessor` and called via `asyncio.to_thread()` to avoid blocking the pipeline event loop. Confirm the `.gguf` embed model is present and loads cleanly before building Phase 2c.
- Whisper ASR output is not clean text — filler words (`um`, `uh`), disfluencies, and proper-noun errors shift the embedding distribution. Exemplars in the classifier must be collected from **real mic input through Whisper**, not typed. Typed text will not transfer. The measurement harness (step 2b) must use the same `VADParams` as the production pipeline — mismatched VAD boundary settings produce different Whisper segmentation and will corrupt the classifier's training distribution.
- The `none / weather` boundary is the highest-risk bleed zone. Cover it explicitly with at least 10 ambiguous utterances in the corpus.
- spaCy `en_core_web_sm` handles location names well but fails on implicit references ("near me", "here"). These fall through to the slow path — do not try to resolve them in the fast path.
- One primary llama-server instance (Gemma 4, port 8080) handles normal Cass turns. The nomic-embed classifier is in-process via `llama-cpp-python` and does not share an inference context with Gemma 4. `AgentProcessor` still holds an `asyncio.Lock` to prevent overlapping Gemma 4 calls between the fast-path formatter and slow-path reasoning loop. Remove the lock after profiling if the overlap never occurs in practice.
- `AgentProcessor` owns context management internally via `ConversationWindow` (from `core.py`). `VoiceConfig.history_tokens` replaces `history_turns`: after each turn, `AgentProcessor` counts the retained context with llama.cpp `/tokenize` and drops oldest turn pairs until within cap. Exactly one component owns the trim invariant; `ConversationWindow` stores turns and `AgentProcessor` performs async token-budget trimming.
- `/tokenize` is a Phase 2 dependency, not just UI trivia. Startup validation now probes `POST /tokenize`; runtime trimming falls back to a conservative character/word estimate if `/tokenize` fails after startup, so history still fails closed. Future streaming integrations should prefer server-provided `usage` fields where available.
- Classifier diagnostics must distinguish real embedding scores from fallback routing. If nomic-embed is unavailable and `Skill.can_handle()` routes heuristically, do not publish confidence as `1.0`; use an explicit route source or sentinel confidence so the UI and corpus analysis do not mistake a keyword fallback for a calibrated classifier result.
- System prompt loading must stop doing global string replacement. The current Hearth-to-Cass migration shim should become an explicit template variable such as `{assistant_name}` in `cass_prompt.txt`, rendered from `VoiceConfig.assistant_name`. Global replacement can silently corrupt future prompts.
- Async wake helpers should use named `asyncio.create_task()` tasks instead of anonymous `ensure_future()` calls. Keep and cancel task handles where lifecycle matters, especially for the awake timeout and ack playback.
- `AgentProcessor` must handle Pipecat interrupts cleanly: when a user barges in while Cass is speaking, cancel the current agent `asyncio.Task`, drain any pending `TextFrame`s, and reset to idle before processing the next `TranscriptionFrame`. The `awake_timeout_seconds` guard from Phase 1 is a safety net, not the primary interrupt mechanism.
- State and diagnostics publishing are part of the agent contract. Emit `thinking` when classification/LLM work begins, `tool_calling` around MCP calls, `speaking` on first emitted `TextFrame`, and `idle` on normal completion or cancellation cleanup. Publish turn metadata that the UI can render later: route/path, skill label, model name, provider (`local`), context token count, output token count when available, tokens/sec when streaming data exists, and latency fields.

**Substeps:**

2a. **Canonical event schemas** — `shared/straylight_shared/events.py`. Define dataclasses: `TranscriptEvent`, `IntentEvent`, `ToolCallEvent`, `ToolResultEvent`, `SpeakingEvent`, `StateEvent`. All include `session_id` and `timestamp_ms`. This is the contract for the gateway UI and future chat adapters — define it before `AgentProcessor` so publishing is consistent from the first working turn. *(Pulled forward from Phase 4a.)*

2b. **Skill base class and registry** — `services/voice/skills/__init__.py`. Define `Skill` abstract base: `name`, `exemplars: list[str]`, `entities(transcript) → dict`, `execute(entities) → str`, `format_prompt`. `AgentProcessor.__init__` accepts `skills: list[Skill]`, builds shared embedding index from all exemplars at startup.

2c. **Measurement harness** — `scripts/measure_router.py`. Records mic input, runs Whisper, scores against embedding index, appends JSONL: `{transcript, skill_label, score, gap, below_threshold}`. Used to calibrate fast-path thresholds before wiring into the live pipeline. Note: the harness currently supports an embedding HTTP endpoint for scoring, while runtime and `scripts/score_corpus.py` use in-process `llama-cpp-python`; keep thresholds aligned by validating with `score_corpus.py` before changing runtime config.

2d. **Corpus collection** — 50–100 spoken utterances via the harness. Minimum 10 per active class and at least 10 `none/weather` boundary utterances with natural disfluencies. Do not proceed to 2e without ≥ 90% accuracy on a held-out 20% split and stable gap distribution.

What still needs to be done for the corpus/accuracy gate:
1. Run the voice environment from repo root: `.venv/bin/python scripts/measure_router.py --count 60 --output corpus_labeled_raw.jsonl` and speak natural utterances through the mic. Include clear `weather`, clear `none`, and ambiguous none/weather boundary phrases such as “is it cold”, “do I need a jacket”, “what is outside like”, and non-weather uses of words like “storm” or “rain”.
2. Label the captured rows into a clean JSONL file accepted by `scripts/score_corpus.py`: each row needs `{"transcript": "...", "label": "weather"}` or `{"transcript": "...", "label": "none"}`. Keep labels to active runtime classes unless more skills are actually registered.
3. Build/update exemplars from real Whisper transcripts, not typed examples. `exemplars.jsonl` can bootstrap collection, but it does not by itself prove transfer to live speech.
4. Run the offline gate with the same in-process embed model used by runtime:

   ```bash
   .venv/bin/python scripts/score_corpus.py \
     --exemplars exemplars.jsonl \
     --corpus corpus_labeled.jsonl \
     --embed-model models/embed/nomic-embed-text-v1.5.f16.gguf \
     --threshold "${CASS_ROUTER_THRESHOLD:-0.80}" \
     --min-gap "${CASS_ROUTER_MIN_GAP:-0.05}" \
     --required-accuracy 0.90
   ```

5. Save the reported held-out accuracy, per-class counts, and confusion matrix near this phase or in a dated note. Phase 2 is complete only when the gate passes and the `none/weather` bleed zone is explicitly represented.

2e. **`AgentProcessor` / Cass runtime** — `services/voice/agent.py`. Custom `FrameProcessor` and first bus runtime. On `TranscriptionFrame`: runs nomic-embed classifier in-process when enabled; if skill match above threshold → fast path (spaCy NER → `skill.execute()` → Gemma 4 single-shot format); else → slow path (Gemma 4 ReAct loop, MCP tool registry). Streams `TextFrame`s to TTS as they arrive. Publishes `TranscriptEvent`, `IntentEvent`, `SpeakingEvent`, `StateEvent`, and diagnostics metadata via `publisher.py` on every turn. Logs `path` (fast/slow), `skill_label`, `model`, `provider`, `classifier_ms`, `agent_ms`, `tokens_per_sec` when available.

2f. **`WeatherSkill` temporary fast-path client** — `services/voice/skills/weather.py`. Exemplars loaded and entity extraction implemented. At Phase 2 close this should be either a deterministic placeholder or clearly documented temporary direct Open-Meteo adapter. It currently lives under `services/voice/skills/`, contrary to the long-term project structure where durable weather logic belongs under `services/tools/weather/`. Phase 3 must extract the adapter/tool boundary to MCP and leave `WeatherSkill` as a thin client.

2g. **Wire into pipeline** — replace `OpenAILLMService` and aggregators in `main.py` with `AgentProcessor(skills=[WeatherSkill()])`. Startup validation adds: Gemma 4 llama-server healthy on port 8080, nomic-embed `Llama` object loads from `models/embed/`.

2h. **`dev_gemma.sh` update** — primary Gemma 4 llama-server on port 8080, health-check it. Voice service loads nomic-embed in-process at startup only when embedding routing/retrieval is enabled. Then docker compose, then voice service. `dev.sh` is archived (was tuned for the prior MoE model).

**A feature is complete when:**
- [x] Event schemas in `shared/straylight_shared/events.py`; all fields match what `AgentProcessor` publishes
- [x] `AgentProcessor` in pipeline; `OpenAILLMService` removed
- [ ] Corpus has ≥ 50 spoken/Whisper utterances, ≥ 10 per active class, covers `none/weather` boundary, and scores ≥ 90% on held-out 20% via `scripts/score_corpus.py` *(current files are seeded/partial and do not prove this gate yet)*
- [ ] Fast/slow path split verified against the scored corpus and a live smoke: weather utterances hit fast path; conversational utterances hit slow path
- [ ] `WeatherSkill` Phase 2 behavior documented as stub or temporary direct adapter; Phase 3 migration target is `services/tools/weather/` MCP
- [x] `path`, `skill_label`, `classifier_ms`, `agent_ms` logged per turn in structured form
- [x] UI-facing diagnostics metadata emitted per turn: model/provider, context tokens, output tokens when available, tokens/sec when available, route/path, latency breakdown *(via `TurnDiagnosticsEvent`)*
- [ ] Fast path adds < 100ms over Phase 1 baseline, proven by a dated measurement run *(embed call is wrapped in `asyncio.to_thread`; `classifier_ms` logged per turn for ongoing validation)*
- [x] `history_tokens` cap enforced by token budget, not turn count; `/tokenize` availability checked at startup; fallback estimate and caching defined
- [x] Diagnostics distinguish `classifier_source="embedding"` from `classifier_source="heuristic"`; heuristic route confidence is not reported as perfect embedding confidence
- [x] Prompt loading uses explicit assistant-name templating instead of global Hearth/Cass string replacement
- [x] Wake ack playback dependency is validated or replaced with an in-process fallback; ack playback failures are visible in logs
- [x] Interrupted/cancelled turns publish a clean return-to-idle state and are marked as interrupted if diagnostics are emitted
- [x] State lifecycle published consistently enough for the gateway and memory worker to infer what Cass is doing without private pipeline state

---

### Phase 3 — Tools & MCP

**Goal:** "What's the weather in London?" works end-to-end from any Cass surface. Weather service is an MCP server on the bus. Both the fast path (`WeatherSkill`) and the slow path (Gemma 4 ReAct) can call it. Docker Compose introduced here.

**Depends on:** Phase 2 hardening complete. `AgentProcessor` stable. `WeatherSkill` temporary client responding. Router corpus gate and token-budget trimming verified. Latency numbers acceptable end-to-end.

**Tricky requirements upfront:**
- The weather service is an **MCP server**, not a plain FastAPI endpoint. Use `fastapi-mcp` to expose the existing FastAPI handler as an MCP tool. This keeps the HTTP handler testable while making it discoverable by the slow-path large LLM via `tools/list`.
- Open-Meteo requires a geocoding step (city name → lat/lon). Current Phase 2 `services/voice/skills/weather.py` has direct geocode/fetch logic as a temporary adapter; Phase 3 should move that durable behavior under `services/tools/weather/` or wrap the existing imported adapter instead of leaving tool logic inside the native voice skill package.
- The **fast path does not trigger a ReAct loop**. `WeatherSkill.execute()` calls the MCP weather tool directly. Gemma 4 (port 8080) only formats the structured tool result in Cass's voice — single-shot, no tool discovery. Fast-path calls are short; verify via llama-server request logs (low token count, no tool schema in prompt).
- The **slow path discovers tools dynamically**. Gemma 4 calls `tools/list` on the MCP registry at the start of each slow-path turn. New tools appear automatically. No hardcoded tool list in the agent loop.
- All tool failures must produce a spoken response. If the weather MCP server is unreachable or returns an error, `AgentProcessor` catches the exception and speaks a fallback. Silent failure is not acceptable.
- The native voice process calls MCP servers at `http://localhost:<port>`. Docker's port mapping makes this work. Confirm explicitly before wiring.
- Every tool call has a bounded timeout. Weather defaults to 5s; future tools choose their own timeout based on expected latency. Use `asyncio.timeout()` or equivalent around MCP calls and formatting calls that depend on tool output.
- Validate tool parameters before execution and validate tool results before injecting them into an LLM prompt. Use Pydantic or MCP schemas for structured validation; reject malformed parameters with a spoken clarification instead of letting the tool fail deep inside its adapter.
- Tool registry is explicit even if discovery is dynamic. Keep a small local registry/config of allowed MCP servers, then call each server's `tools/list`; do not let arbitrary endpoints become tools just because they respond like MCP.
- Publish `ToolCallEvent`, `ToolResultEvent`, and `StateEvent(state="tool_calling")` for both fast-path skill calls and slow-path ReAct tool calls.
- Tool telemetry is product data, not just logs. The Phase 4 UI must be able to render each tool call, arguments (redacted when needed), duration, result shape, error state, and spoken fallback.

**Substeps:**

3a. **Weather MCP server** — `services/tools/weather/main.py`. FastAPI wrapping `imported_code/backend/weather.py` or the temporary adapter logic currently in `services/voice/skills/weather.py`. `fastapi-mcp` exposes `get_weather(location: str)` as an MCP tool. Returns `{temperature, unit, condition, wind_kph, humidity_pct}`. MCP `tools/list` returns the schema.

3b. **Docker Compose: weather** — add `weather` service. Expose port on localhost. `docker compose up weather` must start cleanly with no dependencies on other services.

3c. **`WeatherSkill.execute()` real implementation** — replace stub/direct adapter calls with MCP call to weather server. Pass extracted location entity from spaCy. Validate `location` before the call, handle `location=None` gracefully (ask user to repeat with a location), enforce the weather timeout, and validate the returned shape before formatting.

3d. **Slow path MCP registry** — `AgentProcessor` slow path constructs MCP client, calls `tools/list` on registered servers at turn start. Gemma 4 receives tool schemas in its system context. ReAct loop handles tool call / observation cycles. Registry entries include base URL, allowed tool names, timeout, and whether the tool is enabled.

3e. **Tool isolation and fallback tests** — unit/integration coverage for valid call, invalid params, timeout, MCP server unreachable, malformed tool result, and spoken fallback. Failures must not block the pipeline or leave Cass silent.

3f. **Latency measurement** — log `tool_call_ms` (MCP round-trip), `format_ms` (Gemma 4 formatting), `total_weather_ms` (wake → first TTS audio) per weather turn.

**A feature is complete when:**
- [ ] `docker compose up weather` starts cleanly; `tools/list` returns `get_weather` schema
- [ ] Cass correctly answers "What's the weather in London?" end-to-end by voice
- [ ] Fast path handles the request: no Gemma 4 ReAct loop triggered (verified via llama-server logs; fast-path call is single-shot format, low token count)
- [ ] Slow path also handles weather if fast-path classifier scores below threshold
- [ ] Tool failure spoken: if weather server unreachable, Cass says she can't reach it; no crash, no silence
- [ ] Invalid params and malformed tool results are rejected before LLM prompt injection
- [ ] Timeout path tested: a slow weather server produces a spoken fallback and returns Cass to `idle`
- [ ] Tool call/result events published for fast-path and slow-path weather calls
- [ ] Tool events contain enough metadata for the future UI timeline: tool name, safe args, duration, status, and result summary/shape
- [ ] `tool_call_ms`, `format_ms`, `total_weather_ms` logged per weather turn
- [ ] `docker compose down` + `up` restarts without state issues

---

### Phase 4 — Chat UI, Gateway & Diagnostics

**Goal:** Give Cass a real interface: chat-first browser UI, session list, voice controls, event stream, diagnostics panels, and first-class text input into the same Cass runtime used by voice.

**Depends on:** Phase 2 complete (`AgentProcessor` produces events). Redis introduced here.

> Phase 4 can be built in parallel with Phase 3 if bandwidth allows. It requires `AgentProcessor` to publish events and diagnostics metadata. Weather/tool support enriches the UI timeline but is not required for the first chat loop.

**Tricky requirements upfront:**
- Voice service (native host) and gateway (Docker) are separated by a process boundary. **Redis pub/sub is the correct IPC mechanism.** HTTP polling and shared files do not cross this boundary cleanly.
- The native voice process connects to Redis at `localhost:6379`. Docker services connect at `redis:6379` (internal DNS). Both work via Docker Compose `ports: ["6379:6379"]`. Validate this explicitly.
- SSE connections must survive gateway restarts. The browser `EventSource` auto-reconnects; the gateway resumes publishing without requiring a voice pipeline restart.
- Browser text input is a first-class surface, not a fallback. It submits turns to Cass through the same bus as voice transcripts and receives the same event stream back.
- Session identity moves from process-scoped to turn/session-scoped here. Phase 2 may use one generated `session_id` for the native voice process, but Phase 4 browser tabs, future adapters, and injected turns must carry explicit session IDs through `cass:input`, events, diagnostics, and memory writes.
- Redis channel names are routing metadata. Event payload schemas should not rely on an in-payload `channel` field once live Redis publishing lands; derive publish destination from event type or publisher registry to avoid payload/channel divergence.
- **`POST /input` frame injection requires restructuring `main.py`.** The current entry point blocks on `await runner.run(task)`. Phase 4 changes this to `await asyncio.gather(runner.run(task), redis_input_subscriber(task))`, passing `task` into the subscriber so it can call `await task.queue_frame(TranscriptionFrame(...))`. This reopens Phase 1/2 code — plan for it upfront.
- Diagnostics must be cached/event-driven. Do not make the UI poll llama.cpp or block the voice pipeline for token counts, model stats, or throughput. Cass/runtime components publish what they already know; the gateway caches and serves it.
- Reasoning display means observable trace, not hidden chain-of-thought. Show route, model/provider, tool calls, observations, memory injections, token counts, and optional model-provided reasoning summaries when available.
- Do not add authentication yet. Single user, localhost. The auth system in `imported_code/frontend/` is noted for Phase 6.

**Substeps:**

4a. **Redis publisher** — `services/voice/publisher.py`. `AgentProcessor` publishes to `cass:transcript`, `cass:intent`, `cass:tool_call`, `cass:tool_result`, `cass:speaking`, `cass:state` after each relevant event. Event schemas defined in Phase 2a; this step wires the live Redis connection. All published events conform to schemas in `shared/straylight_shared/events.py`.

4b. **Docker Compose: Redis** — `redis:alpine`, port 6379 exposed on localhost. Config in `infra/redis/redis.conf`.

4c. **Gateway service** — `services/gateway/main.py`. FastAPI. `GET /events` SSE endpoint subscribes to all `cass:*` channels via `aioredis`, fans out to connected SSE clients. `POST /input` accepts `{text: str, mode?: "local" | "cloud", agent?: "cass"}`, publishes to `cass:input` channel. Phase 4 accepts `mode="local"` only; cloud mode is disabled until Phase 6.

4d. **Voice: input subscriber** — `main.py` restructured: `asyncio.gather(runner.run(task), redis_input_subscriber(task))`. Subscriber receives `cass:input` messages and calls `await task.queue_frame(TranscriptionFrame(...))`, bypassing wake word and STT.

4e. **Diagnostics cache** — gateway stores the latest turn metadata from Redis: context tokens/current cap, input/output tokens, tokens/sec, latency waterfall, model/provider, path/skill label, tool timeline, memory injection summary, and active state. Expose via `GET /diagnostics` for UI refresh and debugging. Cached data can lag the live turn; it must not block Cass.

4f. **Browser UI** — `frontend/`. Chat-first Cass surface, borrowing the best Hearth patterns: session sidebar, message stream, text composer, voice enable/stop controls, attachment pattern held for later, and dense dark control surface. It also borrows llama-server-style diagnostics conceptually: context meter, token count, tokens/sec, latency waterfall, model/provider badge, local/cloud mode indicator, route/path, tool timeline, and trace panel. Served on `0.0.0.0` for LAN access. No auth (Phase 6). No browser mic — native wake listener remains the primary voice path.

**A feature is complete when:**
- [ ] Redis starts via `docker compose up redis`; `AgentProcessor` publishes events on every turn
- [ ] `GET /events` streams all turn events to a connected browser tab in real-time
- [ ] Browser UI supports normal chat: session list, message stream, text input, send/stop controls, and streamed Cass responses
- [ ] Browser UI displays transcript, intent, route/path, model/provider, tool calls, memory events, and Cass state badge with no manual refresh
- [ ] Diagnostics panel shows context tokens/cap, token counts, tokens/sec when available, latency waterfall, tool timeline, and local/cloud mode indicator
- [ ] SSE client reconnects automatically after gateway restart; no stale state
- [ ] `POST /input` with `{"text": "what time is it"}` causes Cass to respond through the Cass runtime and publish the same event stream as a voice turn
- [ ] Two SSE clients connected simultaneously both receive all events without loss
- [ ] UI is accessible from a second device on the LAN
- [ ] All published events conform to schemas in `shared/straylight_shared/events.py`
- [ ] Gateway and UI survive missing/unknown future event fields without crashing

---

### Phase 5 — Memory

**Goal:** The bus remembers facts, preferences, and context across conversations. Cass is the first consumer, but memory is designed as a shared service for future agents. Knowledge is stored as a graph, consolidated during idle periods (sleep), retrieved as context before the agent acts, and surfaced in the UI timeline.

**Depends on:** Phase 4 complete. Redis running. Phase 3 tools stable. nomic-embed available for retrieval and offline deduplication.

> Build this phase after the Cass runtime, tool path, and chat UI are stable. The bus boundaries from Phases 2–4 make the retrieval shape obvious. Building memory before the agent loop and UI telemetry are stable produces a moving target.

**Architecture: two stores, one source of truth.**

SQLite holds raw turns and extracted facts. Kuzu holds the graph structure built from those facts. Wipe Kuzu → reload from the `facts` table in SQLite with no LLM calls. The `facts` table is the source of truth for fact text. Kuzu is the graph index.

```
SQLite
├── turns  (id, session_id, timestamp_ms, role, content)
└── facts  (id, session_id, source_turn_ids, subject, relation,
              object, extracted_at, last_consolidated_turn_id)

Kuzu (embedded graph DB, no server process)
├── Entity nodes  (name, type, first_seen_session, last_seen_session)
└── Relationship edges  (label, confidence, source_fact_ids)
```

**Sleep consolidation** runs when the voice pipeline has been idle for N minutes (default: 5). It runs as a separate async process subscribed to `cass:state`. Before each LLM call, it checks Redis `cass:state`; if the pipeline becomes active, it backs off immediately.

**Retrieval** uses nomic-embed to embed the current transcript, finds the nearest entity nodes in Kuzu by vector similarity, traverses relationships up to depth 2, and serializes the subgraph as natural language facts injected into the agent's context. Token cap enforced via llama.cpp `/tokenize` endpoint.

**Memory as MCP.** The retrieval API is an MCP server. Both the fast path (`MemorySkill`) and the slow-path Gemma 4 ReAct loop can query it via `tools/list`. Future agents use the same memory service instead of opening SQLite or Kuzu directly.

**Tricky requirements upfront:**
- Sleep consolidation must be **fully isolated from the voice pipeline**. If the worker crashes or is slow, the voice pipeline is unaffected. The worker subscribes to Redis events; the pipeline never waits on it.
- Deduplication of relationships in Kuzu uses nomic-embed cosine similarity offline during sleep. Two relationship labels that embed close together (e.g., "likes coffee" / "enjoys coffee") are merged — the higher-confidence one survives. This is offline only; not on any latency-sensitive path.
- The **token cap uses the llama.cpp `/tokenize` endpoint**, not tiktoken. Different models tokenize differently. Using the wrong tokenizer silently over-injects context. Cap default: 512 tokens of injected facts.
- Retrieval adds latency before every agent turn. Measure it. If nomic-embed + Kuzu query takes > 150ms (p95), this will stack on top of the classifier. Profile before wiring into the live pipeline.
- Kuzu is embedded (Python library, no server). It runs inside the memory service container. The voice process calls the memory MCP server over HTTP — it never opens Kuzu directly.
- Consolidation is opportunistic. It can use larger/slower models later for long-context extraction, but only from the isolated worker; the live voice turn must never wait for consolidation or share an active LLM call with it.
- Memory activity is visible. Retrieval, injected facts, consolidation start/stop, extracted fact counts, and backoff events publish bus events so the Phase 4 UI can explain why Cass remembered something.

**Substeps:**

5a. **Episodic store** — `services/memory/episodic.py`. SQLite `turns` and `facts` tables as above. Redis subscriber writes every turn to `turns`. Never deletes rows.

5b. **Graph store** — `services/memory/graph.py`. Kuzu interface: `upsert_entity()`, `upsert_relationship()`, `query_neighbors(entity_name, depth=2)`, `rebuild_from_facts(db_path)`. `rebuild_from_facts` reloads all rows from SQLite `facts` table into a fresh Kuzu database — no LLM calls.

5c. **Sleep consolidation worker** — `services/memory/sleep.py`. Separate process. Subscribes to `cass:state` on Redis. On idle timeout (default 5 min): reads unprocessed turns from SQLite since `last_consolidated_turn_id`. Calls Gemma 4 (port 8080) with entity/relationship extraction prompt. Checks `cass:state` before each LLM call; backs off if pipeline active. Writes extracted triples to SQLite `facts`. Upserts into Kuzu. Deduplicates relationships via nomic-embed cosine similarity. Updates `last_consolidated_turn_id`.

5d. **Memory MCP server** — `services/memory/api/main.py`. FastAPI + `fastapi-mcp`. Tool: `retrieve_memory(query: str, limit: int = 5)`. Embeds query via nomic-embed, queries Kuzu entity nodes, traverses neighbors depth-2, serializes subgraph as `[{fact_text, confidence, session_id, extracted_at}]`.

5e. **`MemorySkill`** — `services/voice/skills/memory.py`. Fast-path skill for explicit memory queries ("do you remember when I told you…", "what do I usually…"). Calls `retrieve_memory` MCP tool directly. Gemma 4 formats retrieved facts in Cass's voice.

5f. **Slow-path context injection** — `AgentProcessor` slow path calls `retrieve_memory` at the start of every turn before the Gemma 4 call. Injects serialized facts into the system context. Enforces token cap. Logs `retrieval_ms` and `injected_tokens` per turn.

5g. **Replay command** — `scripts/rebuild_graph.sh`. Calls `graph.rebuild_from_facts()` against the SQLite database. Wipes Kuzu and rebuilds from scratch in a single command. No LLM calls.

**A feature is complete when:**
- [ ] Every turn written to SQLite `turns` with correct `session_id` and `timestamp_ms`; persists across restarts
- [ ] Sleep consolidation extracts facts from 10 test conversations and writes to SQLite `facts` + Kuzu without duplication
- [ ] `retrieve_memory(query="coffee")` returns relevant facts from previous conversations
- [ ] Injected context never exceeds 512 tokens; `injected_tokens` logged per turn via llama.cpp `/tokenize`
- [ ] Retrieval latency < 150ms p95 over 100 sequential queries; `retrieval_ms` logged
- [ ] Cass correctly references a user preference stated in a **previous session** (different `session_id`)
- [ ] `scripts/rebuild_graph.sh` wipes Kuzu and rebuilds from SQLite `facts` with no LLM calls
- [ ] Sleep consolidation worker crash does not affect the voice pipeline
- [ ] `cass:state` backoff works: consolidation pauses when pipeline becomes active mid-extraction
- [ ] UI-visible memory events emitted for retrieval, injection, consolidation, and backoff

---

### Phase 6 — Cloud Escalation & Agent Expansion

**Goal:** Add explicit downstream escalation and prepare for more agents only after the local Cass bus is stable. Gemma 4 remains the default local model. Qwen3.6 is the local heavy-artillery path. External APIs such as Claude or ChatGPT are optional, user-selected escalation paths for genuinely hard questions, not automatic dependencies.

**Depends on:** Phase 4 chat/diagnostics stable, Phase 5 memory stable enough to show retrieval provenance, and local model behavior measured. Do not start Phase 6 while the local loop still has unresolved latency, routing, memory, or UI observability problems.

**Escalation order:**

```
Gemma 4 local primary
  → Qwen3.6 local heavy artillery when explicitly selected or clearly needed
  → External cloud provider only when the user explicitly enables it for a query/session
```

**Tricky requirements upfront:**
- Cloud is opt-in per query or session. No automatic silent fallback. The UI must show provider, privacy/cost warning, and whether the request leaves the machine.
- Cloud providers are downstream services behind a provider interface. Surfaces still submit turns to the bus; no UI calls Claude/ChatGPT directly.
- Qwen3.6 is preferred before cloud for high-context or hard local work when the machine can run it. It should be invoked selectively by Cass/runtime routing, not left always-on unless profiling says it is affordable.
- External agent usage is deferred until local is stable. Phase 6 may define the provider/agent abstractions, but the first implementation should still keep Cass as the primary agent.
- Auth and multi-user prep belong here. Reuse the imported Hearth auth patterns when the system needs multiple users or remote access; do not add auth earlier just because the UI exists.

**Substeps:**

6a. **Provider interface** — define local/cloud provider metadata: name, type (`local`/`cloud`), model, context cap, cost visibility, privacy label, streaming support, token counting method, and enabled state.

6b. **Qwen3.6 local heavy path** — optional second llama-server instance on a separate port. Add UI selector and route metadata. Use for long-context, technical, or multi-step requests where Gemma 4 is insufficient.

6c. **Cloud provider stubs** — implement disabled-by-default provider definitions for Claude/ChatGPT. API keys are user-provided environment/config values. Never require them to run Straylight.

6d. **Explicit escalation UX** — per-query/session toggle in the chat UI. Show provider label, local/cloud status, estimated privacy/cost warning, and route trace.

6e. **Agent registry shape** — define the minimal schema for future named agents: `name`, `system_prompt`, allowed tools, preferred provider, memory access, and UI label. Do not build multiple agents until Cass proves the pattern.

**A feature is complete when:**
- [ ] Straylight still runs fully local with no cloud credentials configured
- [ ] UI can select Gemma 4 primary or Qwen3.6 heavy local path when available
- [ ] Cloud provider is disabled by default and requires explicit user selection
- [ ] Provider/model used for every turn is visible in events, logs, and UI
- [ ] External request path shows privacy/cost warning before use
- [ ] Cass remains the default agent; future agent registry exists as schema/config only unless a concrete second agent is justified
- [ ] Auth/multi-user prep documented and imported Hearth auth patterns evaluated before implementation

---

## Testing Requirements

- Test runner: `pytest`. Run with `.venv/bin/python -m pytest tests/`
- Hardware tests (require mic/speaker): mark `@pytest.mark.hardware`. Excluded from CI.
- Integration tests (require Docker services): mark `@pytest.mark.integration`. Require `docker compose up` before running.
- Each phase must have at minimum:
  - One unit test covering the primary custom `FrameProcessor` or component introduced in that phase
  - One integration test covering the happy path end-to-end without mocks
- Latency measurements are **mandatory**. Each phase exit criterion includes at least one latency metric. Use `time.perf_counter()`. Log in structured form: `{"event": "turn_latency", "classifier_ms": 52, "agent_ms": 210, "ttfb_ms": 890}`.
- No default test should require internet access. All core models and services are local. Cloud-provider tests in Phase 6 must be explicitly marked and skipped unless credentials and opt-in flags are present.
- `AgentProcessor` unit tests must cover: fast path taken, slow path taken, fast path falls through to slow path on low classifier confidence, skill failure produces spoken fallback.

## Performance Baseline

Latency and reliability are phase gates, not afterthoughts. Before starting each major phase, capture a short baseline run and keep the numbers near the phase notes or commit message.

Minimum baseline fields:
- Audio readiness: selected input/output devices, sample rate, startup time, and whether virtual backends were skipped
- Wake loop: false triggers per hour, wake-to-STT latency, barge-in/interrupt behavior
- STT/TTS: transcript latency after speech end, first TTS audio after transcript, 30-turn dropout/crash check
- Agent: classifier latency, agent latency, LLM first token, history token count
- Tools/memory: MCP round-trip, formatter latency, retrieval latency, injected token count
- System load: CPU/GPU/VRAM/RAM snapshot during a live turn and during idle consolidation

The old hard-real-time roadmap is useful as a stress-test menu: jitter, thermal headroom, model load time, and long-run stability are worth measuring. It is not a mandate to replace Pipecat, Redis, or MCP unless the baseline proves they are the bottleneck.

---

## Architecture Decisions

### Retired architecture ideas
The archive is design history, not a backlog. These ideas stay retired unless a later phase explicitly reopens them with measurements:

- Separate dialogue/orchestrator service before `AgentProcessor` outgrows the voice process
- Ollama as the hot-path inference server
- ChromaDB as the primary memory store
- Hand-rolled native audio loop, lock-free ring buffers, ZeroMQ/nng IPC, or SCHED_FIFO as default requirements
- Browser microphone/audio before native audio is proven insufficient
- Authentication before Phase 6 multi-user work
- Plain HTTP tools as the durable tool interface; MCP is the interface
- "Voice is the product" as the core thesis; voice is a premium surface on the bus
- "Browser UI is read-only monitor" as a product direction; the UI is chat, control, and diagnostics
- "Cloud never" as a principle; the actual principle is local-first priority with explicit downstream escalation after local stability
- Premature multi-agent registry before Cass, UI, tools, and memory prove the bus pattern

### Model integrity hardening
Model presence checks are mandatory today; checksum verification is a future hardening step once model downloads stabilize. Target shape: `models/manifest.json` with SHA256, size, source, and expected local path for wake, STT, TTS, embed, and LLM models. Startup validation should warn or fail based on a strictness flag.

### Primary local LLM: Gemma 4 (gemma-4-E4B-it-UD-Q4_K_XL)
Gemma 4 is the primary local LLM. At ~2.5 GB (Q4_K_XL), it fits comfortably alongside the audio pipeline and is fast enough to handle Cass's normal response formatting and slow-path reasoning on a single server (port 8080). `dev_gemma.sh` is the active dev launcher. The default system must work with Gemma 4 alone before any heavier local model or cloud path is introduced.

### Heavy local LLM: Qwen3.6
Qwen3.6 is local heavy artillery. It is reserved for high-context and complex reasoning tasks where Gemma 4's context budget or reasoning depth is insufficient: memory consolidation over long histories, long-document ingestion, deep technical analysis, or multi-hop reasoning chains. If enabled, run it as a second llama-server instance on a separate port and invoke it selectively from the Cass runtime or Phase 6 provider interface. It is not a replacement for Gemma 4 and should not be always-on until profiling proves the machine can afford it.

### External APIs: deferred downstream escalation
Claude, ChatGPT, or similar APIs are out of scope until the local bus is stable. When added in Phase 6, they are explicit user-selected providers per query or session, with visible provider labels, privacy/cost warnings, and no hidden fallback. Straylight must remain fully useful without cloud credentials.

### Embeddings: nomic-embed via llama-cpp-python when required
Researched May 2026. `pyllamacpp` (`nomic-ai/pygpt4all`) is dead — archived May 2023. Do not use.

`llama-cpp-python` (`abetlen/llama-cpp-python`) is actively maintained (10k+ stars, 345 releases, updated continuously). It releases the GIL during inference and is `asyncio.to_thread()` compatible. Decision: use nomic-embed in-process via `llama-cpp-python` (`embedding=True`, model in `models/embed/`) where embeddings clearly help: fast-path skill classification, memory retrieval, and offline deduplication. Do not force embeddings into simple routing cases where heuristics or structured UI state are cleaner.

Gemma 4 stays as an external llama-server process — pre-compiled with GPU flags, independently restartable, and process-isolated from the voice pipeline.

---

## Pro Tips

**Pipecat owns audio. The bus owns turns.** Keep this separation absolute. No reasoning, no tool calls, no memory access inside any Pipecat built-in service. The voice pipeline is audio I/O; Cass/runtime and the bus own routing, tools, memory, diagnostics, and response generation.

**Phase 1 wiring is scaffolding.** `OpenAILLMService` in the Phase 1 pipeline validated the stack. Phase 2 removes it. Do not build on top of the Phase 1 LLM wiring — replace it.

**Phase ordering is load-bearing.** Each phase adds latency to the hot path. Measure Phase 1 TTFB before Phase 2. Measure Phase 2 end-to-end before Phase 3. Fix baseline latency before adding layers.

**Gemma 4 first, Qwen3.6 when warranted, cloud later.** Gemma 4 (port 8080) handles normal Cass turns. nomic-embed runs in-process via `llama-cpp-python` only where useful. Qwen3.6 can run as a second local llama-server for heavy work after Gemma 4 limits are measured. External APIs are Phase 6 opt-in escalation only.

**MCP is the tool interface.** Every tool — weather, memory retrieval, future tools — exposes an MCP endpoint. The slow-path agent discovers them via `tools/list`. Adding a new tool means writing an MCP server and registering it; no changes to `AgentProcessor`.

**SQLite is forever; Kuzu is the graph index.** Never truncate the `turns` or `facts` tables. Kuzu can always be rebuilt from SQLite. If you are unsure where to store something, write it to SQLite first.

**Sleep consolidation backs off aggressively.** If `cass:state` is anything other than `idle`, the consolidation worker stops its current LLM call and waits. The voice pipeline must never compete with consolidation for inference time.

**Retrieval token cap uses the model's own tokenizer.** Call `POST /tokenize` on the llama.cpp server to count tokens before injecting memory context. Never use tiktoken as a proxy — different models tokenize differently and the mismatch is silent.

**The UI is not a monitor bolted on the side.** It is the primary chat/control/diagnostics surface. Build from Hearth's useful chat and voice patterns, plus llama-server-style model transparency, but route every action through the Straylight bus.

**The auth system in `imported_code/frontend/` exists.** scrypt-hashed passwords, bearer tokens — it is ready. Do not add it until Phase 6 (multi-user). Note where it is so it is not re-invented.
