# Straylight — Project Plan

> Local-first voice AI assistant. Single user, localhost. No cloud dependencies on the hot path. The voice is Cass.

---

## Overview

Straylight is an agent bus: a personal voice assistant that runs entirely on local hardware. It listens for a wake word, transcribes speech, routes through an agent framework that decides whether to fast-path through a skill or run a full reasoning loop, optionally calls tools via MCP, and speaks back via streaming TTS. A browser UI on the LAN gives a read-only window into what Cass is doing.

**Two-layer architecture:**
- **Voice I/O layer** — Pipecat pipeline running natively on the host. Owns the mic and speaker. Handles wake word, VAD, STT, TTS, interrupt detection. Knows nothing about intent or tools.
- **Intelligence layer** — `AgentProcessor`, a custom `FrameProcessor` that sits inside the Pipecat pipeline. Receives `TranscriptionFrame`, runs the agent loop (fast path or slow path), emits `TextFrame` to TTS. All reasoning, skill dispatch, tool calls, and memory retrieval happen here.

**Infrastructure split:**
- Voice service (Pipecat + AgentProcessor) runs **natively on the host** — it owns audio.
- All other services (MCP tool servers, memory, gateway, Redis) run in **Docker Compose**.
- The boundary is **Redis pub/sub**: voice publishes events; everything else subscribes.

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Voice I/O | **Pipecat** | Frame-processing pipeline; owns audio only |
| Audio I/O | Pipecat `LocalTransport` (sounddevice) | Mic + speaker directly on host |
| VAD | Pipecat `SileroVAD` | Built-in; gates wake word |
| Wake word | **OpenWakeWord** | Custom `FrameProcessor`; models in `models/wake/` |
| STT | Pipecat `WhisperSTTService` (faster-whisper) | Models in `models/stt/` |
| TTS | Pipecat `PiperTTSService` | Streaming, sentence-by-sentence; models in `models/tts/` |
| Agent | **AgentProcessor** | Custom `FrameProcessor`; bridges Pipecat I/O to intelligence layer |
| Fast-path classifier | **nomic-embed** via llama.cpp | Embedding similarity; inside AgentProcessor; not on conversational path |
| Fast-path NER | **spaCy** (`en_core_web_sm`) | Entity extraction (location, time, quantity) for skills; ~20ms |
| LLM — small | **llama-server** (2–4B model) | Tool response formatting on fast path; port 8081 |
| LLM — large | **llama-server** (Qwen3.6) | Conversational responses + slow-path ReAct loop; port 8080 |
| Tool protocol | **MCP** (Model Context Protocol) | Standard tool interface; all tool services expose MCP endpoints |
| Event bus | **Redis** pub/sub | Introduced Phase 4; voice → gateway IPC bridge |
| Memory: episodic | **SQLite** | Full conversation turn log + extracted facts table; never truncated |
| Memory: graph | **Kuzu** | Personal knowledge graph; entities + relationships; embedded, no server |
| Gateway | **FastAPI** SSE | Browser read-only display; text input fallback |
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
│   │   ├── skills/
│   │   │   ├── __init__.py     # Skill base class + registry
│   │   │   └── weather.py      # WeatherSkill (embed classify + spaCy + MCP + small LLM)
│   │   ├── publisher.py        # Redis event publisher (Phase 4)
│   │   ├── cass_prompt.txt     # System prompt (Cass persona)
│   │   └── requirements.txt
│   ├── memory/
│   │   ├── api/                # MCP server: retrieve facts from Kuzu
│   │   ├── sleep.py            # Consolidation worker: idle → LLM extract → Kuzu merge
│   │   ├── episodic.py         # SQLite: turns + facts tables
│   │   └── graph.py            # Kuzu graph interface: upsert entities + relationships
│   ├── gateway/
│   │   ├── main.py             # FastAPI: GET /events (SSE), POST /input
│   │   └── requirements.txt
│   └── tools/
│       └── weather/
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
├── infra/
│   ├── redis/redis.conf
│   └── kuzu/                   # Kuzu data directory (mounted volume)
├── scripts/
│   ├── download-models.sh
│   └── dev.sh                  # Start both llama-server instances + docker compose + voice
├── tests/
│   ├── test_voice_core.py
│   └── ...
└── docker-compose.yml
```

---

## Agent Framework Design

The `AgentProcessor` is the single intelligence unit. Pipecat sees it as an opaque `FrameProcessor`: `TranscriptionFrame` in, `TextFrame`s out. Internally it runs a two-path decision:

```
TranscriptionFrame arrives
│
├─ nomic-embed classifier (~50ms) — is this a known skill with high confidence?
│
├─ FAST PATH (high-confidence skill match)
│   ├─ spaCy NER extracts entities from transcript (~20ms)
│   ├─ Skill calls MCP tool server(s) directly
│   ├─ Small LLM (port 8081, Cass-prompted) formats response from structured tool result
│   └─ emit TextFrames → TTS
│
└─ SLOW PATH (conversational / ambiguous / multi-step)
    ├─ Large LLM (Qwen3.6, port 8080) with MCP tool registry
    ├─ ReAct loop: reason → call tool → observe → reason → ... → done
    └─ emit TextFrames as they stream → TTS
```

**Skills** are named fast-path bundles. Each `Skill` owns:
- Embedding exemplars for the classifier
- spaCy entity patterns
- MCP tool call(s)
- Small LLM format prompt for Cass's voice

Adding a skill never touches the classifier thresholds — it registers exemplars into the shared embedding index at startup.

**Tools** are MCP servers. Both the fast path and the slow path use the same MCP servers. The slow-path large LLM discovers available tools via `tools/list`.

**When to use which path:**
- Fast path: clear tool intent + extractable entities + structured response (weather, time, timer)
- Slow path: conversational, ambiguous, multi-step, or any query the classifier scores below threshold

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

---

### Phase 2 — Agent Framework

**Goal:** Replace the Phase 1 `OpenAILLMService` scaffolding with `AgentProcessor`. Establish the fast path / slow path split. Implement `WeatherSkill` as the first skill (stub response at end of this phase — real tool call in Phase 3).

**Depends on:** Phase 1 complete and latency numbers acceptable (TTFB < 2s). Phase 2 adds ~70ms to the hot path (nomic-embed + spaCy). Measure Phase 1 baseline before starting.

**Tricky requirements upfront:**
- `AgentProcessor` replaces `OpenAILLMService` in the pipeline. The pipeline becomes: `LocalTransport → WakeWord → VAD → STT → AgentProcessor → PiperTTS → LocalTransport`. The `LLMContextAggregatorPair` used in Phase 1 is replaced — `AgentProcessor` owns context management internally. The `VADProcessor` from the Phase 1 fix remains as an explicit pipeline stage between `WakeWordProcessor` and `WhisperSTTService`; it is not embedded inside an aggregator pair.
- nomic-embed runs via llama.cpp's `/v1/embeddings` endpoint on the same server as the small LLM (port 8081), not a separate process. Confirm the endpoint is available before building Phase 2c.
- Whisper ASR output is not clean text — filler words (`um`, `uh`), disfluencies, and proper-noun errors shift the embedding distribution. Exemplars in the classifier must be collected from **real mic input through Whisper**, not typed. Typed text will not transfer. The measurement harness (step 2b) must use the same `VADParams` as the production pipeline — mismatched VAD boundary settings produce different Whisper segmentation and will corrupt the classifier's training distribution.
- The `none / weather` boundary is the highest-risk bleed zone. Cover it explicitly with at least 10 ambiguous utterances in the corpus.
- spaCy `en_core_web_sm` handles location names well but fails on implicit references ("near me", "here"). These fall through to the slow path — do not try to resolve them in the fast path.
- The slow path large LLM (Qwen3.6) must not run concurrently with the small LLM (port 8081) if RAM is constrained. `AgentProcessor` holds a single `asyncio.Lock` shared between fast and slow paths; neither path starts inference while the other holds the lock. Remove the lock after profiling if concurrent inference is confirmed safe.
- `AgentProcessor` owns context management internally via `ConversationWindow` (from `core.py`). `VoiceConfig.history_tokens` replaces `history_turns`: after each turn call llama.cpp `/tokenize` on the accumulated messages and drop oldest turn pairs until within cap. `ConversationWindow.add_turn()` is the trim mechanism; the unit changes from turns to tokens.
- `AgentProcessor` must handle Pipecat interrupts cleanly: when a user barges in while Cass is speaking, cancel the current agent `asyncio.Task`, drain any pending `TextFrame`s, and reset to idle before processing the next `TranscriptionFrame`. The `awake_timeout_seconds` guard from Phase 1 is a safety net, not the primary interrupt mechanism.

**Substeps:**

2a. **Skill base class and registry** — `services/voice/skills/__init__.py`. Define `Skill` abstract base: `name`, `exemplars: list[str]`, `entities(transcript) → dict`, `execute(entities) → str`, `format_prompt`. `AgentProcessor.__init__` accepts `skills: list[Skill]`, builds shared embedding index from all exemplars at startup.

2b. **Measurement harness** — `scripts/measure_router.py`. Records mic input, runs Whisper, scores against embedding index, appends JSONL: `{transcript, skill_label, score, gap, below_threshold}`. Used to calibrate fast-path thresholds before wiring into the live pipeline.

2c. **Corpus collection** — 50–100 spoken utterances via the harness. Minimum 10 per class: `none`, `weather`. Minimum 10 for the `none/weather` boundary with natural disfluencies. Do not proceed to 2d without ≥ 90% accuracy on a held-out 20% split and stable gap distribution.

2d. **`AgentProcessor`** — `services/voice/agent.py`. Custom `FrameProcessor`. On `TranscriptionFrame`: runs nomic-embed classifier; if skill match above threshold → fast path (spaCy NER → `skill.execute()` → small LLM format); else → slow path (Qwen3.6 ReAct loop, MCP tool registry). Streams `TextFrame`s to TTS as they arrive. Logs `path` (fast/slow), `skill_label`, `classifier_ms`, `agent_ms` per turn.

2e. **`WeatherSkill` stub** — `services/voice/skills/weather.py`. Exemplars loaded, spaCy entity extraction implemented, `execute()` returns a placeholder string. Small LLM formats it in Cass's voice. Real MCP call wired in Phase 3.

2f. **Wire into pipeline** — replace `OpenAILLMService` and aggregators in `main.py` with `AgentProcessor(skills=[WeatherSkill()])`. Startup validation adds: both llama-server instances healthy, nomic-embed `/v1/embeddings` endpoint responding.

2g. **`dev.sh` update** — start small model server (port 8081), health-check it. Start Qwen3.6 (port 8080), health-check it. Then docker compose, then voice service.

**A feature is complete when:**
- [ ] `AgentProcessor` in pipeline; `OpenAILLMService` removed
- [ ] Corpus has ≥ 50 utterances, ≥ 10 per class, covers `none/weather` boundary; accuracy ≥ 90% on held-out 20%
- [ ] Fast/slow path split working: weather utterances hit fast path; conversational utterances hit slow path
- [ ] `WeatherSkill` stub plays a spoken response on weather intent
- [ ] `path`, `skill_label`, `classifier_ms`, `agent_ms` logged per turn in structured form
- [ ] Fast path adds < 100ms over Phase 1 baseline (nomic-embed + spaCy measured separately)
- [ ] `history_tokens` cap enforced; token count logged per turn via llama.cpp `/tokenize` endpoint

---

### Phase 3 — Tools & MCP

**Goal:** "What's the weather in London?" works end-to-end by voice. Weather service is an MCP server. Both the fast path (WeatherSkill) and the slow path (Qwen3.6 ReAct) can call it. Docker Compose introduced here.

**Depends on:** Phase 2 complete. `AgentProcessor` stable. `WeatherSkill` stub responding. Latency numbers acceptable end-to-end.

**Tricky requirements upfront:**
- The weather service is an **MCP server**, not a plain FastAPI endpoint. Use `fastapi-mcp` to expose the existing FastAPI handler as an MCP tool. This keeps the HTTP handler testable while making it discoverable by the slow-path large LLM via `tools/list`.
- Open-Meteo requires a geocoding step (city name → lat/lon). Verify `imported_code/backend/weather.py` handles this before wrapping. Do not re-implement geocoding.
- The **fast path does not call Qwen3.6**. `WeatherSkill.execute()` calls the MCP weather tool directly. The small LLM (port 8081) formats the response in Cass's voice from the structured JSON result. Qwen3.6 is not invoked. Verify this via llama-server request logs on port 8080.
- The **slow path discovers tools dynamically**. Qwen3.6 calls `tools/list` on the MCP registry at the start of each slow-path turn. New tools appear automatically. No hardcoded tool list in the agent loop.
- All tool failures must produce a spoken response. If the weather MCP server is unreachable or returns an error, `AgentProcessor` catches the exception and speaks a fallback. Silent failure is not acceptable.
- The native voice process calls MCP servers at `http://localhost:<port>`. Docker's port mapping makes this work. Confirm explicitly before wiring.

**Substeps:**

3a. **Weather MCP server** — `services/tools/weather/main.py`. FastAPI wrapping `imported_code/backend/weather.py`. `fastapi-mcp` exposes `get_weather(location: str)` as an MCP tool. Returns `{temperature, unit, condition, wind_kph, humidity_pct}`. MCP `tools/list` returns the schema.

3b. **Docker Compose: weather** — add `weather` service. Expose port on localhost. `docker compose up weather` must start cleanly with no dependencies on other services.

3c. **`WeatherSkill.execute()` real implementation** — replace stub with MCP call to weather server. Pass extracted location entity from spaCy. Handle `location=None` gracefully (ask user to repeat with a location).

3d. **Slow path MCP registry** — `AgentProcessor` slow path constructs MCP client, calls `tools/list` on registered servers at turn start. Qwen3.6 receives tool schemas in its system context. ReAct loop handles tool call / observation cycles.

3e. **Latency measurement** — log `tool_call_ms` (MCP round-trip), `format_ms` (small LLM formatting), `total_weather_ms` (wake → first TTS audio) per weather turn.

**A feature is complete when:**
- [ ] `docker compose up weather` starts cleanly; `tools/list` returns `get_weather` schema
- [ ] Cass correctly answers "What's the weather in London?" end-to-end by voice
- [ ] Fast path handles the request: Qwen3.6 **not** called (verified via llama-server logs on port 8080)
- [ ] Slow path also handles weather if fast-path classifier scores below threshold
- [ ] Tool failure spoken: if weather server unreachable, Cass says she can't reach it; no crash, no silence
- [ ] `tool_call_ms`, `format_ms`, `total_weather_ms` logged per weather turn
- [ ] `docker compose down` + `up` restarts without state issues

---

### Phase 4 — Browser Gateway

**Goal:** See what Cass is doing from a phone or second screen on the LAN.

**Depends on:** Phase 2 complete (`AgentProcessor` produces events). Redis introduced here.

> Phase 4 can be built in parallel with Phase 3 if bandwidth allows. It only requires `AgentProcessor` to be publishing events — it does not depend on the weather tool.

**Tricky requirements upfront:**
- Voice service (native host) and gateway (Docker) are separated by a process boundary. **Redis pub/sub is the correct IPC mechanism.** HTTP polling and shared files do not cross this boundary cleanly.
- The native voice process connects to Redis at `localhost:6379`. Docker services connect at `redis:6379` (internal DNS). Both work via Docker Compose `ports: ["6379:6379"]`. Validate this explicitly.
- SSE connections must survive gateway restarts. The browser `EventSource` auto-reconnects; the gateway resumes publishing without requiring a voice pipeline restart.
- **`POST /input` frame injection requires restructuring `main.py`.** The current entry point blocks on `await runner.run(task)`. Phase 4 changes this to `await asyncio.gather(runner.run(task), redis_input_subscriber(task))`, passing `task` into the subscriber so it can call `await task.queue_frame(TranscriptionFrame(...))`. This reopens Phase 1/2 code — plan for it upfront.
- Do not add authentication yet. Single user, localhost. The auth system in `imported_code/frontend/` is noted for Phase 6.

**Substeps:**

4a. **Canonical event schemas** — `shared/straylight_shared/events.py`. Define dataclasses for: `TranscriptEvent`, `IntentEvent`, `ToolCallEvent`, `ToolResultEvent`, `SpeakingEvent`, `StateEvent`. All events include `session_id` and `timestamp_ms`.

4b. **Redis publisher** — `services/voice/publisher.py`. `AgentProcessor` publishes to `cass:transcript`, `cass:intent`, `cass:tool_call`, `cass:tool_result`, `cass:speaking`, `cass:state` after each relevant event. All published events conform to schemas in `events.py`.

4c. **Docker Compose: Redis** — `redis:alpine`, port 6379 exposed on localhost. Config in `infra/redis/redis.conf`.

4d. **Gateway service** — `services/gateway/main.py`. FastAPI. `GET /events` SSE endpoint subscribes to all `cass:*` channels via `aioredis`, fans out to connected SSE clients. `POST /input` accepts `{text: str}`, publishes to `cass:input` channel.

4e. **Voice: input subscriber** — `main.py` restructured: `asyncio.gather(runner.run(task), redis_input_subscriber(task))`. Subscriber receives `cass:input` messages and calls `await task.queue_frame(TranscriptionFrame(...))`, bypassing wake word and STT.

4f. **Browser UI** — `frontend/`. Minimal HTML + vanilla JS. Displays: state badge (idle / listening / thinking / speaking), latest transcript, skill/intent label, tool call log. Auto-reconnects SSE on disconnect. No audio. Bind to `0.0.0.0` for LAN access.

**A feature is complete when:**
- [ ] Redis starts via `docker compose up redis`; `AgentProcessor` publishes events on every turn
- [ ] `GET /events` streams all turn events to a connected browser tab in real-time
- [ ] Browser UI displays transcript, intent, tool calls, and Cass state badge with no manual refresh
- [ ] SSE client reconnects automatically after gateway restart; no stale state
- [ ] `POST /input` with `{"text": "what time is it"}` causes Cass to respond by voice, bypassing wake word
- [ ] Two SSE clients connected simultaneously both receive all events without loss
- [ ] UI is accessible from a second device on the LAN
- [ ] All published events conform to schemas in `shared/straylight_shared/events.py`

---

### Phase 5 — Memory

**Goal:** Cass remembers facts, preferences, and context across conversations. Knowledge is stored as a graph, consolidated during idle periods (sleep), and retrieved as context before the agent acts.

**Depends on:** Phase 4 complete. Redis running. Phase 3 tools stable. nomic-embed available for retrieval and offline deduplication.

> Build this phase last. The agent framework and service boundaries from Phases 2–4 make the retrieval shape obvious. Building memory before the agent loop is stable produces a moving target.

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

**Memory as MCP.** The retrieval API is an MCP server. Both the fast path (`MemorySkill`) and the slow-path Qwen3.6 ReAct loop can query it via `tools/list`.

**Tricky requirements upfront:**
- Sleep consolidation must be **fully isolated from the voice pipeline**. If the worker crashes or is slow, the voice pipeline is unaffected. The worker subscribes to Redis events; the pipeline never waits on it.
- Deduplication of relationships in Kuzu uses nomic-embed cosine similarity offline during sleep. Two relationship labels that embed close together (e.g., "likes coffee" / "enjoys coffee") are merged — the higher-confidence one survives. This is offline only; not on any latency-sensitive path.
- The **token cap uses the llama.cpp `/tokenize` endpoint**, not tiktoken. Different models tokenize differently. Using the wrong tokenizer silently over-injects context. Cap default: 512 tokens of injected facts.
- Retrieval adds latency before every agent turn. Measure it. If nomic-embed + Kuzu query takes > 150ms (p95), this will stack on top of the classifier. Profile before wiring into the live pipeline.
- Kuzu is embedded (Python library, no server). It runs inside the memory service container. The voice process calls the memory MCP server over HTTP — it never opens Kuzu directly.

**Substeps:**

5a. **Episodic store** — `services/memory/episodic.py`. SQLite `turns` and `facts` tables as above. Redis subscriber writes every turn to `turns`. Never deletes rows.

5b. **Graph store** — `services/memory/graph.py`. Kuzu interface: `upsert_entity()`, `upsert_relationship()`, `query_neighbors(entity_name, depth=2)`, `rebuild_from_facts(db_path)`. `rebuild_from_facts` reloads all rows from SQLite `facts` table into a fresh Kuzu database — no LLM calls.

5c. **Sleep consolidation worker** — `services/memory/sleep.py`. Separate process. Subscribes to `cass:state` on Redis. On idle timeout (default 5 min): reads unprocessed turns from SQLite since `last_consolidated_turn_id`. Calls Qwen3.6 (port 8080) with entity/relationship extraction prompt. Checks `cass:state` before each LLM call; backs off if pipeline active. Writes extracted triples to SQLite `facts`. Upserts into Kuzu. Deduplicates relationships via nomic-embed cosine similarity. Updates `last_consolidated_turn_id`.

5d. **Memory MCP server** — `services/memory/api/main.py`. FastAPI + `fastapi-mcp`. Tool: `retrieve_memory(query: str, limit: int = 5)`. Embeds query via nomic-embed, queries Kuzu entity nodes, traverses neighbors depth-2, serializes subgraph as `[{fact_text, confidence, session_id, extracted_at}]`.

5e. **`MemorySkill`** — `services/voice/skills/memory.py`. Fast-path skill for explicit memory queries ("do you remember when I told you…", "what do I usually…"). Calls `retrieve_memory` MCP tool directly. Small LLM formats retrieved facts in Cass's voice.

5f. **Slow-path context injection** — `AgentProcessor` slow path calls `retrieve_memory` at the start of every turn before the Qwen3.6 call. Injects serialized facts into the system context. Enforces token cap. Logs `retrieval_ms` and `injected_tokens` per turn.

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

---

## Testing Requirements

- Test runner: `pytest`. Run with `.venv/bin/python -m pytest tests/`
- Hardware tests (require mic/speaker): mark `@pytest.mark.hardware`. Excluded from CI.
- Integration tests (require Docker services): mark `@pytest.mark.integration`. Require `docker compose up` before running.
- Each phase must have at minimum:
  - One unit test covering the primary custom `FrameProcessor` or component introduced in that phase
  - One integration test covering the happy path end-to-end without mocks
- Latency measurements are **mandatory**. Each phase exit criterion includes at least one latency metric. Use `time.perf_counter()`. Log in structured form: `{"event": "turn_latency", "classifier_ms": 52, "agent_ms": 210, "ttfb_ms": 890}`.
- No test should require internet access. All models are local. All services are local.
- `AgentProcessor` unit tests must cover: fast path taken, slow path taken, fast path falls through to slow path on low classifier confidence, skill failure produces spoken fallback.

---

## Pro Tips

**Pipecat owns audio. `AgentProcessor` owns intelligence.** Keep this separation absolute. No reasoning, no tool calls, no memory access inside any Pipecat built-in service. The pipeline is audio I/O; the agent is everything else.

**Phase 1 wiring is scaffolding.** `OpenAILLMService` in the Phase 1 pipeline validated the stack. Phase 2 removes it. Do not build on top of the Phase 1 LLM wiring — replace it.

**Phase ordering is load-bearing.** Each phase adds latency to the hot path. Measure Phase 1 TTFB before Phase 2. Measure Phase 2 end-to-end before Phase 3. Fix baseline latency before adding layers.

**Two llama-server instances, two ports.** Port 8081 (small model, fast) for skill response formatting. Port 8080 (Qwen3.6) for slow-path reasoning and sleep consolidation. `dev.sh` starts and health-checks both in order. The small model should never receive a slow-path reasoning task; the large model should never be called on the fast path. Verify via server logs.

**MCP is the tool interface.** Every tool — weather, memory retrieval, future tools — exposes an MCP endpoint. The slow-path agent discovers them via `tools/list`. Adding a new tool means writing an MCP server and registering it; no changes to `AgentProcessor`.

**SQLite is forever; Kuzu is the graph index.** Never truncate the `turns` or `facts` tables. Kuzu can always be rebuilt from SQLite. If you are unsure where to store something, write it to SQLite first.

**Sleep consolidation backs off aggressively.** If `cass:state` is anything other than `idle`, the consolidation worker stops its current LLM call and waits. The voice pipeline must never compete with consolidation for inference time.

**Retrieval token cap uses the model's own tokenizer.** Call `POST /tokenize` on the llama.cpp server to count tokens before injecting memory context. Never use tiktoken as a proxy — different models tokenize differently and the mismatch is silent.

**The auth system in `imported_code/frontend/` exists.** scrypt-hashed passwords, bearer tokens — it is ready. Do not add it until Phase 6 (multi-user). Note where it is so it is not re-invented.
