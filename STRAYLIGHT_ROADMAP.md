# Straylight Rebuild Roadmap

## Overall Goal

Refactor Straylight from a Pipecat-subordinate, monolithic `AgentProcessor` into a
bus-native, layered agent runtime (`CassRuntime`) where every concern — classification,
tool execution, conversation management, event reliability, and gateway I/O — lives in
its own module with a typed contract, independently testable, and composable into a
Phase 2 ReAct + MCP toolchain without structural rewrites.

The system must satisfy these invariants at every phase gate:
- Redis is a startup precondition, never a lazy side effect
- The inference hot path (classify → generate → stream) is never blocked by bus I/O
- State ownership is singular: `CassRuntime` owns `ConversationWindow`, nothing else does
- Every inter-module boundary is typed; no raw dicts cross module boundaries
- The classifier is a pure function of (text, exemplar index) — no heuristics, no
  fallback routing logic embedded inside it

---

## Phase 0 — Fix Live Bugs Before Any Refactor

**Rationale:** These are silent data-corruption and diagnostic-poisoning bugs in the
current codebase. They will be carried forward into the refactor if not fixed first,
contaminating the baseline you're measuring against.

---

### Task 0.1 — Fix hardcoded `classifier_source` string

**File:** `agent.py`
**Function:** `_process_turn()` (~line 457)

**Problem:** `IntentEvent` is published with `classifier_source="nomic-embed"` as a
string literal, ignoring the `classifier_source` variable returned by `_classify()`.
The diagnostics stream will always report `"nomic-embed"` even when the heuristic
router or disabled path fired.

**Fix:** Replace the string literal with the variable:
```python
# BEFORE
classifier_source="nomic-embed",

# AFTER
classifier_source=classifier_source,
```

**Acceptance criteria:**
- When embed model is absent, `IntentEvent.classifier_source == "disabled"` in the
  Redis stream
- When heuristic fires, `IntentEvent.classifier_source == "heuristic"`
- No hardcoded string literals remain in `IntentEvent` construction

---

### Task 0.2 — Remove the heuristic router

**File:** `agent.py`
**Functions:** `_heuristic_route()`, `_classify()`

**Problem:** The heuristic router is a routing strategy layered on top of the embedding
classifier, creating a dual-signal merge with coverage gaps (described in detail in
the critique). It exists to compensate for thin exemplar coverage. It is technical
debt that cannot be migrated cleanly to the Phase 2 tool registry.

**Fix:**
- Delete `_heuristic_route()` entirely
- Simplify `_classify()` to: embed → score → threshold check → return result
- If embed model unavailable: return `(None, -1.0, 0, "disabled")` directly
- If embed fails at runtime: return `(None, -1.0, 0, "disabled")` — slow path, no
  fallback routing

**Note on exemplar coverage:** The heuristic was compensating for missing exemplars.
Before deleting it, audit which Skills had `can_handle()` or `score()` implementations
and add equivalent exemplars to `exemplars.jsonl` for those patterns.

**Acceptance criteria:**
- `_heuristic_route()` does not exist
- `_classify()` has a single code path: embed → cosine → threshold gate
- Embed unavailable returns `("disabled", -1.0, 0)` — not a routing result
- All previously heuristic-routed intents now route via exemplars or slow path
- `IntentEvent.classifier_source` is never `"heuristic"`

---

### Task 0.3 — Fix token cache granularity

**File:** `agent.py`
**Functions:** `_count_tokens_for()`, `_count_context_tokens()`, `_token_cache`

**Problem:** The cache is keyed by the full concatenated context string. Every turn
grows the key, producing a cache miss on every call. The cache never evicts. Over a
long session it accumulates unbounded memory and provides zero hit rate.

**Fix:**
- Cache token counts per individual message content string, not per full context dump
- `_count_context_tokens()` sums cached per-message counts rather than re-tokenizing
  the full concatenation each time
- Add a max cache size (e.g. 256 entries) with LRU eviction using `functools.lru_cache`
  or a manual bounded dict

**Acceptance criteria:**
- Cache hit rate > 80% in a 10-turn conversation (same messages don't get re-tokenized)
- Cache size is bounded; no unbounded growth over a long session
- Token count accuracy is unchanged (sum of parts equals whole, within tokenizer rounding)

---

## Phase 1 — Structural Separation (The Critical Gate)

**Rationale:** This phase creates the seams along which Phase 2 (ReAct + MCP) will be
attached. Every task here is about extracting a concern from `AgentProcessor` into its
own module with a typed interface. `AgentProcessor` itself is not deleted yet — it
shrinks to an orchestrator that delegates to the new modules.

**Phase 1 is the gate for all subsequent work.** Nothing in Phase 2 or 3 can be done
cleanly without this foundation.

---

### Task 1.1 — Extract `Classifier` into its own module

**New file:** `classifier.py`
**Affected:** `agent.py` (`AgentProcessor.__init__`, `_ensure_ready`, `_load_embed_model`,
`_embed_pairs_sync`, `_embed_sync`, `_classify`, `_load_llama_sync`)

**Contract:**

```python
@dataclass(frozen=True)
class ClassifierResult:
    tool_name: str | None        # None = slow path
    confidence: float            # cosine similarity; -1.0 if disabled
    source: Literal["embedding", "disabled"]
    latency_ms: int

class Classifier:
    async def startup(self) -> None: ...       # loads embed model, builds index
    async def classify(self, text: str) -> ClassifierResult: ...
    def register_exemplars(
        self, tool_name: str, exemplars: list[str]
    ) -> None: ...
```

**Key design decisions:**
- `Classifier` is constructed with the path to the embed model and an empty exemplar
  index. Exemplars are registered after construction, before `startup()`.
- `startup()` loads the model and embeds all registered exemplars. It is called once
  at `CassRuntime` startup, not lazily.
- `classify()` is a pure async function: text in, `ClassifierResult` out. No side
  effects. No heuristics. No fallback routing.
- The embed model runs in-process via `asyncio.to_thread()` (preserves the existing
  no-network-hop design for the hot path).
- `tool_name` in the result matches the key in the `ToolRegistry` (Task 1.2), not a
  Skill class name. This is the seam that connects classifier output to tool dispatch.

**Acceptance criteria:**
- `Classifier` can be imported and tested without importing `agent.py`
- `Classifier.classify()` returns `ClassifierResult` for arbitrary text input
- When embed model path does not exist, `startup()` logs a warning and
  `classify()` always returns `ClassifierResult(tool_name=None, source="disabled")`
- `AgentProcessor._classify()` is deleted; replaced by a call to `Classifier.classify()`
- The exemplar index is built from `exemplars.jsonl` at startup, not lazily

---

### Task 1.2 — Extract `ToolRegistry` into its own module

**New file:** `tools.py`
**Affected:** `agent.py` (`_run_fast_path`, `_generate_response`), `skills.py` (Skill base class)

**Contract:**

```python
@dataclass(frozen=True)
class ToolResult:
    content: str                 # Raw string result for LLM injection
    structured: dict | None      # Parsed structured data if available; None otherwise
    tool_name: str
    latency_ms: int

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str             # For LLM tool manifest in Phase 2
    input_schema: dict           # JSON Schema for the tool's inputs
    execute: Callable[[dict], Awaitable[ToolResult]]

class ToolRegistry:
    def register(self, spec: ToolSpec) -> None: ...
    async def call(self, tool_name: str, args: dict) -> ToolResult: ...
    def manifest(self) -> list[dict]: ...   # OpenAI-format tool list for Phase 2
```

**Key design decisions:**
- `ToolSpec.execute` takes a typed `dict` of args (validated against `input_schema`)
  and returns a `ToolResult`. This is the execution concern extracted from the Skill class.
- `manifest()` returns the OpenAI `tools` array format so the Phase 2 ReAct loop can
  inject it directly into the LLM request payload without transformation.
- Existing Skills are migrated: their `execute()` logic becomes a `ToolSpec.execute`
  callable. Their exemplars move to `exemplars.jsonl` under the tool name. Their
  `format_prompt` is retired — the system prompt handles formatting from Phase 1 onward.
- `ToolRegistry.call()` publishes `ToolCallEvent` and `ToolResultEvent` to the bus.
  This is the only place those events are emitted.

**Acceptance criteria:**
- `ToolRegistry` can be constructed and tested without importing `agent.py`
- Calling an unregistered tool name raises a typed `ToolNotFoundError`, not `KeyError`
- `ToolResult` is returned for all registered tools; exceptions from tool execution are
  caught and re-raised as `ToolExecutionError` with the original exception chained
- `manifest()` returns valid OpenAI tool schema format (verifiable by passing it to
  the llama.cpp server in a test request)
- `ToolCallEvent` and `ToolResultEvent` appear in the Redis stream for every tool call

---

### Task 1.3 — Harden `bus.py` — connection as startup precondition

**File:** `bus.py`
**Affected:** `RedisEventBus`, `get_event_bus()`, `publish()`

**Problem (detailed):** The global `_event_bus` singleton is initialized lazily on the
first `publish()` call during a live turn. Redis failure during a turn triggers a
`RuntimeError` inside `_process_turn()`'s exception handler, which calls `publish()`
again, which tries to reconnect, which fails again. Error handling depends on the
failing resource.

**Fix:**

```python
class RedisEventBus:
    def __init__(self, redis_url: str): ...
    async def startup(self) -> None:
        """Connect and PING. Raises RuntimeError if Redis is unreachable."""
    async def shutdown(self) -> None: ...
    async def publish(self, event: Event) -> None:
        """
        Publish event. On connection failure:
        - Attempt one reconnect
        - On second failure, enqueue to bounded in-memory buffer (max 64 events)
        - Log error; never raise into caller
        """
    async def _drain_buffer(self) -> None:
        """Attempt to flush buffered events after reconnect."""
```

**Key design decisions:**
- `startup()` is called by `CassRuntime.startup()` before any turns are processed.
  If `startup()` fails, the process exits — Redis is not optional.
- `publish()` never raises into the caller. Bus I/O failures are logged and buffered;
  the inference path is never blocked or aborted by bus failure.
- The in-memory buffer is bounded at 64 events. When full, oldest events are dropped
  with a warning log. This protects against unbounded memory growth during extended
  Redis outages.
- Reconnection uses exponential backoff: 100ms, 200ms, 400ms, cap at 5s.

**Acceptance criteria:**
- Starting the process with Redis unavailable raises `RuntimeError` at startup, not
  during the first turn
- Killing Redis mid-turn does not raise an exception in `_process_turn()`
- Events published during Redis outage appear in the stream after Redis recovers
  (buffer drains on reconnect)
- Buffer never exceeds 64 events; overflow is logged with count of dropped events

---

### Task 1.4 — Fix `main.py` SSE implementation

**File:** `main.py`
**Function:** `stream_events()`, `event_generator()`

**Problems:**
1. `psubscribe("*")` matches all Redis channels including `cass:input` (echo) and
   any Redis keyspace notifications
2. `get_message(timeout=1.0)` + `asyncio.sleep(0.01)` polls at ~100 iterations/sec
   per client doing nothing
3. Redis connection is lazily initialised in `get_redis_client()` — same startup
   precondition failure as `bus.py`
4. `request.is_disconnected()` is called inside the tight generator loop

**Fix:**

```python
# Explicit channel subscription — no wildcards
SUBSCRIBE_CHANNELS = [
    "cass:transcript", "cass:intent", "cass:tool_call",
    "cass:tool_result", "cass:speaking", "cass:state", "cass:diagnostics"
]

async def event_generator(request: Request, pubsub) -> AsyncGenerator[str, None]:
    disconnect_task = asyncio.create_task(_watch_disconnect(request))
    try:
        async for message in pubsub.listen():
            if disconnect_task.done():
                break
            if message["type"] != "message":
                continue
            channel = message["channel"].decode()
            data = message["data"].decode()
            yield f"event: {channel}\ndata: {data}\n\n"
    finally:
        disconnect_task.cancel()
        await pubsub.unsubscribe()
        await pubsub.close()

async def _watch_disconnect(request: Request) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(0.5)
```

**Key design decisions:**
- `pubsub.listen()` is an async iterator that blocks until a message arrives.
  No polling loop, no `asyncio.sleep()`, no busy waiting.
- `cass:input` is not in `SUBSCRIBE_CHANNELS` — the gateway publishes to it, clients
  must not receive their own input echoed back.
- Disconnect is monitored in a separate task checking every 500ms, not inline with
  message processing.
- Redis connection for the gateway is established in a `lifespan` context manager on
  `app` startup, not lazily in `get_redis_client()`.
- SSE `media_type` is `"text/event-stream"`, not `"text/plain"`.

**Acceptance criteria:**
- `cass:input` events do not appear in the SSE stream
- Connecting a client and sending no input produces no CPU burn (verifiable with
  `top` or profiler)
- Client disconnect is detected within 1 second and the pubsub connection is closed
- All 7 `cass:*` channels appear correctly typed in the SSE stream
- `media_type="text/event-stream"` in the `StreamingResponse`

---

### Task 1.5 — Introduce `CassRuntime` skeleton

**New file:** `runtime.py`
**Affected:** `agent.py` (shrinks), `bus.py` (startup called here), `main.py`
(runtime injected via dependency)

**Contract:**

```python
class CassRuntime:
    def __init__(self, config: RuntimeConfig): ...

    async def startup(self) -> None:
        """
        1. Connect bus (raises on failure)
        2. Load classifier embed model and build exemplar index
        3. Register tools in ToolRegistry
        4. Subscribe to cass:input
        5. Emit StateEvent(idle)
        """

    async def shutdown(self) -> None: ...

    async def handle_input(self, text: str, session_id: str) -> None:
        """
        Called when cass:input fires. Owns the full turn:
        classify → fast or slow path → stream to bus → update conversation
        """

    # Internal — not called externally
    async def _run_turn(self, text: str, session_id: str) -> None: ...
    async def _fast_path(self, text: str, tool_name: str, session_id: str) -> str: ...
    async def _slow_path(self, text: str, session_id: str) -> AsyncIterator[str]: ...
```

**Key design decisions:**
- `CassRuntime` owns: `RedisEventBus`, `Classifier`, `ToolRegistry`, `ConversationWindow`
- `CassRuntime` does not inherit from `FrameProcessor` or any Pipecat class. It is
  a plain Python class with an async lifecycle.
- The `cass:input` subscription is managed inside `CassRuntime.startup()`. The
  gateway posts to `cass:input`; `CassRuntime` listens and calls `handle_input()`.
- `ConversationWindow` lives here, not in `AgentProcessor`. This is the singular owner
  of session state.
- `AgentProcessor` is kept as a thin shim calling `CassRuntime` for backward
  compatibility with any existing Pipecat wiring, until voice is re-integrated in
  Phase 3.

**Acceptance criteria:**
- `CassRuntime.startup()` completes without error when Redis and llama.cpp are running
- Posting to `POST /input` results in a full turn: `cass:transcript` →
  `cass:intent` → `cass:state:thinking` → `cass:speaking:start` →
  `cass:speaking:stop` → `cass:state:idle` appear in the SSE stream in order
- `CassRuntime` can be instantiated and started without any Pipecat imports
- `ConversationWindow` state persists correctly across multiple turns in the same session

---

## Phase 2 — ReAct Loop + MCP Integration

**Prerequisite:** All Phase 1 tasks complete and acceptance criteria verified.

**Rationale:** With the classifier and tool registry as clean separate modules, the
ReAct loop is additive — it slots in as the `_slow_path` implementation inside
`CassRuntime`, replacing the current direct `ConversationWindow → LLM stream` call.
MCP tools register into `ToolRegistry` using the same `ToolSpec` interface as local tools.

---

### Task 2.1 — Implement ReAct loop in `_slow_path`

**File:** `runtime.py`
**Function:** `CassRuntime._slow_path()`

**Contract:**

The slow path becomes a loop rather than a single streaming call:

```
ITERATION_CAP = 5

loop:
    messages = conversation.build_messages(text) + tool_results_so_far
    response = llm_stream(messages, tools=tool_registry.manifest())

    if response contains tool_call:
        publish ToolCallEvent
        result = tool_registry.call(tool_name, args)
        publish ToolResultEvent
        append tool_call + tool_result to messages
        iteration += 1
        if iteration >= ITERATION_CAP:
            yield "I've hit my reasoning limit. Try rephrasing."
            break
        continue

    else:  # natural language response
        yield chunks
        break
```

**Key design decisions:**
- The iteration cap is a hard safety gate, not a configurable threshold. 5 iterations
  is sufficient for all anticipated use cases; if a query needs more, the tool
  decomposition is wrong.
- Tool calls from the LLM are parsed from the streaming response. The llama.cpp server
  emits tool calls in the `choices[0].delta.tool_calls` field when the model requests
  one. This requires the model to have been fine-tuned for tool use (Qwen 3 and Gemma 3
  both qualify).
- The fast path remains unchanged: classifier fires → one `ToolRegistry.call()` →
  one formatting LLM call. The ReAct loop is only for the slow path.
- `StateEvent("tool_calling")` is emitted at the start of each tool dispatch iteration.

**Acceptance criteria:**
- A query that requires one tool call produces the correct `ToolCallEvent` →
  `ToolResultEvent` → `SpeakingEvent` sequence in the Redis stream
- A query requiring no tool call goes through the loop exactly once and exits normally
- A pathological query that would cause the model to loop forever is capped at 5
  iterations with a fallback response
- Iteration count appears in `TurnDiagnosticsEvent` (add `react_iterations: int` field)

---

### Task 2.2 — Add MCP client adapter to `ToolRegistry`

**New file:** `mcp_client.py`
**Affected:** `tools.py` (`ToolRegistry.register_mcp_server()`)

**Contract:**

```python
class MCPClient:
    def __init__(self, server_url: str, server_name: str): ...
    async def startup(self) -> list[ToolSpec]:
        """
        1. Connect to MCP server
        2. Fetch tool manifest (list of tool names, descriptions, schemas)
        3. Return list[ToolSpec] ready for ToolRegistry.register()
        """
    async def call(self, tool_name: str, args: dict) -> ToolResult: ...
```

```python
# In ToolRegistry:
async def register_mcp_server(self, client: MCPClient) -> None:
    specs = await client.startup()
    for spec in specs:
        self.register(spec)
```

**Key design decisions:**
- MCP tools and local tools are indistinguishable from `CassRuntime`'s perspective
  after registration. The `ToolRegistry` interface is the only seam.
- `MCPClient.startup()` fetches the manifest at runtime. If the MCP server is
  unavailable at startup, log a warning and register zero tools from that server —
  do not block `CassRuntime` startup.
- MCP server URLs are configured in `RuntimeConfig`, not hardcoded.
- One `MCPClient` instance per MCP server. Multiple MCP servers are supported.

**Acceptance criteria:**
- A running MCP server's tools appear in `ToolRegistry.manifest()` after
  `register_mcp_server()` is called
- An MCP tool call dispatched by the ReAct loop produces the correct
  `ToolCallEvent` + `ToolResultEvent` in the Redis stream
- MCP server unavailability at startup does not prevent `CassRuntime` from starting
- MCP tool names do not collide with local tool names (enforce uniqueness in
  `ToolRegistry.register()`)

---

### Task 2.3 — Migrate existing Skills to ToolSpecs

**Affected:** All existing files in `skills/`, `exemplars.jsonl`, `ToolRegistry`

For each existing Skill:

1. Extract `execute()` logic into a standalone async callable
2. Define `ToolSpec` with name, description, and JSON Schema for inputs
   (derived from the existing `entities()` parsing)
3. Add tool name as exemplar label in `exemplars.jsonl` (replacing Skill class name)
4. Delete the Skill class
5. Register the `ToolSpec` in `CassRuntime.startup()`

**Acceptance criteria:**
- `skills/` directory is empty or deleted
- All previously Skill-routed intents now route via `Classifier` → `ToolRegistry`
- Fast path still produces sub-200ms TTFB for known tool calls (benchmark before/after)
- `exemplars.jsonl` covers all tool names with at least 8 exemplars per tool

---

## Phase 3 — Gateway Hardening + Session Model

**Prerequisite:** Phase 2 complete.

---

### Task 3.1 — Proper session management

**File:** `main.py`, `runtime.py`

**Problem:** Currently `session_id` defaults to the string `"default"` if not provided
by the caller. Multiple simultaneous callers share a session, meaning their conversation
history is entangled.

**Fix:**
- `POST /input` requires `session_id` in the request body (not optional, not defaulted)
- `CassRuntime` maintains a `dict[str_session_id, ConversationWindow]`
- Sessions are evicted after a configurable idle timeout (default: 30 minutes)
- `GET /events` accepts an optional `session_id` query parameter; without it, all
  sessions are streamed (useful for debugging); with it, only that session's events
  are forwarded

**Acceptance criteria:**
- Two concurrent clients with different `session_id` values have independent
  conversation histories
- A session idle for > 30 minutes is evicted from memory (verifiable via memory
  profiling or explicit eviction log)
- `GET /events?session_id=X` only yields events where `event.session_id == X`

---

### Task 3.2 — `RuntimeConfig` replaces `VoiceConfig` for runtime concerns

**File:** `core.py` (`VoiceConfig`), new file `config.py`

**Problem:** `VoiceConfig` is a 30-field dataclass mixing voice pipeline configuration
(sample rate, wake word paths, TTS model path) with agent runtime configuration
(LLM URL, embed model path, history tokens, router thresholds). These concerns have
different owners: voice pipeline vs. `CassRuntime`.

**Fix:**
- Create `RuntimeConfig` in `config.py` containing only the fields `CassRuntime` needs:
  `llm_base_url`, `llm_model`, `embed_model_path`, `router_exemplars_path`,
  `router_threshold`, `router_min_gap`, `history_tokens`, `llm_ctx_size`,
  `llm_output_size`, `redis_url`, `mcp_server_urls`
- `VoiceConfig` retains only voice pipeline fields
- Both have `from_env()` classmethods
- `CassRuntime.__init__` takes `RuntimeConfig`, not `VoiceConfig`

**Acceptance criteria:**
- `CassRuntime` can be constructed with only `RuntimeConfig`; it imports nothing from
  the voice pipeline
- `VoiceConfig` and `RuntimeConfig` have zero overlapping fields
- All env var names are unchanged (backward compatible)

---

## Phase 4 — Observability + Exemplar Pipeline

**Prerequisite:** Phase 3 complete.

---

### Task 4.1 — Slow-path transcript collector

**Rationale:** The heuristic router was deleted in Phase 0. The exemplar set is the
only coverage mechanism. It will have gaps. This task builds the pipeline to close them.

**New file:** `collector.py`

**Behaviour:**
- Subscribe to `cass:intent`
- When `IntentEvent.path == "slow"` and `IntentEvent.classifier_source == "disabled"`
  or confidence is below 0.6: append `{text, session_id, timestamp_ms}` to a JSONL
  file (`slow_transcripts.jsonl`)
- This file is the raw material for exemplar expansion

**Acceptance criteria:**
- Running the system for 10 turns with varied slow-path queries produces entries in
  `slow_transcripts.jsonl`
- The collector does not affect turn latency (it is a separate subscriber, not in the
  hot path)
- Collector can be started and stopped independently of `CassRuntime`

---

### Task 4.2 — Add `react_iterations` to `TurnDiagnosticsEvent`

**File:** `events.py`, `runtime.py`

Add `react_iterations: int` field to `TurnDiagnosticsEvent`. Set to `1` for fast
path, actual iteration count for slow path ReAct loop.

**Acceptance criteria:**
- `TurnDiagnosticsEvent` always contains `react_iterations`
- Fast path always emits `react_iterations=1`
- ReAct loop emits the correct iteration count

---

## Dependency Order (Critical Path)

```
0.1 → 0.2 → 0.3
          ↓
      1.1 + 1.2 (can run in parallel)
          ↓
         1.3
          ↓
         1.4
          ↓
         1.5  ← Phase 1 gate
          ↓
      2.1 + 2.2 (can run in parallel)
          ↓
         2.3  ← Phase 2 gate
          ↓
      3.1 + 3.2 (can run in parallel)
          ↓
      4.1 + 4.2 (can run in parallel)
```

---

## File Map: Before → After

| Before | After | Change |
|---|---|---|
| `agent.py` | `agent.py` (shim) | Shrinks to orchestrator; delegates to `runtime.py` |
| `agent.py` | `classifier.py` | Extracted: embed model, exemplar index, cosine scoring |
| `agent.py` | `tools.py` | Extracted: ToolSpec, ToolRegistry, ToolResult |
| `agent.py` | `runtime.py` | New: CassRuntime, owns lifecycle and turn orchestration |
| `bus.py` | `bus.py` | Hardened: startup precondition, buffer, reconnect |
| `main.py` | `main.py` | Fixed: SSE, startup lifecycle, session_id enforcement |
| `core.py` | `core.py` + `config.py` | Split: VoiceConfig (voice) + RuntimeConfig (runtime) |
| `events.py` | `events.py` | Additive only: `react_iterations` on `TurnDiagnosticsEvent` |
| `skills/*.py` | `tools.py` + `exemplars.jsonl` | Migrated: execute logic → ToolSpec; exemplars → JSONL |
| _(new)_ | `mcp_client.py` | New: MCPClient adapter for external MCP servers |
| _(new)_ | `collector.py` | New: slow-path transcript collector for exemplar expansion |

---

## What Is Explicitly Out of Scope

The following are known future concerns that must not creep into this roadmap:

- Memory architecture (episodic SQLite, semantic ChromaDB) — deferred
- Wake word training pipeline — deferred
- Voice re-integration (Pipecat, faster-whisper, Piper) — deferred to after Phase 3
- Multi-session persistence across restarts — deferred
- Hearth thin-client integration — deferred to after Phase 2 gate
- Cloud LLM provider fallback — deferred
