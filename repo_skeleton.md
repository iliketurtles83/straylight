# Repository Map
.
├── AGENTS.md
├── archive
│   ├── architecture.md
│   ├── imported_code
│   │   ├── backend
│   │   │   ├── auth.py
│   │   │   ├── embedding_router.py
│   │   │   ├── hearth_coder_prompt.txt
│   │   │   ├── hearth_prompt.txt
│   │   │   ├── kokoro.py
│   │   │   ├── piper.py
│   │   │   ├── wake.py
│   │   │   └── weather.py
│   │   └── frontend
│   │       ├── audio-processor.js
│   │       ├── auth.js
│   │       ├── index.html
│   │       ├── message.js
│   │       ├── style.css
│   │       └── voice.js
│   ├── initial_draft.md
│   ├── phase_a.json
│   ├── phase_a_remediation.json
│   ├── PLAN.md
│   ├── plan_v1.md
│   └── review.md
├── audio
├── core
│   ├── agent_core.py
│   ├── classifier.py
│   ├── observer.py
│   ├── runtime.py
│   └── tools
│       ├── __init__.py
│       ├── local
│       │   ├── __init__.py
│       │   └── weather.py
│       ├── mcp
│       │   ├── __init__.py
│       │   └── mcp_client.py
│       └── registry.py
├── create-skeleton.sh
├── docker-compose.yml
├── Dockerfile
├── eval
│   ├── corpus_scored.jsonl
│   ├── measure_router.py
│   └── score_corpus.py
├── exemplars.jsonl
├── infra
│   └── redis
│       └── redis.conf
├── models
│   ├── tts
│   │   ├── ack.mp3
│   │   ├── en_US-amy-medium.onnx
│   │   └── en_US-amy-medium.onnx.json
│   └── wake
│       ├── computer_v2.onnx
│       ├── embedding_model.onnx
│       └── melspectrogram.onnx
├── new_skeleton.md
├── phase_a2_runtime_fix.json
├── phase_b_remediation.json
├── repo_skeleton.md
├── requirements.txt
├── schemas
│   └── events.py
├── scripts
│   ├── download-models.sh
│   └── download-tts-models.sh
├── surfaces
│   ├── cli
│   ├── telegram
│   ├── voice
│   │   ├── agent.py
│   │   ├── clients.py
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── wake.py
│   └── web
│       └── main.py
├── tests
│   ├── core
│   │   └── test_gateway.py
│   ├── eval
│   │   ├── detect_from_microphone.py
│   │   ├── echo_loop.py
│   │   ├── piper_test.py
│   │   ├── test-docker-setup.sh
│   │   └── test_docker_setup.sh
│   └── surfaces
│       ├── test_browser.py
│       └── test_voice_core.py
└── verify_setup.sh

27 directories, 70 files

## File Archetypes & Signatures
./archive/imported_code/backend/wake.py:def get_oww_model():
./archive/imported_code/backend/wake.py:def get_whisper_model():
./archive/imported_code/backend/piper.py:class Engine:
./archive/imported_code/backend/piper.py:    def __init__(self) -> None:
./archive/imported_code/backend/piper.py:    def _parse_float(name: str, default: float) -> float:
./archive/imported_code/backend/piper.py:    def _parse_optional_float(name: str) -> float | None:
./archive/imported_code/backend/piper.py:    def _parse_optional_int(name: str) -> int | None:
./archive/imported_code/backend/piper.py:    def _build_cmd(self, output_path: Path) -> list[str]:
./archive/imported_code/backend/piper.py:    def _run_piper(self, text: str) -> bytes:
./archive/imported_code/backend/piper.py:    def _apply_pitch_shift(self, wav_bytes: bytes, pitch: float) -> bytes:
./archive/imported_code/backend/auth.py:class AuthError(Exception):
./archive/imported_code/backend/auth.py:    def __init__(self, message: str, code: str, status: int = 400) -> None:
./archive/imported_code/backend/auth.py:class AuthService:
./archive/imported_code/backend/auth.py:    def __init__(self, db_path: str) -> None:
./archive/imported_code/backend/auth.py:    def _init_db(self) -> None:
./archive/imported_code/backend/auth.py:    def _hash_password(password: str, salt: bytes) -> str:
./archive/imported_code/backend/auth.py:    def _new_salt() -> bytes:
./archive/imported_code/backend/auth.py:    def _new_token() -> str:
./archive/imported_code/backend/auth.py:    def _user_id_from_username(username: str) -> str:
./archive/imported_code/backend/auth.py:    def register(
./archive/imported_code/backend/auth.py:    def login(
./archive/imported_code/backend/auth.py:    def verify_token(self, token: str) -> str | None:
./archive/imported_code/backend/auth.py:    def revoke_token(self, token: str) -> bool:
./archive/imported_code/backend/auth.py:    def revoke_all_tokens(self, user_id: str) -> int:
./archive/imported_code/backend/auth.py:    def get_user(self, user_id: str) -> dict | None:
./archive/imported_code/backend/auth.py:    def purge_expired_tokens(self) -> int:
./archive/imported_code/backend/embedding_router.py:class ClassifierResult:
./archive/imported_code/backend/embedding_router.py:class DualClassifierResult:
./archive/imported_code/backend/embedding_router.py:class ExemplarIndex:
./archive/imported_code/backend/embedding_router.py:    def from_embeddings(
./archive/imported_code/backend/embedding_router.py:class EmbeddingRouterSnapshot:
./archive/imported_code/backend/embedding_router.py:def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
./archive/imported_code/backend/embedding_router.py:def _normalize_vector(vector: np.ndarray) -> np.ndarray:
./archive/imported_code/backend/embedding_router.py:def _classify_index(
./archive/imported_code/backend/embedding_router.py:class EmbeddingIntentRouter:
./archive/imported_code/backend/embedding_router.py:    def __init__(
./archive/imported_code/backend/embedding_router.py:    def classify_embedding(self, query_embedding: np.ndarray) -> DualClassifierResult:
./archive/imported_code/backend/embedding_router.py:    def classify_text(self, text: str, embed_text: Callable[[str], np.ndarray]) -> DualClassifierResult:
./archive/imported_code/backend/embedding_router.py:def get_embedding_router() -> EmbeddingIntentRouter | None:
./archive/imported_code/backend/embedding_router.py:def get_embedding_router_snapshot() -> EmbeddingRouterSnapshot | None:
./archive/imported_code/backend/embedding_router.py:def get_embedding_router_error() -> str:
./archive/imported_code/backend/embedding_router.py:def embedding_router_ready() -> bool:
./archive/imported_code/backend/kokoro.py:class Engine:
./archive/imported_code/backend/kokoro.py:    def __init__(self) -> None:
./archive/imported_code/backend/kokoro.py:    def _parse_float(name: str, default: float) -> float:
./archive/imported_code/backend/kokoro.py:    def _parse_int(name: str, default: int) -> int:
./archive/imported_code/backend/kokoro.py:    def _load_runtime(self) -> Any:
./archive/imported_code/backend/kokoro.py:    def _call_runtime(self, text: str) -> tuple[Any, int]:
./archive/imported_code/backend/kokoro.py:    def _extract_audio(self, result: Any) -> tuple[Any, int]:
./archive/imported_code/backend/kokoro.py:    def _coerce_pcm_float(samples: Any) -> list[float]:
./archive/imported_code/backend/kokoro.py:    def _float_to_wav_bytes(samples: list[float], sample_rate: int) -> bytes:
./archive/imported_code/backend/kokoro.py:    def _synthesize_sync(self, text: str) -> bytes:
./archive/imported_code/backend/kokoro.py:def create_engine() -> Engine:
./archive/imported_code/backend/weather.py:def wmo_condition(code: int) -> str:
./archive/imported_code/backend/weather.py:def extract_location(prompt: str) -> str | None:
./archive/imported_code/backend/weather.py:def is_weather_reasoning(prompt: str) -> bool:
./archive/imported_code/backend/weather.py:def format_weather_response(data: dict) -> str:
./archive/imported_code/backend/weather.py:    def _to_celsius(value: float | None) -> float | None:
./eval/measure_router.py:def _cosine_similarity(a: list[float], b: list[float]) -> float:
./eval/measure_router.py:class ScoringProcessor(FrameProcessor):
./eval/measure_router.py:    def __init__(
./eval/measure_router.py:    def count(self) -> int:
./eval/measure_router.py:        def _on_signal() -> None:
./eval/score_corpus.py:def _cosine(a: list[float], b: list[float]) -> float:
./eval/score_corpus.py:def _load_jsonl(path: Path) -> list[dict]:
./eval/score_corpus.py:def _text_of(row: dict) -> str | None:
./eval/score_corpus.py:def main(args: argparse.Namespace) -> int:
./eval/score_corpus.py:    def embed(text: str) -> list[float]:
./core/runtime.py:class RuntimeConfig:
./core/runtime.py:class CassRuntime:
./core/runtime.py:    def __init__(self, config: RuntimeConfig, observer: TurnObserver | None = None):
./core/runtime.py:    def _create_weather_tool_spec(self, skill: WeatherSkill) -> ToolSpec:
./core/runtime.py:    def _get_skill_for_tool(self, tool_name: str) -> Skill | None:
./core/agent_core.py:class VoiceConfig:
./core/agent_core.py:    def from_env(cls) -> "VoiceConfig":
./core/agent_core.py:class TranscriptTurn:
./core/agent_core.py:class ConversationWindow:
./core/agent_core.py:    def build_messages(self, user_text: str) -> list[dict[str, str]]:
./core/agent_core.py:    def add_turn(self, user_text: str, assistant_text: str) -> None:
./core/agent_core.py:    def drop_oldest_turn_pair(self) -> bool:
./core/agent_core.py:def _env_int(name: str, default: int) -> int:
./core/agent_core.py:def _env_float(name: str, default: float) -> float:
./core/agent_core.py:def _env_optional_str(name: str) -> str | None:
./core/agent_core.py:def load_system_prompt(prompt_path: Path, assistant_name: str = "Cass") -> str:
./core/agent_core.py:def normalize_reply_text(text: str) -> str:
./core/agent_core.py:def pcm16_to_float32(samples: Sequence[int]) -> list[float]:
./core/agent_core.py:def float32_to_pcm16(samples: Sequence[float]) -> bytes:
./core/agent_core.py:def audio_to_wav_bytes(samples: Sequence[float], sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
./core/tools/local/weather.py:def _wmo_condition(code: int) -> str:
./core/tools/local/weather.py:def _extract_location_regex(transcript: str) -> str | None:
./core/tools/local/weather.py:class WeatherSkill(Skill):
./core/tools/local/weather.py:    def __init__(self) -> None:
./core/tools/local/weather.py:    def name(self) -> str:
./core/tools/local/weather.py:    def exemplars(self) -> list[str]:
./core/tools/local/weather.py:    def format_prompt(self) -> str:
./core/tools/local/weather.py:    def score(self, transcript: str) -> float:
./core/tools/local/weather.py:    def entities(self, transcript: str) -> dict[str, Any]:
./core/tools/mcp/mcp_client.py:class MCPClient:
./core/tools/mcp/mcp_client.py:    def __init__(self, server_url: str, server_name: str):
./core/tools/mcp/mcp_client.py:    def _create_tool_caller(self, tool_name: str):
./core/tools/registry.py:class ToolResult:
./core/tools/registry.py:class ToolSpec:
./core/tools/registry.py:class ToolNotFoundError(Exception):
./core/tools/registry.py:class ToolExecutionError(Exception):
./core/tools/registry.py:class ToolRegistry:
./core/tools/registry.py:    def __init__(self, observer: TurnObserver | None = None) -> None:
./core/tools/registry.py:    def register(self, spec: ToolSpec) -> None:
./core/tools/registry.py:    def manifest(self) -> list[dict]:
./core/classifier.py:class ClassifierResult:
./core/classifier.py:class Classifier:
./core/classifier.py:    def __init__(self, embed_model_path: Path):
./core/classifier.py:    def _embed_sync(self, text: str) -> list[float]:
./core/classifier.py:    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
./core/classifier.py:    def register_exemplars(self, tool_name: str, exemplars: list[str]) -> None:
./core/observer.py:class TurnObserver(Protocol):
./core/observer.py:    def notify(self, event: Event) -> None:
./core/observer.py:class InMemoryObserver:
./core/observer.py:    def __init__(self) -> None:
./core/observer.py:    def notify(self, event: Event) -> None:
./core/observer.py:    def events(self) -> list[Event]:
./surfaces/voice/clients.py:class VoiceDependencyError(RuntimeError):
./surfaces/voice/clients.py:class OpenWakeWordDetector:
./surfaces/voice/clients.py:    def __init__(
./surfaces/voice/clients.py:    def _normalize_label(label: str) -> str:
./surfaces/voice/clients.py:    def _resolve_prediction_key(self, prediction: dict[str, Any]) -> str | None:
./surfaces/voice/clients.py:    def _coerce_score(raw: Any) -> float:
./surfaces/voice/clients.py:    def _ensure_model(self) -> Any:
./surfaces/voice/clients.py:    def reset(self) -> None:
./surfaces/voice/clients.py:    def score(self, frame: Sequence[int]) -> float:
./surfaces/voice/clients.py:    def triggered(self, frame: Sequence[int]) -> tuple[float, bool]:
./surfaces/voice/wake.py:class _State(Enum):
./surfaces/voice/wake.py:class WakeWordFrame(Frame):
./surfaces/voice/wake.py:class _LatencyMarker:
./surfaces/voice/wake.py:class WakeWordProcessor(FrameProcessor):
./surfaces/voice/wake.py:    def __init__(
./surfaces/voice/wake.py:    def _rms_energy(pcm_bytes: bytes) -> float:
./surfaces/voice/wake.py:    def _to_mono_pcm16(frame: InputAudioRawFrame) -> np.ndarray:
./surfaces/voice/wake.py:    def _resample_to_16k(pcm: np.ndarray, input_rate: int) -> np.ndarray:
./surfaces/voice/wake.py:    def t_wake(self) -> float:
./surfaces/voice/wake.py:    def t_transcript(self) -> float:
./surfaces/voice/wake.py:    def t_transcript(self, value: float) -> None:
./surfaces/voice/wake.py:    def notify_bot_audio_active(self) -> None:
./surfaces/voice/wake.py:    def notify_bot_audio_stopped(self) -> None:
./surfaces/voice/wake.py:    def _suppress_input_audio(self) -> bool:
./surfaces/voice/wake.py:    def _cancel_task(task: asyncio.Task | None) -> None:
./surfaces/voice/wake.py:    def _on_awake_timeout_done(self, task: asyncio.Task) -> None:
./surfaces/voice/wake.py:    def _on_ack_task_done(self, task: asyncio.Task) -> None:
./surfaces/voice/agent.py:class AgentProcessor(FrameProcessor):
./surfaces/voice/agent.py:    def __init__(self, runtime: CassRuntime):
./surfaces/voice/main.py:class LatencyObserver(FrameProcessor):
./surfaces/voice/main.py:    def __init__(self, wake_processor: WakeWordProcessor | None = None, **kwargs) -> None:
./surfaces/voice/main.py:class WakeResetRelay(FrameProcessor):
./surfaces/voice/main.py:    def __init__(self, wake_processor: WakeWordProcessor | None = None, **kwargs) -> None:
./surfaces/voice/main.py:def _validate_ack_player_binary(config: VoiceConfig) -> None:
./surfaces/voice/main.py:def _select_audio_devices(
./surfaces/voice/main.py:    def _supports_input(idx: int) -> bool:
./surfaces/voice/main.py:    def _supports_output(idx: int) -> bool:
./surfaces/voice/main.py:    def _match_preferred_device(preferred_name: str, want_input: bool) -> int | None:
./surfaces/voice/main.py:def _load_none_exemplars(path: Path) -> list[str]:
./surfaces/voice/main.py:def build_pipeline(config: VoiceConfig) -> tuple[Pipeline, WakeWordProcessor | None]:
./tests/eval/detect_from_microphone.py:def _default_model_dir() -> Path:
./tests/eval/detect_from_microphone.py:def parse_args() -> argparse.Namespace:
./tests/eval/detect_from_microphone.py:def main() -> int:
./tests/eval/piper_test.py:def resolve_piper_binary() -> str | None:
./tests/eval/piper_test.py:def parse_args() -> argparse.Namespace:
./tests/eval/piper_test.py:def require_binary(name: str) -> None:
./tests/eval/piper_test.py:def main() -> int:
./tests/eval/echo_loop.py:class EchoProcessor(FrameProcessor):
./tests/eval/echo_loop.py:    def __init__(self, ack_path: Path, **kwargs) -> None:
./tests/eval/echo_loop.py:class VADTraceProcessor(FrameProcessor):
./tests/eval/echo_loop.py:def _validate(config: VoiceConfig) -> None:
./tests/eval/echo_loop.py:def _select_audio_devices(
./tests/eval/echo_loop.py:    def _supports_input(idx: int) -> bool:
./tests/eval/echo_loop.py:    def _supports_output(idx: int) -> bool:
./tests/eval/echo_loop.py:def _build_pipeline(
./tests/eval/echo_loop.py:    def _request_shutdown(sig_name: str) -> None:
./tests/core/test_gateway.py:def test_imports():
./tests/surfaces/test_voice_core.py:class VoiceCoreTests(unittest.TestCase):
./tests/surfaces/test_voice_core.py:    def test_prompt_loader_renders_assistant_name_placeholder(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_prompt_loader_leaves_non_template_words_unchanged(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_conversation_window_keeps_history_until_agent_trims(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_conversation_window_drops_oldest_pair_on_request(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_normalization_collapses_whitespace(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_audio_to_wav_bytes_produces_riff_header(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_voice_config_reads_preferred_audio_device_names(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_voice_config_reads_history_token_budget(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_voice_config_reads_bot_audio_drain_ms(self) -> None:
./tests/surfaces/test_voice_core.py:class WakeWordProcessorTests(unittest.TestCase):
./tests/surfaces/test_voice_core.py:    def _make_processor(self, triggered_result: tuple[float, bool] = (0.0, False)):
./tests/surfaces/test_voice_core.py:    def test_wake_word_processor_sleeping_drops_frames(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_wake_word_processor_sleeping_drops_upstream_input(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_wake_word_processor_emits_wake_frame_on_trigger(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_startup_validation_fails_on_missing_model(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_llama_server_validation_requires_tokenize_endpoint(self) -> None:
./tests/surfaces/test_voice_core.py:        def handler(request: httpx.Request) -> httpx.Response:
./tests/surfaces/test_voice_core.py:    def test_llama_server_validation_accepts_tokenize_tokens(self) -> None:
./tests/surfaces/test_voice_core.py:        def handler(request: httpx.Request) -> httpx.Response:
./tests/surfaces/test_voice_core.py:    def test_startup_validation_requires_ack_player_binary(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_to_mono_pcm16_uses_first_channel(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_resample_to_16k_from_48k(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_post_wake_flush_clamp_prevents_negative(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_stuck_awake_timeout_resets_to_sleeping(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_wake_processor_suppresses_input_while_bot_audio_active(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_wake_processor_applies_post_stop_drain(self) -> None:
./tests/surfaces/test_voice_core.py:class WakeResetRelayTests(unittest.TestCase):
./tests/surfaces/test_voice_core.py:    def test_wake_reset_relay_pushes_bot_stop_upstream_on_tts_stop(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_wake_reset_relay_notifies_wake_on_tts_audio_and_stop(self) -> None:
./tests/surfaces/test_voice_core.py:class WeatherSkillTests(unittest.TestCase):
./tests/surfaces/test_voice_core.py:    def test_weather_skill_extracts_location_with_preposition(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_weather_skill_can_handle_weather_queries(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_weather_skill_execute_missing_location(self) -> None:
./tests/surfaces/test_voice_core.py:class AgentRouterTests(unittest.TestCase):
./tests/surfaces/test_voice_core.py:    def test_agent_uses_skill_heuristic_without_embed_model(self) -> None:
./tests/surfaces/test_voice_core.py:        class DummySkill(Skill):
./tests/surfaces/test_voice_core.py:            def name(self) -> str:
./tests/surfaces/test_voice_core.py:            def exemplars(self) -> list[str]:
./tests/surfaces/test_voice_core.py:            def can_handle(self, transcript: str) -> bool:
./tests/surfaces/test_voice_core.py:            def entities(self, transcript: str) -> dict:
./tests/surfaces/test_voice_core.py:    def test_agent_trims_conversation_to_token_budget(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_normalize_stream_chunk_inserts_separator(self) -> None:
./tests/surfaces/test_voice_core.py:class AgentDiagnosticsTests(unittest.TestCase):
./tests/surfaces/test_voice_core.py:    def _make_agent(self, skill: Skill | None = None) -> tuple[AgentProcessor, list]:
./tests/surfaces/test_voice_core.py:    def test_turn_publishes_diagnostics_event_on_slow_path(self) -> None:
./tests/surfaces/test_voice_core.py:    def test_skill_failure_falls_back_to_spoken_message(self) -> None:
./tests/surfaces/test_voice_core.py:        class BrokenSkill(Skill):
./tests/surfaces/test_voice_core.py:            def name(self) -> str:
./tests/surfaces/test_voice_core.py:            def exemplars(self) -> list[str]:
./tests/surfaces/test_voice_core.py:            def can_handle(self, transcript: str) -> bool:
./tests/surfaces/test_voice_core.py:            def entities(self, transcript: str) -> dict:
./tests/surfaces/test_voice_core.py:    def test_cancelled_turn_publishes_idle_cleanup(self) -> None:
./tests/surfaces/test_browser.py:def test_imports():
./schemas/events.py:class Event:
./schemas/events.py:def _now_ms() -> int:
./schemas/events.py:class TranscriptEvent(Event):
./schemas/events.py:class IntentEvent(Event):
./schemas/events.py:class ToolCallEvent(Event):
./schemas/events.py:class ToolResultEvent(Event):
./schemas/events.py:class SpeakingEvent(Event):
./schemas/events.py:class StateEvent(Event):
./schemas/events.py:class TurnDiagnosticsEvent(Event):
