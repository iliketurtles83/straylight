from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from services.voice.agent import AgentProcessor
from services.voice.core import ConversationWindow, TranscriptTurn, VoiceConfig, audio_to_wav_bytes, load_system_prompt, normalize_reply_text, trim_to_last_turns
from services.voice.clients import VoiceDependencyError
from services.voice.skills import Skill
from services.voice.skills.weather import WeatherSkill


class VoiceCoreTests(unittest.TestCase):
    def test_prompt_loader_relabels_hearth_to_cass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_path = Path(temp_dir) / "prompt.txt"
            prompt_path.write_text("You are Hearth. Speak plainly.", encoding="utf-8")

            prompt = load_system_prompt(prompt_path)

        self.assertIn("Cass", prompt)
        self.assertNotIn("Hearth", prompt)

    def test_conversation_window_trims_history(self) -> None:
        window = ConversationWindow(system_prompt="system", history_turns=2)
        window.add_turn("one", "reply one")
        window.add_turn("two", "reply two")
        window.add_turn("three", "reply three")

        self.assertEqual(len(window.turns), 4)
        self.assertEqual(window.turns[0].content, "two")
        messages = window.build_messages("latest question")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["content"], "latest question")

    def test_normalization_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_reply_text("  hello\nworld   from  Cass  "), "hello world from Cass")

    def test_audio_to_wav_bytes_produces_riff_header(self) -> None:
        audio = [0.0] * 16000
        wav_bytes = audio_to_wav_bytes(audio)

        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        self.assertIn(b"WAVE", wav_bytes[:16])

    def test_trim_to_last_turns_returns_latest_pairs(self) -> None:
        turns = [
            TranscriptTurn(role="user", content="a"),
            TranscriptTurn(role="assistant", content="b"),
            TranscriptTurn(role="user", content="c"),
            TranscriptTurn(role="assistant", content="d"),
            TranscriptTurn(role="user", content="e"),
            TranscriptTurn(role="assistant", content="f"),
        ]

        trimmed = trim_to_last_turns(turns, 2)

        self.assertEqual([turn.content for turn in trimmed], ["c", "d", "e", "f"])

    def test_voice_config_reads_preferred_audio_device_names(self) -> None:
        from unittest.mock import patch

        env = {
            "CASS_INPUT_DEVICE_NAME": "Wireless Stereo Headset",
            "CASS_OUTPUT_DEVICE_NAME": "USB Audio",
        }

        with patch.dict(os.environ, env, clear=False):
            from services.voice.core import VoiceConfig

            config = VoiceConfig.from_env()

        self.assertEqual(config.input_device_name, "Wireless Stereo Headset")
        self.assertEqual(config.output_device_name, "USB Audio")


class WakeWordProcessorTests(unittest.TestCase):
    def _make_processor(self, triggered_result: tuple[float, bool] = (0.0, False)):
        """Build a WakeWordProcessor with a mocked detector.

        vad_threshold=0.0 disables the RMS energy gate so test audio content
        doesn't need to simulate real speech levels — tests focus on wake
        detection logic, not the energy gate itself.
        """
        from services.voice.wake import WakeWordProcessor

        detector = MagicMock()
        detector.triggered.return_value = triggered_result
        detector.reset.return_value = None

        ack_path = Path(tempfile.mktemp(suffix=".mp3"))

        proc = WakeWordProcessor(detector=detector, ack_path=ack_path, vad_threshold=0.0)
        return proc, detector

    def test_wake_word_processor_sleeping_drops_frames(self) -> None:
        """Frames emitted while sleeping should NOT be pushed downstream."""
        from pipecat.frames.frames import InputAudioRawFrame
        from pipecat.processors.frame_processor import FrameDirection
        from services.voice.wake import WakeWordProcessor

        proc, detector = self._make_processor(triggered_result=(0.0, False))

        received: list = []

        async def run() -> None:
            # Monkey-patch push_frame to capture calls.
            async def capture(frame, direction=FrameDirection.DOWNSTREAM):
                received.append(frame)
            proc.push_frame = capture  # type: ignore[method-assign]

            frame = InputAudioRawFrame(
                # 3 x 512-sample VAD windows -> 1536 samples, enough to build
                # one 1280-sample OWW window in the staged buffering path.
                audio=b"\x00" * 3072,
                sample_rate=16000,
                num_channels=1,
            )
            await proc.process_frame(frame, FrameDirection.DOWNSTREAM)

        asyncio.run(run())
        # With triggered=False and sleeping state, no downstream frames should be emitted.
        from services.voice.wake import WakeWordFrame
        self.assertFalse(any(isinstance(f, WakeWordFrame) for f in received))
        self.assertFalse(any(isinstance(f, InputAudioRawFrame) for f in received))

    def test_wake_word_processor_sleeping_drops_upstream_input(self) -> None:
        """Input audio should be gated while sleeping regardless of direction."""
        from pipecat.frames.frames import InputAudioRawFrame
        from pipecat.processors.frame_processor import FrameDirection

        proc, _detector = self._make_processor(triggered_result=(0.0, False))
        received: list = []

        async def run() -> None:
            async def capture(frame, direction=FrameDirection.DOWNSTREAM):
                received.append((frame, direction))

            proc.push_frame = capture  # type: ignore[method-assign]

            frame = InputAudioRawFrame(
                audio=b"\x00" * 3072,
                sample_rate=16000,
                num_channels=1,
            )
            await proc.process_frame(frame, FrameDirection.UPSTREAM)

        asyncio.run(run())
        self.assertEqual(received, [])

    def test_wake_word_processor_emits_wake_frame_on_trigger(self) -> None:
        """A triggered OWW detection should emit WakeWordFrame and switch to awake."""
        from pipecat.frames.frames import InputAudioRawFrame
        from pipecat.processors.frame_processor import FrameDirection
        from services.voice.wake import WakeWordFrame, _State

        proc, detector = self._make_processor(triggered_result=(0.85, True))

        received: list = []

        async def run() -> None:
            async def capture(frame, direction=FrameDirection.DOWNSTREAM):
                received.append(frame)

            proc.push_frame = capture  # type: ignore[method-assign]

            # Suppress ack playback side-effect.
            async def no_ack():
                pass
            proc._play_ack = no_ack  # type: ignore[method-assign]

            frame = InputAudioRawFrame(
                # 3 x 512-sample VAD windows -> 1536 samples, enough to build
                # one 1280-sample OWW window in the staged buffering path.
                audio=b"\x00" * 3072,
                sample_rate=16000,
                num_channels=1,
            )
            await proc.process_frame(frame, FrameDirection.DOWNSTREAM)

        asyncio.run(run())
        wake_frames = [f for f in received if isinstance(f, WakeWordFrame)]
        self.assertEqual(len(wake_frames), 1)
        self.assertAlmostEqual(wake_frames[0].score, 0.85)
        self.assertEqual(proc._state, _State.awake)

    def test_startup_validation_fails_on_missing_model(self) -> None:
        """validate_startup() should raise VoiceDependencyError when wake models are absent."""
        from services.voice.core import VoiceConfig
        from services.voice.main import validate_startup

        with tempfile.TemporaryDirectory() as empty_dir:
            config = VoiceConfig(
                wake_model_dir=Path(empty_dir),
                tts_model_path=Path(empty_dir) / "nonexistent.onnx",
                ack_sound_path=Path(empty_dir) / "nonexistent.mp3",
            )

            with self.assertRaises(VoiceDependencyError):
                asyncio.run(validate_startup(config))

    def test_to_mono_pcm16_uses_first_channel(self) -> None:
        from pipecat.frames.frames import InputAudioRawFrame
        from services.voice.wake import WakeWordProcessor

        # Interleaved stereo: L=[1000, 2000], R=[-1000, -2000].
        frame = InputAudioRawFrame(
            audio=(b"\xe8\x03\x18\xfc\xd0\x07\x30\xf8"),
            sample_rate=16000,
            num_channels=2,
        )

        mono = WakeWordProcessor._to_mono_pcm16(frame)
        self.assertEqual(mono.tolist(), [1000, 2000])

    def test_resample_to_16k_from_48k(self) -> None:
        from services.voice.wake import WakeWordProcessor

        # 48k input should downsample by ~3x when normalized to 16k.
        src = list(range(480))
        import numpy as np

        out = WakeWordProcessor._resample_to_16k(np.asarray(src, dtype=np.int16), 48000)
        self.assertTrue(158 <= len(out) <= 162)

    def test_post_wake_flush_clamp_prevents_negative(self) -> None:
        """_flush_remaining must not go negative when a frame is larger than the flush window."""
        from pipecat.frames.frames import InputAudioRawFrame
        from pipecat.processors.frame_processor import FrameDirection
        from services.voice.wake import _State, WakeWordProcessor

        proc, _detector = self._make_processor(triggered_result=(0.85, True))

        async def run() -> None:
            async def capture(frame, direction=FrameDirection.DOWNSTREAM):
                pass

            proc.push_frame = capture  # type: ignore[method-assign]

            async def no_ack() -> None:
                pass

            proc._play_ack = no_ack  # type: ignore[method-assign]
            proc._awake_timeout = no_ack  # suppress timeout task

            # First frame: triggers wake; sets _flush_remaining = OWW_CHUNK_BYTES (2560 bytes).
            trigger_frame = InputAudioRawFrame(
                audio=b"\x00" * 3072, sample_rate=16000, num_channels=1
            )
            await proc.process_frame(trigger_frame, FrameDirection.DOWNSTREAM)
            self.assertEqual(proc._state, _State.awake)

            # Second frame: much larger than remaining flush window.
            big_frame = InputAudioRawFrame(
                audio=b"\x00" * 10000, sample_rate=16000, num_channels=1
            )
            await proc.process_frame(big_frame, FrameDirection.DOWNSTREAM)

        asyncio.run(run())
        self.assertGreaterEqual(proc._flush_remaining, 0, "_flush_remaining must not go negative")

    def test_stuck_awake_timeout_resets_to_sleeping(self) -> None:
        """WakeWordProcessor must reset to sleeping if BotStoppedSpeakingFrame never arrives."""
        from pathlib import Path
        import tempfile
        from pipecat.frames.frames import InputAudioRawFrame
        from pipecat.processors.frame_processor import FrameDirection
        from services.voice.wake import _State, WakeWordProcessor
        from unittest.mock import MagicMock

        detector = MagicMock()
        detector.triggered.return_value = (0.85, True)
        detector.reset.return_value = None

        ack_path = Path(tempfile.mktemp(suffix=".mp3"))
        proc = WakeWordProcessor(
            detector=detector,
            ack_path=ack_path,
            vad_threshold=0.0,
            awake_timeout_seconds=0.05,  # very short for test speed
        )

        async def run() -> None:
            async def capture(frame, direction=FrameDirection.DOWNSTREAM):
                pass

            proc.push_frame = capture  # type: ignore[method-assign]

            async def no_ack() -> None:
                pass

            proc._play_ack = no_ack  # type: ignore[method-assign]

            frame = InputAudioRawFrame(
                audio=b"\x00" * 3072, sample_rate=16000, num_channels=1
            )
            await proc.process_frame(frame, FrameDirection.DOWNSTREAM)
            self.assertEqual(proc._state, _State.awake)

            # Wait longer than the timeout — no BotStoppedSpeakingFrame sent.
            await asyncio.sleep(0.15)

        asyncio.run(run())
        self.assertEqual(
            proc._state, _State.sleeping,
            "processor must reset to sleeping after awake timeout with no BotStoppedSpeakingFrame",
        )


class ContextTrimmerTests(unittest.TestCase):
    def test_wake_reset_relay_pushes_bot_stop_upstream_on_tts_stop(self) -> None:
        """WakeResetRelay should mirror TTSStoppedFrame upstream as BotStoppedSpeakingFrame."""
        from pipecat.frames.frames import BotStoppedSpeakingFrame, TTSStoppedFrame
        from pipecat.processors.frame_processor import FrameDirection
        from services.voice.main import WakeResetRelay

        relay = WakeResetRelay()
        seen: list[tuple[object, FrameDirection]] = []

        async def run() -> None:
            async def capture(frame, direction=FrameDirection.DOWNSTREAM):
                seen.append((frame, direction))

            relay.push_frame = capture  # type: ignore[method-assign]
            await relay.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)

        asyncio.run(run())

        self.assertEqual(len(seen), 2)
        self.assertIsInstance(seen[0][0], BotStoppedSpeakingFrame)
        self.assertEqual(seen[0][1], FrameDirection.UPSTREAM)
        self.assertIsInstance(seen[1][0], TTSStoppedFrame)
        self.assertEqual(seen[1][1], FrameDirection.DOWNSTREAM)

    def test_context_trimmer_caps_non_system_messages(self) -> None:
        """ContextTrimmer trims oldest turns, preserving the system message."""
        from pipecat.frames.frames import BotStoppedSpeakingFrame
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.frame_processor import FrameDirection
        from services.voice.main import ContextTrimmer

        context = LLMContext(messages=[
            {"role": "system", "content": "you are cass"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "u4"},
            {"role": "assistant", "content": "a4"},
        ])
        trimmer = ContextTrimmer(context=context, history_turns=2)

        async def run() -> None:
            async def capture(frame, direction=FrameDirection.DOWNSTREAM):
                pass

            trimmer.push_frame = capture  # type: ignore[method-assign]
            await trimmer.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        asyncio.run(run())

        non_system = [m for m in context.messages if m.get("role") != "system"]
        # history_turns=2 → keep last 4 messages (2 user + 2 assistant)
        self.assertEqual(len(non_system), 4)
        self.assertEqual(non_system[0]["content"], "u3")
        self.assertEqual(non_system[-1]["content"], "a4")
        # System message must survive trimming
        self.assertEqual(context.messages[0]["role"], "system")
        self.assertEqual(context.messages[0]["content"], "you are cass")

    def test_context_trimmer_leaves_short_history_untouched(self) -> None:
        """ContextTrimmer must not modify context already within the turn limit."""
        from pipecat.frames.frames import BotStoppedSpeakingFrame
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.frame_processor import FrameDirection
        from services.voice.main import ContextTrimmer

        context = LLMContext(messages=[
            {"role": "system", "content": "you are cass"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
        original_len = len(context.messages)
        trimmer = ContextTrimmer(context=context, history_turns=6)

        async def run() -> None:
            async def capture(frame, direction=FrameDirection.DOWNSTREAM):
                pass

            trimmer.push_frame = capture  # type: ignore[method-assign]
            await trimmer.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        asyncio.run(run())
        self.assertEqual(len(context.messages), original_len)


class WeatherSkillTests(unittest.TestCase):
    def test_weather_skill_extracts_location_with_preposition(self) -> None:
        skill = WeatherSkill()
        entities = skill.entities("what's the weather in London tomorrow?")
        self.assertEqual(entities["location"], "London")

    def test_weather_skill_can_handle_weather_queries(self) -> None:
        skill = WeatherSkill()
        self.assertTrue(skill.can_handle("Do I need an umbrella in Seattle?"))
        self.assertFalse(skill.can_handle("Tell me a joke."))

    def test_weather_skill_execute_missing_location(self) -> None:
        skill = WeatherSkill()

        async def run() -> str:
            return await skill.execute({"location": None})

        result = asyncio.run(run())
        self.assertIn("missing_location", result)


class AgentRouterTests(unittest.TestCase):
    def test_agent_uses_skill_heuristic_without_embed_model(self) -> None:
        class DummySkill(Skill):
            @property
            def name(self) -> str:
                return "dummy"

            @property
            def exemplars(self) -> list[str]:
                return []

            def can_handle(self, transcript: str) -> bool:
                return "route-me" in transcript

            def entities(self, transcript: str) -> dict:
                return {}

            async def execute(self, entities: dict) -> str:
                return "ok"

        agent = AgentProcessor(
            config=VoiceConfig(),
            skills=[DummySkill()],
            embed_model_path=Path("/tmp/not-there.gguf"),
        )

        async def run() -> tuple[str | None, float, int]:
            return await agent._classify("please route-me now")

        label, score, classifier_ms = asyncio.run(run())
        self.assertEqual(label, "dummy")
        self.assertEqual(score, 1.0)
        self.assertEqual(classifier_ms, 0)


class AgentDiagnosticsTests(unittest.TestCase):
    """End-to-end turn coverage: events + skill fallback."""

    def _make_agent(self, skill: Skill | None = None) -> tuple[AgentProcessor, list]:
        from shared.straylight_shared import events as ev
        from services.voice import publisher

        captured: list = []

        async def fake_publish(event):
            captured.append(event)

        publisher.publish = fake_publish  # type: ignore[assignment]

        agent = AgentProcessor(
            config=VoiceConfig(),
            skills=[skill] if skill else [],
            embed_model_path=Path("/tmp/not-there.gguf"),
            session_id="t",
        )
        # Stub LLM helpers — no network.
        async def _stream(messages):
            for c in ("hello ", "there"):
                yield c
        async def _single(messages):
            return "weather formatted reply"
        async def _count(text=None):
            return 7
        agent._llm_stream = _stream  # type: ignore[method-assign]
        agent._llm_single_shot = _single  # type: ignore[method-assign]
        agent._count_tokens = lambda: _count()  # type: ignore[assignment]
        agent._count_tokens_for = lambda text: _count(text)  # type: ignore[assignment]
        return agent, captured

    def test_turn_publishes_diagnostics_event_on_slow_path(self) -> None:
        from shared.straylight_shared.events import TurnDiagnosticsEvent, IntentEvent

        agent, captured = self._make_agent()

        async def run() -> None:
            await agent._process_turn("hello there")

        asyncio.run(run())
        diag = [e for e in captured if isinstance(e, TurnDiagnosticsEvent)]
        intents = [e for e in captured if isinstance(e, IntentEvent)]
        self.assertEqual(len(diag), 1)
        self.assertEqual(diag[0].path, "slow")
        self.assertEqual(diag[0].provider, "local")
        self.assertEqual(diag[0].output_tokens, 7)
        self.assertGreaterEqual(diag[0].agent_ms, 0)
        self.assertEqual(intents[0].path, "slow")

    def test_skill_failure_falls_back_to_spoken_message(self) -> None:
        from services.voice.skills import SkillExecutionError
        from shared.straylight_shared.events import TurnDiagnosticsEvent

        class BrokenSkill(Skill):
            @property
            def name(self) -> str:
                return "weather"

            @property
            def exemplars(self) -> list[str]:
                return []

            def can_handle(self, transcript: str) -> bool:
                return "weather" in transcript

            def entities(self, transcript: str) -> dict:
                return {"location": "London"}

            async def execute(self, entities: dict) -> str:
                raise SkillExecutionError("tool down")

        agent, captured = self._make_agent(BrokenSkill())
        pushed: list = []

        async def capture(frame, direction):
            pushed.append(frame)
        agent.push_frame = capture  # type: ignore[method-assign]

        async def run() -> None:
            await agent._process_turn("weather in london")

        asyncio.run(run())
        texts = [getattr(f, "text", "") for f in pushed]
        joined = " ".join(texts)
        self.assertTrue(joined.strip(), "expected a spoken fallback")
        diag = [e for e in captured if isinstance(e, TurnDiagnosticsEvent)]
        self.assertEqual(len(diag), 1)
        self.assertEqual(diag[0].path, "fast")
        self.assertEqual(diag[0].skill_label, "weather")


if __name__ == "__main__":
    unittest.main()
