from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from services.voice.core import ConversationWindow, TranscriptTurn, audio_to_wav_bytes, load_system_prompt, normalize_reply_text, trim_to_last_turns
from services.voice.clients import VoiceDependencyError


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


class WakeWordProcessorTests(unittest.TestCase):
    def _make_processor(self, triggered_result: tuple[float, bool] = (0.0, False)):
        """Build a WakeWordProcessor with a mocked detector and SileroVAD."""
        from services.voice.wake import WakeWordProcessor

        detector = MagicMock()
        detector.triggered.return_value = triggered_result
        detector.reset.return_value = None

        ack_path = Path(tempfile.mktemp(suffix=".mp3"))

        # Patch SileroVADAnalyzer so it doesn't load the ONNX model on init.
        with patch("services.voice.wake.SileroVADAnalyzer") as MockVAD:
            vad_instance = MagicMock()
            vad_instance.voice_confidence.return_value = 1.0  # Always "loud" so OWW is called.
            MockVAD.return_value = vad_instance
            proc = WakeWordProcessor(detector=detector, ack_path=ack_path)

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


if __name__ == "__main__":
    unittest.main()

