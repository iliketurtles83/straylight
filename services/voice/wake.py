"""WakeWordProcessor — Pipecat FrameProcessor wrapping OpenWakeWord.

States:
    sleeping — audio frames are inspected for wake word but not passed downstream.
    awake    — all audio frames pass through until BotStoppedSpeakingFrame resets to sleeping.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import numpy as np
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .clients import OpenWakeWordDetector


class _State(Enum):
    sleeping = auto()
    awake = auto()


@dataclass
class WakeWordFrame(Frame):
    """Emitted once when the wake word is detected."""

    score: float = 0.0


@dataclass
class _LatencyMarker:
    """Internal timing; shared with LatencyObserver via reference."""

    t_wake: float = field(default=0.0)
    t_transcript: float = field(default=0.0)


class WakeWordProcessor(FrameProcessor):
    """Gates pipeline audio behind wake word detection.

    Sleeping: consumes InputAudioRawFrame; runs SileroVAD to gate OWW calls.
    Awake: passes InputAudioRawFrame downstream until bot finishes speaking.

    Args:
        detector:      Configured OpenWakeWordDetector instance.
        ack_path:      Path to the acknowledgement sound (.mp3 or .wav).
        vad_threshold: Voice-confidence minimum before calling OWW. Saves CPU on silence.
    """

    def __init__(
        self,
        detector: OpenWakeWordDetector,
        ack_path: Path,
        vad_threshold: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._detector = detector
        self._ack_path = ack_path
        self._vad_threshold = vad_threshold
        self._state = _State.sleeping
        self._marker = _LatencyMarker()
        self._vad = SileroVADAnalyzer(sample_rate=16000)
        self._vad.set_sample_rate(16000)
        # Silero VAD in Pipecat expects 512 samples at 16kHz (32ms).
        self._VAD_CHUNK_BYTES: int = 512 * 2
        # OpenWakeWord performs best on 80ms windows at 16kHz.
        self._OWW_CHUNK_BYTES: int = 1280 * 2
        self._vad_buffer: bytes = b""
        self._oww_buffer: bytes = b""
        self._POST_WAKE_FLUSH_BYTES: int = int(16000 * 0.4 * 2)
        self._flush_remaining: int = 0
        # Diagnostics: count chunks processed and track peak OWW score.
        self._chunks_processed: int = 0
        self._peak_oww_score: float = 0.0

    # ------------------------------------------------------------------
    # Public timing access (used by LatencyObserver in main.py)
    # ------------------------------------------------------------------

    @property
    def t_wake(self) -> float:
        return self._marker.t_wake

    @property
    def t_transcript(self) -> float:
        return self._marker.t_transcript

    @t_transcript.setter
    def t_transcript(self, value: float) -> None:
        self._marker.t_transcript = value

    # ------------------------------------------------------------------
    # Pipecat FrameProcessor interface
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._state is _State.sleeping:
                await self._check_wake(frame)
            elif self._flush_remaining > 0:
                # Post-wake flush: silently discard audio until the wake word
                # utterance has fully passed so Whisper doesn't hear "computer".
                self._flush_remaining -= len(frame.audio)
            else:
                # Awake — pass audio through so Whisper can transcribe it.
                await self.push_frame(frame, direction)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._state is _State.awake:
                logger.debug("wake: bot done speaking → back to sleeping")
            self._state = _State.sleeping
            self._vad_buffer = b""
            self._oww_buffer = b""
            self._flush_remaining = 0
            self._detector.reset()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _check_wake(self, frame: InputAudioRawFrame) -> None:
        # Normalize incoming audio to mono PCM16 before buffering so detector
        # behavior is stable across devices that expose stereo capture.
        pcm = np.frombuffer(frame.audio, dtype=np.int16)
        num_channels = int(getattr(frame, "num_channels", 1) or 1)
        if num_channels > 1 and pcm.size >= num_channels:
            usable = (pcm.size // num_channels) * num_channels
            pcm = pcm[:usable].reshape(-1, num_channels).mean(axis=1).astype(np.int16)

        # Feed VAD in 512-sample windows; only speech windows are accumulated
        # for OWW scoring in 1280-sample windows.
        self._vad_buffer += pcm.tobytes()
        while len(self._vad_buffer) >= self._VAD_CHUNK_BYTES:
            vad_chunk = self._vad_buffer[: self._VAD_CHUNK_BYTES]
            self._vad_buffer = self._vad_buffer[self._VAD_CHUNK_BYTES :]

            self._chunks_processed += 1
            # Heartbeat every ~5s (5000ms / 32ms per chunk ≈ 156 chunks).
            if self._chunks_processed % 156 == 0:
                logger.debug(
                    f"wake: listening... chunks={self._chunks_processed} "
                    f"peak_oww={self._peak_oww_score:.3f} threshold={self._detector.threshold:.2f}"
                )

            # VAD gate — skip OWW on silence.
            try:
                confidence = float(np.asarray(self._vad.voice_confidence(vad_chunk)).flat[0])
            except Exception:
                confidence = 1.0

            if confidence < self._vad_threshold:
                continue

            self._oww_buffer += vad_chunk
            while len(self._oww_buffer) >= self._OWW_CHUNK_BYTES:
                oww_chunk = self._oww_buffer[: self._OWW_CHUNK_BYTES]
                self._oww_buffer = self._oww_buffer[self._OWW_CHUNK_BYTES :]

                raw = np.frombuffer(oww_chunk, dtype=np.int16)
                score, fired = self._detector.triggered(raw)
                self._peak_oww_score = max(self._peak_oww_score, float(score))
                logger.debug(f"wake: voice chunk — vad={float(confidence):.2f} oww={float(score):.3f}")

                if not fired:
                    continue

                self._marker.t_wake = time.monotonic()
                logger.info(f"wake: trigger detected (score={float(score):.3f})")
                self._state = _State.awake
                self._flush_remaining = self._POST_WAKE_FLUSH_BYTES

                # Emit WakeWordFrame downstream so LatencyObserver can record t_wake.
                await self.push_frame(WakeWordFrame(score=score), FrameDirection.DOWNSTREAM)

                # Play ack concurrently; don't block the pipeline.
                asyncio.ensure_future(self._play_ack())
                return

    async def _play_ack(self) -> None:
        try:
            await asyncio.to_thread(
                lambda: subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(self._ack_path)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        except Exception as exc:
            logger.warning(f"wake: ack playback failed: {exc}")
