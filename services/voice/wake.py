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

    Sleeping: consumes InputAudioRawFrame; runs RMS energy gate then OWW.
    Awake: passes InputAudioRawFrame downstream until bot finishes speaking.

    Args:
        detector:      Configured OpenWakeWordDetector instance.
        ack_path:      Path to the acknowledgement sound (.mp3 or .wav).
        vad_threshold: Normalised RMS energy minimum [0.0–1.0] before calling OWW.
                       Saves CPU on silence. Default 0.3 works well for 16-bit PCM.
    """

    def __init__(
        self,
        detector: OpenWakeWordDetector,
        ack_path: Path,
        vad_threshold: float = 0.3,
        awake_timeout_seconds: float = 30.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._detector = detector
        self._ack_path = ack_path
        self._vad_threshold = vad_threshold
        self._awake_timeout_seconds = awake_timeout_seconds
        self._awake_timeout_task: asyncio.Task | None = None
        self._state = _State.sleeping
        self._marker = _LatencyMarker()
        # RMS energy gate: 512-sample windows at 16kHz (32ms) — same granularity
        # as Silero VAD, avoids the Pipecat pipeline-component API mismatch.
        self._VAD_CHUNK_BYTES: int = 512 * 2
        # OpenWakeWord performs best on 80ms windows at 16kHz.
        self._OWW_CHUNK_BYTES: int = 1280 * 2
        self._oww_buffer: bytes = b""
        # Flush exactly one OWW window after wake — skips the trailing artifact
        # of the wake phrase without eating the first syllable of the command.
        self._POST_WAKE_FLUSH_BYTES: int = self._OWW_CHUNK_BYTES
        self._flush_remaining: int = 0
        # Diagnostics: count chunks processed and track peak OWW score.
        self._chunks_processed: int = 0
        self._peak_oww_score: float = 0.0
        self._peak_rms_energy: float = 0.0

    @staticmethod
    def _rms_energy(pcm_bytes: bytes) -> float:
        """Return normalised RMS energy of raw int16 PCM bytes in [0.0, 1.0]."""
        if not pcm_bytes:
            return 0.0
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(samples)))) / 32768.0

    @staticmethod
    def _to_mono_pcm16(frame: InputAudioRawFrame) -> np.ndarray:
        """Convert frame audio to a mono int16 stream for wake processing."""
        pcm = np.frombuffer(frame.audio, dtype=np.int16)
        num_channels = int(getattr(frame, "num_channels", 1) or 1)
        if num_channels <= 1 or pcm.size < num_channels:
            return pcm

        usable = (pcm.size // num_channels) * num_channels
        if usable == 0:
            return np.empty(0, dtype=np.int16)

        # Use channel 0 rather than averaging channels; some capture stacks can
        # produce phase-skewed stereo where averaging suppresses wake features.
        interleaved = pcm[:usable].reshape(-1, num_channels)
        return interleaved[:, 0].astype(np.int16, copy=False)

    @staticmethod
    def _resample_to_16k(pcm: np.ndarray, input_rate: int) -> np.ndarray:
        """Resample mono int16 PCM to 16kHz using linear interpolation."""
        if input_rate == 16000 or pcm.size == 0:
            return pcm.astype(np.int16, copy=False)
        if input_rate <= 0:
            return pcm.astype(np.int16, copy=False)

        out_len = max(1, int(round(pcm.size * 16000.0 / float(input_rate))))
        if out_len == pcm.size:
            return pcm.astype(np.int16, copy=False)

        x_old = np.arange(pcm.size, dtype=np.float32)
        x_new = np.linspace(0.0, float(max(pcm.size - 1, 0)), out_len, dtype=np.float32)
        resampled = np.interp(x_new, x_old, pcm.astype(np.float32))
        clipped = np.clip(np.rint(resampled), -32768, 32767)
        return clipped.astype(np.int16)

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
        # logger.debug("wake: got frame type={}", type(frame).__name__)

        if isinstance(frame, InputAudioRawFrame):
            if self._state is _State.sleeping:
                await self._check_wake(frame)
            elif self._flush_remaining > 0:
                # Post-wake flush: silently discard audio until the wake word
                # utterance has fully passed so Whisper doesn't hear "computer".
                self._flush_remaining = max(0, self._flush_remaining - len(frame.audio))
            else:
                # Awake — pass audio through so Whisper can transcribe it.
                await self.push_frame(frame, direction)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._awake_timeout_task is not None:
                self._awake_timeout_task.cancel()
                self._awake_timeout_task = None
            if self._state is _State.awake:
                logger.debug("wake: bot done speaking → back to sleeping")
            self._state = _State.sleeping
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
        # Normalize to mono 16k PCM16 before buffering so detector behavior is
        # stable across devices that expose stereo and/or non-16k capture.
        pcm = self._to_mono_pcm16(frame)
        in_rate = int(getattr(frame, "sample_rate", 16000) or 16000)
        pcm = self._resample_to_16k(pcm, in_rate)

        # Accumulate ALL audio into the OWW buffer — including silence.
        # OWW's internal LSTM/transformer requires contiguous audio to build up
        # its feature state. Gating out silence chunks breaks temporal continuity
        # and collapses scores to the ~0.001 noise floor.
        # The RMS gate is kept only for log suppression, not for data gating.
        self._oww_buffer += pcm.tobytes()

        while len(self._oww_buffer) >= self._OWW_CHUNK_BYTES:
            oww_chunk = self._oww_buffer[: self._OWW_CHUNK_BYTES]
            self._oww_buffer = self._oww_buffer[self._OWW_CHUNK_BYTES :]

            self._chunks_processed += 1
            # Heartbeat every ~5s (5000ms / 80ms per chunk ≈ 62 chunks).
            if self._chunks_processed % 62 == 0:
                logger.debug(
                    f"wake: listening... chunks={self._chunks_processed} "
                    f"peak_oww={self._peak_oww_score:.3f} peak_rms={self._peak_rms_energy:.3f} "
                    f"threshold={self._detector.threshold:.2f}"
                )

            raw = np.frombuffer(oww_chunk, dtype=np.int16)
            score, fired = self._detector.triggered(raw)
            self._peak_oww_score = max(self._peak_oww_score, float(score))

            # Log only during voice activity to reduce noise.
            confidence = self._rms_energy(oww_chunk)
            self._peak_rms_energy = max(self._peak_rms_energy, float(confidence))
            if confidence >= self._vad_threshold:
                logger.debug(
                    "wake: voice chunk — in_sr={} vad={:.2f} oww={:.3f}",
                    in_rate,
                    float(confidence),
                    float(score),
                )

            if not fired:
                continue

            self._marker.t_wake = time.monotonic()
            logger.info(f"wake: trigger detected (score={float(score):.3f})")
            self._state = _State.awake
            self._flush_remaining = self._POST_WAKE_FLUSH_BYTES

            # Emit WakeWordFrame downstream so LatencyObserver can record t_wake.
            await self.push_frame(WakeWordFrame(score=score), FrameDirection.DOWNSTREAM)

            # Stuck-awake guard: if BotStoppedSpeakingFrame never arrives (LLM/TTS
            # failure, interruption without clean shutdown), force-reset to sleeping
            # after a timeout rather than passing all mic audio downstream indefinitely.
            self._awake_timeout_task = asyncio.ensure_future(self._awake_timeout())
            # Play ack concurrently; don't block the pipeline.
            asyncio.ensure_future(self._play_ack())
            return

    async def _awake_timeout(self) -> None:
        """Force-reset to sleeping if BotStoppedSpeakingFrame never arrives."""
        try:
            await asyncio.sleep(self._awake_timeout_seconds)
        except asyncio.CancelledError:
            return  # Normal path: BotStoppedSpeakingFrame cancelled us.
        if self._state is _State.awake:
            logger.warning(
                "wake: awake for {:.0f}s with no BotStoppedSpeakingFrame — "
                "forcing reset to sleeping (LLM/TTS failure?)",
                self._awake_timeout_seconds,
            )
            self._state = _State.sleeping
            self._oww_buffer = b""
            self._flush_remaining = 0
            self._detector.reset()
        self._awake_timeout_task = None

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
