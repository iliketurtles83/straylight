"""Straylight voice service — Phase 1 entry point.

Wires a Pipecat pipeline:
  LocalAudioTransport.input()
  → [WakeWordProcessor]   (omitted in --listen mode)
  → WhisperSTTService
  → LLMContextAggregator (user)
  → OpenAILLMService   (llama-server backend)
  → PiperTTSService
  → LatencyObserver
  → LocalAudioTransport.output()
  → LLMContextAggregator (assistant)

Run via:
  python -m voice.main [--no-validate] [--listen]

--listen: skip wake word detection; pipeline always active.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import time
from pathlib import Path

import httpx
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from .clients import OpenWakeWordDetector, VoiceDependencyError
from .core import VoiceConfig, load_system_prompt
from .wake import WakeWordFrame, WakeWordProcessor


# ---------------------------------------------------------------------------
# Latency observer
# ---------------------------------------------------------------------------

class LatencyObserver(FrameProcessor):
    """Logs turn latency: wake → transcript → first TTS audio byte.

    Placed between PiperTTSService and transport.output() so it sees
    TTSAudioRawFrame before playback.
    """

    def __init__(self, wake_processor: WakeWordProcessor | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._wake = wake_processor
        self._t_transcript: float = 0.0
        self._ttfb_logged: bool = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, WakeWordFrame) and self._wake is not None:
            # Reset per-turn state when a new wake fires.
            self._t_transcript = 0.0
            self._ttfb_logged = False

        elif isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            self._t_transcript = time.monotonic()

        elif isinstance(frame, TTSAudioRawFrame) and not self._ttfb_logged and self._t_transcript:
            t_now = time.monotonic()
            t_wake = self._wake.t_wake if self._wake is not None else None

            wake_ms = round((self._t_transcript - t_wake) * 1000) if t_wake else None
            ttfb_ms = round((t_now - self._t_transcript) * 1000)

            logger.info(
                json.dumps({
                    "event": "turn_latency",
                    "wake_to_stt_ms": wake_ms,
                    "stt_to_ttfb_ms": ttfb_ms,
                })
            )
            self._ttfb_logged = True

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._ttfb_logged = False

        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

async def validate_startup(config: VoiceConfig) -> None:
    """Fail fast if any required asset or service is missing."""

    if not config.listen_mode:
        wake_files = [
            config.wake_model_dir / config.wakeword_model_file,
            config.wake_model_dir / config.wakeword_melspec_model_file,
            config.wake_model_dir / config.wakeword_embedding_model_file,
        ]
        for p in wake_files:
            if not p.exists():
                raise VoiceDependencyError(f"Wake word model missing: {p}")

        if not config.ack_sound_path.exists():
            raise VoiceDependencyError(f"Ack sound missing: {config.ack_sound_path}")

    if not config.tts_model_path.exists():
        raise VoiceDependencyError(f"TTS model missing: {config.tts_model_path}")

    try:
        import sounddevice as sd
        sd.query_devices(kind="input")
    except Exception as exc:
        raise VoiceDependencyError(f"No audio input device: {exc}") from exc

    health_url = f"{config.llm_base_url}/health"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(health_url, timeout=5.0)
        if resp.status_code != 200:
            raise VoiceDependencyError(
                f"llama-server health check failed: HTTP {resp.status_code} from {health_url}"
            )
    except httpx.ConnectError as exc:
        raise VoiceDependencyError(
            f"llama-server not reachable at {health_url} — is it running?"
        ) from exc

    logger.info("startup validation passed")


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(config: VoiceConfig) -> tuple[Pipeline, WakeWordProcessor | None]:
    system_prompt = load_system_prompt(config.prompt_path)

    # --- Transport -----------------------------------------------------------
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=config.sample_rate,
            audio_out_enabled=True,
        )
    )

    # --- Wake word (skipped in listen mode) ----------------------------------
    if not config.listen_mode:
        detector = OpenWakeWordDetector(
            model_dir=config.wake_model_dir,
            wakeword_model_file=config.wakeword_model_file,
            melspec_model_file=config.wakeword_melspec_model_file,
            embedding_model_file=config.wakeword_embedding_model_file,
            wakeword_label=config.wakeword_model_file.replace(".onnx", ""),
            threshold=config.wake_threshold,
        )
        wake: WakeWordProcessor | None = WakeWordProcessor(
            detector=detector, ack_path=config.ack_sound_path
        )
        wake_stages: list = [wake]
    else:
        wake = None
        wake_stages = []
        logger.info("listen mode: wake word disabled, pipeline always active")

    # --- STT -----------------------------------------------------------------
    stt = WhisperSTTService(
        device=config.stt_device,
        compute_type=config.stt_compute_type,
        settings=WhisperSTTService.Settings(
            model=config.stt_model,
        ),
    )

    # --- LLM context & aggregators -------------------------------------------
    context = LLMContext(messages=[{"role": "system", "content": system_prompt}])
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(sample_rate=config.sample_rate),
        ),
    )

    # --- LLM -----------------------------------------------------------------
    llm = OpenAILLMService(
        api_key="not-needed",
        base_url=f"{config.llm_base_url}/v1",
        settings=OpenAILLMService.Settings(
            model=config.llm_model,
            temperature=0.2,
        ),
    )

    # --- TTS -----------------------------------------------------------------
    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice=config.tts_model_path.stem),
        download_dir=config.tts_model_path.parent,
    )

    # --- Latency observer ----------------------------------------------------
    latency_observer = LatencyObserver(wake_processor=wake)

    # --- Pipeline ------------------------------------------------------------
    pipeline = Pipeline([
        transport.input(),
        *wake_stages,
        stt,
        aggregators.user(),
        llm,
        tts,
        latency_observer,
        transport.output(),
        aggregators.assistant(),
    ])

    return pipeline, wake


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(validate: bool = True, listen_mode: bool = False) -> None:
    config = VoiceConfig.from_env()
    if listen_mode:
        config = dataclasses.replace(config, listen_mode=True)
    logger.info(
        f"straylight voice service starting (llm={config.llm_base_url}, "
        f"mode={'listen' if config.listen_mode else 'wake'})"
    )

    if validate:
        await validate_startup(config)

    pipeline, _wake = build_pipeline(config)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=config.sample_rate),
    )
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Straylight voice service")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip startup validation (useful for testing without live services)",
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Listen mode: skip wake word, pipeline always active",
    )
    args = parser.parse_args()
    asyncio.run(main(validate=not args.no_validate, listen_mode=args.listen))
