"""Echo loop — no-LLM pipeline diagnostic.

Runs the full voice pipeline stack without an LLM:

    LocalAudioTransport → WakeWordProcessor → WhisperSTTService
        → EchoProcessor → PiperTTSService → LocalAudioTransport.output()

WhisperSTTService in Pipecat is segmented STT, so it requires VAD speaking
events. This script inserts a Silero VAD processor before STT and logs
start/stop events to make capture behavior observable.

EchoProcessor catches TranscriptionFrame, plays ack.mp3 (bridging the TTS
generation latency), then forwards the transcript text to Piper to be spoken
back verbatim.

WakeWordProcessor's built-in ack is suppressed (non-existent sentinel path);
ack is owned by EchoProcessor and fires after STT, not at wake detection.

Usage:
    python tests/echo_loop.py           # "computer" wake word required
    python tests/echo_loop.py --listen  # mic always active, no wake word
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import signal
import subprocess
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path when the script is run directly
# (i.e. `python tests/echo_loop.py`) rather than as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from services.voice.clients import OpenWakeWordDetector
from services.voice.core import VoiceConfig
from services.voice.wake import WakeWordProcessor


# ---------------------------------------------------------------------------
# EchoProcessor
# ---------------------------------------------------------------------------

class EchoProcessor(FrameProcessor):
    """Catches TranscriptionFrame, plays ack.mp3, echoes text to Piper.

    Ack fires here — after STT completes — so the user hears the confirmation
    while Piper is generating audio. This bridges the TTS generation latency
    gap rather than the wake→speech gap.
    """

    def __init__(self, ack_path: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._ack_path = ack_path

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            text = frame.text.strip()
            if text:
                logger.info("echo: heard = {!r}", text)
                # Fire ack concurrently — don't stall the frame push.
                asyncio.ensure_future(self._play_ack())
                await self.push_frame(TextFrame(text=text), direction)
        else:
            await self.push_frame(frame, direction)

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
            logger.warning("echo: ack playback failed: {}", exc)


class VADTraceProcessor(FrameProcessor):
    """Logs VAD state transitions for utterance diagnostics."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, VADUserStartedSpeakingFrame):
                logger.info("vad: user started speaking")
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                logger.info("vad: user stopped speaking")
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def _validate(config: VoiceConfig) -> None:
    """Lightweight asset check — no HTTP calls, no sounddevice query."""
    errors: list[str] = []

    if not config.tts_model_path.exists():
        errors.append(f"TTS model missing: {config.tts_model_path}")
    if not config.ack_sound_path.exists():
        errors.append(f"Ack sound missing: {config.ack_sound_path}")

    if not config.listen_mode:
        for model_file in [
            config.wake_model_dir / config.wakeword_model_file,
            config.wake_model_dir / config.wakeword_melspec_model_file,
            config.wake_model_dir / config.wakeword_embedding_model_file,
        ]:
            if not model_file.exists():
                errors.append(f"Wake model missing: {model_file}")

    if errors:
        for e in errors:
            logger.error(e)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

# Sentinel path used to suppress WakeWordProcessor's built-in ack.
# The path is intentionally absent — ffplay fails silently (check=False).
# EchoProcessor owns the ack responsibility in this pipeline.
_NO_ACK = Path("/tmp/straylight-echo-no-ack")


def _select_audio_devices(
    input_device_index: int | None,
    output_device_index: int | None,
    sample_rate: int,
) -> tuple[int | None, int | None]:
    """Resolve stable audio device indices, avoiding flaky virtual defaults."""
    try:
        import sounddevice as sd
    except Exception as exc:
        logger.warning("echo: sounddevice unavailable for device selection: {}", exc)
        return input_device_index, output_device_index

    if input_device_index is not None and output_device_index is not None:
        return input_device_index, output_device_index

    devices = sd.query_devices()
    fallback_in: int | None = None
    fallback_out: int | None = None

    def _supports_input(idx: int) -> bool:
        try:
            sd.check_input_settings(device=idx, channels=1, samplerate=sample_rate)
            return True
        except Exception:
            return False

    def _supports_output(idx: int) -> bool:
        try:
            sd.check_output_settings(device=idx, channels=1, samplerate=sample_rate)
            return True
        except Exception:
            return False

    if input_device_index is not None and not _supports_input(input_device_index):
        logger.warning(
            "echo: requested input device {} does not support {} Hz; auto-selecting",
            input_device_index,
            sample_rate,
        )
        input_device_index = None
    if output_device_index is not None and not _supports_output(output_device_index):
        logger.warning(
            "echo: requested output device {} does not support {} Hz; auto-selecting",
            output_device_index,
            sample_rate,
        )
        output_device_index = None

    for idx, dev in enumerate(devices):
        name = str(dev.get("name", "")).lower()
        max_in = int(dev.get("max_input_channels", 0) or 0)
        max_out = int(dev.get("max_output_channels", 0) or 0)
        is_virtual = any(token in name for token in ("default", "pipewire", "pulse", "jack"))
        supports_in = max_in > 0 and _supports_input(idx)
        supports_out = max_out > 0 and _supports_output(idx)

        if fallback_in is None and supports_in:
            fallback_in = idx
        if fallback_out is None and supports_out:
            fallback_out = idx

        if input_device_index is None and supports_in and not is_virtual:
            input_device_index = idx
        if output_device_index is None and supports_out and not is_virtual:
            output_device_index = idx

        if input_device_index is not None and output_device_index is not None:
            break

    if input_device_index is None:
        input_device_index = fallback_in
    if output_device_index is None:
        output_device_index = fallback_out

    if input_device_index is None or output_device_index is None:
        logger.warning(
            "echo: no fully compatible audio device pair found at {} Hz (input={}, output={})",
            sample_rate,
            input_device_index,
            output_device_index,
        )

    input_name = None
    output_name = None
    try:
        if input_device_index is not None:
            input_name = str(devices[input_device_index].get("name", ""))
        if output_device_index is not None:
            output_name = str(devices[output_device_index].get("name", ""))
    except Exception:
        input_name = None
        output_name = None

    logger.info(
        "echo: audio devices selected input={} ({}) output={} ({}) @{}Hz",
        input_device_index,
        input_name or "unknown",
        output_device_index,
        output_name or "unknown",
        sample_rate,
    )
    return input_device_index, output_device_index


def _build_pipeline(
    config: VoiceConfig,
    input_device_index: int | None = None,
    output_device_index: int | None = None,
) -> Pipeline:
    t0 = time.monotonic()
    input_device_index, output_device_index = _select_audio_devices(
        input_device_index=input_device_index,
        output_device_index=output_device_index,
        sample_rate=config.sample_rate,
    )
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=config.sample_rate,
            input_device_index=input_device_index,
            output_device_index=output_device_index,
        )
    )
    logger.info("echo: transport ready ({} ms)", round((time.monotonic() - t0) * 1000))

    if config.listen_mode:
        wake_stages: list = []
        logger.info("listen mode: wake word disabled, mic always active")
    else:
        detector = OpenWakeWordDetector(
            model_dir=config.wake_model_dir,
            wakeword_model_file=config.wakeword_model_file,
            melspec_model_file=config.wakeword_melspec_model_file,
            embedding_model_file=config.wakeword_embedding_model_file,
            wakeword_label=config.wakeword_model_file.replace(".onnx", ""),
            threshold=config.wake_threshold,
        )
        wake_stages = [WakeWordProcessor(detector=detector, ack_path=_NO_ACK)]

    stt = WhisperSTTService(
        device=config.stt_device,
        compute_type=config.stt_compute_type,
        settings=WhisperSTTService.Settings(model=config.stt_model),
    )
    logger.info(
        "echo: stt ready model={} device={} compute={} ({} ms)",
        config.stt_model,
        config.stt_device,
        config.stt_compute_type,
        round((time.monotonic() - t0) * 1000),
    )

    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            sample_rate=config.sample_rate,
            params=VADParams(
                confidence=0.6,
                start_secs=0.15,
                stop_secs=0.25,
                min_volume=0.2,
            ),
        )
    )
    vad_trace = VADTraceProcessor()

    echo = EchoProcessor(ack_path=config.ack_sound_path)

    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice=config.tts_model_path.stem),
        download_dir=config.tts_model_path.parent,
    )
    logger.info("echo: tts ready voice={} ({} ms)", config.tts_model_path.stem, round((time.monotonic() - t0) * 1000))

    pipeline = Pipeline([
        transport.input(),
        *wake_stages,
        vad,
        vad_trace,
        stt,
        echo,
        tts,
        transport.output(),
    ])
    logger.info("echo: pipeline built ({} ms)", round((time.monotonic() - t0) * 1000))
    return pipeline


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(
    listen_mode: bool = False,
    validate: bool = True,
    stt_model: str | None = None,
    stt_device: str | None = None,
    stt_compute_type: str | None = None,
    input_device_index: int | None = None,
    output_device_index: int | None = None,
) -> None:
    config = VoiceConfig.from_env()
    if listen_mode:
        config = dataclasses.replace(config, listen_mode=True)
    if stt_model:
        config = dataclasses.replace(config, stt_model=stt_model)
    if stt_device:
        config = dataclasses.replace(config, stt_device=stt_device)
    if stt_compute_type:
        config = dataclasses.replace(config, stt_compute_type=stt_compute_type)

    if validate:
        _validate(config)

    logger.info(
        "echo loop starting — mode={} stt={} tts={}",
        "listen" if config.listen_mode else "wake",
        config.stt_model,
        config.tts_model_path.name,
    )

    t0 = time.monotonic()
    pipeline = _build_pipeline(
        config,
        input_device_index=input_device_index,
        output_device_index=output_device_index,
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=config.sample_rate),
        enable_rtvi=False,
        enable_turn_tracking=False,
    )
    runner = PipelineRunner()

    shutdown_requested = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_shutdown(sig_name: str) -> None:
        logger.info("echo: signal {} received, cancelling pipeline", sig_name)
        shutdown_requested.set()

    signal_handlers_registered = False
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown, sig.name)
            signal_handlers_registered = True
        except NotImplementedError:
            pass

    runner_task = asyncio.create_task(runner.run(task), name="echo-loop-runner")
    logger.info("echo: waiting for pipeline ready (startup {} ms)", round((time.monotonic() - t0) * 1000))

    if signal_handlers_registered:
        shutdown_task = asyncio.create_task(shutdown_requested.wait(), name="echo-loop-shutdown")
        done, pending = await asyncio.wait(
            {runner_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if runner_task not in done and shutdown_requested.is_set():
            await task.cancel()
            try:
                await asyncio.wait_for(runner_task, timeout=3.0)
            except TimeoutError:
                logger.warning("echo: pipeline cancellation timed out; forcing runner cancel")
                runner.cancel()
        else:
            await runner_task
        for pending_task in pending:
            pending_task.cancel()
    else:
        try:
            await runner_task
        except KeyboardInterrupt:
            logger.info("echo: keyboard interrupt received, cancelling pipeline")
            await task.cancel()
            try:
                await asyncio.wait_for(runner_task, timeout=3.0)
            except TimeoutError:
                logger.warning("echo: pipeline cancellation timed out; forcing runner cancel")
                runner.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.remove_signal_handler(sig)
        except NotImplementedError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Echo loop — voice pipeline without LLM")
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Skip wake word detection; mic always active",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip model/asset checks on startup",
    )
    parser.add_argument(
        "--stt-model",
        default="tiny.en",
        help="Whisper model for echo test (default: tiny.en for fast startup)",
    )
    parser.add_argument(
        "--stt-device",
        default=None,
        help="Override STT device (cpu/cuda)",
    )
    parser.add_argument(
        "--stt-compute-type",
        default=None,
        help="Override STT compute type (int8/float16/etc)",
    )
    parser.add_argument(
        "--input-device-index",
        type=int,
        default=None,
        help="Audio input device index (sounddevice); auto-picks a non-virtual input by default",
    )
    parser.add_argument(
        "--output-device-index",
        type=int,
        default=None,
        help="Audio output device index (sounddevice); auto-picks a non-virtual output by default",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            listen_mode=args.listen,
            validate=not args.no_validate,
            stt_model=args.stt_model,
            stt_device=args.stt_device,
            stt_compute_type=args.stt_compute_type,
            input_device_index=args.input_device_index,
            output_device_index=args.output_device_index,
        )
    )
