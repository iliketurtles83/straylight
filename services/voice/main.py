"""Straylight voice service entry point.

Wires a Pipecat pipeline:
  LocalAudioTransport.input()
  → [WakeWordProcessor]   (omitted in --listen mode)
  → VADProcessor
  → WhisperSTTService
  → AgentProcessor
  → PiperTTSService
  → WakeResetRelay
  → LatencyObserver
  → LocalAudioTransport.output()

Run via:
  python -m services.voice.main [--no-validate] [--listen]

--listen: skip wake word detection; pipeline always active.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import shutil
import time
from pathlib import Path

import httpx
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    TTSStoppedFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from agent import AgentProcessor
from .clients import OpenWakeWordDetector, VoiceDependencyError
from .core import VoiceConfig
from .skills.weather import WeatherSkill
from .wake import WakeWordProcessor


# ---------------------------------------------------------------------------
# Latency observer
# ---------------------------------------------------------------------------

class LatencyObserver(FrameProcessor):
    """Logs turn latency: wake → transcript → first TTS audio byte.

    Placed between PiperTTSService and transport.output().
    WakeWordFrame does not flow this far downstream, so new-wake detection
    is done by comparing the WakeWordProcessor's t_wake against a stored
    sentinel rather than waiting for the frame itself.
    """

    def __init__(self, wake_processor: WakeWordProcessor | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._wake = wake_processor
        self._t_transcript: float = 0.0
        self._ttfb_logged: bool = False
        self._last_wake_t: float = 0.0  # sentinel: detect new wake events

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # Detect a new wake event: t_wake advances every time OWW fires.
        if self._wake is not None:
            current_t_wake = self._wake.t_wake
            if current_t_wake != self._last_wake_t:
                self._last_wake_t = current_t_wake
                self._t_transcript = 0.0
                self._ttfb_logged = False

        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
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


class WakeResetRelay(FrameProcessor):
    """Mirrors TTS completion upstream so WakeWordProcessor can reset promptly.

    In the live pipeline, TTSStoppedFrame is observed downstream of the wake
    gate, and the LocalAudioOutputTransport emits BotStoppedSpeakingFrame even
    further downstream. WakeWordProcessor never sees that downstream bot-stop
    signal, so it can remain awake until the timeout fires. Relay the stop
    event upstream as BotStoppedSpeakingFrame while preserving the original
    downstream TTSStoppedFrame for the rest of the pipeline.
    """

    def __init__(self, wake_processor: WakeWordProcessor | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._wake = wake_processor

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if (
            self._wake is not None
            and isinstance(frame, TTSAudioRawFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
            self._wake.notify_bot_audio_active()

        if isinstance(frame, TTSStoppedFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._wake is not None:
                self._wake.notify_bot_audio_stopped()
            await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

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
            config.wakeword_embedding_model_file,
        ]
        for p in wake_files:
            if not p.exists():
                raise VoiceDependencyError(f"Wake word model missing: {p}")

        if not config.ack_sound_path.exists():
            raise VoiceDependencyError(f"Ack sound missing: {config.ack_sound_path}")
        _validate_ack_player_binary(config)

    if not config.tts_model_path.exists():
        raise VoiceDependencyError(f"TTS model missing: {config.tts_model_path}")

    if not config.embed_model_path.exists():
        logger.warning(
            "startup: embed model missing at {} — router will use heuristic fallback",
            config.embed_model_path,
        )
    if not config.router_exemplars_path.exists():
        logger.warning(
            "startup: exemplars file missing at {} — router will run without negative corpus",
            config.router_exemplars_path,
        )

    try:
        import sounddevice as sd
        sd.query_devices(kind="input")
        sd.check_input_settings(channels=1, samplerate=config.sample_rate)
    except Exception as exc:
        raise VoiceDependencyError(
            f"No audio input device compatible with {config.sample_rate}Hz: {exc}"
        ) from exc

    try:
        sd.check_output_settings(channels=1, samplerate=config.sample_rate)
    except Exception as exc:
        logger.warning(
            "audio: no output device compatible with {}Hz: {}", config.sample_rate, exc
        )

    # Check if LLM server is reachable (but don't require it to be running)
    async with httpx.AsyncClient() as client:
        try:
            await _validate_llama_server_health(config, client)
        except VoiceDependencyError:
            # If the server is not reachable, warn but don't fail the startup
            # This allows Straylight to connect to an external server
            logger.warning(
                "llama-server not reachable at {} — assuming external server connection",
                config.llm_base_url
            )

    logger.info("startup validation passed")


def _validate_ack_player_binary(config: VoiceConfig) -> None:
    """Ensure the configured ack playback binary is available and executable."""

    resolved = shutil.which(config.ack_player_bin)
    if resolved is None:
        raise VoiceDependencyError(
            "Ack player binary not found: "
            f"{config.ack_player_bin}. Install ffplay or set CASS_ACK_PLAYER_BIN."
        )


async def _validate_llama_server_health(config: VoiceConfig, client: httpx.AsyncClient) -> None:
    """Validate llama-server health endpoint only."""

    health_url = f"{config.llm_base_url}/health"
    try:
        resp = await client.get(health_url, timeout=5.0)
        if resp.status_code != 200:
            raise VoiceDependencyError(
                f"llama-server health check failed: HTTP {resp.status_code} from {health_url}"
            )
    except httpx.RequestError as exc:
        raise VoiceDependencyError(
            f"llama-server not reachable at {health_url} — is it running?"
        ) from exc


async def _validate_llama_server(config: VoiceConfig, client: httpx.AsyncClient) -> None:
    """Validate llama-server endpoints needed by the hot path."""

    # Check health first
    await _validate_llama_server_health(config, client)

    # Validate tokenize endpoint
    tokenize_url = f"{config.llm_base_url}/tokenize"
    try:
        resp = await client.post(
            tokenize_url,
            json={"content": "tokenize startup probe"},
            timeout=5.0,
        )
        if resp.status_code != 200:
            raise VoiceDependencyError(
                f"llama-server tokenize check failed: HTTP {resp.status_code} from {tokenize_url}"
            )
        tokens = resp.json().get("tokens")
        if not isinstance(tokens, list):
            raise VoiceDependencyError(
                f"llama-server tokenize check failed: malformed response from {tokenize_url}"
            )
    except VoiceDependencyError:
        raise
    except (httpx.RequestError, ValueError) as exc:
        raise VoiceDependencyError(
            f"llama-server tokenize endpoint not usable at {tokenize_url}: {exc}"
        ) from exc

    # Pre-warm: llama-server can pass /health while the model is still paging in.
    # A cheap one-token call ensures first-turn latency is not a cold-start surprise.
    warmup_url = f"{config.llm_base_url}/v1/chat/completions"
    try:
        await client.post(
            warmup_url,
            json={
                "model": config.llm_model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "temperature": 0.0,
            },
            timeout=20.0,
        )
        logger.info("startup: llama-server warmup probe complete")
    except Exception as exc:
        logger.warning(
            "startup: llama-server warmup probe failed ({}); first turn may be slow", exc
        )


# ---------------------------------------------------------------------------
# Audio device selection
# ---------------------------------------------------------------------------

def _select_audio_devices(
    sample_rate: int,
    input_device_index: int | None = None,
    output_device_index: int | None = None,
    preferred_input_name: str | None = None,
    preferred_output_name: str | None = None,
) -> tuple[int | None, int | None]:
    """Resolve stable audio device indices, preferring physical over virtual backends.

    Avoids PulseAudio/PipeWire/JACK virtual devices that can cause 30s+ startup
    hangs on Linux. Falls back to any 16kHz-compatible device if no physical
    device is found.
    """
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover
        logger.warning("audio: sounddevice unavailable for device selection: {}", exc)
        return input_device_index, output_device_index

    if input_device_index is not None and output_device_index is not None:
        return input_device_index, output_device_index

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
            "audio: requested input device {} does not support {}Hz; auto-selecting",
            input_device_index, sample_rate,
        )
        input_device_index = None
    if output_device_index is not None and not _supports_output(output_device_index):
        logger.warning(
            "audio: requested output device {} does not support {}Hz; auto-selecting",
            output_device_index, sample_rate,
        )
        output_device_index = None

    devices = sd.query_devices()
    fallback_in: int | None = None
    fallback_out: int | None = None

    def _match_preferred_device(preferred_name: str, want_input: bool) -> int | None:
        preferred = preferred_name.strip().lower()
        if not preferred:
            return None
        for idx, dev in enumerate(devices):
            name = str(dev.get("name", ""))
            supports = _supports_input(idx) if want_input else _supports_output(idx)
            channels = int(dev.get("max_input_channels" if want_input else "max_output_channels", 0) or 0)
            if channels > 0 and supports and preferred in name.lower():
                return idx
        return None

    if input_device_index is None and preferred_input_name:
        matched_input = _match_preferred_device(preferred_input_name, want_input=True)
        if matched_input is not None:
            input_device_index = matched_input
        else:
            logger.warning(
                "audio: preferred input device {!r} not found at {}Hz; auto-selecting",
                preferred_input_name,
                sample_rate,
            )

    if output_device_index is None and preferred_output_name:
        matched_output = _match_preferred_device(preferred_output_name, want_input=False)
        if matched_output is not None:
            output_device_index = matched_output
        else:
            logger.warning(
                "audio: preferred output device {!r} not found at {}Hz; auto-selecting",
                preferred_output_name,
                sample_rate,
            )

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
            "audio: no fully compatible device pair found at {}Hz (input={} output={})",
            sample_rate, input_device_index, output_device_index,
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
        "audio: selected input={} ({}) output={} ({}) @{}Hz",
        input_device_index,
        input_name or "unknown",
        output_device_index,
        output_name or "unknown",
        sample_rate,
    )
    return input_device_index, output_device_index


# ---------------------------------------------------------------------------
# Router exemplars
# ---------------------------------------------------------------------------

def _load_none_exemplars(path: Path) -> list[str]:
    """Load negative ('none') exemplars from JSONL.

    Expected line format: {"text": "...", "label": "none"}.
    """
    if not path.exists():
        logger.warning("router: exemplars file not found at {} (continuing)", path)
        return []

    none_exemplars: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            if row.get("label") == "none" and isinstance(row.get("text"), str):
                text = row["text"].strip()
                if text:
                    none_exemplars.append(text)
        except Exception:
            logger.debug("router: skipping malformed exemplar at {}:{}", path, lineno)
    return none_exemplars


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(config: VoiceConfig) -> tuple[Pipeline, WakeWordProcessor | None]:
    # --- Transport -----------------------------------------------------------
    input_device_index, output_device_index = _select_audio_devices(
        sample_rate=config.sample_rate,
        preferred_input_name=config.input_device_name,
        preferred_output_name=config.output_device_name,
    )
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=config.sample_rate,
            audio_out_enabled=True,
            input_device_index=input_device_index,
            output_device_index=output_device_index,
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
            detector=detector,
            ack_path=config.ack_sound_path,
            ack_player_bin=config.ack_player_bin,
            bot_audio_drain_ms=config.bot_audio_drain_ms,
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

    # --- AgentProcessor ------------------------------------------------------
    none_exemplars = _load_none_exemplars(config.router_exemplars_path)
    agent = AgentProcessor(
        config=config,
        skills=[WeatherSkill()],
        embed_model_path=config.embed_model_path,
        none_exemplars=none_exemplars,
        threshold=config.router_threshold,
        min_gap=config.router_min_gap,
    )

    # --- TTS -----------------------------------------------------------------
    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice=config.tts_model_path.stem),
        download_dir=config.tts_model_path.parent,
    )

    # --- VAD (gates Whisper turn segmentation) --------------------------------
    # Segmented WhisperSTTService only emits TranscriptionFrame after receiving
    # VADUserStartedSpeakingFrame / VADUserStoppedSpeakingFrame.  The explicit
    # VADProcessor must sit upstream of STT in the pipeline stage list.
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

    # --- Latency observer ----------------------------------------------------
    latency_observer = LatencyObserver(wake_processor=wake)

    # --- Wake reset relay ----------------------------------------------------
    wake_reset_relay = WakeResetRelay(wake_processor=wake)

    # --- Pipeline ------------------------------------------------------------
    pipeline = Pipeline([
        transport.input(),
        *wake_stages,
        vad,
        stt,
        agent,
        tts,
        wake_reset_relay,
        latency_observer,
        transport.output(),
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
