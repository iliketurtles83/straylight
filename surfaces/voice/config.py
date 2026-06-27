from __future__ import annotations

import io
import os
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

_BASE_DIR = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_SAMPLES = 1280
DEFAULT_WAKE_THRESHOLD = 0.5
DEFAULT_HISTORY_TOKENS = 4096


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {name}: {raw}") from exc


def _env_optional_str(name: str) -> str | None:
    raw = os.getenv(name, "").strip()
    return raw or None


# ---------------------------------------------------------------------------
# VoiceConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VoiceConfig:
    llm_base_url: str = "http://localhost:8080"
    assistant_name: str = "Cass"
    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_samples: int = DEFAULT_FRAME_SAMPLES
    wake_threshold: float = DEFAULT_WAKE_THRESHOLD
    wake_guard_seconds: float = 1.5
    prompt_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "cass_prompt.txt")
    wake_model_dir: Path = _BASE_DIR / "models" / "wake"
    wakeword_model_file: str = "computer_v2.onnx"
    wakeword_melspec_model_file: str = "melspectrogram.onnx"
    wakeword_embedding_model_file: str = "embedding_model.onnx"
    stt_model: str = "base.en"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    tts_model_path: Path = _BASE_DIR / "models" / "tts" / "en_US-amy-medium.onnx"
    ack_sound_path: Path = _BASE_DIR / "models" / "tts" / "ack.mp3"
    ack_player_bin: str = "ffplay"
    bot_audio_drain_ms: int = 450
    input_device_name: str | None = None
    output_device_name: str | None = None
    listen_mode: bool = False
    history_tokens: int = DEFAULT_HISTORY_TOKENS

    @classmethod
    def from_env(cls) -> "VoiceConfig":
        prompt_path = Path(os.getenv("CASS_PROMPT_PATH", str(_BASE_DIR / "core" / "cass_prompt.txt")))
        wake_model_dir = Path(os.getenv("CASS_WAKE_MODEL_DIR", str(_BASE_DIR / "models" / "wake")))
        tts_model_path = Path(os.getenv("TTS_PIPER_MODEL", str(_BASE_DIR / "models" / "tts" / "en_US-amy-medium.onnx")))
        ack_raw = os.getenv("CASS_ACK_SOUND_PATH", "").strip()
        ack_sound_path = Path(ack_raw) if ack_raw else _BASE_DIR / "models" / "tts" / "ack.mp3"

        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:8080").strip() or "http://localhost:8080",
            assistant_name=os.getenv("CASS_ASSISTANT_NAME", "Cass").strip() or "Cass",
            sample_rate=_env_int("CASS_SAMPLE_RATE", DEFAULT_SAMPLE_RATE),
            frame_samples=_env_int("CASS_FRAME_SAMPLES", DEFAULT_FRAME_SAMPLES),
            wake_threshold=_env_float("CASS_WAKE_THRESHOLD", DEFAULT_WAKE_THRESHOLD),
            wake_guard_seconds=_env_float("CASS_WAKE_GUARD_SECONDS", 1.5),
            prompt_path=prompt_path,
            wake_model_dir=wake_model_dir,
            wakeword_model_file=os.getenv("WAKEWORD_MODEL_FILE", "computer_v2.onnx").strip() or "computer_v2.onnx",
            wakeword_melspec_model_file=os.getenv("OWW_MELSPEC_MODEL_FILE", "melspectrogram.onnx").strip() or "melspectrogram.onnx",
            wakeword_embedding_model_file=os.getenv("OWW_EMBEDDING_MODEL_FILE", "embedding_model.onnx").strip() or "embedding_model.onnx",
            stt_model=os.getenv("WHISPER_MODEL", "base.en").strip() or "base.en",
            stt_device=os.getenv("WHISPER_DEVICE", "cpu").strip() or "cpu",
            stt_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip() or "int8",
            tts_model_path=tts_model_path,
            ack_sound_path=ack_sound_path,
            ack_player_bin=os.getenv("CASS_ACK_PLAYER_BIN", "ffplay").strip() or "ffplay",
            bot_audio_drain_ms=_env_int("CASS_BOT_AUDIO_DRAIN_MS", 450),
            input_device_name=_env_optional_str("CASS_INPUT_DEVICE_NAME"),
            output_device_name=_env_optional_str("CASS_OUTPUT_DEVICE_NAME"),
            listen_mode=os.getenv("CASS_LISTEN_MODE", "").strip().lower() in ("1", "true", "yes"),
            history_tokens=_env_int("CASS_HISTORY_TOKENS", DEFAULT_HISTORY_TOKENS),
        )


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def pcm16_to_float32(samples: Sequence[int]) -> list[float]:
    return [int(sample) / 32768.0 for sample in samples]


def float32_to_pcm16(samples: Sequence[float]) -> bytes:
    pcm = array("h")
    for sample in samples:
        clipped = max(-1.0, min(1.0, float(sample)))
        if clipped < 0:
            value = int(clipped * 32768.0)
        else:
            value = int(clipped * 32767.0)
        pcm.append(value)
    return pcm.tobytes()


def audio_to_wav_bytes(samples: Sequence[float], sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    pcm = float32_to_pcm16(samples)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buffer.getvalue()
