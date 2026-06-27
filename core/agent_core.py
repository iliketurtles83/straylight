from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import re

# Re-export from surfaces.voice.config for backwards compatibility.
from surfaces.voice.config import (
    DEFAULT_FRAME_SAMPLES,
    DEFAULT_HISTORY_TOKENS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WAKE_THRESHOLD,
    VoiceConfig,
)

_BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TranscriptTurn:
    role: str
    content: str


@dataclass
class ConversationWindow:
    system_prompt: str
    turns: list[TranscriptTurn] = field(default_factory=list)

    def build_messages(self, user_text: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend({"role": turn.role, "content": turn.content} for turn in self.turns)
        normalized = normalize_reply_text(user_text)
        if normalized:
            messages.append({"role": "user", "content": normalized})
        return messages

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        user = normalize_reply_text(user_text)
        assistant = normalize_reply_text(assistant_text)
        if user:
            self.turns.append(TranscriptTurn(role="user", content=user))
        if assistant:
            self.turns.append(TranscriptTurn(role="assistant", content=assistant))

    def drop_oldest_turn_pair(self) -> bool:
        """Drop the oldest user/assistant pair. Returns True if anything changed."""
        if not self.turns:
            return False

        if len(self.turns) >= 2 and self.turns[0].role == "user" and self.turns[1].role == "assistant":
            del self.turns[:2]
            return True

        del self.turns[0]
        return True



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


def load_system_prompt(prompt_path: Path, assistant_name: str = "Cass") -> str:
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    prompt = prompt.replace("{assistant_name}", assistant_name)
    return prompt.strip()


def normalize_reply_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

