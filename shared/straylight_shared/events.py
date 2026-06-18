"""Canonical event schemas for the Straylight event bus.

All events published to Redis by AgentProcessor, and consumed by the
gateway SSE endpoint, memory service, and future chat adapters, must
conform to these dataclasses.

Redis channels:
    cass:transcript  — TranscriptEvent
    cass:intent      — IntentEvent
    cass:tool_call   — ToolCallEvent
    cass:tool_result — ToolResultEvent
    cass:speaking    — SpeakingEvent
    cass:state       — StateEvent
    cass:input       — text injection from gateway POST /input (not a dataclass)

Serialisation: use dataclasses.asdict() + json.dumps() for publishing;
json.loads() + dataclass(**kwargs) for consuming.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


# Base Event class for all events
class Event:
    """Base Event class for all events in the system."""
    pass


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class TranscriptEvent(Event):
    """Emitted when Whisper produces a transcript for a turn."""

    text: str
    session_id: str
    timestamp_ms: int = field(default_factory=_now_ms)
    channel: str = "cass:transcript"


@dataclass
class IntentEvent(Event):
    """Emitted when AgentProcessor classifies a transcript and picks a path."""

    path: Literal["fast", "slow"]
    skill_label: str | None          # None on slow path
    confidence: float                # Cosine score; -1 when classifier not used
    classifier_source: Literal["embedding", "heuristic", "disabled"]
    classifier_ms: int               # Time taken by nomic-embed classifier
    session_id: str
    timestamp_ms: int = field(default_factory=_now_ms)
    channel: str = "cass:intent"


@dataclass
class ToolCallEvent(Event):
    """Emitted when a skill or slow-path ReAct loop invokes an MCP tool."""

    tool: str
    args: dict[str, Any]
    session_id: str
    timestamp_ms: int = field(default_factory=_now_ms)
    channel: str = "cass:tool_call"


@dataclass
class ToolResultEvent(Event):
    """Emitted when an MCP tool returns a result."""

    tool: str
    result: dict[str, Any]
    tool_call_ms: int                # Round-trip time for the MCP call
    session_id: str
    timestamp_ms: int = field(default_factory=_now_ms)
    channel: str = "cass:tool_result"


@dataclass
class SpeakingEvent(Event):
    """Emitted when TTS output starts or stops."""

    state: Literal["start", "stop"]
    text: str                        # Full response text (available on stop)
    session_id: str
    timestamp_ms: int = field(default_factory=_now_ms)
    channel: str = "cass:speaking"


@dataclass
class StateEvent(Event):
    """Emitted on every pipeline state transition.

    Gateway UI uses this to drive the state badge.
    Sleep consolidation worker uses this to decide when to run.
    """

    state: Literal["idle", "listening", "thinking", "tool_calling", "speaking"]
    session_id: str
    timestamp_ms: int = field(default_factory=_now_ms)
    channel: str = "cass:state"


@dataclass
class TurnDiagnosticsEvent(Event):
    """End-of-turn UI metadata. Drives the diagnostics panel in Phase 4.

    Populated by AgentProcessor after the response stream finishes (or after
    cancellation). All latency fields are integer milliseconds; token counts
    are -1 when the underlying source was unavailable.
    """

    session_id: str
    path: Literal["fast", "slow"]
    skill_label: str | None
    model: str
    provider: Literal["local", "cloud"]
    context_tokens: int
    output_tokens: int
    tokens_per_sec: float
    classifier_confidence: float
    classifier_source: Literal["embedding", "heuristic", "disabled"]
    classifier_ms: int
    agent_ms: int
    ttfb_ms: int
    timestamp_ms: int = field(default_factory=_now_ms)
    channel: str = "cass:diagnostics"
