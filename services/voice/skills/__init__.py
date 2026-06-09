"""Straylight skill base class and registry.

Each Skill is a named fast-path bundle that owns:
  - Embedding exemplars for the classifier
  - spaCy entity extraction
  - MCP tool call(s)
  - A format prompt for the small LLM response formatter

AgentProcessor builds a shared embedding index from all registered
skill exemplars at startup. Skill.execute() is called on the fast
path after entity extraction. It must be an async coroutine.

Adding a new skill:
  1. Subclass Skill in services/voice/skills/<name>.py
  2. Collect exemplars from real mic input via scripts/measure_router.py
     (do NOT use typed text — Whisper disfluencies shift the distribution)
  3. Register with AgentProcessor(skills=[..., MySkill()])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Skill(ABC):
    """Abstract base class for all Straylight skills."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique skill identifier used in logs and IntentEvents."""
        ...

    @property
    @abstractmethod
    def exemplars(self) -> list[str]:
        """Representative utterances for the embedding classifier.

        IMPORTANT: Collect these from real mic input through Whisper,
        not typed text. Whisper's disfluencies and ASR errors shift the
        embedding distribution; typed exemplars do not transfer.
        See scripts/measure_router.py.
        """
        ...

    @abstractmethod
    def entities(self, transcript: str) -> dict[str, Any]:
        """Extract structured entities from the Whisper transcript.

        Args:
            transcript: Raw Whisper output for this turn.

        Returns:
            Dict of entity name → value. May be empty if nothing was
            extracted — execute() must handle this gracefully (e.g. ask
            the user to repeat with a location).
        """
        ...

    def can_handle(self, transcript: str) -> bool:
        """Heuristic router fallback when embed classification is unavailable.

        Default is False so skills opt-in explicitly. AgentProcessor uses this
        only when nomic-embed is not loaded or no exemplar index exists.
        """
        return False

    @abstractmethod
    async def execute(self, entities: dict[str, Any]) -> str:
        """Invoke the skill and return a raw result string.

        This string is passed to the primary LLM (Gemma on port 8080) for
        formatting in Cass's voice. Return structured data or a
        plain-language result, not a final user-facing response.

        Args:
            entities: Output of self.entities() for this turn.

        Raises:
            SkillExecutionError: If the MCP call or tool fails.
                AgentProcessor catches this and routes to the slow path
                or speaks a spoken fallback. Never raise silently.
        """
        ...

    def score(self, transcript: str) -> float:
        """Return a confidence score (0-1) that this transcript matches
        this skill. Used by the heuristic router to compete with the
        embedding classifier instead of bypassing it.

        Default: 0.0 — skills with no keyword vocabulary get no heuristic
        boost. Override in subclasses.
        """
        return 0.0

    @property
    def format_prompt(self) -> str:
        """System prompt fragment for the small LLM response formatter.

        Override in subclasses to give skill-specific formatting
        instructions. Default: generic Cass persona.
        """
        return (
            "Format the following tool result as a concise spoken response "
            "in Cass's voice. Keep it under two sentences. Avoid repeating "
            "the question. Speak naturally, not like a list."
        )


class SkillExecutionError(Exception):
    """Raised by Skill.execute() when a tool call fails."""


class SkillRegistry:
    """Holds all registered skills and their exemplars.

    Used by AgentProcessor to build the nomic-embed index at startup.
    All exemplars from all skills are indexed together so the classifier
    can score any utterance against any skill in a single pass.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a skill. Raises ValueError on duplicate name."""
        if skill.name in self._skills:
            raise ValueError(f"Skill already registered: {skill.name!r}")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    @property
    def exemplar_corpus(self) -> list[tuple[str, str]]:
        """All (exemplar_text, skill_name) pairs across registered skills.

        The caller must supply negative exemplars labelled 'none'
        separately to prevent the classifier from routing everything
        to the highest-scoring skill regardless of confidence.
        """
        corpus: list[tuple[str, str]] = []
        for skill in self._skills.values():
            for ex in skill.exemplars:
                corpus.append((ex, skill.name))
        return corpus
