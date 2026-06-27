from abc import ABC, abstractmethod
from typing import Any


class Skill(ABC):
    """Abstract base class for all skills in the system."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the skill."""
        pass

    @property
    @abstractmethod
    def exemplars(self) -> list[str]:
        """Return a list of example utterances that trigger this skill."""
        pass

    @property
    def format_prompt(self) -> str:
        """Return a prompt for formatting the tool result."""
        return ""

    @abstractmethod
    def score(self, transcript: str) -> float:
        """Score how well this skill matches a transcript."""
        pass

    @abstractmethod
    def entities(self, transcript: str) -> dict[str, Any]:
        """Extract entities from the transcript."""
        pass

    async def execute(self, entities: dict[str, Any]) -> str:
        """Execute the skill with given entities."""
        raise NotImplementedError("execute method must be implemented by subclasses")


class SkillExecutionError(Exception):
    """Raised when a skill execution fails."""
    pass