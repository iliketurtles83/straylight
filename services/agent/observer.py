"""Turn observer protocol and implementation for Straylight agent service.

This module defines the TurnObserver protocol and an InMemoryObserver
implementation for local testing and debugging.
"""

from __future__ import annotations

from typing import Protocol
from shared.straylight_shared.events import Event


class TurnObserver(Protocol):
    """Protocol for observing turn events in the agent runtime."""

    def notify(self, event: Event) -> None:
        """Notify the observer of a turn event.
        
        This method is called synchronously with no I/O operations.
        
        Args:
            event: The event to observe
        """
        ...


class InMemoryObserver:
    """In-memory implementation of TurnObserver for testing and local debugging."""
    
    def __init__(self) -> None:
        self._events: list[Event] = []
    
    def notify(self, event: Event) -> None:
        """Record the event in memory.
        
        Args:
            event: The event to record
        """
        self._events.append(event)
    
    @property
    def events(self) -> list[Event]:
        """Return the recorded events."""
        return self._events.copy()