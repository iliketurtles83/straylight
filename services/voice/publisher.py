"""Event publisher for Straylight voice service.

This module provides the interface that AgentProcessor uses to publish events.
In Phase 2, this forwards events to the Redis-based event bus in the agent service.
"""

from __future__ import annotations

from typing import Any

# Import the Redis-based publisher from the agent service
from agent.bus import publish as agent_publish


async def publish(event: Any) -> None:
    """Publish a Straylight event via the agent service's event bus.

    This forwards events from the voice pipeline to the central event bus.
    """
    await agent_publish(event)
