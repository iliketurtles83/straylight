"""Event publisher — Phase 2 stub.

Provides the same interface that Phase 4 will wire to Redis. All
AgentProcessor publish() calls go through this module so Phase 4 only
needs to replace the implementation here, not touch agent.py.

Phase 4 implementation: aioredis client, publishes JSON-serialised
dataclass to the channel named in event.channel.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from loguru import logger


async def publish(event: Any) -> None:
    """Publish a Straylight event.

    Stub: logs at DEBUG level. Phase 4 replaces this with Redis pub/sub.
    """
    try:
        logger.debug(
            "event: {} {}",
            event.__class__.__name__,
            json.dumps(asdict(event), default=str),
        )
    except Exception:
        pass
