"""Event bus implementation for Straylight agent service.

This module implements a Redis-based event bus that publishes events
to Redis channels. It provides a clean interface for publishing events
from the agent service to the central event bus.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any

import redis.asyncio as aioredis
from loguru import logger

from shared.straylight_shared.events import Event


class RedisEventBus:
    """Redis-based event bus implementation."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis_client = None

    async def connect(self):
        """Initialize Redis connection."""
        self._redis_client = aioredis.from_url(self.redis_url)
        logger.info("Connected to Redis at {}", self.redis_url)

    async def disconnect(self):
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close()
            logger.info("Disconnected from Redis")

    async def publish(self, event: Event) -> None:
        """Publish an event to Redis channel.

        Args:
            event: The event to publish
        """
        if not self._redis_client:
            raise RuntimeError("Redis client not connected")

        try:
            # Get channel name from event
            channel = event.channel
            if not channel:
                # Fallback to event class name if no channel specified
                channel = event.__class__.__name__.lower()

            # Serialize event data
            event_data = json.dumps(asdict(event), default=str)
            
            # Publish to Redis channel
            await self._redis_client.publish(channel, event_data)
            logger.debug("Published event {} to channel {}", event.__class__.__name__, channel)
            
        except Exception as e:
            logger.error("Failed to publish event {}: {}", event.__class__.__name__, str(e))
            raise


# Global event bus instance
_event_bus = None


async def get_event_bus() -> RedisEventBus:
    """Get the global Redis event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = RedisEventBus()
        await _event_bus.connect()
    return _event_bus


async def publish(event: Event) -> None:
    """Publish an event via the event bus.

    This function is the main interface that AgentProcessor uses to publish events.
    It gets the global event bus instance and publishes the event.

    Args:
        event: The event to publish
    """
    bus = await get_event_bus()
    await bus.publish(event)