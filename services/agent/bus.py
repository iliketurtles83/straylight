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
        self._connection_lock = asyncio.Lock()
        self._buffer = []
        self._buffer_max_size = 64

    async def startup(self) -> None:
        """Connect to Redis and verify connection.
        
        Raises:
            RuntimeError: If Redis is unreachable
        """
        async with self._connection_lock:
            if self._redis_client is None:
                try:
                    self._redis_client = aioredis.from_url(self.redis_url)
                    # Test connection
                    await self._redis_client.ping()
                    logger.info("Connected to Redis at {}", self.redis_url)
                except Exception as e:
                    raise RuntimeError(f"Failed to connect to Redis at {self.redis_url}: {e}") from e

    async def shutdown(self) -> None:
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close()
            logger.info("Disconnected from Redis")

    async def publish(self, event: Event) -> None:
        """Publish an event to Redis channel.
        
        If Redis is unavailable, events are buffered in memory (bounded at 64).
        Failed attempts to publish are logged but don't raise exceptions.
        
        Args:
            event: The event to publish
        """
        if not self._redis_client:
            # Try to reconnect
            try:
                await self.startup()
            except RuntimeError:
                logger.warning("Redis unavailable, buffering event: {}", event.__class__.__name__)
                self._buffer_event(event)
                return
        
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
            logger.warning("Failed to publish event {}: {} - buffering", event.__class__.__name__, str(e))
            self._buffer_event(event)
    
    def _buffer_event(self, event: Event) -> None:
        """Buffer event in memory if Redis is unavailable."""
        if len(self._buffer) >= self._buffer_max_size:
            # Drop oldest event if buffer is full
            dropped = self._buffer.pop(0)
            logger.warning("Buffer full, dropped event: {}", dropped.__class__.__name__)
        
        self._buffer.append(event)
    
    async def _drain_buffer(self) -> None:
        """Attempt to flush buffered events after reconnect."""
        if not self._buffer:
            return
            
        logger.info("Attempting to drain {} buffered events", len(self._buffer))
        
        # Try to flush buffer with exponential backoff
        retry_delay = 0.1  # seconds
        for event in self._buffer[:]:  # Copy to avoid modification during iteration
            try:
                await self.publish(event)
                self._buffer.remove(event)
                retry_delay = 0.1  # Reset delay on success
            except Exception as e:
                logger.warning("Still failing to publish buffered event: {}", str(e))
                retry_delay = min(retry_delay * 2, 5.0)  # Exponential backoff, max 5s
                await asyncio.sleep(retry_delay)
        
        if self._buffer:
            logger.warning("Failed to flush {} events after buffer drain attempt", len(self._buffer))


# Global event bus instance
_event_bus = None


async def get_event_bus() -> RedisEventBus:
    """Get the global Redis event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = RedisEventBus()
        await _event_bus.startup()
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