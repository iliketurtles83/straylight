"""Gateway service for Straylight.

This service provides API access to the central event bus via FastAPI.
It exposes endpoints for:
- POST /input: Submit input to the system
- GET /events: Subscribe to event stream via Server-Sent Events (SSE)
- GET /diagnostics: Get system diagnostics information
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
import redis.asyncio as redis

from shared.straylight_shared.events import Event

# Global Redis connection
_redis_client = None


async def get_redis_client() -> redis.Redis:
    """Get Redis client instance."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url("redis://localhost:6379")
        logger.info("Connected to Redis")
    return _redis_client


app = FastAPI(
    title="Straylight Gateway Service",
    description="API access to the Straylight event bus",
    version="0.1.0"
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Straylight Gateway Service"}


@app.post("/input")
async def submit_input(request: Request):
    """Submit input to the system."""
    try:
        # Parse the request body
        body = await request.json()
        text = body.get("text", "")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Publish to Redis as cass:input event
        redis_client = await get_redis_client()
        event_data = json.dumps({
            "text": text,
            "session_id": body.get("session_id", "default"),
            "timestamp_ms": int(asyncio.get_event_loop().time() * 1000)
        })
        await redis_client.publish("cass:input", event_data)
        
        return {"status": "success", "message": "Input submitted", "text": text}
    except Exception as e:
        logger.error("Error submitting input: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/events")
async def stream_events(request: Request) -> StreamingResponse:
    """Stream events from the Redis event bus."""
    try:
        redis_client = await get_redis_client()
        
        async def event_generator() -> AsyncGenerator[str, None]:
            """Generate events from Redis."""
            pubsub = redis_client.pubsub()
            # Subscribe to all channels (this is a simplified approach)
            # In a real implementation, you might want to subscribe to specific channels
            await pubsub.psubscribe("*")
            
            logger.info("Client connected to event stream")
            
            try:
                while True:
                    # Check if client is still connected
                    if await request.is_disconnected():
                        logger.info("Client disconnected from event stream")
                        break
                    
                    # Wait for messages (with timeout)
                    message = await pubsub.get_message(timeout=1.0)
                    if message and message["type"] == "pmessage":
                        channel = message["channel"].decode()
                        data = message["data"].decode()
                        
                        # Format as SSE - channel is part of the event data, not a separate field
                        yield f"data: {data}\nevent: {channel}\n\n"
                    
                    # Add small delay to prevent busy waiting
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error("Error in event stream: {}", e)
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
            finally:
                await pubsub.close()
                logger.info("Event stream connection closed")
        
        return StreamingResponse(event_generator(), media_type="text/plain")
        
    except Exception as e:
        logger.error("Error in stream_events: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/diagnostics")
async def get_diagnostics():
    """Get system diagnostics."""
    try:
        redis_client = await get_redis_client()
        # Simple health check
        await redis_client.ping()
        
        return {
            "status": "healthy",
            "redis": "connected",
            "version": "0.1.0"
        }
    except Exception as e:
        logger.error("Redis health check failed: {}", e)
        return {
            "status": "unhealthy",
            "redis": "disconnected",
            "error": str(e)
        }


if __name__ == "__main__":
    # This is for development only
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)