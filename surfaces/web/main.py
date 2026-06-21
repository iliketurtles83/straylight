"""Browser UI service for Straylight.

Minimal browser interface with textarea + SSE event stream.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
from loguru import logger

app = FastAPI(
    title="Straylight Browser UI",
    description="Minimal browser interface for Straylight",
    version="0.1.0"
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve the main HTML page."""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Straylight Browser UI</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #events { 
            border: 1px solid #ccc; 
            padding: 10px; 
            height: 300px; 
            overflow-y: auto;
            background-color: #f5f5f5;
        }
        #input-area { margin-top: 20px; }
        #input-text { width: 300px; padding: 5px; }
        #send-btn { padding: 5px 10px; margin-left: 10px; }
        .event { margin-bottom: 10px; padding: 5px; border-bottom: 1px solid #eee; }
        .event-type { font-weight: bold; color: #007bff; }
        .event-data { font-family: monospace; }
    </style>
</head>
<body>
    <h1>Straylight Browser UI</h1>
    
    <div id="input-area">
        <input type="text" id="input-text" placeholder="Enter text to send">
        <button id="send-btn">Send</button>
    </div>
    
    <div id="events">
        <p>Events will appear here...</p>
    </div>

    <script>
        // Connect to SSE endpoint
        const eventSource = new EventSource('/events');
        const eventsDiv = document.getElementById('events');
        
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            const eventDiv = document.createElement('div');
            eventDiv.className = 'event';
            eventDiv.innerHTML = `
                <div class="event-type">${data.type || 'Unknown'}</div>
                <div class="event-data">${JSON.stringify(data, null, 2)}</div>
            `;
            eventsDiv.appendChild(eventDiv);
            eventsDiv.scrollTop = eventsDiv.scrollHeight;
        };
        
        eventSource.onerror = function(err) {
            console.error("EventSource error:", err);
        };
        
        // Handle send button
        document.getElementById('send-btn').addEventListener('click', function() {
            const input = document.getElementById('input-text');
            const text = input.value;
            if (text.trim()) {
                fetch('/input', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({text: text})
                });
                input.value = '';
            }
        });
        
        // Handle Enter key in input
        document.getElementById('input-text').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                document.getElementById('send-btn').click();
            }
        });
    </script>
</body>
</html>
"""
    return html_content


@app.get("/events", response_class=StreamingResponse)
async def stream_events(request: Request) -> StreamingResponse:
    """Stream events from the gateway service."""
    try:
        # In a real implementation, this would connect to the Redis event bus
        # For now, we'll return a simple test stream
        
        async def event_generator() -> AsyncGenerator[str, None]:
            """Generate test events for demo purposes."""
            counter = 0
            try:
                while True:
                    # Check if client is still connected
                    if await request.is_disconnected():
                        logger.info("Client disconnected from event stream")
                        break
                    
                    # Generate a simple event (this would connect to actual Redis in real implementation)
                    event_data = {
                        "type": "test_event",
                        "counter": counter,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    
                    yield f"data: {str(event_data)}\n\n"
                    counter += 1
                    
                    # Wait before next event
                    await asyncio.sleep(2.0)
                    
            except Exception as e:
                logger.error("Error in event stream: {}", e)
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        
        return StreamingResponse(event_generator(), media_type="text/plain")
        
    except Exception as e:
        logger.error("Error in stream_events: {}", e)
        raise httpx.HTTPError(str(e))


@app.post("/input")
async def submit_input(text: dict):
    """Submit input to the system."""
    try:
        # In a real implementation, this would forward to the gateway /input endpoint
        # For now, we'll just return a success response
        logger.info("Input received: {}", text)
        return {"status": "success", "message": "Input submitted"}
    except Exception as e:
        logger.error("Error submitting input: {}", e)
        raise httpx.HTTPError(str(e))


if __name__ == "__main__":
    # This is for development only
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)