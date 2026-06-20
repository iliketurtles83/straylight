"""Simple weather tool for demonstration purposes."""

import asyncio
from dataclasses import dataclass
from typing import Dict, Any
from services.agent.tools import ToolSpec, ToolResult


async def _weather_tool_execute(args: Dict[str, Any]) -> ToolResult:
    """Execute the weather tool."""
    # Simple mock implementation
    location = args.get("location", "unknown")
    weather = "sunny"  # mock weather data
    temperature = 22  # mock temperature
    
    result = f"The weather in {location} is currently {weather} with a temperature of {temperature}°C."
    
    return ToolResult(
        content=result,
        structured={
            "location": location,
            "weather": weather,
            "temperature": temperature
        },
        tool_name="weather",
        latency_ms=50  # mock latency
    )


# Tool specification for weather tool
WEATHER_TOOL = ToolSpec(
    name="weather",
    description="Get current weather information for a location",
    input_schema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location to get weather for"
            }
        },
        "required": ["location"]
    },
    execute=_weather_tool_execute
)