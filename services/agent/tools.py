"""Tool registry implementation for Straylight agent service.

This module implements the ToolRegistry class which manages tools
and their execution.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Awaitable, Dict, List

from shared.straylight_shared.events import ToolCallEvent, ToolResultEvent


@dataclass(frozen=True)
class ToolResult:
    content: str
    structured: dict | None
    tool_name: str
    latency_ms: int


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    execute: Callable[[dict], Awaitable[ToolResult]]


class ToolNotFoundError(Exception):
    """Raised when a tool is not found in the registry."""
    pass


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""
    pass


class ToolRegistry:
    """Registry for managing tools and their execution."""
    
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}
    
    def register(self, spec: ToolSpec) -> None:
        """Register a tool specification.
        
        Args:
            spec: Tool specification to register
            
        Raises:
            ValueError: If tool name is already registered
        """
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec
        print(f"Registered tool: {spec.name}")
    
    async def call(self, tool_name: str, args: dict) -> ToolResult:
        """Execute a tool with the given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            args: Arguments to pass to the tool
            
        Returns:
            ToolResult containing the execution result
            
        Raises:
            ToolNotFoundError: If tool is not found
            ToolExecutionError: If tool execution fails
        """
        tool = self._tools.get(tool_name)
        if not tool:
            raise ToolNotFoundError(f"Tool not found: {tool_name}")
        
        try:
            # Get event bus to publish events
            from services.agent.bus import get_event_bus
            event_bus = await get_event_bus()
            
            # Publish tool call event
            tool_call_event = ToolCallEvent(
                tool=tool_name,
                args=args,
                session_id="default",
                timestamp_ms=int(asyncio.get_event_loop().time() * 1000),
            )
            await event_bus.publish(tool_call_event)
            
            # Execute the tool
            start_time = asyncio.get_event_loop().time()
            result = await tool.execute(args)
            latency_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            # Update result with latency
            result = ToolResult(
                content=result.content,
                structured=result.structured,
                tool_name=result.tool_name,
                latency_ms=latency_ms
            )
            
            # Publish tool result event
            tool_result_event = ToolResultEvent(
                tool=tool_name,
                result={
                    "content": result.content,
                    "structured": result.structured
                },
                tool_call_ms=latency_ms,
                session_id="default",
                timestamp_ms=int(asyncio.get_event_loop().time() * 1000),
            )
            await event_bus.publish(tool_result_event)
            
            return result
            
        except Exception as e:
            # Wrap any tool execution errors
            raise ToolExecutionError(f"Tool {tool_name} execution failed: {str(e)}") from e
    
    def manifest(self) -> list[dict]:
        """Return OpenAI-style tool manifest for the LLM.
        
        Returns:
            List of tool specifications in OpenAI format
        """
        manifest = []
        for tool_spec in self._tools.values():
            manifest.append({
                "type": "function",
                "function": {
                    "name": tool_spec.name,
                    "description": tool_spec.description,
                    "parameters": tool_spec.input_schema
                }
            })
        return manifest