"""MCP client implementation for Straylight agent service.

This module implements the MCPClient class which provides an adapter for
external MCP servers.
"""

from __future__ import annotations

import asyncio
import json
from typing import List, Dict, Any

import httpx
from loguru import logger

from services.agent.tools import ToolSpec


class MCPClient:
    """Client for connecting to MCP servers."""

    def __init__(self, server_url: str, server_name: str):
        self.server_url = server_url
        self.server_name = server_name
        self._client = None

    async def startup(self) -> List[ToolSpec]:
        """Connect to MCP server and fetch tool manifest.
        
        Returns:
            List of ToolSpec objects ready for registration
        """
        try:
            self._client = httpx.AsyncClient()
            
            # Fetch tool manifest from MCP server
            manifest_url = f"{self.server_url}/tools"
            response = await self._client.get(manifest_url)
            response.raise_for_status()
            
            manifest_data = response.json()
            
            # Convert manifest data to ToolSpec objects
            tool_specs = []
            for tool_data in manifest_data.get("tools", []):
                tool_spec = ToolSpec(
                    name=tool_data["name"],
                    description=tool_data["description"],
                    input_schema=tool_data["input_schema"],
                    execute=self._create_tool_caller(tool_data["name"])
                )
                tool_specs.append(tool_spec)
            
            logger.info(f"Successfully connected to MCP server {self.server_name} with {len(tool_specs)} tools")
            return tool_specs
            
        except Exception as e:
            logger.warning(f"Failed to connect to MCP server {self.server_name}: {e}")
            return []

    def _create_tool_caller(self, tool_name: str):
        """Create a tool caller function for the given tool name."""
        async def tool_caller(args: dict) -> dict:
            # This would make the actual HTTP call to the MCP server
            # For now, return a dummy result
            return {"content": f"Result from {tool_name} tool", "structured": {}}
        return tool_caller

    async def call(self, tool_name: str, args: dict) -> dict:
        """Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            args: Arguments to pass to the tool
            
        Returns:
            Tool result
        """
        if not self._client:
            raise RuntimeError("MCPClient not initialized")
        
        try:
            call_url = f"{self.server_url}/tools/{tool_name}"
            response = await self._client.post(call_url, json=args)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Tool call failed for {tool_name}: {e}")
            raise