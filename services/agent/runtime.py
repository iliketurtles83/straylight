"""CassRuntime implementation for Straylight agent service.

This module implements the CassRuntime class which serves as the main runtime
for the agent service, managing the lifecycle and orchestration of the agent.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List

from services.agent.bus import get_event_bus
from services.agent.classifier import Classifier
from services.agent.tools import ToolRegistry
from services.agent.agent_core import VoiceConfig, ConversationWindow
from shared.straylight_shared.events import StateEvent


@dataclass
class RuntimeConfig:
    """Configuration for the CassRuntime."""
    llm_base_url: str
    llm_model: str
    embed_model_path: str
    router_exemplars_path: str
    router_threshold: float
    router_min_gap: float
    history_tokens: int
    llm_ctx_size: int
    llm_output_size: int
    redis_url: str
    mcp_server_urls: List[str]


class CassRuntime:
    """Main runtime for the Straylight agent service.
    
    This class orchestrates the agent lifecycle and handles turn processing
    by delegating to specialized modules like Classifier and ToolRegistry.
    """

    def __init__(self, config: RuntimeConfig):
        self._config = config
        self._event_bus = None
        self._classifier = None
        self._tool_registry = None
        self._conversation_windows: Dict[str, ConversationWindow] = {}
        self._shutdown = False

    async def startup(self) -> None:
        """Initialize the runtime components.
        
        This method:
        1. Connects to the event bus (raises on failure)
        2. Loads classifier embed model and builds exemplar index
        3. Registers tools in ToolRegistry
        4. Subscribes to cass:input
        5. Emits StateEvent(idle)
        """
        # 1. Connect bus
        self._event_bus = await get_event_bus()
        
        # 2. Load classifier embed model and build exemplar index
        self._classifier = Classifier(self._config.embed_model_path)
        await self._classifier.startup()
        
        # 3. Register tools in ToolRegistry
        self._tool_registry = ToolRegistry()
        # In a real implementation, tools would be registered here
        
        # 4. Subscribe to cass:input (this will be handled by the gateway)
        # We don't need to do anything here since the gateway will publish to cass:input
        
        # 5. Emit StateEvent(idle)
        await self._event_bus.publish(StateEvent(
            state="idle",
            session_id="default",
        ))
        
        print("CassRuntime started successfully")

    async def shutdown(self) -> None:
        """Shut down the runtime components."""
        self._shutdown = True
        if self._event_bus:
            await self._event_bus.shutdown()
        print("CassRuntime shut down")

    async def handle_input(self, text: str, session_id: str) -> None:
        """Handle input from the gateway.
        
        This is called when cass:input fires. It owns the full turn:
        classify → fast or slow path → stream to bus → update conversation
        
        Args:
            text: The input text
            session_id: The session identifier
        """
        await self._run_turn(text, session_id)

    async def _run_turn(self, text: str, session_id: str) -> None:
        """Execute a single turn of conversation.
        
        Args:
            text: The input text
            session_id: The session identifier
        """
        # Get or create conversation window for session
        if session_id not in self._conversation_windows:
            self._conversation_windows[session_id] = ConversationWindow(
                system_prompt="",
                turns=[],
            )
        
        conversation = self._conversation_windows[session_id]
        
        # Classify the intent
        classifier_result = await self._classifier.classify(text)
        
        # In a real implementation, we would now route to fast or slow path
        # For now, we'll just simulate the process
        
        # Update conversation window
        conversation.add_turn(text)
        
        # For now, just simulate processing
        print(f"Processing turn for session {session_id}: {text}")

    async def _fast_path(self, text: str, tool_name: str, session_id: str) -> str:
        """Execute the fast path for a tool call.
        
        Args:
            text: The input text
            tool_name: The tool to execute
            session_id: The session identifier
            
        Returns:
            The result of the tool execution
        """
        # In a real implementation, this would call the tool registry
        # For now, we'll simulate a tool execution
        print(f"Executing fast path tool: {tool_name}")
        return f"Result from {tool_name} tool"

    async def _slow_path(self, text: str, session_id: str) -> str:
        """Execute the slow path using LLM.
        
        Args:
            text: The input text
            session_id: The session identifier
            
        Returns:
            The LLM response
        """
        # In a real implementation, this would stream responses from LLM
        # For now, we'll simulate a response
        print(f"Executing slow path for: {text}")
        return "I don't have a specific answer for that right now."