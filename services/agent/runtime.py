"""CassRuntime implementation for Straylight agent service.

This module implements the CassRuntime class which serves as the main runtime
for the agent service, managing the lifecycle and orchestration of the agent.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List

from services.agent.classifier import Classifier
from services.agent.tools import ToolRegistry, ToolSpec, ToolResult
from services.agent.agent_core import VoiceConfig, ConversationWindow
from services.agent.skills.weather import WeatherSkill
from services.agent.skills import Skill
from services.agent.observer import TurnObserver
from shared.straylight_shared.events import StateEvent, IntentEvent


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
    prompt_path: str


class CassRuntime:
    """Main runtime for the Straylight agent service.
    
    This class orchestrates the agent lifecycle and handles turn processing
    by delegating to specialized modules like Classifier and ToolRegistry.
    """

    def __init__(self, config: RuntimeConfig, observer: TurnObserver | None = None):
        self._config = config
        self._observer = observer
        self._classifier = None
        self._tool_registry = None
        self._conversation_windows: Dict[str, ConversationWindow] = {}
        self._shutdown = False

    async def startup(self) -> None:
        """Initialize the runtime components.
        
        This method:
        1. Loads classifier embed model and builds exemplar index
        2. Registers tools in ToolRegistry
        3. Emits StateEvent(idle)
        """
        # 1. Load classifier embed model and build exemplar index
        self._classifier = Classifier(self._config.embed_model_path)
        await self._classifier.startup()
        
        # Load and register exemplars from exemplars.jsonl
        await self._load_exemplars()
        
        # 2. Register tools in ToolRegistry
        self._tool_registry = ToolRegistry(observer=self._observer)
        
        # Register the weather tool (real implementation)
        weather_skill = WeatherSkill()
        weather_tool_spec = self._create_weather_tool_spec(weather_skill)
        self._tool_registry.register(weather_tool_spec)
        
        # 3. Emit StateEvent(idle)
        if self._observer:
            self._observer.notify(StateEvent(
                state="idle",
                session_id="default",
            ))
        
        print("CassRuntime started successfully")

    def _create_weather_tool_spec(self, skill: WeatherSkill) -> ToolSpec:
        """Create a ToolSpec from the WeatherSkill."""
        async def execute_weather_tool(args: dict) -> ToolResult:
            # Extract location using skill's entity extraction
            entities = skill.entities(args.get("text", ""))
            # Execute the skill with the extracted entities
            result_string = await skill.execute(entities)
            return ToolResult(
                content=result_string,
                structured={"entities": entities, "result": result_string},
                tool_name="weather",
                latency_ms=0  # Would measure actual latency in production
            )
        
        return ToolSpec(
            name="weather",
            description="Get current weather information for a location",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The input text containing location information"
                    }
                },
                "required": ["text"]
            },
            execute=execute_weather_tool
        )

    async def _load_exemplars(self) -> None:
        """Load exemplars from exemplars.jsonl and register them with the classifier."""
        from pathlib import Path
        import json
        
        exemplars_path = Path(self._config.router_exemplars_path)
        if not exemplars_path.exists():
            print(f"Warning: exemplars file not found at {exemplars_path}")
            return
            
        # Group exemplars by label
        exemplar_groups: dict[str, list[str]] = {}
        try:
            with open(exemplars_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        text = data.get('text', '')
                        label = data.get('label', '')
                        if text and label:
                            if label not in exemplar_groups:
                                exemplar_groups[label] = []
                            exemplar_groups[label].append(text)
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON line in exemplars file: {line}")
                        
        except Exception as e:
            print(f"Error loading exemplars: {e}")
            return
            
        # Register exemplars with classifier
        for label, exemplars in exemplar_groups.items():
            self._classifier.register_exemplars(label, exemplars)
            print(f"Registered {len(exemplars)} exemplars for label '{label}'")

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
            from services.agent.agent_core import load_system_prompt
            system_prompt = load_system_prompt(Path(self._config.prompt_path))
            self._conversation_windows[session_id] = ConversationWindow(
                system_prompt=system_prompt,
                turns=[],
            )
        
        conversation = self._conversation_windows[session_id]
        
        # Classify the intent
        classifier_result = await self._classifier.classify(text)
        
        # Emit IntentEvent for the classification result
        if self._observer:
            await self._observer.notify(IntentEvent(
                path="fast" if classifier_result.tool_name else "slow",
                skill_label=classifier_result.tool_name,
                confidence=classifier_result.confidence,
                classifier_ms=classifier_result.classifier_ms,
                session_id=session_id,
            ))
        
        # Route to fast or slow path based on classification result
        if classifier_result.tool_name is not None:
            # Fast path: execute the tool
            try:
                result = await self._fast_path(text, classifier_result.tool_name, session_id)
                print(f"Fast path result for '{text}': {result}")
            except Exception as e:
                print(f"Fast path execution failed: {e}")
                # Fall back to slow path
                result = await self._slow_path(text, session_id)
        else:
            # Slow path: use LLM
            result = await self._slow_path(text, session_id)
        
        # Update conversation window
        conversation.add_turn(text, result)
        
        print(f"Processing turn for session {session_id}: {text}")

async def _fast_path(self, text: str, tool_name: str, session_id: str) -> str:
        """Execute the fast path using a tool.
        
        Args:
            text: The input text
            tool_name: The name of the tool to execute
            session_id: The session identifier
            
        Returns:
            The result of the tool execution
        """
        try:
            # Get the skill for this tool
            skill = self._get_skill_for_tool(tool_name)
            if skill is None:
                raise ValueError(f"No skill found for tool: {tool_name}")
            
            # Extract entities from text using skill's entity extraction
            entities = skill.entities(text)
            
            # Call the tool registry with the already extracted entities
            result = await self._tool_registry.call(tool_name, {"text": text, "entities": entities}, session_id)
            return result.content
        except Exception as e:
            print(f"Fast path execution failed: {e}")
            raise

    def _get_skill_for_tool(self, tool_name: str) -> Skill | None:
        """Get the skill associated with a tool name."""
        # This is a simplified approach - in the future we might have a more 
        # sophisticated mapping between tools and skills
        if tool_name == "weather":
            return WeatherSkill()
        return None

    async def _slow_path(self, text: str, session_id: str) -> str:
        """Execute the slow path using LLM.
        
        Args:
            text: The input text
            session_id: The session identifier
            
        Returns:
            The LLM response
        """
        # Make HTTP call to LLM
        import httpx
        import json
        
        print(f"Executing slow path for: {text}")
        
        # Construct the request to the LLM endpoint
        headers = {
            "Content-Type": "application/json"
        }
        
        # Get the conversation window for session
        conversation = self._conversation_windows[session_id]
        
        # Create the message payload
        payload = {
            "model": self._config.llm_model,
            "messages": [
                {"role": "system", "content": conversation.system_prompt},
                {"role": "user", "content": text}
            ],
            "stream": False
        }
        
        try:
            # Make the HTTP request to the LLM endpoint
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._config.llm_base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "I encountered an error processing your request."