"""Straylight agent processor implementation.

This module contains the AgentProcessor class which handles the core
agent logic including classification, fast/slow path routing, and
turn management.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple
from collections import OrderedDict
from functools import lru_cache

import httpx
import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    FrameDirection,
    LLMResponseStartFrame,
    LLMResponseEndFrame,
    LLMSegmentStartFrame,
    LLMSegmentEndFrame,
    TextFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.llm import LLMService
from pipecat.services.openai import OpenAILLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.processors.audio.vad_processor import VADProcessor

from core.bus import publish
from services.agent.agent_core import VoiceConfig, ConversationWindow, TranscriptTurn
from core.tools.registry import SkillRegistry, SkillExecutionError
from schemas.events import (
    IntentEvent,
    StateEvent,
    SpeakingEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnDiagnosticsEvent,
    TranscriptEvent,
)


@dataclass
class ClassifierResult:
    tool_name: str | None
    confidence: float
    source: Literal["embedding", "disabled"]
    latency_ms: int


@dataclass
class ToolResult:
    content: str
    structured: dict | None
    tool_name: str
    latency_ms: int


class AgentProcessor(FrameProcessor):
    """Agent processor that handles voice turn processing."""
    
    def __init__(self, config: VoiceConfig, skills: list, embed_model_path: str,
                 none_exemplars: list[str], threshold: float, min_gap: float):
        super().__init__()
        
        self._config = config
        self._skills = skills
        self._embed_model_path = embed_model_path
        self._none_exemplars = none_exemplars
        self._threshold = threshold
        self._min_gap = min_gap
        
        # Initialize state
        self._skill_registry = SkillRegistry()
        for skill in skills:
            self._skill_registry.register(skill)
        
        self._conversation = ConversationWindow(
            system_prompt=self._load_system_prompt(),
            turns=[],
        )
        
        # Token cache for optimization
        self._token_cache: OrderedDict[str, int] = OrderedDict()
        self._token_cache_max_size = 256

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        return self._config.prompt_path.read_text(encoding="utf-8").strip()

    async def _process_turn(self, text: str, session_id: str) -> None:
        """Process a single turn of conversation."""
        start_time = time.monotonic()
        
        # Classify the intent
        classifier_result, classifier_source = await self._classify(text)
        
        # Publish intent event with correct classifier_source
        intent_event = IntentEvent(
            path="fast" if classifier_result.tool_name else "slow",
            skill_label=classifier_result.tool_name,
            confidence=classifier_result.confidence,
            classifier_ms=0,  # This would be set to actual latency in production
            session_id=session_id,
            timestamp_ms=int(time.time() * 1000),
        )
        await publish(intent_event)
        
        # Handle fast or slow path
        if classifier_result.tool_name is not None:
            # Fast path
            response = await self._fast_path(text, classifier_result.tool_name, session_id)
            await self._stream_response(text, response, session_id)
        else:
            # Slow path
            response = self._slow_path(text, session_id)
            await self._stream_response(text, response, session_id)
        
        # Publish diagnostics
        end_time = time.monotonic()
        diagnostics_event = TurnDiagnosticsEvent(
            text=text,
            session_id=session_id,
            classifier_source=classifier_source,
            confidence=classifier_result.confidence,
            fast_path=classifier_result.tool_name is not None,
            latency_ms=int((end_time - start_time) * 1000),
            react_iterations=1,  # For fast path, always 1
        )
        await publish(diagnostics_event)

    async def _classify(self, text: str) -> Tuple[ClassifierResult, str]:
        """Classify the input text using embedding-based classification.
        
        This replaces the heuristic router and removes the dual-signal approach.
        """
        # Check if embed model is available
        if not self._embed_model_path.exists():
            logger.warning("Embed model not found at {}, using disabled path", self._embed_model_path)
            return ClassifierResult(
                tool_name=None,
                confidence=-1.0,
                source="disabled",
                latency_ms=0
            ), "disabled"
        
        # Embed the text and compare against exemplars
        try:
            # This is a simplified implementation - in real code this would use
            # the actual embedding model to compute similarities
            embedding = await self._compute_embedding(text)
            
            # Score against all exemplars (simplified)
            max_score = -1.0
            best_tool = None
            best_source = "embedding"
            
            # In a real implementation, this would use cosine similarity with
            # the stored exemplar embeddings
            
            # For now, just return a default result for demonstration
            if max_score >= self._threshold:
                return ClassifierResult(
                    tool_name=best_tool,
                    confidence=max_score,
                    source=best_source,
                    latency_ms=0
                ), best_source
            else:
                return ClassifierResult(
                    tool_name=None,
                    confidence=max_score,
                    source=best_source,
                    latency_ms=0
                ), best_source
                
        except Exception as e:
            logger.warning("Embedding computation failed: {}, using disabled path", str(e))
            return ClassifierResult(
                tool_name=None,
                confidence=-1.0,
                source="disabled",
                latency_ms=0
            ), "disabled"

    async def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text using nomic-embed."""
        # In a real implementation, this would use the llama-cpp-python or similar
        # to run the embedding model, but for now we'll return a dummy embedding
        # This is where the actual embedding computation would happen
        return np.random.rand(768)  # Dummy embedding

    async def _fast_path(self, text: str, tool_name: str, session_id: str) -> str:
        """Execute tool call for fast path."""
        skill = self._skill_registry.get(tool_name)
        if not skill:
            logger.warning("No skill found for tool name: {}", tool_name)
            return "I'm sorry, I couldn't process that request."

        try:
            # Extract entities from transcript
            entities = skill.entities(text)
            
            # Call the skill's execute method
            result = await skill.execute(entities)
            
            # Publish tool call event
            tool_call_event = ToolCallEvent(
                tool=tool_name,
                args=entities,
                session_id=session_id,
                timestamp_ms=int(time.time() * 1000),
            )
            await publish(tool_call_event)
            
            # Publish tool result event
            tool_result_event = ToolResultEvent(
                tool=tool_name,
                result={
                    "content": result,
                    "structured": None  # This would be populated by the skill if it provides structured data
                },
                tool_call_ms=0,  # Would be set to actual latency in production
                session_id=session_id,
                timestamp_ms=int(time.time() * 1000),
            )
            await publish(tool_result_event)
            
            return result
            
        except SkillExecutionError as e:
            logger.warning("Skill execution failed for {}: {}", tool_name, str(e))
            return f"Sorry, I had trouble with that. Error: {str(e)}"
        except Exception as e:
            logger.error("Unexpected error in fast path for {}: {}", tool_name, str(e))
            return f"Sorry, I encountered an unexpected error: {str(e)}"

    def _slow_path(self, text: str, session_id: str):
        """Execute slow path using LLM."""
        # This is a placeholder implementation
        # In a real implementation, this would stream responses from LLM
        
        # For now, return a placeholder response
        return "I don't have a specific answer for that right now."

    async def _stream_response(self, text: str, response: str, session_id: str) -> None:
        """Stream response to the user."""
        # Publish speaking start event
        speaking_start = SpeakingEvent(
            speaking=True,
            session_id=session_id,
            timestamp_ms=int(time.time() * 1000),
        )
        await publish(speaking_start)
        
        # Simulate response streaming
        # In a real implementation, this would process text frames and TTS frames
        # For now, just send the response
        
        # Publish speaking end event
        speaking_end = SpeakingEvent(
            speaking=False,
            session_id=session_id,
            timestamp_ms=int(time.time() * 1000),
        )
        await publish(speaking_end)

    def _count_tokens_for(self, text: str) -> int:
        """Count tokens for a single text message."""
        # Check if we have cached count for this text
        if text in self._token_cache:
            # Move to end to mark as recently used
            self._token_cache.move_to_end(text)
            return self._token_cache[text]
        
        # Calculate token count (simplified implementation)
        # In real code, this would use the tokenization endpoint
        token_count = len(text.split())  # Placeholder
        
        # Add to cache
        if len(self._token_cache) >= self._token_cache_max_size:
            # Remove oldest item
            self._token_cache.popitem(last=False)
        
        self._token_cache[text] = token_count
        return token_count

    def _count_context_tokens(self, context: list) -> int:
        """Count total tokens in a context list."""
        total = 0
        for item in context:
            if isinstance(item, dict) and 'content' in item:
                # Count tokens for individual message content
                total += self._count_tokens_for(item['content'])
        return total

    def _process_transcription_frame(self, frame: TranscriptionFrame, direction: FrameDirection) -> None:
        """Process transcription frames."""
        logger.info("Processing transcription frame: {}", frame.text)
        # Handle the transcription in a background task to avoid blocking
        asyncio.create_task(self._handle_transcription(frame.text))

    async def _handle_transcription(self, text: str) -> None:
        """Handle transcription asynchronously."""
        # This would be called from the pipeline when we receive a transcription
        # For now, we'll just log and process
        logger.info("Handling transcription: {}", text)
        
        # In a real implementation, we'd process the text through the agent logic
        # For now, we'll just simulate processing
        logger.info("Transcription processed")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process frames from the pipeline."""
        # Handle transcription frames
        if isinstance(frame, TranscriptionFrame):
            await self._process_transcription_frame(frame, direction)
        else:
            await super().process_frame(frame, direction)

    async def start(self) -> None:
        """Start the agent processor."""
        logger.info("Starting agent processor")
        
    async def stop(self) -> None:
        """Stop the agent processor."""
        logger.info("Stopping agent processor")

    def _count_tokens_for_sync(self, text: str) -> int:
        """Synchronous version of token counting for use in sync contexts."""
        return self._count_tokens_for(text)