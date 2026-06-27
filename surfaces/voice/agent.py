"""Straylight agent processor implementation.

This module contains the AgentProcessor class which handles the core
agent logic including classification, fast/slow path routing, and
turn management.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from core.runtime import CassRuntime


class AgentProcessor(FrameProcessor):
    """Agent processor that handles voice turn processing."""
    
    def __init__(self, runtime: CassRuntime):
        super().__init__()
        
        self._runtime = runtime

    async def _process_transcription_frame(self, frame: TranscriptionFrame, direction: FrameDirection) -> None:
        """Process transcription frames by delegating to CassRuntime."""
        logger.info("Processing transcription frame: {}", frame.text)
        
        # Extract text from the frame
        text = frame.text
        
        # Delegate to CassRuntime's single entrypoint
        response = await self._runtime.handle_input(text, frame.session_id)
        await self.push_frame(TextFrame(text=response), direction)

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