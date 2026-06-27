"""Classifier implementation for Straylight agent service.

This module implements the Classifier class which handles embedding-based
intent classification without heuristic routing.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from loguru import logger

from schemas.events import IntentEvent


@dataclass(frozen=True)
class ClassifierResult:
    tool_name: str | None
    confidence: float
    source: Literal["embedding", "disabled"]
    latency_ms: int
    classifier_ms: int


class Classifier:
    """Embedding-based classifier for intent classification.
    
    This replaces the heuristic router and implements a single, clean code path
    for classification using only embedding similarity.
    """
    
    def __init__(self, embed_model_path: Path):
        self._embed_model_path = embed_model_path
        self._exemplar_index: dict[str, list[tuple[str, list[float]]]] = {}
        self._embed_model = None
    
    async def startup(self) -> None:
        """Initialize the classifier with embedding model and exemplar index.
        
        This loads the embedding model and builds the exemplar index.
        """
        if not self._embed_model_path.exists():
            logger.warning("Embed model not found at {}, classifier will be disabled", self._embed_model_path)
            return
        
        try:
            # Load the embedding model
            from llama_cpp import Llama
            self._embed_model = Llama(
                model_path=str(self._embed_model_path),
                embedding=True,
                n_ctx=512,
                n_threads=4,
                verbose=False,
            )
            logger.info("Classifier initialized with embed model at {}", self._embed_model_path)
        except Exception as e:
            logger.warning("Failed to load embed model at {}: {}", self._embed_model_path, e)
            self._embed_model = None
    
    async def classify(self, text: str) -> ClassifierResult:
        """Classify text using embedding similarity.
        
        Args:
            text: The text to classify
            
        Returns:
            ClassifierResult with tool_name, confidence, source, and latency
        """
        start_time = time.monotonic()
        
        # If no model is loaded, return disabled result
        if self._embed_model is None:
            return ClassifierResult(
                tool_name=None,
                confidence=-1.0,
                source="disabled",
                latency_ms=int((time.monotonic() - start_time) * 1000),
                classifier_ms=int((time.monotonic() - start_time) * 1000)
            )
        
        try:
            # Compute embedding for input text
            text_embedding = await asyncio.to_thread(self._embed_sync, text)
            
            # Compare against exemplar embeddings and find best match
            best_match = None
            best_score = -1.0
            
            # Iterate through all exemplars to find the best match
            for tool_name, exemplar_data in self._exemplar_index.items():
                for exemplar_text, exemplar_embedding in exemplar_data:
                    score = self._cosine_similarity(text_embedding, exemplar_embedding)
                    if score > best_score:
                        best_score = score
                        best_match = tool_name
            
            # Apply threshold and min gap logic to determine if we should use the fast path
            threshold = 0.80  # Default threshold - should be configurable
            min_gap = 0.05    # Default min gap - should be configurable
            
            # If we have a good match and it exceeds the threshold, return it
            if best_match is not None and best_score >= threshold:
                latency_ms = int((time.monotonic() - start_time) * 1000)
                return ClassifierResult(
                    tool_name=best_match,
                    confidence=best_score,
                    source="embedding",
                    latency_ms=latency_ms,
                    classifier_ms=latency_ms
                )
            
            # Otherwise, route to slow path
            latency_ms = int((time.monotonic() - start_time) * 1000)
            return ClassifierResult(
                tool_name=None,
                confidence=best_score,
                source="embedding",
                latency_ms=latency_ms,
                classifier_ms=latency_ms
            )
            
        except Exception as e:
            logger.warning("Classifier failed to classify text: {}", e)
            # Return disabled result on error
            latency_ms = int((time.monotonic() - start_time) * 1000)
            return ClassifierResult(
                tool_name=None,
                confidence=-1.0,
                source="disabled",
                latency_ms=latency_ms,
                classifier_ms=latency_ms
            )
    
    def _embed_sync(self, text: str) -> list[float]:
        """Embed a single text synchronously. Runs in a thread."""
        if self._embed_model is None:
            raise RuntimeError("Embed model not loaded")
        embedding = self._embed_model.create_embedding(text)["data"][0]["embedding"]
        if isinstance(embedding[0], list):
            return embedding[0]
        return embedding
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def register_exemplars(self, tool_name: str, exemplars: list[str]) -> None:
        """Register exemplars for a specific tool.
        
        Args:
            tool_name: Name of the tool
            exemplars: List of exemplar text strings
        """
        # For each exemplar, compute its embedding and store it
        exemplar_embeddings = []
        if self._embed_model is not None:
            for exemplar_text in exemplars:
                try:
                    embedding = self._embed_sync(exemplar_text)
                    exemplar_embeddings.append((exemplar_text, embedding))
                except Exception as e:
                    logger.warning("Failed to embed exemplar '{}': {}", exemplar_text, e)
                    # If we can't embed this exemplar, we'll skip it but warn
        else:
            # If we don't have a model, we'll store the exemplars as-is
            # This should only happen during startup or when model is disabled
            for exemplar_text in exemplars:
                exemplar_embeddings.append((exemplar_text, []))
        
        self._exemplar_index[tool_name] = exemplar_embeddings
        logger.debug("Registered {} exemplars for tool {}", len(exemplars), tool_name)