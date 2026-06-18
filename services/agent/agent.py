"""AgentProcessor — Phase 2 intelligence layer for Straylight.

Replaces OpenAILLMService in the Pipecat pipeline.
Receives TranscriptionFrame, routes via fast or slow path, emits TextFrames
for PiperTTSService.

Fast path (nomic-embed confidence >= threshold, known skill):
    nomic-embed classifier → Skill.entities() → Skill.execute() →
    Gemma 4 single-shot formatter → TextFrame

Slow path (low confidence, no skill match, or no embed model):
    ConversationWindow → Gemma 4 streaming → TextFrame per chunk

Pipeline position:
    ... → WhisperSTTService → AgentProcessor → PiperTTSService → ...

AgentProcessor consumes TranscriptionFrame (does not push it forward).
All other frame types are passed through unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import time
import uuid
from pathlib import Path
from typing import Literal
from typing import AsyncIterator

import httpx
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from shared.straylight_shared.events import (
    IntentEvent,
    SpeakingEvent,
    StateEvent,
    TranscriptEvent,
    TurnDiagnosticsEvent,
)

# Import the new Redis event bus publisher
from agent.bus import publish as agent_publish
from .core import ConversationWindow, TranscriptTurn, VoiceConfig, load_system_prompt
from .skills import Skill, SkillExecutionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# AgentProcessor
# ---------------------------------------------------------------------------

class AgentProcessor(FrameProcessor):
    """Custom FrameProcessor: TranscriptionFrame in, TextFrame(s) out.

    Holds the nomic-embed Llama object (in-process, embedding=True) and
    calls it via asyncio.to_thread() to avoid blocking the event loop.
    Holds an asyncio.Lock to prevent overlapping Gemma 4 calls between the
    fast-path formatter and the slow-path ReAct loop.

    Args:
        config:          VoiceConfig from environment.
        skills:          Registered Skill instances. AgentProcessor builds
                         the embedding index from their exemplars at startup.
        embed_model_path: Path to nomic-embed .gguf file. If None or missing,
                         the classifier is disabled and all queries route to
                         the slow path.
        none_exemplars:  Negative exemplars (label="none") for the classifier.
                         Collect from real mic input via measure_router.py.
        threshold:       Minimum cosine similarity score for a fast-path hit.
        min_gap:         Minimum score gap between 1st and 2nd exemplar match.
                         Prevents low-confidence routing when two skills score
                         similarly.
        session_id:      Unique identifier for this session. Included in all
                         published events. Auto-generated if not provided.
    """

    DEFAULT_THRESHOLD: float = 0.80
    DEFAULT_MIN_GAP: float = 0.05
    CLASSIFIER_SOURCE_EMBEDDING: Literal["embedding"] = "embedding"
    CLASSIFIER_SOURCE_HEURISTIC: Literal["heuristic"] = "heuristic"
    CLASSIFIER_SOURCE_DISABLED: Literal["disabled"] = "disabled"

    def __init__(
        self,
        *,
        config: VoiceConfig,
        skills: list[Skill] | None = None,
        embed_model_path: Path | None = None,
        none_exemplars: list[str] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        min_gap: float = DEFAULT_MIN_GAP,
        session_id: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)

        self._config = config
        self._skills: list[Skill] = list(skills or [])
        self._embed_model_path = embed_model_path
        self._none_exemplars: list[str] = list(none_exemplars or [])
        self._threshold = threshold
        self._min_gap = min_gap
        self._session_id = session_id or uuid.uuid4().hex[:12]

        # Conversation context (AgentProcessor owns this; no LLMContextAggregatorPair)
        system_prompt = load_system_prompt(config.prompt_path, config.assistant_name)
        self._conversation = ConversationWindow(system_prompt=system_prompt)
        self._token_cache: dict[str, int] = {}

        # Lazy-initialised embed state
        self._llama: object | None = None           # llama_cpp.Llama | None
        self._exemplar_index: list[tuple[str, str, list[float]]] = []
        self._initialized: bool = False
        self._init_lock: asyncio.Lock = asyncio.Lock()

        # Serialise Gemma 4 calls (fast-path formatter + slow-path loop share one server)
        self._llm_lock: asyncio.Lock = asyncio.Lock()

        # Current processing task — cancelled on barge-in / interrupt
        self._current_task: asyncio.Task | None = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # FrameProcessor protocol
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            # Cancel any in-flight turn before starting the next one.
            await self._cancel_current()
            text = frame.text.strip()
            if text:
                self._current_task = asyncio.create_task(
                    self._process_turn(text),
                    name=f"agent-turn-{self._session_id}",
                )
                self._current_task.add_done_callback(self._on_current_task_done)
            # Do not push TranscriptionFrame further; AgentProcessor is the consumer.
            return

        if isinstance(frame, (UserStartedSpeakingFrame, InterruptionFrame)):
            # Barge-in or explicit interrupt — cancel current processing.
            await self._cancel_current()
            # Pass through so WakeWordProcessor and TTS can react.
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    # Interrupt / cancel
    # ------------------------------------------------------------------

    async def _cancel_current(self) -> None:
        task = self._current_task
        if task is None:
            return

        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug("agent: cancelled task ended with error: {}", exc)

        if self._current_task is task:
            self._current_task = None

    def _on_current_task_done(self, task: asyncio.Task) -> None:
        if task is self._current_task:
            self._current_task = None

        with contextlib.suppress(asyncio.CancelledError):
            exc = task.exception()
            if exc is not None:
                logger.warning("agent: turn task failed: {}", exc)

    # ------------------------------------------------------------------
    # Lazy initialisation — embed model + exemplar index
    # ------------------------------------------------------------------

    async def _ensure_ready(self) -> None:
        """Load the embed model and build the exemplar index on first use."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self._load_embed_model()
            self._initialized = True

    async def _load_embed_model(self) -> None:
        if not self._embed_model_path or not self._embed_model_path.exists():
            logger.info(
                "agent: embed model not found at {}; slow path for all queries",
                self._embed_model_path,
            )
            return

        try:
            self._llama = await asyncio.to_thread(self._load_llama_sync)
            logger.info("agent: embed model loaded from {}", self._embed_model_path)
        except Exception as exc:
            logger.warning("agent: embed model load failed ({}); slow path only", exc)
            self._llama = None
            return

        all_pairs: list[tuple[str, str]] = []
        for skill in self._skills:
            all_pairs.extend((ex, skill.name) for ex in skill.exemplars)
        for ex in self._none_exemplars:
            all_pairs.append((ex, "none"))

        if not all_pairs:
            logger.info("agent: no exemplars registered; slow path for all queries")
            return

        try:
            self._exemplar_index = await asyncio.to_thread(
                self._embed_pairs_sync, all_pairs
            )
            skill_count = sum(1 for _, label, _ in self._exemplar_index if label != "none")
            none_count = len(self._exemplar_index) - skill_count
            logger.info(
                "agent: {} exemplars indexed ({} skill, {} none)",
                len(self._exemplar_index),
                skill_count,
                none_count,
            )
        except Exception as exc:
            logger.warning("agent: exemplar indexing failed ({}); slow path only", exc)
            self._exemplar_index = []

    def _load_llama_sync(self) -> object:
        """Synchronous Llama constructor — runs in a thread to avoid blocking."""
        from llama_cpp import Llama  # type: ignore[import]
        return Llama(
            model_path=str(self._embed_model_path),
            embedding=True,
            n_ctx=512,
            n_threads=4,
            verbose=False,
        )

    def _embed_pairs_sync(
        self, pairs: list[tuple[str, str]]
    ) -> list[tuple[str, str, list[float]]]:
        """Embed all (text, label) pairs synchronously. Runs in a thread."""
        result: list[tuple[str, str, list[float]]] = []
        for text, label in pairs:
            emb: list[float] = self._llama.create_embedding(text)["data"][0]["embedding"]  # type: ignore[union-attr]
            result.append((text, label, emb))
        return result

    def _embed_sync(self, text: str) -> list[float]:
        """Embed a single text synchronously. Runs in a thread."""
        return self._llama.create_embedding(text)["data"][0]["embedding"]  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    async def _classify(
        self, text: str
    ) -> tuple[str | None, float, int, Literal["embedding", "heuristic", "disabled"]]:
        """Classify a transcript against registered skill exemplars.

        Returns:
            (skill_label, score, classifier_ms, classifier_source)
            skill_label is None when routing to the slow path.
        """
        heuristic_skill, heuristic_score = self._heuristic_route(text)

        if self._llama is None or not self._exemplar_index:
            if heuristic_skill is not None:
                return (
                    heuristic_skill,
                    heuristic_score,
                    0,
                    self.CLASSIFIER_SOURCE_HEURISTIC,
                )
            return None, -1.0, 0, self.CLASSIFIER_SOURCE_DISABLED

        t0 = time.perf_counter()
        try:
            emb = await asyncio.to_thread(self._embed_sync, text)
        except Exception as exc:
            logger.warning("agent: embed failed ({}); slow path", exc)
            if heuristic_skill is not None:
                return (
                    heuristic_skill,
                    heuristic_score,
                    0,
                    self.CLASSIFIER_SOURCE_HEURISTIC,
                )
            return None, -1.0, 0, self.CLASSIFIER_SOURCE_DISABLED
        classifier_ms = round((time.perf_counter() - t0) * 1000)

        scored = sorted(
            (
                (label, _cosine_similarity(emb, ex_emb))
                for _, label, ex_emb in self._exemplar_index
            ),
            key=lambda t: t[1],
            reverse=True,
        )

        best_label, best_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0
        gap = best_score - second_score

        logger.debug(
            "classifier: best={!r} score={:.3f} gap={:.3f} ms={} heuristic={:.2f}",
            best_label, best_score, gap, classifier_ms, heuristic_score,
        )

        # --- Merge embedding + heuristic signals -------------------------
        # If the heuristic found a candidate, let it compete with the
        # embedding classifier. We treat heuristic_score as an additive
        # boost to the best embedding match when they agree on a skill.
        if heuristic_skill is not None:
            # Heuristic matched something — give it a chance to override
            # the embedding if it's the top result, or at least check
            # whether it provides a stronger signal than the embed model.
            if best_label == heuristic_skill:
                # Both agree: use embedding (more principled), but ensure
                # it still meets the threshold.
                embed_meets = (
                    best_label != "none"
                    and best_score >= self._threshold
                    and gap >= self._min_gap
                )
                if embed_meets:
                    return (
                        best_label,
                        best_score,
                        classifier_ms,
                        self.CLASSIFIER_SOURCE_EMBEDDING,
                    )
            else:
                # Disagreement: heuristic found a skill the embed model
                # didn't rank highly. If heuristic confidence is strong
                # enough, trust it — it caught something the embedding
                # model missed.
                if heuristic_score >= 0.7:
                    return (
                        heuristic_skill,
                        heuristic_score,
                        classifier_ms,
                        self.CLASSIFIER_SOURCE_HEURISTIC,
                    )

        # Fallback: use embedding results as-is.
        if (
            best_label != "none"
            and best_score >= self._threshold
            and gap >= self._min_gap
        ):
            skill = next((s for s in self._skills if s.name == best_label), None)
            if skill is not None:
                return (
                    best_label,
                    best_score,
                    classifier_ms,
                    self.CLASSIFIER_SOURCE_EMBEDDING,
                )

        return None, best_score, classifier_ms, self.CLASSIFIER_SOURCE_EMBEDDING

    def _heuristic_route(self, text: str) -> tuple[str | None, float]:
        """Score-based heuristic router.

        Primary mechanism: ``score(text)`` — a 0-1 confidence the
        transcript matches this skill. Allows heuristic candidates to
        compete with the embedding classifier.

        Fallback for legacy skills: ``can_handle(text)`` — if a skill
        doesn't implement ``score()`` (returns 0.0), falls back to
        ``can_handle()`` with a neutral score of 0.5 so it still
        participates in routing.
        """
        best_name: str | None = None
        best_score: float = 0.0

        for skill in self._skills:
            try:
                s = skill.score(text)
            except Exception as exc:
                logger.debug("agent: heuristic score failed for {} ({})", skill.name, exc)
                s = 0.0

            if s <= 0.0:
                # Legacy skill — no score() impl. Fall back to can_handle()
                # with a neutral confidence so it still routes.
                try:
                    if skill.can_handle(text):
                        s = 0.5
                    else:
                        continue
                except Exception:
                    continue

            if s > best_score:
                best_score = s
                best_name = skill.name

        if best_name is not None:
            return (best_name, best_score)
        return (None, 0.0)

    # ------------------------------------------------------------------
    # Main turn handler
    # ------------------------------------------------------------------

    async def _process_turn(self, transcript: str) -> None:
        await self._ensure_ready()
        t_turn_start = time.perf_counter()

        # --- Classify ---------------------------------------------------
        skill_label, score, classifier_ms, classifier_source = await self._classify(
            transcript
        )
        path = "fast" if skill_label is not None else "slow"

       # --- Publish pre-LLM events -----------------------------------
        await agent_publish(
            TranscriptEvent(text=transcript, session_id=self._session_id)
        )
        await agent_publish(
            IntentEvent(
                path=path,
                skill_label=skill_label,
                confidence=score,
                classifier_source="nomic-embed",
                classifier_ms=classifier_ms,
                session_id=self._session_id,
            )
        )
        await agent_publish(StateEvent(state="thinking", session_id=self._session_id))

        # --- Generate and stream response ----------------------------
        full_response_parts: list[str] = []
        t_first_chunk: float | None = None
        speaking_started = False
        last_emitted_char: str = ""

        try:
            async with self._llm_lock:
                async for chunk in self._generate_response(transcript, path, skill_label):
                    chunk = self._normalize_stream_chunk(chunk, last_emitted_char)
                    if not chunk:
                        continue
                    if not full_response_parts:
                        t_first_chunk = time.perf_counter()
                        # First chunk: signal start of speaking
                        await agent_publish(
                            SpeakingEvent(
                                state="start", text="", session_id=self._session_id
                            )
                        )
                        await agent_publish(
                            StateEvent(state="speaking", session_id=self._session_id)
                        )
                        speaking_started = True
                    full_response_parts.append(chunk)
                    last_emitted_char = chunk[-1]
                    await self.push_frame(
                        TextFrame(text=chunk), FrameDirection.DOWNSTREAM
                    )

        except asyncio.CancelledError:
            if speaking_started:
                await agent_publish(
                    SpeakingEvent(state="stop", text="", session_id=self._session_id)
                )
            await agent_publish(StateEvent(state="idle", session_id=self._session_id))
            logger.debug("agent: turn cancelled (barge-in)")
            raise

        except Exception as exc:
            if speaking_started:
                await agent_publish(
                    SpeakingEvent(state="stop", text="", session_id=self._session_id)
                )
            await agent_publish(StateEvent(state="idle", session_id=self._session_id))
            logger.error("agent: unhandled error in turn: {}", exc)
            raise

        full_response = "".join(full_response_parts)

        # --- Update conversation history --------------------------------
        if full_response:
            self._conversation.add_turn(transcript, full_response)
            context_tokens = await self._trim_conversation_to_token_budget()
        else:
            context_tokens = await self._count_context_tokens()

        # --- Token counts (context + output) ---------------------------
        output_tokens = await self._count_tokens_for(full_response) if full_response else 0

        # --- Latency math ---------------------------------------------
        t_end = time.perf_counter()
        agent_ms = round((t_end - t_turn_start) * 1000)
        ttfb_ms = (
            round((t_first_chunk - t_turn_start) * 1000) if t_first_chunk else agent_ms
        )
        generation_s = (t_end - t_first_chunk) if t_first_chunk else 0.0
        tokens_per_sec = (
            round(output_tokens / generation_s, 2)
            if output_tokens > 0 and generation_s > 0
            else 0.0
        )

        # --- Publish post-turn events ----------------------------------
        await agent_publish(
            SpeakingEvent(
                state="stop", text=full_response, session_id=self._session_id
            )
        )
        await agent_publish(StateEvent(state="idle", session_id=self._session_id))
        await agent_publish(
            TurnDiagnosticsEvent(
                session_id=self._session_id,
                path=path,  # type: ignore[arg-type]
                skill_label=skill_label,
                model=self._config.llm_model,
                provider="local",
                context_tokens=context_tokens,
                output_tokens=output_tokens,
                tokens_per_sec=tokens_per_sec,
                classifier_confidence=score,
                classifier_source=classifier_source,
                classifier_ms=classifier_ms,
                agent_ms=agent_ms,
                ttfb_ms=ttfb_ms,
            )
        )

        logger.info(
            json.dumps(
                {
                    "event": "agent_turn",
                    "path": path,
                    "skill_label": skill_label,
                    "model": self._config.llm_model,
                    "provider": "local",
                    "classifier_source": classifier_source,
                    "classifier_confidence": score,
                    "classifier_ms": classifier_ms,
                    "agent_ms": agent_ms,
                    "ttfb_ms": ttfb_ms,
                    "context_tokens": context_tokens,
                    "output_tokens": output_tokens,
                    "tokens_per_sec": tokens_per_sec,
                }
            )
        )

    # ------------------------------------------------------------------
    # Response generator — unified entry for fast and slow paths
    # ------------------------------------------------------------------

    async def _generate_response(
        self, transcript: str, path: str, skill_label: str | None
    ) -> AsyncIterator[str]:
        """Yield response chunks. Falls back to slow path on skill failure."""
        if path == "fast" and skill_label is not None:
            skill = next((s for s in self._skills if s.name == skill_label), None)
            if skill is not None:
                try:
                    text = await self._run_fast_path(transcript, skill)
                    if text:
                        yield text
                    return
                except SkillExecutionError as exc:
                    logger.warning(
                        "agent: skill {!r} failed ({}); speaking fallback",
                        skill_label, exc,
                    )
                    if skill_label == "weather":
                        yield (
                            "I hit turbulence reaching the weather grid. "
                            "Try again in a moment."
                        )
                    else:
                        yield "Tooling glitched. Try that again in a moment."
                    return
                except Exception as exc:
                    logger.warning(
                        "agent: fast path error ({}); speaking fallback", exc
                    )
                    yield "Something glitched in the fast lane. Say it again."
                    return

        # Slow path (Gemma 4 streaming)
        messages = self._conversation.build_messages(transcript)
        async for chunk in self._llm_stream(messages):
            yield chunk

    # ------------------------------------------------------------------
    # Fast path
    # ------------------------------------------------------------------

    async def _run_fast_path(self, transcript: str, skill: Skill) -> str:
        """Run entity extraction → skill execute → Gemma 4 format."""
        entities = skill.entities(transcript)
        raw_result = await skill.execute(entities)

        history = self._conversation_history_turns()
        messages = self._build_messages_with_history(transcript, history, skill.format_prompt)
        messages.append({"role": "user", "content": raw_result})
        return await self._llm_single_shot(messages)

    def _conversation_history_turns(self) -> list[TranscriptTurn]:
        """Return the last 2 conversation turns for fast-path context."""
        # Each turn pair is 2 entries (user + assistant), so we grab the last 4.
        return self._conversation.turns[-4:]

    @staticmethod
    def _build_messages_with_history(
        transcript: str,
        history: list[TranscriptTurn],
        format_prompt: str,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": format_prompt},
        ]
        messages.extend({"role": turn.role, "content": turn.content} for turn in history)
        messages.append({"role": "user", "content": transcript})
        return messages

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    async def _llm_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream Gemma 4 completions, yielding text chunks as they arrive."""
        url = f"{self._config.llm_base_url}/v1/chat/completions"
        payload = {
            "model": self._config.llm_model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
        }
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=payload, timeout=60.0) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def _llm_single_shot(self, messages: list[dict]) -> str:
        """Single non-streaming Gemma 4 call for fast-path response formatting."""
        url = f"{self._config.llm_base_url}/v1/chat/completions"
        payload = {
            "model": self._config.llm_model,
            "messages": messages,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 200,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=30.0)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    async def _trim_conversation_to_token_budget(self) -> int:
        """Drop oldest turn pairs until the retained context fits history_tokens."""
        budget = max(1, int(self._config.history_tokens))
        context_tokens = await self._count_context_tokens()
        trimmed_pairs = 0

        while context_tokens > budget and self._conversation.drop_oldest_turn_pair():
            trimmed_pairs += 1
            context_tokens = await self._count_context_tokens()

        if trimmed_pairs:
            logger.debug(
                "context: trimmed {} oldest pair(s) to {} tokens (budget={})",
                trimmed_pairs,
                context_tokens,
                budget,
            )

        return context_tokens

    async def _count_tokens(self) -> int:
        """Backward-compatible wrapper for tests and diagnostics."""
        return await self._count_context_tokens()

    async def _count_context_tokens(self) -> int:
        """Count tokens in retained conversation history."""
        text = self._messages_to_token_text(self._conversation.build_messages(""))
        return await self._count_tokens_for(text)

    async def _count_tokens_for(self, text: str) -> int:
        """Tokenize arbitrary text via llama.cpp /tokenize, with fallback estimate."""
        if not text:
            return 0
        cached = self._token_cache.get(text)
        if cached is not None:
            return cached

        url = f"{self._config.llm_base_url}/tokenize"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url, json={"content": text}, timeout=5.0
                )
                resp.raise_for_status()
                count = len(resp.json().get("tokens", []))
                self._token_cache[text] = count
                return count
        except Exception as exc:
            estimate = self._estimate_token_count(text)
            logger.debug(
                "tokenize: falling back to estimate={} for {} chars ({})",
                estimate,
                len(text),
                exc,
            )
            self._token_cache[text] = estimate
            return estimate

    @staticmethod
    def _normalize_stream_chunk(chunk: str, last_emitted_char: str) -> str:
        """Insert a separator for adjacent text chunks that would otherwise join."""
        if not chunk:
            return ""
        if (
            last_emitted_char
            and not last_emitted_char.isspace()
            and chunk[0].isalnum()
        ):
            return f" {chunk}"
        return chunk

    @staticmethod
    def _messages_to_token_text(messages: list[dict[str, str]]) -> str:
        return "\n".join(
            f"{message.get('role', '')}: {message.get('content', '')}"
            for message in messages
            if message.get("content")
        )

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        if not text:
            return 0
        char_estimate = math.ceil(len(text) / 3)
        word_estimate = math.ceil(len(text.split()) * 1.5)
        return max(1, char_estimate, word_estimate)
