#!/usr/bin/env python3
"""Measurement harness for fast-path classifier calibration (Phase 2c).

Records mic utterances through the same VAD + Whisper configuration used
in production, optionally scores each transcript against exemplars via the
nomic-embed API, and appends structured JSONL records for corpus analysis
and threshold calibration.

The VAD parameters here must stay in sync with services/voice/main.py.
Mismatched VAD settings produce different Whisper segmentation and corrupt
the classifier's training distribution.

Usage
-----
  # Collect raw transcripts (no embed server required):
  python scripts/measure_router.py --output corpus.jsonl

  # Score against exemplars (llama-server with nomic-embed on port 8081):
  python scripts/measure_router.py --exemplars exemplars.jsonl --output corpus.jsonl

  # Stop automatically after N utterances:
  python scripts/measure_router.py --count 20 --output corpus.jsonl

Exemplar file format — one JSON object per line:
  {"text": "what's the weather like in London", "label": "weather"}
  {"text": "will it rain tomorrow", "label": "weather"}
  {"text": "tell me something interesting", "label": "none"}

Output JSONL fields:
  transcript       Whisper output, verbatim
  skill_label      Best-matching exemplar label (null in transcript-only mode)
  score            Cosine similarity to best exemplar (null in transcript-only mode)
  gap              Score difference between 1st and 2nd best (null in transcript-only mode)
  below_threshold  True if score < threshold (null in transcript-only mode)
  timestamp_ms     Unix epoch milliseconds at time of transcription

IMPORTANT — exemplar collection:
  Collect exemplars from real mic input through this harness, NOT from typed
  text. Whisper disfluencies and ASR errors shift the embedding distribution;
  typed exemplars do not transfer to live speech. Run in transcript-only mode
  first to build a raw corpus, then label it and load it back as --exemplars.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import signal
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from services.voice.core import VoiceConfig
from services.voice.main import _select_audio_devices

# ---------------------------------------------------------------------------
# VAD parameters — keep in sync with services/voice/main.py build_pipeline().
# Changing these produces different Whisper segmentation and corrupts the
# classifier's training distribution.
# ---------------------------------------------------------------------------
_VAD_PARAMS = VADParams(
    confidence=0.6,
    start_secs=0.15,
    stop_secs=0.25,
    min_volume=0.2,
)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _embed_text(
    text: str,
    embed_url: str,
    client: httpx.AsyncClient,
    model: str,
) -> list[float] | None:
    try:
        resp = await client.post(
            embed_url,
            json={"model": model, "input": text},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    except Exception as exc:
        logger.warning("embed: request failed ({}) for {!r}", exc, text[:60])
        return None


async def _precompute_exemplar_embeddings(
    pairs: list[tuple[str, str]],
    embed_url: str,
    model: str,
) -> list[tuple[str, str, list[float]]]:
    """Pre-compute (text, label, embedding) triples for all exemplars at startup."""
    results: list[tuple[str, str, list[float]]] = []
    async with httpx.AsyncClient() as client:
        for text, label in pairs:
            emb = await _embed_text(text, embed_url, client, model)
            if emb is not None:
                results.append((text, label, emb))
                logger.debug("embed: exemplar ({}) {!r}", label, text[:60])
            else:
                logger.warning("embed: skipping exemplar ({}) {!r} — no embedding returned", label, text[:60])
    return results


# ---------------------------------------------------------------------------
# Scoring FrameProcessor
# ---------------------------------------------------------------------------

class ScoringProcessor(FrameProcessor):
    """Catches TranscriptionFrame, scores via embedding, writes JSONL output."""

    def __init__(
        self,
        *,
        output_path: Path,
        exemplar_embeddings: list[tuple[str, str, list[float]]],
        embed_url: str | None,
        embed_model: str,
        threshold: float,
        count_limit: int | None,
        stop_event: asyncio.Event,
        http_client: httpx.AsyncClient | None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._output_path = output_path
        self._exemplar_embeddings = exemplar_embeddings
        self._embed_url = embed_url
        self._embed_model = embed_model
        self._threshold = threshold
        self._count_limit = count_limit
        self._stop_event = stop_event
        self._http = http_client
        self._count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            text = frame.text.strip()
            if text:
                await self._handle(text)
        await self.push_frame(frame, direction)

    async def _handle(self, text: str) -> None:
        record: dict = {
            "transcript": text,
            "timestamp_ms": int(time.time() * 1000),
            "skill_label": None,
            "score": None,
            "gap": None,
            "below_threshold": None,
        }

        if self._embed_url and self._http and self._exemplar_embeddings:
            emb = await _embed_text(text, self._embed_url, self._http, self._embed_model)
            if emb is not None:
                scored = sorted(
                    (
                        (label, _cosine_similarity(emb, ex_emb))
                        for _, label, ex_emb in self._exemplar_embeddings
                    ),
                    key=lambda t: t[1],
                    reverse=True,
                )
                best_label, best_score = scored[0]
                second_score = scored[1][1] if len(scored) > 1 else 0.0
                gap = best_score - second_score
                below = best_score < self._threshold

                record.update({
                    "skill_label": best_label,
                    "score": round(best_score, 4),
                    "gap": round(gap, 4),
                    "below_threshold": below,
                })

                tag = "MISS" if below else "HIT "
                logger.info(
                    '[{}] "{}" → {} (score={:.3f} gap={:.3f})',
                    tag, text, best_label, best_score, gap,
                )
            else:
                logger.info('[----] "{}" (embed unavailable)', text)
        else:
            logger.info('[----] "{}"', text)

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._count += 1
        if self._count_limit is not None and self._count >= self._count_limit:
            logger.info("measure_router: count limit ({}) reached — stopping", self._count_limit)
            self._stop_event.set()

    @property
    def count(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    config = VoiceConfig.from_env()

    # Normalise embed URL to the full /v1/embeddings endpoint.
    raw_url = args.embed_url.rstrip("/")
    if raw_url.endswith("/embeddings"):
        embed_url = raw_url
    elif raw_url.endswith("/v1"):
        embed_url = raw_url + "/embeddings"
    else:
        embed_url = raw_url + "/v1/embeddings"
    parsed = urlparse(embed_url)
    health_url = f"{parsed.scheme}://{parsed.netloc}/health"

    output_path = Path(args.output)

    # --- Load exemplars ---------------------------------------------------
    exemplar_pairs: list[tuple[str, str]] = []
    if args.exemplars:
        ep = Path(args.exemplars)
        if not ep.exists():
            logger.error("measure_router: exemplars file not found: {}", ep)
            sys.exit(1)
        with ep.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    exemplar_pairs.append((obj["text"], obj["label"]))
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("measure_router: exemplars line {}: {} — skipping", lineno, exc)
        logger.info("measure_router: {} exemplars loaded from {}", len(exemplar_pairs), ep)

    # --- Pre-compute exemplar embeddings ---------------------------------
    exemplar_embeddings: list[tuple[str, str, list[float]]] = []
    embed_active: str | None = None

    if exemplar_pairs:
        try:
            async with httpx.AsyncClient(timeout=3.0) as probe:
                r = await probe.get(health_url)
                r.raise_for_status()
            logger.info(
                "measure_router: embed server healthy — pre-computing {} exemplars",
                len(exemplar_pairs),
            )
            exemplar_embeddings = await _precompute_exemplar_embeddings(
                exemplar_pairs, embed_url, args.embed_model,
            )
            if exemplar_embeddings:
                embed_active = embed_url
                logger.info("measure_router: {} of {} exemplars embedded", len(exemplar_embeddings), len(exemplar_pairs))
            else:
                logger.warning("measure_router: no exemplars embedded — transcript-only mode")
        except Exception as exc:
            logger.warning(
                "measure_router: embed server unavailable ({}) — transcript-only mode", exc,
            )

    # --- Audio device selection ------------------------------------------
    input_idx, _ = _select_audio_devices(
        sample_rate=config.sample_rate,
        preferred_input_name=config.input_device_name,
    )

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=config.sample_rate,
            audio_out_enabled=False,
            input_device_index=input_idx,
        )
    )

    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            sample_rate=config.sample_rate,
            params=_VAD_PARAMS,
        )
    )

    stt = WhisperSTTService(
        device=args.stt_device,
        compute_type=args.stt_compute_type,
        settings=WhisperSTTService.Settings(model=args.stt_model),
    )

    stop_event = asyncio.Event()
    count_limit = args.count if args.count > 0 else None

    async with httpx.AsyncClient() as http_client:
        scorer = ScoringProcessor(
            output_path=output_path,
            exemplar_embeddings=exemplar_embeddings,
            embed_url=embed_active,
            embed_model=args.embed_model,
            threshold=args.threshold,
            count_limit=count_limit,
            stop_event=stop_event,
            http_client=http_client if embed_active else None,
        )

        pipeline = Pipeline([transport.input(), vad, stt, scorer])
        task = PipelineTask(
            pipeline,
            params=PipelineParams(audio_in_sample_rate=config.sample_rate),
        )
        runner = PipelineRunner()

        loop = asyncio.get_running_loop()

        def _on_signal() -> None:
            logger.info("measure_router: stopping (signal)")
            stop_event.set()

        loop.add_signal_handler(signal.SIGINT, _on_signal)
        loop.add_signal_handler(signal.SIGTERM, _on_signal)

        mode = "scoring" if embed_active else "transcript-only"
        logger.info(
            "measure_router: ready — mode={} threshold={} output={}",
            mode, args.threshold, output_path.resolve(),
        )
        if count_limit:
            logger.info("measure_router: will stop after {} utterances", count_limit)
        logger.info("measure_router: speak naturally — Ctrl-C to stop")

        async def _cancel_on_stop() -> None:
            await stop_event.wait()
            await task.cancel()

        await asyncio.gather(runner.run(task), _cancel_on_stop(), return_exceptions=True)

    logger.info(
        "measure_router: done — {} utterances written to {}",
        scorer.count, output_path.resolve(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure fast-path classifier routing on real mic utterances.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:

  Collect raw transcripts (no embed server required):
    python scripts/measure_router.py --output corpus.jsonl

  Score against exemplars (llama-server on port 8081 must be running):
    python scripts/measure_router.py --exemplars exemplars.jsonl --output corpus.jsonl

  Collect exactly 20 utterances then stop:
    python scripts/measure_router.py --count 20 --output corpus.jsonl

  Test a different threshold:
    python scripts/measure_router.py --exemplars exemplars.jsonl --threshold 0.75

Exemplar file format (JSONL, one per line):
  {"text": "what's the weather like in London", "label": "weather"}
  {"text": "will it rain tomorrow", "label": "weather"}
  {"text": "tell me something interesting", "label": "none"}
  {"text": "um hey what is going on", "label": "none"}

Note: collect exemplars from real mic input through this harness, NOT from
typed text. Whisper's disfluencies shift the embedding distribution.
""",
    )
    parser.add_argument(
        "--exemplars", metavar="FILE",
        help="JSONL file of {text, label} pairs for scoring (optional; omit for transcript-only mode)",
    )
    parser.add_argument(
        "--output", metavar="FILE", default="corpus.jsonl",
        help="JSONL file to append utterance records to (default: corpus.jsonl)",
    )
    parser.add_argument(
        "--embed-url", metavar="URL", default="http://127.0.0.1:8081/v1/embeddings",
        help="llama.cpp embeddings endpoint (default: http://127.0.0.1:8081/v1/embeddings)",
    )
    parser.add_argument(
        "--embed-model", metavar="NAME", default="nomic-embed-text",
        help="Model name to pass to the embeddings API (default: nomic-embed-text)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.70,
        help="Classifier threshold to test against (default: 0.70)",
    )
    parser.add_argument(
        "--count", type=int, default=0,
        help="Stop after N utterances; 0 = run until Ctrl-C (default: 0)",
    )
    parser.add_argument(
        "--stt-model", metavar="NAME", default="base.en",
        help="Whisper model name (default: base.en; must match production)",
    )
    parser.add_argument(
        "--stt-device", metavar="DEVICE", default="cpu",
        help="Whisper inference device (default: cpu)",
    )
    parser.add_argument(
        "--stt-compute-type", metavar="TYPE", default="int8",
        help="Whisper compute type (default: int8)",
    )
    asyncio.run(main(parser.parse_args()))
