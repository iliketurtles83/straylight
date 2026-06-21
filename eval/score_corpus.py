#!/usr/bin/env python3
"""Offline corpus scorer for Phase 2 fast-path classifier calibration.

Loads exemplars and a labeled corpus, embeds everything in-process with
llama-cpp-python (same path used by `AgentProcessor` at runtime), then
reports accuracy on a held-out 20% split plus per-class confusion and
gap-distribution stats. No mic, no live llama-server, no /tokenize.

Usage:
  python scripts/score_corpus.py \
      --exemplars exemplars.jsonl \
      --corpus corpus_labeled.jsonl \
      [--embed-model models/embed/nomic-embed-text-v1.5.f16.gguf] \
      [--threshold 0.80] [--min-gap 0.05] [--test-frac 0.2] [--seed 0]

Both files are JSONL. Required fields per line:
  exemplars: {"text": "...", "label": "weather" | "none" | ...}
  corpus:    {"transcript": "...", "label": "weather" | "none" | ...}
             (or {"text": "...", "label": "..."} — both accepted)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            print(f"warn: {path}:{lineno}: {exc}", file=sys.stderr)
    return rows


def _text_of(row: dict) -> str | None:
    for key in ("transcript", "text"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def main(args: argparse.Namespace) -> int:
    try:
        from llama_cpp import Llama
    except ImportError:
        print("error: llama-cpp-python not installed. pip install llama-cpp-python", file=sys.stderr)
        return 2

    embed_path = Path(args.embed_model)
    if not embed_path.exists():
        print(f"error: embed model missing: {embed_path}", file=sys.stderr)
        return 2

    exemplar_rows = _load_jsonl(Path(args.exemplars))
    corpus_rows = _load_jsonl(Path(args.corpus))

    exemplars: list[tuple[str, str]] = []
    for row in exemplar_rows:
        text = _text_of(row)
        label = row.get("label")
        if text and isinstance(label, str):
            exemplars.append((text, label))

    corpus: list[tuple[str, str]] = []
    for row in corpus_rows:
        text = _text_of(row)
        label = row.get("label")
        if text and isinstance(label, str):
            corpus.append((text, label))

    if not exemplars:
        print("error: no usable exemplars", file=sys.stderr)
        return 2
    if not corpus:
        print("error: no usable labeled corpus rows", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    by_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item in corpus:
        by_label[item[1]].append(item)

    test: list[tuple[str, str]] = []
    train: list[tuple[str, str]] = []
    for label, items in by_label.items():
        rng.shuffle(items)
        n_test = max(1, int(round(len(items) * args.test_frac))) if len(items) > 1 else 0
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    print(f"exemplars: {len(exemplars)} | corpus: {len(corpus)} (train={len(train)} test={len(test)})")
    print(f"per-class: {dict(Counter(l for _, l in corpus))}")

    print(f"loading embed model: {embed_path}")
    llama = Llama(
        model_path=str(embed_path),
        embedding=True,
        n_ctx=512,
        n_threads=4,
        verbose=False,
    )

    def embed(text: str) -> list[float]:
        return llama.create_embedding(text)["data"][0]["embedding"]  # type: ignore[index]

    print("embedding exemplars...")
    indexed: list[tuple[str, list[float]]] = [(label, embed(text)) for text, label in exemplars]

    print("scoring test split...")
    correct = 0
    confusion: dict[tuple[str, str], int] = Counter()
    gaps: list[float] = []
    near_threshold = 0

    for text, expected in test:
        emb = embed(text)
        scored = sorted(
            ((label, _cosine(emb, ex_emb)) for label, ex_emb in indexed),
            key=lambda t: t[1],
            reverse=True,
        )
        best_label, best_score = scored[0]
        second = scored[1][1] if len(scored) > 1 else 0.0
        gap = best_score - second
        gaps.append(gap)

        if best_score >= args.threshold and gap >= args.min_gap:
            predicted = best_label
        else:
            predicted = "none"
            near_threshold += 1

        if predicted == expected:
            correct += 1
        confusion[(expected, predicted)] += 1

    acc = correct / len(test) if test else 0.0
    print()
    print(f"held-out accuracy: {acc:.1%} ({correct}/{len(test)})")
    print(f"mean gap: {sum(gaps) / len(gaps):.3f}" if gaps else "mean gap: n/a")
    print(f"rejected (below threshold/gap → 'none'): {near_threshold}")
    print()
    print("confusion (expected → predicted: count):")
    for (exp, pred), n in sorted(confusion.items()):
        marker = "  " if exp == pred else "!!"
        print(f"  {marker} {exp:>10} -> {pred:<10} {n}")

    target = args.required_accuracy
    print()
    if acc >= target:
        print(f"PASS — accuracy {acc:.1%} ≥ required {target:.0%}")
        return 0
    print(f"FAIL — accuracy {acc:.1%} < required {target:.0%}")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exemplars", required=True, help="JSONL exemplar file")
    parser.add_argument("--corpus", required=True, help="JSONL labeled corpus file")
    parser.add_argument(
        "--embed-model",
        default="models/embed/nomic-embed-text-v1.5.f16.gguf",
        help="Path to nomic-embed .gguf",
    )
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--min-gap", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--required-accuracy",
        type=float,
        default=0.90,
        help="Exit non-zero if held-out accuracy falls below this (default 0.90)",
    )
    sys.exit(main(parser.parse_args()))
