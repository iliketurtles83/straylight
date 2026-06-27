from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _default_model_dir() -> Path:
    # Repo root: tests/ -> ../
    return Path(__file__).resolve().parents[1] / "models" / "wake"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wakeword smoke test for straylight. "
            "Runs production OpenWakeWordDetector against microphone audio."
        )
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=_default_model_dir(),
        help="Directory containing computer_v2.onnx, melspectrogram.onnx, embedding_model.onnx",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="computer_v2",
        help="Wakeword label to match in model predictions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1280,
        help="Audio chunk size in samples at 16kHz",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate (keep at 16000 for openWakeWord)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Maximum run time for smoke test",
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=20,
        help="Print score heartbeat every N chunks",
    )
    parser.add_argument(
        "--require-detection",
        action="store_true",
        help="Exit non-zero if no wakeword was detected before timeout",
    )
    return parser.parse_args()


def main() -> int:
    from surfaces.voice.clients import OpenWakeWordDetector, VoiceDependencyError

    args = parse_args()

    if args.chunk_size <= 0:
        print("chunk-size must be > 0")
        return 2
    if args.sample_rate != 16000:
        print("sample-rate must be 16000 for this smoke test")
        return 2
    if args.seconds <= 0:
        print("seconds must be > 0")
        return 2

    detector = OpenWakeWordDetector(
        model_dir=args.model_dir,
        wakeword_label=args.label,
        threshold=args.threshold,
    )

    # Force model load up front so missing files/deps fail before mic capture.
    try:
        detector.reset()
    except VoiceDependencyError as exc:
        print(f"[FAIL] dependency/model setup error: {exc}")
        return 2

    try:
        import pyaudio
    except Exception as exc:
        print(f"[FAIL] pyaudio is required for microphone capture: {exc}")
        return 2

    print(f"[INFO] model dir: {args.model_dir}")
    print(f"[INFO] label: {args.label}  threshold: {args.threshold:.2f}")
    print(f"[INFO] listening for up to {args.seconds:.1f}s ...")

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=args.sample_rate,
        input=True,
        frames_per_buffer=args.chunk_size,
    )

    started = time.monotonic()
    chunks = 0
    best_score = 0.0
    triggered = False

    try:
        while time.monotonic() - started < args.seconds:
            frame = np.frombuffer(
                stream.read(args.chunk_size, exception_on_overflow=False),
                dtype=np.int16,
            )
            score, is_triggered = detector.triggered(frame.tolist())
            chunks += 1
            if score > best_score:
                best_score = score
            if is_triggered:
                print(f"[PASS] wakeword detected (score={score:.3f}) after {chunks} chunks")
                triggered = True
                break
            if args.status_every > 0 and chunks % args.status_every == 0:
                print(f"[INFO] heartbeat chunks={chunks} best_score={best_score:.3f}")
    except KeyboardInterrupt:
        print("[INFO] interrupted by user")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    if triggered:
        return 0

    print(f"[INFO] timeout reached. best_score={best_score:.3f}")
    if args.require_detection:
        print("[FAIL] wakeword was not detected before timeout")
        return 1

    print("[PASS] smoke completed: detector and microphone pipeline stayed healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
