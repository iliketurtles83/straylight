from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = Path("/path/to/straylight/models/tts/en_US-amy-medium.onnx")
DEFAULT_TEXT = "Ghost in the Shell and Akira are awesome."


def resolve_piper_binary() -> str | None:
    # Prefer PATH, then fall back to the active interpreter's bin directory.
    piper_bin = shutil.which("piper")
    if piper_bin:
        return piper_bin

    interpreter_dir = Path(sys.executable).parent
    candidate = interpreter_dir / "piper"
    if candidate.exists() and candidate.is_file():
        return str(candidate)

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a standalone Piper TTS test.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to Piper .onnx model")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="PCM sample rate expected by the model",
    )
    return parser.parse_args()


def require_binary(name: str) -> None:
    if shutil.which(name):
        return
    print(f"Missing required binary: {name}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    args = parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 1

    piper_bin = resolve_piper_binary()
    if not piper_bin:
        print(
            "Missing required binary: piper (not found on PATH or next to active Python).",
            file=sys.stderr,
        )
        return 1
    require_binary("aplay")

    synth = subprocess.run(
        [piper_bin, "--model", str(args.model), "--output-raw"],
        input=args.text.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if synth.returncode != 0:
        print("Piper synthesis failed:", file=sys.stderr)
        if synth.stderr:
            print(synth.stderr.decode("utf-8", errors="replace"), file=sys.stderr)
        return synth.returncode

    if not synth.stdout:
        print("Piper returned no audio bytes.", file=sys.stderr)
        return 1

    play = subprocess.run(
        ["aplay", "-r", str(args.sample_rate), "-f", "S16_LE", "-c", "1"],
        input=synth.stdout,
        check=False,
    )
    if play.returncode != 0:
        print("Audio playback failed (aplay).", file=sys.stderr)
        return play.returncode

    print("Piper isolation test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())