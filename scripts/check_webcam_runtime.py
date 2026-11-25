"""Offline smoke test for the isolated webcam feature child; never opens a camera."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from context_operator.providers.webcam import MODEL, RUNTIME as PYTHON

RUNTIME = ROOT / "context_operator/providers/webcam_runtime.py"
SCHEMA = "contextoperator.webcam.v1"
HEADER = struct.Struct("<4sQdII")


def _read_lines(stream, output: queue.Queue) -> None:
    try:
        for line in iter(stream.readline, b""):
            output.put(line)
    finally:
        output.put(None)


def _next_json(lines: queue.Queue, timeout: float) -> dict | None:
    try:
        line = lines.get(timeout=timeout)
    except queue.Empty:
        return None
    if line is None:
        return None
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def run(python: Path, model: Path, fixture: Path) -> tuple[dict, int]:
    from PIL import Image

    with Image.open(fixture) as source:
        image = source.convert("RGB")
        image.thumbnail((640, 480), Image.Resampling.LANCZOS)
        width, height = image.size
        frame = image.tobytes()

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        [str(python), str(RUNTIME), "--model", str(model)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        shell=False,
        bufsize=0,
    )
    lines: queue.Queue = queue.Queue()
    reader = threading.Thread(target=_read_lines, args=(process.stdout, lines), daemon=True)
    reader.start()
    summary = {"result_count": 0, "latency_ms": None, "feature_shape": []}
    exit_code = 2
    try:
        ready = _next_json(lines, 45.0)
        if ready != {"schema": SCHEMA, "ready": True}:
            return summary, exit_code

        captured = time.monotonic()
        packet = HEADER.pack(b"WCF1", 1, captured, width, height) + frame
        started = time.monotonic()
        process.stdin.write(packet)
        process.stdin.flush()
        result = _next_json(lines, 45.0)
        latency_ms = (time.monotonic() - started) * 1000.0
        if (not isinstance(result, dict)
                or set(result) != {"schema", "seq", "captured", "features", "reason"}
                or result.get("schema") != SCHEMA
                or result.get("seq") != 1
                or result.get("captured") != captured
                or not isinstance(result.get("features"), list)):
            return summary, exit_code
        features = result["features"]
        if (len(features) != 8
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in features)
                or result.get("reason") != ""):
            summary = {"result_count": 1, "latency_ms": round(latency_ms, 2),
                       "feature_shape": [len(features)]}
            return summary, exit_code
        summary = {"result_count": 1, "latency_ms": round(latency_ms, 2),
                   "feature_shape": [len(features)]}

        # Closing stdin verifies clean EOF handling after a complete frame.
        process.stdin.close()
        exit_code = process.wait(timeout=15.0)
        return summary, 0 if exit_code == 0 else 2
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.stdout:
            process.stdout.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--fixture", type=Path, required=True, help="Path to a local test photograph")
    args = parser.parse_args(argv)
    summary, code = run(args.python, args.model, args.fixture)
    print(json.dumps(summary, separators=(",", ":"), allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
