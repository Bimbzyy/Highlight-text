
"""
Silence detection using ffmpeg's silencedetect filter.

Usage:
    python3 detect_silence.py INPUT_AUDIO.(m4a|aac|mp3|wav) output/silence.json

Notes:
    - Works directly with your iPhone .m4a file.
    - Requires ffmpeg installed and in PATH.
    - No heavy Python dependencies (only stdlib).
"""

import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

# Regex to capture lines like:
#   silence_start: 3.024
#   silence_end: 4.512 | silence_duration: 1.488
SILENCE_RE = re.compile(r"silence_(start|end):\s+([0-9.]+)")

logger = logging.getLogger("silence_detector")


def sec_to_timestamp(sec: float) -> str:
    """Convert seconds to mm:ss.mmm format."""
    if sec < 0:
        sec = 0.0
    total_ms = int(round(sec * 1000))
    minutes = total_ms // 60000
    remaining = total_ms % 60000
    seconds = remaining // 1000
    millis = remaining % 1000
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def run_ffmpeg_silencedetect(
    audio_path: Path,
    noise_threshold_db: int = -35,
    min_silence_duration: float = 0.30,
) -> str:
    """
    Run ffmpeg silencedetect and return its stderr output as text.

    noise_threshold_db: dB below full scale, e.g., -35, -40, etc.
    min_silence_duration: minimum silence duration in seconds.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={noise_threshold_db}dB:d={min_silence_duration}",
        "-f",
        "null",
        "-",  # output to null, we only care about logs
    ]

    logger.info("Running ffmpeg: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install ffmpeg and ensure it is in PATH.")
        raise

    if proc.returncode != 0:
        logger.warning(
            "ffmpeg exited with non-zero code %s. "
            "Sometimes this still produces usable silencedetect output.",
            proc.returncode,
        )

    return proc.stderr


def parse_silence_intervals(ffmpeg_log: str) -> List[Dict[str, Any]]:
    """
    Parse ffmpeg silencedetect output and return list of silence intervals.

    Each interval:
        {
          "index": 0,
          "start_sec": 1.234,
          "end_sec": 2.345,
          "start": "00:01.234",
          "end": "00:02.345"
        }
    """
    intervals: List[Dict[str, Any]] = []
    current_start = None

    for line in ffmpeg_log.splitlines():
        m = SILENCE_RE.search(line)
        if not m:
            continue

        event_type, value = m.groups()
        t = float(value)

        if event_type == "start":
            current_start = t
            logger.debug("Detected silence_start at %.3f sec", t)
        elif event_type == "end":
            if current_start is None:
                # ffmpeg sometimes prints an end without a start
                logger.debug("silence_end at %.3f with no matching start; skipping", t)
                continue
            logger.debug("Detected silence_end at %.3f sec", t)
            interval = {
                "start_sec": round(current_start, 3),
                "end_sec": round(t, 3),
                "start": sec_to_timestamp(current_start),
                "end": sec_to_timestamp(t),
            }
            intervals.append(interval)
            current_start = None

    # In rare cases, we might end with a start but no end (e.g. trailing silence)
    if current_start is not None:
        logger.info(
            "File ends with an open silence_start at %.3f sec; "
            "treating audio end as silence_end.",
            current_start,
        )
        # You can optionally pass total duration and close this interval properly.

    return intervals


def build_silence_json(
    audio_path: Path,
    noise_threshold_db: int,
    min_silence_duration: float,
    intervals: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Wrap intervals into a structured JSON object."""
    for i, it in enumerate(intervals):
        it["index"] = i

    return {
        "audio_file": audio_path.name,
        "noise_threshold_db": noise_threshold_db,
        "min_silence_duration_sec": min_silence_duration,
        "total_silence_intervals": len(intervals),
        "silences": intervals,
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect silence intervals in an audio file using ffmpeg."
    )
    parser.add_argument("input_audio", help="Path to input audio file (.m4a, .aac, .mp3, .wav)")
    parser.add_argument("output_json", help="Path to output JSON file")
    parser.add_argument(
        "--noise-db",
        type=int,
        default=-35,
        help="Noise threshold in dB for silencedetect (default: -35)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.30,
        help="Minimum silence duration in seconds (default: 0.30)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    audio_path = Path(args.input_audio)
    if not audio_path.is_file():
        logger.error("Input audio file does not exist: %s", audio_path)
        sys.exit(1)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Input audio: %s", audio_path)
    logger.info("Output JSON: %s", output_path)
    logger.info("noise_db=%d, min_duration=%.3f", args.noise_db, args.min_duration)

    ffmpeg_log = run_ffmpeg_silencedetect(
        audio_path=audio_path,
        noise_threshold_db=args.noise_db,
        min_silence_duration=args.min_duration,
    )

    intervals = parse_silence_intervals(ffmpeg_log)
    logger.info("Detected %d silence intervals", len(intervals))

    result_json = build_silence_json(
        audio_path=audio_path,
        noise_threshold_db=args.noise_db,
        min_silence_duration=args.min_duration,
        intervals=intervals,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    logger.info("Silence JSON written to %s", output_path)


if __name__ == "__main__":
    main()
