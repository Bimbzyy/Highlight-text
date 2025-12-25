#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate non-silence segments from a silence.json file.

DEFAULT BEHAVIOUR:
    - Uses ffprobe on the provided audio file to get the real duration.
    - Builds non-silence segments as complement of silences in [0, audio_duration].

USAGE:

    # Default: USE ffprobe (recommended)
    python3 generate_non_silence.py \
        output/silence.json \
        output/non_silence.json \
        assets/input_path.m4a

    # Explicitly DISABLE ffprobe (pure JSON -> JSON, stop at last_silence_end)
    python3 generate_non_silence.py \
        output/silence.json \
        output/non_silence.json \
        assets/input_path.m4a \
        --no-ffprobe

Other options:
    --min-seg-duration  Minimum non-silence duration in seconds (default: 0.01)
    --debug              Enable debug logging
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("non_silence")


def sec_to_timestamp(sec: float) -> str:
    """Convert seconds (float) to 'MM:SS.mmm'."""
    if sec < 0:
        sec = 0.0
    total_ms = int(round(sec * 1000))
    minutes = total_ms // 60000
    remaining = total_ms % 60000
    seconds = remaining // 1000
    millis = remaining % 1000
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def load_silence_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    silences = data.get("silences", [])
    # ensure sorted & normalized
    silences = sorted(silences, key=lambda s: float(s["start_sec"]))
    data["silences"] = silences
    return data


def get_audio_duration_ffprobe(audio_path: Path) -> Optional[float]:
    """
    Get duration in seconds using ffprobe.

    Returns float or None on failure.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    logger.info("Running ffprobe for duration: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        logger.warning("ffprobe not found; skipping duration refinement.")
        return None
    except subprocess.CalledProcessError as e:
        logger.warning("ffprobe failed: %s", e.stderr.strip())
        return None

    try:
        return float(proc.stdout.strip())
    except ValueError:
        logger.warning("Unable to parse ffprobe duration: %r", proc.stdout)
        return None


def build_non_silence_segments(
    silences: List[Dict[str, Any]],
    timeline_start: float,
    timeline_end: float,
    min_segment_duration: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Compute non-silence segments as complement of silences in [timeline_start, timeline_end].

    - Start from 'cursor' = timeline_start.
    - For each silence:
        create [cursor, silence.start_sec] if long enough.
        move cursor to max(cursor, silence.end_sec).
    - After last silence:
        create [cursor, timeline_end] if long enough.
    """
    segments: List[Dict[str, Any]] = []

    if timeline_end <= timeline_start:
        return segments

    cursor = timeline_start

    for s in silences:
        s_start = float(s["start_sec"])
        s_end = float(s["end_sec"])

        # clamp silences to [timeline_start, timeline_end]
        if s_end <= timeline_start or s_start >= timeline_end:
            continue

        s_start = max(s_start, timeline_start)
        s_end = min(s_end, timeline_end)

        # gap [cursor, s_start] is non-silence
        if cursor < s_start:
            seg_start = cursor
            seg_end = s_start
            dur = seg_end - seg_start
            if dur >= min_segment_duration:
                segments.append(
                    {
                        "start_sec": round(seg_start, 3),
                        "end_sec": round(seg_end, 3),
                        "duration_sec": round(dur, 3),
                        "start": sec_to_timestamp(seg_start),
                        "end": sec_to_timestamp(seg_end),
                    }
                )

        # move cursor past this silence
        cursor = max(cursor, s_end)

    # tail after last silence -> timeline_end
    if cursor < timeline_end:
        seg_start = cursor
        seg_end = timeline_end
        dur = seg_end - seg_start
        if dur >= min_segment_duration:
            segments.append(
                {
                    "start_sec": round(seg_start, 3),
                    "end_sec": round(seg_end, 3),
                    "duration_sec": round(dur, 3),
                    "start": sec_to_timestamp(seg_start),
                    "end": sec_to_timestamp(seg_end),
                }
            )

    # index them
    for i, seg in enumerate(segments):
        seg["index"] = i

    return segments


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate non-silence segments from a silence.json file."
    )
    parser.add_argument("silence_json", help="Path to silence.json")
    parser.add_argument("output_json", help="Path to non_silence.json")
    parser.add_argument("audio_path", help="Path to original audio file (.m4a/.aac/.mp3/.wav)")
    parser.add_argument(
        "--no-ffprobe",
        action="store_true",
        help="Do NOT use ffprobe; stop at last_silence_end instead of real file duration",
    )
    parser.add_argument(
        "--min-seg-duration",
        type=float,
        default=0.01,
        help="Minimum non-silence duration in seconds (default: 0.01)",
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

    silence_path = Path(args.silence_json)
    output_path = Path(args.output_json)
    audio_path = Path(args.audio_path)

    if not silence_path.is_file():
        logger.error("silence.json file does not exist: %s", silence_path)
        sys.exit(1)

    if not audio_path.is_file():
        logger.error("Audio file does not exist: %s", audio_path)
        sys.exit(1)

    data = load_silence_json(silence_path)
    silences = data.get("silences", [])
    audio_file = audio_path.name

    if not silences:
        logger.warning("No silences found; cannot infer non-silence segments.")
        sys.exit(1)

    # Base timeline from 0 to last_silence_end
    last_silence_end = float(silences[-1]["end_sec"])
    timeline_start = 0.0
    timeline_end = last_silence_end
    used_ffprobe = False

    if not args.no_ffprobe:
        duration = get_audio_duration_ffprobe(audio_path)
        if duration is not None:
            timeline_end = max(last_silence_end, duration)
            used_ffprobe = True
        else:
            logger.warning(
                "ffprobe failed or missing; falling back to last_silence_end as timeline_end."
            )

    logger.info("timeline_start_sec = %.3f", timeline_start)
    logger.info("timeline_end_sec   = %.3f", timeline_end)
    logger.info("last_silence_end   = %.3f", last_silence_end)
    logger.info("used_ffprobe       = %s", used_ffprobe)

    segments = build_non_silence_segments(
        silences=silences,
        timeline_start=timeline_start,
        timeline_end=timeline_end,
        min_segment_duration=args.min_seg_duration,
    )

    result = {
        "audio_file": audio_file,
        "source_silence_json": silence_path.name,
        "timeline_start_sec": round(timeline_start, 3),
        "timeline_end_sec": round(timeline_end, 3),
        "last_silence_end_sec": round(last_silence_end, 3),
        "used_ffprobe_for_timeline_end": used_ffprobe,
        "total_non_silence_intervals": len(segments),
        "segments": segments,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Non-silence JSON written to %s", output_path)


if __name__ == "__main__":
    main()
