#!/usr/bin/env python3
"""Validate a high-resolution replay metadata file without contacting AirSim."""

import argparse
import json
import math
from pathlib import Path


def close(a, b, eps=1e-6):
    return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(
        math.isfinite(float(x)) and math.isfinite(float(y)) and abs(float(x) - float(y)) <= eps
        for x, y in zip(a, b)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    args = parser.parse_args()
    data = json.loads(args.metadata.read_text(encoding="utf-8"))
    required = ("endpoint", "recorded_endpoint", "frame_count", "fps", "width", "height", "success_20m_with_surface")
    missing = [key for key in required if key not in data]
    if missing:
        raise SystemExit(f"missing fields: {', '.join(missing)}")
    if not data["success_20m_with_surface"]:
        raise SystemExit("replay is not marked success_20m_with_surface")
    if not close(data["endpoint"], data["recorded_endpoint"]):
        raise SystemExit("rendered endpoint differs from recorded endpoint")
    if int(data["width"]) < 960 or int(data["height"]) < 540:
        raise SystemExit("resolution is below the presentation minimum")
    if int(data["frame_count"]) <= 0 or float(data["fps"]) <= 0:
        raise SystemExit("invalid frame count or fps")
    print(json.dumps({"ok": True, "session_id": data.get("session_id"), "resolution": f"{data['width']}x{data['height']}", "frames": data["frame_count"], "fps": data["fps"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
