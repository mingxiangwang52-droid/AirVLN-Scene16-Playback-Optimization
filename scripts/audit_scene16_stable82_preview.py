#!/usr/bin/env python3
"""Audit the strict Scene16 stable82 presentation manifest without mutating it."""

import argparse
import collections
import json
from pathlib import Path


PROFILE = "airsim_preview_final_v3_reuse_scene"


def resolve_path(root, value):
    raw = str(value or "")
    if raw.startswith("/agent-sessions/"):
        return root / "live_server" / "runtime" / "agent_sessions" / raw[len("/agent-sessions/"):]
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path


def audit(manifest_path, pool_path, profile):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    pool_cases = pool.get("cases") if isinstance(pool, dict) else []
    expected = {str(item.get("episode_id")) for item in pool_cases if isinstance(item, dict)}
    entries = [
        item for item in manifest.get("entries", [])
        if isinstance(item, dict) and item.get("runtime_profile") == profile
    ]
    by_episode = {str(item.get("episode_id")): item for item in entries}
    missing = sorted(expected - set(by_episode))
    duplicate_count = len(entries) - len(by_episode)
    reasons = collections.Counter(str(item.get("eligibility_reason") or "unknown") for item in entries)
    status = collections.Counter(str(item.get("status") or "unknown") for item in entries)
    strict_success = [item for item in entries if item.get("ok_for_frontend") is True]
    problems = []
    for item in entries:
        episode_id = str(item.get("episode_id") or "")
        metadata = item.get("video_metadata") if isinstance(item.get("video_metadata"), dict) else {}
        video_path = resolve_path(manifest_path.parent.parent.parent, item.get("video_path"))
        if item.get("ok_for_frontend") is True:
            if item.get("status") != "done":
                problems.append({"episode_id": episode_id, "problem": "frontend_entry_not_done"})
            if item.get("success_20m_with_surface") is not True:
                problems.append({"episode_id": episode_id, "problem": "frontend_entry_not_strict_success"})
            try:
                if float(item.get("distance_to_goal")) > 20.0:
                    problems.append({"episode_id": episode_id, "problem": "frontend_entry_distance_over_20m"})
            except (TypeError, ValueError):
                problems.append({"episode_id": episode_id, "problem": "frontend_entry_missing_distance"})
            source_width = int(metadata.get("source_width") or 0)
            source_height = int(metadata.get("source_height") or 0)
            video_width = int(metadata.get("width") or 0)
            video_height = int(metadata.get("height") or 0)
            aspect = source_width / max(1, source_height)
            if source_width < 640 or source_height < 360:
                problems.append({"episode_id": episode_id, "problem": "source_resolution", "width": source_width, "height": source_height})
            if not 1.6 <= aspect <= 1.85:
                problems.append({"episode_id": episode_id, "problem": "source_aspect", "aspect": aspect})
            if str(metadata.get("interp") or "").lower() != "flow":
                problems.append({"episode_id": episode_id, "problem": "video_interpolation_not_flow"})
            if str(metadata.get("codec") or "").lower() != "h264":
                problems.append({"episode_id": episode_id, "problem": "video_codec_not_h264"})
            if video_width < 960 or video_height < 540:
                problems.append({"episode_id": episode_id, "problem": "video_resolution", "width": video_width, "height": video_height})
            if not item.get("video_path"):
                problems.append({"episode_id": episode_id, "problem": "frontend_entry_without_video_path"})
            elif not video_path.exists():
                # Paths in a remote manifest may be session-relative and are
                # checked again by the server; preserve the diagnostic here.
                problems.append({"episode_id": episode_id, "problem": "video_path_missing", "path": str(video_path)})
    return {
        "manifest": str(manifest_path),
        "profile": profile,
        "expected_pool_count": len(expected),
        "active_entry_count": len(entries),
        "strict_success_count": len(strict_success),
        "failed_or_excluded_count": len(entries) - len(strict_success),
        "missing_episode_count": len(missing),
        "missing_episodes": missing,
        "duplicate_entry_count": duplicate_count,
        "status_counts": dict(status),
        "eligibility_reasons": dict(reasons),
        "problems": problems,
        "audit_ok": bool(
            len(expected) == 82
            and len(entries) == len(expected)
            and not missing
            and duplicate_count == 0
            and not problems
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--profile", default=PROFILE)
    args = parser.parse_args()
    report = audit(args.manifest, args.pool, args.profile)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["audit_ok"] else 2)


if __name__ == "__main__":
    main()
