#!/usr/bin/env python3
"""Regenerate Scene16 stable82 flights with high-resolution preview frames."""

import argparse
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT / "live_server/runtime/scene16_single_scene_stable_pool_v055_entity_endpoint_stable82_20260822.json"
DEFAULT_MANIFEST = ROOT / "live_server/runtime/scene16_stable82_preview_manifest_20260825.json"
DEFAULT_CHECKPOINT = (
    os.environ.get("AIRVLN_PHASE_A_CKPT", "")
)
RUNTIME_PROFILE = "airsim_preview_final_v3_reuse_scene"

_REPORT_CACHE = {}


def _report_path(root, value):
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path


def _find_case(payload, episode_id):
    if not isinstance(payload, dict):
        return None
    for item in payload.get("cases") or []:
        if isinstance(item, dict) and str(item.get("episode_id") or "").strip() == episode_id:
            return item
    return None


def resolve_original_case_config(case, root):
    """Recover the route-prior/controller config used by the source success."""
    episode_id = str(case.get("episode_id") or "").strip()
    visited = set()

    def visit(report_name):
        path = _report_path(root, report_name)
        if path is None or path in visited or not path.exists():
            return None
        visited.add(path)
        try:
            payload = _REPORT_CACHE.get(path)
            if payload is None:
                payload = json.loads(path.read_text(encoding="utf-8"))
                _REPORT_CACHE[path] = payload
        except (OSError, ValueError):
            return None
        item = _find_case(payload, episode_id)
        if isinstance(item, dict):
            route_prior = item.get("eval_scene16_route_prior")
            source_session_id = str(item.get("session_id") or "").strip()
            session_summary = None
            if source_session_id:
                summary_path = root / "live_server/runtime/agent_sessions" / source_session_id / "summary.json"
                try:
                    session_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    session_summary = None
            ridge = item.get("eval_segment_grounding_ridge") or item.get("eval_segment_grounding_final_ridge")
            final_ridge = item.get("eval_segment_grounding_final_ridge")
            if isinstance(session_summary, dict):
                ridge = session_summary.get("segment_grounding_ridge") or ridge
            return {
                "route_prior": str(route_prior or ""),
                "enable_route_prior": bool(route_prior),
                "support_ranker": str(item.get("eval_scene16_support_ranker") or ""),
                "enable_support_ranker": bool(item.get("eval_enable_scene16_support_ranker", False)),
                "enable_oracle_goal_homing": bool(item.get("eval_enable_oracle_goal_homing", False)),
                "enable_oracle_path_homing": bool(item.get("eval_enable_oracle_path_homing", False)),
                "enable_learned_segment_homing": bool(item.get("eval_enable_learned_segment_homing", True)),
                "segment_grounding_ridge": str(ridge or ""),
                "segment_grounding_final_ridge": str(final_ridge or ""),
                "max_steps": int((session_summary or {}).get("max_steps") or payload.get("max_steps") or 1200),
                "params": item.get("eval_learned_segment_params") or {},
                "source_report": str(report_name),
            }
            nested = item.get("source_report")
            if nested:
                resolved = visit(nested)
                if resolved:
                    return resolved
        for nested in payload.get("source_reports") or []:
            resolved = visit(nested)
            if resolved:
                return resolved
        return None

    resolved = visit(case.get("source_report"))
    if resolved:
        return resolved
    return {
        "route_prior": "",
        "enable_route_prior": False,
        "support_ranker": "",
        "enable_support_ranker": False,
        "enable_oracle_goal_homing": False,
        "enable_oracle_path_homing": False,
        "enable_learned_segment_homing": True,
        "segment_grounding_ridge": "",
        "segment_grounding_final_ridge": "",
        "max_steps": 1200,
        "params": {},
        "source_report": str(case.get("source_report") or "unknown"),
        "fallback": True,
    }


def request_json(url, payload=None, timeout=30):
    body = None
    headers = {}
    method = "GET"
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def error_text(exc):
    if isinstance(exc, urllib.error.HTTPError):
        try:
            return exc.read().decode("utf-8", errors="replace")[:4000]
        except Exception:
            pass
    return str(exc)


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_manifest(path, pool_name):
    if not path.exists():
        manifest = {
            "format": "scene16_stable82_preview_manifest_v1",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": None,
            "source_stable_pool": pool_name,
            "video_profile": {
                "format": "mp4",
                "codec": "h264",
                "fps": 14,
                "smooth": True,
                "interp": "flow",
                "smooth_multiplier": 2.8,
                "stride": 1,
                "max_width": 960,
                "upscale": True,
                "sharpen": 0.16,
            },
            "regeneration_mode": {
                "purpose": "presentation_only",
                "oracle_goal_homing": False,
                "oracle_path_homing": False,
                "learned_segment_homing": True,
                "scene16_support_ranker": True,
                "scene16_route_prior": "per_case_source_report",
                "preview_camera": "preview_0",
                "preview_width": 768,
                "preview_height": 432,
                "jpeg_quality": 90,
            },
            "entries": [],
        }
        return manifest
    manifest = json.loads(path.read_text(encoding="utf-8"))
    # Keep resumable entries, but always advertise the active presentation profile.
    manifest["format"] = "scene16_stable82_preview_manifest_v1"
    manifest["source_stable_pool"] = pool_name
    manifest["video_profile"] = {
        "format": "mp4", "codec": "h264", "fps": 14, "smooth": True,
        "interp": "flow", "smooth_multiplier": 2.8, "stride": 1,
        "max_width": 960, "upscale": True, "sharpen": 0.16,
    }
    manifest["regeneration_mode"] = {
        "purpose": "presentation_only",
        "oracle_goal_homing": False,
        "oracle_path_homing": False,
        "learned_segment_homing": True,
        "scene16_support_ranker": True,
        "scene16_route_prior": "per_case_source_report",
        "preview_camera": "preview_0", "preview_width": 768,
        "preview_height": 432, "jpeg_quality": 90,
    }
    return manifest


def load_cases(pool_path):
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or len(cases) != 82:
        raise RuntimeError(f"expected 82 stable cases, got {len(cases or [])}")
    return [case for case in cases if isinstance(case, dict)]


def wait_for_no_active(base_url, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            active = request_json(f"{base_url}/api/agent/sessions/active?t={time.time()}", timeout=10)
            if active.get("status") in ("done", "error", "stopped"):
                return
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return
            raise
        time.sleep(2)
    raise RuntimeError("another agent session remained active")


def _stagnation_snapshot(latest):
    """Extract conservative recovery signals from a live session status."""
    snapshot = {
        "resamples": 0,
        "segment_safety_turns": 0,
    }
    for override in latest.get("safety_override") or []:
        if not isinstance(override, dict):
            continue
        learned = override.get("learned_segment_homing")
        if isinstance(learned, dict):
            snapshot["resamples"] = max(
                snapshot["resamples"], int(learned.get("resamples") or 0)
            )
            snapshot["segment_safety_turns"] = max(
                snapshot["segment_safety_turns"],
                int(learned.get("segment_safety_turn_count") or 0),
            )
    return snapshot


def poll_session(base_url, session_id, timeout, args):
    deadline = time.monotonic() + timeout
    latest = {}
    best_distance = None
    last_material_progress_step = 0
    while time.monotonic() < deadline:
        latest = request_json(f"{base_url}/api/agent/sessions/{urllib.parse.quote(session_id)}?t={time.time()}", timeout=20)
        step = int(latest.get("step") or 0)
        try:
            distance = float(latest.get("distance_to_goal"))
        except (TypeError, ValueError):
            distance = None
        if distance is not None and (best_distance is None or distance <= best_distance - args.stagnation_min_progress_m):
            best_distance = distance
            last_material_progress_step = step
        print(
            "PROGRESS",
            session_id,
            latest.get("status"),
            "step", latest.get("step"),
            "distance", latest.get("distance_to_goal"),
            "frames", latest.get("stream_frame_count"),
            flush=True,
        )
        if latest.get("status") in ("done", "error", "stopped"):
            return latest
        recovery = _stagnation_snapshot(latest)
        stalled_steps = step - last_material_progress_step
        if (
            args.stagnation_stop_enabled
            and distance is not None
            and distance >= args.stagnation_min_distance_m
            and step >= args.stagnation_min_steps
            and stalled_steps >= args.stagnation_window_steps
            and recovery["resamples"] >= args.stagnation_min_resamples
            and recovery["segment_safety_turns"] >= args.stagnation_min_safety_turns
        ):
            print(
                "EARLY_STOP_STAGNATION", session_id,
                json.dumps({
                    "step": step,
                    "distance": distance,
                    "stalled_steps": stalled_steps,
                    **recovery,
                }, sort_keys=True),
                flush=True,
            )
            request_json(f"{base_url}/api/agent/stop/{urllib.parse.quote(session_id)}", {}, timeout=15)
            latest = request_json(f"{base_url}/api/agent/sessions/{urllib.parse.quote(session_id)}?t={time.time()}", timeout=20)
            latest["stopped_for_stagnation"] = True
            return latest
        time.sleep(10)
    try:
        request_json(f"{base_url}/api/agent/stop/{urllib.parse.quote(session_id)}", {}, timeout=15)
    except Exception:
        pass
    latest = request_json(f"{base_url}/api/agent/sessions/{urllib.parse.quote(session_id)}?t={time.time()}", timeout=20)
    latest["timed_out"] = True
    return latest


def build_video(base_url, session_id):
    query = urllib.parse.urlencode({
        "format": "mp4",
        "video_codec": "h264",
        "smooth": 1,
        "interp": "flow",
        "smooth_multiplier": 2.8,
        "stride": 1,
        "fps": 14,
        "limit": 10000,
        "max_width": 960,
        "upscale": 1,
        "sharpen": 0.16,
    })
    return request_json(
        f"{base_url}/api/agent/sessions/{urllib.parse.quote(session_id)}/replay-video?{query}",
        timeout=600,
    )


def flight_eligibility(latest):
    """Check flight completion before spending time encoding a replay video."""
    if latest.get("status") != "done":
        return False, "flight_not_done"
    if latest.get("success_20m_with_surface") is not True:
        return False, "flight_not_successful_with_surface"
    try:
        if float(latest.get("distance_to_goal")) > 20.0:
            return False, "distance_over_20m"
    except (TypeError, ValueError):
        return False, "missing_final_distance"
    return True, "success_20m_with_surface"


def frontend_eligibility(latest, video):
    """A replay is selectable only when the flight itself reached the goal."""
    flight_ok, reason = flight_eligibility(latest)
    if not flight_ok:
        return False, reason
    metadata = video.get("metadata") if isinstance(video, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    if not video.get("ok"):
        return False, "video_encode_failed"
    if not metadata.get("frame_count") or not metadata.get("width"):
        return False, "video_metadata_invalid"
    if str(metadata.get("interp") or "").lower() != "flow":
        return False, "video_interpolation_not_flow"
    if str(metadata.get("codec") or "").lower() != "h264":
        return False, "video_codec_not_h264"
    try:
        video_width = int(metadata.get("width") or 0)
        video_height = int(metadata.get("height") or 0)
    except (TypeError, ValueError):
        video_width = video_height = 0
    if video_width < 960 or video_height < 540:
        return False, "video_resolution_too_low"
    try:
        source_width = int(metadata.get("source_width") or 0)
        source_height = int(metadata.get("source_height") or 0)
    except (TypeError, ValueError):
        source_width = source_height = 0
    if source_width < 640 or source_height < 360:
        return False, "source_preview_resolution_too_low"
    aspect = source_width / max(1, source_height)
    if aspect < 1.6 or aspect > 1.85:
        return False, "source_preview_aspect_invalid"
    return True, reason


def existing_entry_is_frontend_ready(entry):
    if not isinstance(entry, dict) or not entry.get("video_path"):
        return False
    ok, _reason = frontend_eligibility(
        entry,
        {"ok": True, "metadata": entry.get("video_metadata") or {}},
    )
    return ok


def attach_video(base_url, entry):
    """Encode a replay for a flight that has already passed its arrival checks."""
    try:
        video = build_video(base_url, entry["replay_session_id"])
        metadata = video.get("metadata") or {}
        ok_for_frontend, eligibility_reason = frontend_eligibility(entry, video)
        entry.update({
            "video_path": video.get("url"),
            "video_metadata": metadata,
            "video_width": metadata.get("width"),
            "video_height": metadata.get("height"),
            "video_fps": metadata.get("fps"),
            "video_frame_count": metadata.get("frame_count"),
            "video_size": metadata.get("size"),
            "ok_for_frontend": ok_for_frontend,
            "eligibility_reason": eligibility_reason,
        })
    except Exception as exc:
        entry["video_error"] = error_text(exc)
        entry["eligibility_reason"] = "video_encode_failed"
    return entry


def run_case(base_url, case, args, encode_video=True):
    original = resolve_original_case_config(case, args.root)
    params = original.get("params") or {}
    route_prior = original.get("route_prior") or ""
    support_ranker = original.get("support_ranker") or ""
    payload = {
        "instruction": str(case.get("instruction") or "").strip(),
        "scene_id": int(case.get("scene_id") or 16),
        "max_steps": original.get("max_steps") or args.max_steps,
        "mode": "agent",
        "confirm_control": True,
        "episode_id": str(case.get("episode_id") or ""),
        "split": "train",
        "checkpoint": args.checkpoint,
        "simulator_tool_port": args.simulator_port,
        "gpu_id": args.gpu_id,
        "save_frames": True,
        "enable_oracle_goal_homing": original.get("enable_oracle_goal_homing", False),
        "enable_oracle_path_homing": original.get("enable_oracle_path_homing", False),
        "enable_learned_segment_homing": original.get("enable_learned_segment_homing", True),
        "enable_scene16_support_ranker": original.get("enable_support_ranker", False),
        "scene16_support_ranker": support_ranker,
        "enable_scene16_route_prior": original.get("enable_route_prior", False),
        "scene16_route_prior": route_prior,
        "learned_segment_target_radius": params.get("learned_segment_target_radius", 12),
        "learned_segment_max_age": params.get("learned_segment_max_age", 34),
        "learned_segment_z_clamp": params.get("learned_segment_z_clamp", 6),
        "learned_segment_min_xy_step": params.get("learned_segment_min_xy_step", 24),
        "learned_segment_max_xy_step": params.get("learned_segment_max_xy_step", 54),
        "learned_segment_resample_worse_streak": params.get("learned_segment_resample_worse_streak", 4),
        "learned_segment_resample_regression": params.get("learned_segment_resample_regression", 55),
        "learned_segment_cruise_altitude": params.get("learned_segment_cruise_altitude", 42),
        "learned_segment_descent_altitude": params.get("learned_segment_descent_altitude", 18),
        "segment_switch_confidence": params.get("segment_switch_confidence", 0.48),
        "segment_min_steps": params.get("segment_min_steps", 3),
        "segment_progress_distance": params.get("segment_progress_distance", 22),
    }
    if original.get("segment_grounding_ridge"):
        payload["segment_grounding_ridge"] = original["segment_grounding_ridge"]
    if original.get("segment_grounding_final_ridge"):
        payload["segment_grounding_final_ridge"] = original["segment_grounding_final_ridge"]
    started = request_json(f"{base_url}/api/agent/start", payload, timeout=40)
    session_id = str(started["session_id"])
    latest = poll_session(base_url, session_id, args.case_timeout_sec, args)
    entry = {
        "runtime_profile": args.runtime_profile,
        "episode_id": payload["episode_id"],
        "instruction": payload["instruction"],
        "source_stable_pool": str(args.pool),
        "source_report": case.get("source_report"),
        "route_prior": route_prior,
        "original_config_fallback": bool(original.get("fallback")),
        "replay_session_id": session_id,
        "status": latest.get("status"),
        "success_20m_with_surface": latest.get("success_20m_with_surface"),
        "surface_aligned": latest.get("surface_aligned"),
        "distance_to_goal": latest.get("distance_to_goal"),
        "surface_clearance": latest.get("surface_clearance"),
        "stream_frame_count": latest.get("stream_frame_count"),
        "black_frame_drop_count": latest.get("black_frame_drop_count"),
        "collision_rollback_count": latest.get("collision_rollback_count"),
        "model_error": latest.get("error"),
        "stopped_for_stagnation": bool(latest.get("stopped_for_stagnation")),
        "ok_for_frontend": False,
        "eligibility_reason": "flight_not_successful_with_surface",
    }
    flight_ok, eligibility_reason = flight_eligibility(latest)
    if not flight_ok:
        entry["eligibility_reason"] = eligibility_reason
        return entry
    if not encode_video:
        entry["eligibility_reason"] = "video_pending"
        return entry
    return attach_video(base_url, entry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=82)
    parser.add_argument("--gpu-id", type=int, default=2)
    parser.add_argument("--simulator-port", type=int, default=30014)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    # The stable-pool source evaluations used 1200 steps. 900 truncates longer routes.
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--case-timeout-sec", type=int, default=600)
    parser.add_argument("--stagnation-stop-enabled", type=int, choices=(0, 1), default=1)
    parser.add_argument("--stagnation-min-steps", type=int, default=300)
    parser.add_argument("--stagnation-window-steps", type=int, default=150)
    parser.add_argument("--stagnation-min-progress-m", type=float, default=4.0)
    parser.add_argument("--stagnation-min-distance-m", type=float, default=80.0)
    parser.add_argument("--stagnation-min-resamples", type=int, default=6)
    parser.add_argument("--stagnation-min-safety-turns", type=int, default=80)
    parser.add_argument("--runtime-profile", default=RUNTIME_PROFILE)
    parser.add_argument(
        "--reset-resource-lock-errors",
        action="store_true",
        help="remove recoverable stale-lock batch errors before processing",
    )
    args = parser.parse_args()
    args.pool = args.pool if args.pool.is_absolute() else ROOT / args.pool
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    args.root = ROOT
    cases = load_cases(args.pool)
    manifest = load_manifest(args.out, args.pool.name)
    if args.reset_resource_lock_errors:
        before = list(manifest.get("entries") or [])
        manifest["entries"] = [
            entry for entry in before
            if not (
                isinstance(entry, dict)
                and entry.get("status") == "batch_error"
                and "agent resource lock exists" in str(entry.get("batch_error") or "")
            )
        ]
        removed = len(before) - len(manifest["entries"])
        if removed:
            print("RESET_RECOVERABLE_LOCK_ERRORS", removed, flush=True)
    manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_write(args.out, manifest)
    entries = {str(item.get("episode_id")): item for item in manifest.get("entries", []) if item.get("episode_id")}
    selected = cases[max(0, args.start): max(0, args.start) + max(0, args.count)]
    print("BATCH_START", json.dumps({"total": len(cases), "selected": len(selected), "manifest": str(args.out)}, ensure_ascii=False), flush=True)
    def record_entry(entry):
        entries[str(entry.get("episode_id") or "")] = entry
        manifest["entries"] = [entries[key] for key in sorted(entries)]
        manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest["completed_count"] = sum(bool(item.get("ok_for_frontend")) for item in manifest["entries"])
        manifest["failed_or_pending_count"] = len(cases) - manifest["completed_count"]
        atomic_write(args.out, manifest)
        print("CASE_RESULT", json.dumps(entry, ensure_ascii=False, sort_keys=True), flush=True)

    pending = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview-video") as video_encoder:
        for index, case in enumerate(selected, start=max(0, args.start)):
            episode_id = str(case.get("episode_id") or "")
            existing = entries.get(episode_id)
            if (
                existing
                and existing.get("runtime_profile") == args.runtime_profile
                and existing_entry_is_frontend_ready(existing)
            ):
                print("SKIP", index, episode_id, "already_ok", flush=True)
                continue
            if (
                existing
                and existing.get("runtime_profile") == args.runtime_profile
                and existing.get("status") in ("done", "error", "stopped", "batch_error")
                and not existing.get("ok_for_frontend")
            ):
                print("SKIP", index, episode_id, "already_failed_under_profile", flush=True)
                continue
            print("CASE_START", index, episode_id, flush=True)
            try:
                wait_for_no_active(args.base_url)
                entry = run_case(args.base_url, case, args, encode_video=False)
            except Exception as exc:
                entry = {
                    "episode_id": episode_id,
                    "instruction": case.get("instruction"),
                    "source_stable_pool": str(args.pool),
                    "status": "batch_error",
                    "batch_error": error_text(exc),
                    "ok_for_frontend": False,
                }
            if pending is not None:
                record_entry(pending.result())
                pending = None
            flight_ok, _ = flight_eligibility(entry)
            if flight_ok:
                pending = video_encoder.submit(attach_video, args.base_url, entry)
            else:
                record_entry(entry)
        if pending is not None:
            record_entry(pending.result())
    print("BATCH_DONE", json.dumps({"completed": manifest.get("completed_count"), "total": len(cases), "out": str(args.out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
