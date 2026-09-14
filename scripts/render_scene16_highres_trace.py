#!/usr/bin/env python3
"""Render a successful AirVLN trace at the simulator's native preview size.

This is deliberately an offline presentation renderer.  It never invokes the
model and never changes the recorded success decision: every rendered pose uses
the position stored in the successful model trace.  Heading is reconstructed
from the episode's initial rotation plus the recorded 15-degree turn actions.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import msgpackrpc
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "live_server" / "runtime"
TURN_RADIANS = math.radians(15.0)
MAX_AIRSIM_IMAGE_BYTES = 4 * 1024 * 1024


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def locate_trace(session_dir):
    candidates = sorted(session_dir.glob("model_runs/*/trace.jsonl"))
    if not candidates:
        raise RuntimeError(f"no model trace found under {session_dir}")
    return candidates[-1]


def load_trace(trace_path):
    entries = []
    for line in Path(trace_path).read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            position = item.get("position")
            if isinstance(position, list) and len(position) == 3:
                item["position"] = [float(value) for value in position]
                entries.append(item)
        except (TypeError, ValueError):
            continue
    if len(entries) < 2:
        raise RuntimeError(f"trace has too few position entries: {trace_path}")
    return entries


def initial_yaw(episode_id):
    # The AirVLN dataset serializes the initial quaternion as [w, x, y, z].
    for candidate in sorted((ROOT / "live_server").glob("**/*.json")):
        # Runtime reports are very numerous and do not contain episode geometry.
        if "runtime" in candidate.parts:
            continue
        try:
            payload = read_json(candidate)
        except (OSError, ValueError):
            continue
        episodes = payload.get("episodes") if isinstance(payload, dict) else payload
        if not isinstance(episodes, list):
            continue
        for episode in episodes:
            if not isinstance(episode, dict) or str(episode.get("episode_id") or "") != episode_id:
                continue
            rotation = episode.get("start_rotation")
            if isinstance(rotation, list) and len(rotation) == 4:
                return math.atan2(2.0 * (float(rotation[0]) * float(rotation[3])), 1.0 - 2.0 * (float(rotation[3]) ** 2))
    # The production dataset is outside the git worktree; ask the existing
    # app's dataset index only after the local convenience search above.
    sys.path.insert(0, str(ROOT / "live_server"))
    import app  # pylint: disable=import-outside-toplevel

    for record in app.cached_dataset_episode_index():
        episode = record.get("episode") or {}
        if str(episode.get("episode_id") or "") != episode_id:
            continue
        rotation = episode.get("start_rotation")
        if isinstance(rotation, list) and len(rotation) == 4:
            return math.atan2(2.0 * (float(rotation[0]) * float(rotation[3])), 1.0 - 2.0 * (float(rotation[3]) ** 2))
    raise RuntimeError(f"could not resolve start_rotation for {episode_id}")


def reconstructed_poses(entries, start_yaw):
    yaw = float(start_yaw)
    poses = []
    for index, entry in enumerate(entries):
        action = str(entry.get("action") or "").upper()
        # trace rows report the pose after their action; the initial row uses
        # INIT and therefore preserves the start yaw.
        if index > 0:
            if action == "TURN_LEFT":
                yaw += TURN_RADIANS
            elif action == "TURN_RIGHT":
                yaw -= TURN_RADIANS
        poses.append((entry["position"], yaw, entry))
    return poses


def wait_for_scene(tool_port, scene_id, timeout):
    client = msgpackrpc.Client(msgpackrpc.Address("127.0.0.1", tool_port), timeout=30)
    started = time.monotonic()
    result = client.call("reopen_scenes", "127.0.0.1", [int(scene_id)])
    if not result or not result[0]:
        raise RuntimeError(f"simulator tool did not open scene {scene_id}: {result}")
    host, ports = result[1]
    if not ports:
        raise RuntimeError(f"simulator tool returned no AirSim port: {result}")
    if isinstance(host, bytes):
        host = host.decode("utf-8")
    return str(host), int(ports[0]), round(time.monotonic() - started, 2)


def connect_airsim(host, port, timeout):
    import airsim  # pylint: disable=import-outside-toplevel

    # AirSim's bundled msgpackrpc initializes an Unpacker with a 1 MB binary
    # limit. A 1280x720 JPEG commonly exceeds that even though it is much
    # smaller than the 2.76 MB raw RGB frame. Patch only this offline process.
    import msgpackrpc.transport.tcp as rpc_tcp  # pylint: disable=import-outside-toplevel

    original_unpacker = rpc_tcp.msgpack.Unpacker

    def highres_unpacker(*args, **kwargs):
        kwargs.setdefault("max_bin_len", MAX_AIRSIM_IMAGE_BYTES)
        kwargs.setdefault("max_buffer_size", MAX_AIRSIM_IMAGE_BYTES * 2)
        return original_unpacker(*args, **kwargs)

    rpc_tcp.msgpack.Unpacker = highres_unpacker

    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            client = airsim.VehicleClient(ip=host, port=port, timeout_value=10)
            client.confirmConnection()
            return client
        except Exception as exc:  # AirSim may be starting after the tool response.
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"AirSim did not become ready on {host}:{port}: {last_error}")


def capture(client, position, yaw, camera_name):
    import airsim  # pylint: disable=import-outside-toplevel

    pose = airsim.Pose(
        airsim.Vector3r(*position),
        airsim.to_quaternion(0.0, 0.0, yaw),
    )
    client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name="Drone_1")
    response = client.simGetImages(
        # Raw 1280x720 RGB is 2.76 MB per RPC and stalls this AirSim build.
        # AirSim's JPEG response keeps the same render resolution while making
        # the transport bounded and responsive.
        [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=True)],
        vehicle_name="Drone_1",
    )[0]
    width, height = int(response.width), int(response.height)
    raw = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if width <= 0 or height <= 0 or image is None or image.shape[1] != width or image.shape[0] != height:
        raise RuntimeError(f"invalid JPEG image response {width}x{height}, {raw.size} bytes")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_video(frame_dir, video_path, fps):
    import imageio_ffmpeg  # pylint: disable=import-outside-toplevel

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg, "-y", "-framerate", str(fps), "-i", str(frame_dir / "frame_%06d.jpg"),
        "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(video_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def highres_video_metadata(metadata, video_path):
    return {
        "cached": True,
        "codec": "h264",
        "requested_video_codec": "h264",
        "h264_available": True,
        "format": "mp4",
        "mimetype": "video/mp4",
        "fps": metadata["fps"],
        "frame_count": metadata["frame_count"],
        "source_frame_count": metadata["frame_count"],
        "original_source_frame_count": metadata["frame_count"],
        "stride": 1,
        "smooth": False,
        "smooth_multiplier": 1.0,
        "interp": "hold",
        "width": metadata["width"],
        "height": metadata["height"],
        "max_width": metadata["width"],
        "source_width": metadata["width"],
        "source_height": metadata["height"],
        "upscale": False,
        "sharpen": 0.0,
        "size": video_path.stat().st_size,
    }


def update_manifest(manifest_path, metadata, video_path):
    if not manifest_path:
        return
    try:
        payload = read_json(manifest_path)
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ValueError("entries is not a list")
        episode_id = str(metadata["episode_id"])
        entry = next((item for item in entries if str(item.get("episode_id") or "") == episode_id), None)
        if entry is None:
            raise ValueError(f"episode {episode_id} is absent from manifest")
        session_id = metadata["session_id"]
        entry["video_path"] = f"/agent-sessions/{session_id}/replay/{video_path.name}"
        entry["video_metadata"] = highres_video_metadata(metadata, video_path)
        entry["video_width"] = metadata["width"]
        entry["video_height"] = metadata["height"]
        entry["video_fps"] = metadata["fps"]
        entry["video_frame_count"] = metadata["frame_count"]
        entry["video_size"] = video_path.stat().st_size
        entry["video_reencoded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        temporary = Path(f"{manifest_path}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"could not update manifest {manifest_path}: {exc}") from exc


def render_session(session_id, args, client, scene_open_sec):
    session_dir = RUNTIME / "agent_sessions" / session_id
    summary = read_json(session_dir / "summary.json")
    trace_path = locate_trace(session_dir)
    entries = load_trace(trace_path)
    yaw = initial_yaw(str(summary["episode_id"]))
    poses = reconstructed_poses(entries, yaw)
    # Keep the finished asset in the established replay directory so the
    # frontend's existing static replay route can serve it without a bypass.
    out_dir = session_dir / "replay"
    frame_dir = out_dir / "frames"
    video_path = out_dir / "replay_h264_highres_8fps.mp4"
    if args.skip_existing and video_path.exists() and video_path.stat().st_size > 0:
        print(f"SKIPPED {session_id} existing={video_path}", flush=True)
        return None
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    started = time.monotonic()
    for index, (position, heading, entry) in enumerate(poses):
        image = capture(client, position, heading, args.camera)
        path = frame_dir / f"frame_{index:06d}.jpg"
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 96]):
            raise RuntimeError(f"failed to write {path}")
        rendered.append({"trace_step": entry.get("step"), "action": entry.get("action"), "position": position, "yaw": heading})
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(poses):
            print(f"RENDERED {session_id} {index + 1}/{len(poses)} {image.shape[1]}x{image.shape[0]}", flush=True)

    write_video(frame_dir, video_path, args.fps)
    metadata = {
        "session_id": session_id,
        "episode_id": summary["episode_id"],
        "scene_id": summary["scene_id"],
        "trace": str(trace_path),
        "endpoint": poses[-1][0],
        "recorded_endpoint": summary.get("position"),
        "recorded_distance_to_goal": summary.get("distance_to_goal"),
        "success_20m_with_surface": summary.get("success_20m_with_surface"),
        "camera": args.camera,
        "frame_count": len(poses),
        "fps": args.fps,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "scene_open_sec": scene_open_sec,
        "render_sec": round(time.monotonic() - started, 2),
        "video": str(video_path),
        "poses": rendered,
    }
    (out_dir / "replay_h264_highres_8fps.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    update_manifest(args.manifest, metadata, video_path)
    if not args.keep_frames:
        shutil.rmtree(frame_dir)
    print(
        f"COMPLETE {session_id} frames={metadata['frame_count']} render_sec={metadata['render_sec']} "
        f"endpoint={metadata['endpoint']}",
        flush=True,
    )
    return metadata


def eligible_session_ids(manifest_path):
    entries = read_json(manifest_path).get("entries")
    if not isinstance(entries, list):
        raise RuntimeError(f"invalid manifest entries: {manifest_path}")
    return [
        str(item.get("replay_session_id") or "").strip()
        for item in entries
        if isinstance(item, dict)
        and item.get("ok_for_frontend")
        and item.get("success_20m_with_surface") is True
        and str(item.get("replay_session_id") or "").strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", action="append", default=[])
    parser.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--manifest", type=Path, default=RUNTIME / "scene16_stable82_preview_manifest_20260825.json")
    parser.add_argument("--tool-port", type=int, default=30014)
    parser.add_argument("--camera", default="preview_0")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--scene-timeout", type=float, default=150.0)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    session_ids = list(args.session_id)
    if args.all_eligible:
        session_ids.extend(eligible_session_ids(args.manifest))
    session_ids = list(dict.fromkeys(session_id for session_id in session_ids if session_id))
    if not session_ids:
        parser.error("provide --session-id or --all-eligible")

    client = None
    active_scene_id = None
    scene_open_sec = 0.0
    failures = []
    for session_id in session_ids:
        try:
            summary = read_json(RUNTIME / "agent_sessions" / session_id / "summary.json")
            scene_id = int(summary["scene_id"])
            if client is None or active_scene_id != scene_id:
                host, scene_port, scene_open_sec = wait_for_scene(args.tool_port, scene_id, args.scene_timeout)
                print(f"SCENE_READY scene={scene_id} {host}:{scene_port} open_sec={scene_open_sec}", flush=True)
                client = connect_airsim(host, scene_port, args.scene_timeout)
                active_scene_id = scene_id
                print("AIRSIM_CONNECTED", flush=True)
            render_session(session_id, args, client, scene_open_sec)
        except Exception as exc:  # Keep later verified demonstrations rendering.
            failures.append({"session_id": session_id, "error": str(exc)})
            print(f"FAILED {session_id}: {exc}", flush=True)
    print(json.dumps({"complete": len(session_ids) - len(failures), "failed": failures}, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
