import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
from collections import deque
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageEnhance, ImageStat


DEFAULT_PHASE_A_SPLIT = "train_scene16_segments_high_conf_1k"
DEFAULT_PHASE_A_EPISODE_ID = "304SM51WAC2LJX66KU06PFQOK2ISBK__seg006"
DEFAULT_PHASE_A_CKPT = (
    os.environ.get("AIRVLN_PHASE_A_CKPT", "")
)
DEFAULT_PHASE_A_SIMULATOR_PORT = 30003
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _flight_target_telemetry_manifest(meta):
    manifest = meta.get("flight_target_telemetry") if isinstance(meta, dict) else None
    if not isinstance(manifest, dict) or set(manifest) != {"path", "sha256", "event_count"}:
        return None
    if (
        not isinstance(manifest["path"], str)
        or not manifest["path"]
        or not isinstance(manifest["sha256"], str)
        or not _SHA256_RE.fullmatch(manifest["sha256"])
        or isinstance(manifest["event_count"], bool)
        or not isinstance(manifest["event_count"], int)
        or manifest["event_count"] < 0
    ):
        return None
    return dict(manifest)


def _percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    index = round((float(pct) / 100.0) * (len(ordered) - 1))
    index = min(len(ordered) - 1, max(0, int(index)))
    return ordered[index]


def stream_pacing_metrics(frame_times, window_size=45):
    times = [float(item) for item in frame_times or []]
    if len(times) < 2:
        return {
            "stream_recent_frame_count": len(times),
            "stream_recent_fps": None,
            "stream_recent_gap_avg_sec": None,
            "stream_recent_gap_p95_sec": None,
            "stream_recent_gap_max_sec": None,
            "stream_recent_gaps_over_0_2s": 0,
            "stream_recent_gaps_over_0_5s": 0,
            "stream_pacing_label": "waiting",
        }
    times = times[-max(2, int(window_size)):]
    gaps = [right - left for left, right in zip(times, times[1:]) if right > left]
    elapsed = times[-1] - times[0]
    fps = ((len(times) - 1) / elapsed) if elapsed > 0 else None
    max_gap = max(gaps) if gaps else None
    over_02 = sum(1 for gap in gaps if gap > 0.2)
    over_05 = sum(1 for gap in gaps if gap > 0.5)
    if max_gap is None:
        label = "waiting"
    elif over_05 > 0 or max_gap > 0.5:
        label = "stalled"
    elif over_02 > 2 or (fps is not None and fps < 18.0):
        label = "degraded"
    else:
        label = "smooth"
    return {
        "stream_recent_frame_count": len(times),
        "stream_recent_fps": round(fps, 1) if fps is not None else None,
        "stream_recent_gap_avg_sec": round(sum(gaps) / len(gaps), 4) if gaps else None,
        "stream_recent_gap_p95_sec": round(_percentile(gaps, 95), 4) if gaps else None,
        "stream_recent_gap_max_sec": round(max_gap, 4) if max_gap is not None else None,
        "stream_recent_gaps_over_0_2s": over_02,
        "stream_recent_gaps_over_0_5s": over_05,
        "stream_pacing_label": label,
    }


class AgentSession:
    ACTION_ID_TO_NAME = {
        0: "STOP",
        1: "MOVE_FORWARD",
        2: "TURN_LEFT",
        3: "TURN_RIGHT",
        4: "GO_UP",
        5: "GO_DOWN",
        6: "MOVE_LEFT",
        7: "MOVE_RIGHT",
    }

    TERMINAL_STATUSES = ("done", "error", "stopped")

    def __init__(
        self,
        session_id,
        instruction,
        scene_id,
        max_steps,
        runtime_dir,
        model_script,
        episode_id=None,
        split=DEFAULT_PHASE_A_SPLIT,
        checkpoint=None,
        simulator_tool_port=DEFAULT_PHASE_A_SIMULATOR_PORT,
        save_frames=True,
        gpu_id=0,
        enable_oracle_goal_homing=False,
        enable_oracle_path_homing=False,
        enable_learned_segment_homing=False,
        enable_scene16_support_ranker=False,
        scene16_support_ranker="",
        enable_scene16_route_prior=False,
        scene16_route_prior="",
        segment_grounding_ridge="",
        model_run_id=None,
        model_extra_args=None,
        v014_namespace=None,
    ):
        self.session_id = session_id
        self.instruction = instruction
        self.scene_id = int(scene_id)
        self.max_steps = int(max_steps)
        self.mode = "agent"
        self.episode_id = episode_id or DEFAULT_PHASE_A_EPISODE_ID
        self.split = split or DEFAULT_PHASE_A_SPLIT
        self.checkpoint = checkpoint or DEFAULT_PHASE_A_CKPT
        self.simulator_tool_port = int(simulator_tool_port)
        self.save_frames = bool(save_frames)
        self.gpu_id = int(gpu_id)
        self.enable_oracle_goal_homing = bool(enable_oracle_goal_homing)
        self.enable_oracle_path_homing = bool(enable_oracle_path_homing)
        self.enable_learned_segment_homing = bool(enable_learned_segment_homing)
        self.enable_scene16_support_ranker = bool(enable_scene16_support_ranker)
        self.scene16_support_ranker = str(scene16_support_ranker or "")
        self.enable_scene16_route_prior = bool(enable_scene16_route_prior)
        self.scene16_route_prior = str(scene16_route_prior or "")
        self.segment_grounding_ridge = str(segment_grounding_ridge or "")
        self.model_run_id = model_run_id
        self.model_extra_args = dict(model_extra_args or {})
        self.cuda_visible_devices = str(os.environ.get("AIRVLN_AGENT_CUDA_VISIBLE_DEVICES", self.gpu_id))
        self.trainer_gpu_device = int(os.environ.get("AIRVLN_AGENT_TRAINER_GPU_DEVICE", "0"))
        self.runtime_dir = Path(runtime_dir)
        self.model_script = Path(model_script)
        self.session_dir = self.runtime_dir / "agent_sessions" / session_id
        self.frames_dir = self.session_dir / "frames"
        self.preview_frames_dir = self.session_dir / "preview_frames"
        self.model_out_root = self.session_dir / "model_runs"
        self.logs_dir = self.session_dir / "logs"
        self.v014_namespace = v014_namespace
        if v014_namespace is None:
            for item in (self.frames_dir, self.preview_frames_dir, self.model_out_root, self.logs_dir):
                item.mkdir(parents=True, exist_ok=True)
        else:
            expected = {
                "session_dir": self.session_dir,
                "model_out_root": self.model_out_root,
                "model_run_dir": self.model_out_root / model_run_id,
            }
            if any(v014_namespace.get(key) != value for key, value in expected.items()):
                raise ValueError("invalid v014 pre-reserved namespace")
            for item in (self.session_dir, self.frames_dir, self.preview_frames_dir, self.model_out_root, self.logs_dir, expected["model_run_dir"]):
                info = os.lstat(item)
                if os.path.islink(item) or not os.path.isdir(item) or bool(getattr(info, "st_file_attributes", 0) & 0x400):
                    raise ValueError("v014 namespace was redirected")
                if v014_namespace["identities"].get(str(item)) != (info.st_dev, info.st_ino):
                    raise ValueError("v014 namespace identity changed")

        self.status = "initializing"
        self.error = None
        self.step = 0
        self.action = "INIT"
        self.position = None
        self.distance_to_goal = None
        self.initial_distance = None
        self.best_distance = None
        self.best_distance_step = None
        self.success_distance = 20.0
        self.success_20m = False
        self.success_20m_with_altitude = False
        self.success_20m_with_surface = False
        self.oracle_success_20m = False
        self.distance_3d_to_goal = None
        self.vertical_error_to_goal = None
        self.altitude_aligned = False
        self.surface_clearance = None
        self.surface_aligned = False
        self.surface_probe_ok = False
        self.surface_probe_error = None
        self.progress_to_goal = None
        self.navigation_phase = "initializing"
        self.goal_position = None
        self.path = []
        self.safety_override = None
        self.collision_rollback_count = 0
        self.last_collision = None
        self.stop_reason = None
        self.last_frame = None
        self.last_frame_at = 0.0
        self.first_frame_at = 0.0
        self.frame_sequence = 0
        try:
            frame_queue_maxlen = int(os.environ.get("AIRVLN_LIVE_FRAME_QUEUE_MAXLEN", "120"))
        except (TypeError, ValueError):
            frame_queue_maxlen = 120
        self.frame_queue_maxlen = min(240, max(8, frame_queue_maxlen))
        self.frame_queue = deque(maxlen=self.frame_queue_maxlen)
        self.frame_times = deque(maxlen=120)
        self.use_model_frames_for_stream = os.environ.get("AIRVLN_LIVE_USE_MODEL_FRAMES", "0").lower() not in (
            "0",
            "false",
            "no",
        )
        self.black_frame_drop_count = 0
        self.low_light_frame_enhance_count = 0
        self.created_at = time.time()
        self.started_at = None
        self.ended_at = None
        self.process = None
        self.model_run_dir = self.model_out_root / model_run_id if model_run_id else None
        self.flight_target_telemetry = None
        self._bridge_frame_dir = None
        self._bridge_frame_index = 0
        self._trace_cache_path = None
        self._trace_cache_offset = 0
        self._trace_items_by_step = {}
        self._refresh_trace_path = None
        self._refresh_trace_offset = 0
        self._last_model_frame_scan_at = 0.0
        try:
            self.model_frame_scan_interval = float(os.environ.get("AIRVLN_LIVE_MODEL_FRAME_SCAN_INTERVAL", "0.20"))
        except (TypeError, ValueError):
            self.model_frame_scan_interval = 0.20
        self.model_frame_scan_interval = min(1.0, max(0.02, self.model_frame_scan_interval))
        self.model_log_path = self.logs_dir / "model_live_once.log"
        self.model_log_tail = None
        self.stop_event = threading.Event()
        self.frame_condition = threading.Condition()
        self.acquired_locks = []
        self.thread = threading.Thread(target=self.run, name=f"agent-session-{session_id}", daemon=True)

    def start(self):
        self.write_meta()
        self.thread.start()

    def write_meta(self):
        payload = {
            "session_id": self.session_id,
            "mode": self.mode,
            "instruction": self.instruction,
            "episode_id": self.episode_id,
            "split": self.split,
            "scene_id": self.scene_id,
            "max_steps": self.max_steps,
            "checkpoint": self.checkpoint,
            "simulator_tool_port": self.simulator_tool_port,
            "save_frames": self.save_frames,
            "gpu_id": self.gpu_id,
            "enable_oracle_goal_homing": self.enable_oracle_goal_homing,
            "enable_oracle_path_homing": self.enable_oracle_path_homing,
            "enable_learned_segment_homing": self.enable_learned_segment_homing,
            "segment_grounding_ridge": self.segment_grounding_ridge,
            "model_run_id": self.model_run_id,
            "flight_target_telemetry": self.flight_target_telemetry,
            "cuda_visible_devices": self.cuda_visible_devices,
            "trainer_gpu_device": self.trainer_gpu_device,
            "created_at": self.created_at,
        }
        (self.session_dir / "meta.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def append_trace(self, extra=None):
        item = {
            "time": time.time(),
            "step": self.step,
            "action": self.action,
            "position": self.position,
            "distance_to_goal": self.distance_to_goal,
            "initial_distance": self.initial_distance,
            "best_distance": self.best_distance,
            "best_distance_step": self.best_distance_step,
            "success_distance": self.success_distance,
            "success_20m": self.success_20m,
            "success_20m_with_altitude": self.success_20m_with_altitude,
            "success_20m_with_surface": self.success_20m_with_surface,
            "oracle_success_20m": self.oracle_success_20m,
            "distance_3d_to_goal": self.distance_3d_to_goal,
            "vertical_error_to_goal": self.vertical_error_to_goal,
            "altitude_aligned": self.altitude_aligned,
            "surface_clearance": self.surface_clearance,
            "surface_aligned": self.surface_aligned,
            "surface_probe_ok": self.surface_probe_ok,
            "surface_probe_error": self.surface_probe_error,
            "progress_to_goal": self.progress_to_goal,
            "navigation_phase": self.navigation_phase,
            "goal_position": self.goal_position,
            "safety_override": self.safety_override,
            "collision_rollback_count": self.collision_rollback_count,
            "last_collision": self.last_collision,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "error": self.error,
        }
        if extra:
            item.update(extra)
        with (self.session_dir / "trajectory.jsonl").open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")

    def snapshot(self):
        self._refresh_path_from_model_trace()
        frame_times = list(self.frame_times)
        frame_intervals = [
            current - previous
            for previous, current in zip(frame_times, frame_times[1:])
            if 0 < current - previous <= 1.5
        ]
        median_interval = None
        if frame_intervals:
            ordered_intervals = sorted(frame_intervals)
            median_interval = ordered_intervals[len(ordered_intervals) // 2]
        pacing = stream_pacing_metrics(frame_times)
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "instruction": self.instruction,
            "episode_id": self.episode_id,
            "split": self.split,
            "scene_id": self.scene_id,
            "max_steps": self.max_steps,
            "checkpoint": self.checkpoint,
            "gpu_id": self.gpu_id,
            "enable_oracle_goal_homing": self.enable_oracle_goal_homing,
            "enable_oracle_path_homing": self.enable_oracle_path_homing,
            "enable_learned_segment_homing": self.enable_learned_segment_homing,
            "segment_grounding_ridge": self.segment_grounding_ridge,
            "cuda_visible_devices": self.cuda_visible_devices,
            "trainer_gpu_device": self.trainer_gpu_device,
            "status": self.status,
            "error": self.error,
            "step": self.step,
            "action": self.action,
            "position": self.position,
            "distance_to_goal": self.distance_to_goal,
            "initial_distance": self.initial_distance,
            "best_distance": self.best_distance,
            "best_distance_step": self.best_distance_step,
            "success_distance": self.success_distance,
            "success_20m": self.success_20m,
            "success_20m_with_altitude": self.success_20m_with_altitude,
            "success_20m_with_surface": self.success_20m_with_surface,
            "oracle_success_20m": self.oracle_success_20m,
            "distance_3d_to_goal": self.distance_3d_to_goal,
            "vertical_error_to_goal": self.vertical_error_to_goal,
            "altitude_aligned": self.altitude_aligned,
            "surface_clearance": self.surface_clearance,
            "surface_aligned": self.surface_aligned,
            "surface_probe_ok": self.surface_probe_ok,
            "surface_probe_error": self.surface_probe_error,
            "progress_to_goal": self.progress_to_goal,
            "navigation_phase": self.navigation_phase,
            "goal_position": self.goal_position,
            "safety_override": self.safety_override,
            "collision_rollback_count": self.collision_rollback_count,
            "last_collision": self.last_collision,
            "stop_reason": self.stop_reason,
            "last_frame_at": self.last_frame_at,
            "frame_sequence": self.frame_sequence,
            "frame_queue_maxlen": self.frame_queue_maxlen,
            "stream_frame_count": self.frame_sequence,
            "stream_fps": round(1.0 / median_interval, 1) if median_interval else None,
            "stream_average_fps": round(
                (self.frame_sequence - 1) / (self.last_frame_at - self.first_frame_at), 1
            ) if self.frame_sequence > 1 and self.last_frame_at > self.first_frame_at else None,
            "stream_frame_interval_ms": round(median_interval * 1000.0, 1) if median_interval else None,
            **pacing,
            "black_frame_drop_count": self.black_frame_drop_count,
            "low_light_frame_enhance_count": self.low_light_frame_enhance_count,
            "model_run_dir": str(self.model_run_dir) if self.model_run_dir else None,
            "model_run_id": self.model_run_id,
            "flight_target_telemetry": self.flight_target_telemetry,
            "model_log": str(self.model_log_path),
            "model_frame_scan_interval_sec": self.model_frame_scan_interval,
            "model_log_tail": self.model_log_tail,
            "session_dir": str(self.session_dir),
            "locks": [str(path) for path in self.acquired_locks],
            "age_sec": round(time.time() - self.created_at, 1),
        }

    def set_frame(self, data, captured_at=None):
        with self.frame_condition:
            self.last_frame = data
            frame_at = float(captured_at or time.time())
            if self.last_frame_at > 0:
                frame_at = max(frame_at, self.last_frame_at + 1e-6)
            self.last_frame_at = frame_at
            if self.first_frame_at <= 0:
                self.first_frame_at = frame_at
            self.frame_times.append(self.last_frame_at)
            self.frame_sequence += 1
            self.frame_queue.append((self.frame_sequence, self.last_frame_at, data))
            self.frame_condition.notify_all()

    def prepare_stream_frame(self, data):
        try:
            image = Image.open(BytesIO(data)).convert("RGB")
            gray = image.convert("L").resize((32, 32))
            stat = ImageStat.Stat(gray)
            mean_luma = float(stat.mean[0])
            std_luma = float(stat.stddev[0])
            histogram = gray.histogram()
            black_ratio = float(sum(histogram[:8])) / float(32 * 32)
            if mean_luma < 3.0 or (mean_luma < 10.0 and std_luma < 2.0) or black_ratio > 0.92:
                self.black_frame_drop_count += 1
                return None
            if mean_luma >= 48.0:
                return data
            factor = min(2.25, max(1.15, 58.0 / max(mean_luma, 1.0)))
            image = ImageEnhance.Brightness(image).enhance(factor)
            image = ImageEnhance.Contrast(image).enhance(1.05)
            output = BytesIO()
            image.save(output, format="JPEG", quality=80)
            self.low_light_frame_enhance_count += 1
            return output.getvalue()
        except Exception:
            self.black_frame_drop_count += 1
            return None

    def get_frame_packet(self, after_sequence=0, timeout=5.0, latest_only=False):
        deadline = time.monotonic() + timeout
        with self.frame_condition:
            while self.status not in self.TERMINAL_STATUSES:
                candidates = [item for item in self.frame_queue if item[0] > after_sequence]
                packet = candidates[-1] if latest_only and candidates else (candidates[0] if candidates else None)
                if packet is not None:
                    return packet
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.frame_condition.wait(timeout=remaining)
            candidates = [item for item in self.frame_queue if item[0] > after_sequence]
            return candidates[-1] if latest_only and candidates else (candidates[0] if candidates else None)

    def get_frame(self, timeout=5.0):
        with self.frame_condition:
            if self.last_frame is None and self.status not in self.TERMINAL_STATUSES:
                self.frame_condition.wait(timeout=timeout)
            return self.last_frame

    def request_stop(self):
        self.stop_event.set()
        self.stop_reason = "user_stop"
        if self.status not in self.TERMINAL_STATUSES:
            self.status = "stopped"
        self._terminate_process()
        with self.frame_condition:
            self.frame_condition.notify_all()

    def run(self):
        self.started_at = time.time()
        try:
            self.status = "acquiring_resources"
            self._acquire_lock()
            self.status = "running"
            self.append_trace({"event": "start"})
            self._run_model_subprocess()
            if self.status == "running":
                self.status = "done"
                self.stop_reason = self.stop_reason or "model_finished"
        except Exception as exc:
            self.status = "error"
            self.error = str(exc)
            self.stop_reason = self.stop_reason or "error"
            (self.session_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            self.ended_at = time.time()
            self._release_lock()
            self._write_summary()
            self.append_trace({"event": "end"})
            with self.frame_condition:
                self.frame_condition.notify_all()

    def _lock_dir(self):
        locks_dir = self.runtime_dir / "locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        return locks_dir

    def _lock_paths(self):
        locks_dir = self._lock_dir()
        endpoint = f"{os.environ.get('AIRSIM_HOST', '127.0.0.1')}-{self.simulator_tool_port}"
        return [
            locks_dir / "agent-active.lock",
            locks_dir / f"airsim-{endpoint}.lock",
            locks_dir / "drone-Drone_1.lock",
            locks_dir / f"gpu-{self.gpu_id}.lock",
        ]

    def _acquire_lock(self):
        payload = json.dumps(
            {
                "session_id": self.session_id,
                "pid": os.getpid(),
                "time": time.time(),
                "scene_id": self.scene_id,
                "episode_id": self.episode_id,
            },
            indent=2,
        )
        for lock_path in self._lock_paths():
            if lock_path.exists():
                try:
                    data = json.loads(lock_path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
                active = data.get("session_id")
                if active != self.session_id:
                    raise RuntimeError(f"resource lock exists: {lock_path.name} session={active or 'unknown'}")
                self.acquired_locks.append(lock_path)
                continue
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as fp:
                    fp.write(payload)
                self.acquired_locks.append(lock_path)
            except FileExistsError:
                raise RuntimeError(f"resource lock exists: {lock_path.name}")

    def _release_lock(self):
        for lock_path in list(self.acquired_locks):
            try:
                if lock_path.exists():
                    data = json.loads(lock_path.read_text(encoding="utf-8"))
                    if data.get("session_id") == self.session_id:
                        lock_path.unlink()
            except Exception:
                pass
        self.acquired_locks = []

    def _run_model_subprocess(self):
        if self.v014_namespace is not None:
            for item in (
                self.session_dir,
                self.frames_dir,
                self.preview_frames_dir,
                self.model_out_root,
                self.model_run_dir,
                self.logs_dir,
            ):
                info = os.lstat(item)
                if os.path.islink(item) or not os.path.isdir(item) or bool(getattr(info, "st_file_attributes", 0) & 0x400) or self.v014_namespace["identities"].get(str(item)) != (info.st_dev, info.st_ino):
                    raise RuntimeError("v014 namespace identity changed before model launch")
        python_executable = self._model_python_executable()
        cmd = [
            python_executable,
            str(self.model_script),
            "--episode-id",
            self.episode_id,
            "--split",
            self.split,
            "--ckpt",
            self.checkpoint,
            "--scene-id",
            str(self.scene_id),
            "--instruction",
            self.instruction,
            "--max-steps",
            str(self.max_steps),
            "--simulator-tool-port",
            str(self.simulator_tool_port),
            "--out-root",
            str(self.model_out_root),
            "--trainer_gpu_device",
            str(self.trainer_gpu_device),
        ]
        if self.model_run_id:
            cmd += ["--model-run-id", self.model_run_id]
        if self.v014_namespace is not None:
            cmd.append("--pre-reserved-model-run")
        if self.enable_oracle_goal_homing:
            cmd.append("--enable-oracle-goal-homing")
        if self.enable_oracle_path_homing:
            cmd.append("--enable-oracle-path-homing")
        if self.enable_learned_segment_homing:
            cmd.append("--enable-learned-segment-homing")
        if self.enable_scene16_support_ranker:
            cmd.append("--enable-scene16-support-ranker")
        if self.scene16_support_ranker:
            cmd += ["--scene16-support-ranker", self.scene16_support_ranker]
        if self.enable_scene16_route_prior:
            cmd.append("--enable-scene16-route-prior")
        if self.scene16_route_prior:
            cmd += ["--scene16-route-prior", self.scene16_route_prior]
        if self.segment_grounding_ridge:
            cmd += ["--segment-grounding-ridge", self.segment_grounding_ridge]
        for key, value in sorted(self.model_extra_args.items()):
            if value is None or value == "" or value is False:
                continue
            flag = "--" + str(key).strip().replace("_", "-")
            cmd.append(flag)
            if value is not True:
                cmd.append(str(value))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        env["OMP_NUM_THREADS"] = str(os.environ.get("AIRVLN_AGENT_OMP_NUM_THREADS", "4"))
        env["AIRVLN_LIVE_PREVIEW_DIR"] = str(self.preview_frames_dir)
        # Preview frames are captured only at the final pose. Keep the motion
        # sweep distance-bounded, but do not create twelve redundant pose RPCs
        # for every small model action.
        env["AIRVLN_LIVE_POSE_TWEEN_STEPS"] = str(os.environ.get("AIRVLN_LIVE_POSE_TWEEN_STEPS", "1"))
        env["AIRVLN_LIVE_POSE_TWEEN_MAX_DISTANCE"] = str(os.environ.get("AIRVLN_LIVE_POSE_TWEEN_MAX_DISTANCE", "120"))
        env["AIRVLN_LIVE_POSE_TWEEN_METERS_PER_STEP"] = str(os.environ.get("AIRVLN_LIVE_POSE_TWEEN_METERS_PER_STEP", "4"))
        env["AIRVLN_LIVE_POSE_TWEEN_MAX_STEPS"] = str(os.environ.get("AIRVLN_LIVE_POSE_TWEEN_MAX_STEPS", "36"))
        # Keep the model's front_0 input at 224px, but retain the separate
        # preview_0 capture at its native 768x432 presentation resolution.
        env["AIRVLN_LIVE_JPEG_QUALITY"] = str(os.environ.get("AIRVLN_LIVE_JPEG_QUALITY", "90"))
        env["AIRVLN_LIVE_PREVIEW_MAX_WIDTH"] = str(os.environ.get("AIRVLN_LIVE_PREVIEW_MAX_WIDTH", "768"))
        env["AIRVLN_LIVE_PREVIEW_CAMERA"] = str(os.environ.get("AIRVLN_LIVE_PREVIEW_CAMERA", "preview_0"))
        env["AIRVLN_LIVE_PREVIEW_CAPTURE_IN_OBSERVATION"] = str(
            os.environ.get("AIRVLN_LIVE_PREVIEW_CAPTURE_IN_OBSERVATION", "0")
        )
        env["AIRVLN_LIVE_LOW_LIGHT_ENHANCE"] = str(os.environ.get("AIRVLN_LIVE_LOW_LIGHT_ENHANCE", "0"))
        env["AIRVLN_LIVE_MODEL_FRAME_STRIDE"] = str(os.environ.get("AIRVLN_LIVE_MODEL_FRAME_STRIDE", "0"))
        env["AIRVLN_LIVE_PREVIEW_DEPTH_GUARD"] = str(os.environ.get("AIRVLN_LIVE_PREVIEW_DEPTH_GUARD", "0"))
        # The fast path keeps collision checks on every action and performs an
        # exact pose read periodically; set to 0 to force the older RPC-heavy
        # verification behavior for diagnostics.
        env["AIRVLN_LIVE_FAST_POSE_VERIFY"] = str(os.environ.get("AIRVLN_LIVE_FAST_POSE_VERIFY", "0"))
        env["AIRVLN_LIVE_FAST_POSE_VERIFY_INTERVAL"] = str(
            os.environ.get("AIRVLN_LIVE_FAST_POSE_VERIFY_INTERVAL", "8")
        )
        # Only enabled by the controlled single-scene presentation batch.
        # The session lock guarantees exclusive ownership of this simulator.
        env["AIRVLN_ATTACH_EXISTING_SCENE"] = str(
            os.environ.get("AIRVLN_ATTACH_EXISTING_SCENE", "0")
        )
        env.pop("LOCAL_RANK", None)
        env.pop("RANK", None)
        env.pop("WORLD_SIZE", None)
        with self.model_log_path.open("wb") as log:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.model_script.parent.parent),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
            self._monitor_process()

    def _model_python_executable(self):
        override_python = os.environ.get("AIRVLN_AGENT_PYTHON")
        if override_python:
            return override_python

        marker_path = Path(os.environ.get("AIRVLN_BLACKWELL_RUNTIME_MARKER", str(self.runtime_dir / "blackwell_probe" / "sm120_ok.json")))
        if marker_path.exists():
            try:
                info = json.loads(marker_path.read_text(encoding="utf-8"))
                python_path = Path(str(info.get("python") or ""))
                if info.get("ok") and python_path.exists():
                    return str(python_path)
            except Exception:
                pass
        return sys_executable()

    def _monitor_process(self):
        seen = set()
        while True:
            if self.stop_event.is_set():
                self.status = "stopped"
                self.stop_reason = "user_stop"
                self._terminate_process()
                break

            self._bridge_preview_frames()
            self._bridge_new_frames(seen)
            code = self.process.poll()
            if code is not None:
                self._bridge_preview_frames()
                self._bridge_new_frames(seen)
                self._load_final_model_state()
                if code != 0 and self.status != "stopped":
                    self.status = "error"
                    self.stop_reason = "model_process_failed"
                    self.model_log_tail = self._read_model_log_tail()
                    detail = f": {self.model_log_tail}" if self.model_log_tail else ""
                    self.error = f"model_live_once exited with code {code}{detail}"
                break
            time.sleep(0.02)

    def _read_model_log_tail(self, max_chars=2000):
        if not self.model_log_path.exists():
            return None
        try:
            text = self.model_log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        text = text.strip()
        if not text:
            return None
        return text[-max_chars:]

    def _terminate_process(self):
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

    def _bridge_new_frames(self, seen):
        now = time.monotonic()
        if now - self._last_model_frame_scan_at < self.model_frame_scan_interval:
            return
        self._last_model_frame_scan_at = now
        if self.model_run_id:
            self.model_run_dir = self.model_out_root / self.model_run_id
            if not self.model_run_dir.is_dir():
                return
        else:
            run_dirs = sorted((path for path in self.model_out_root.glob("*") if path.is_dir()), key=lambda path: path.stat().st_mtime)
            if not run_dirs:
                return
            self.model_run_dir = run_dirs[-1]
        self._refresh_path_from_model_trace()
        frame_dir = self.model_run_dir / "frames"
        if not frame_dir.exists():
            return
        if self._bridge_frame_dir != frame_dir:
            self._bridge_frame_dir = frame_dir
            self._bridge_frame_index = 0
            seen.clear()
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        for frame_path in frame_paths[self._bridge_frame_index :]:
            if frame_path in seen:
                continue
            seen.add(frame_path)
            self.model_run_dir = frame_path.parent.parent
            if self.use_model_frames_for_stream:
                data = self.prepare_stream_frame(frame_path.read_bytes())
                if data is not None:
                    self.set_frame(data, captured_at=frame_path.stat().st_mtime)
            self._update_from_model_meta(frame_path)
            if self.save_frames:
                shutil.copy2(frame_path, self.frames_dir / frame_path.name)
            self.append_trace({"frame": str(frame_path)})
        self._bridge_frame_index = len(frame_paths)

    def _bridge_preview_frames(self):
        for frame_path in sorted(self.preview_frames_dir.glob("preview_*.jpg")):
            try:
                # Preview frames were already validated and exposure-corrected
                # before the atomic write in AirVLNSimulatorClientTool.
                captured_at = frame_path.stat().st_mtime
                data = frame_path.read_bytes()
                self.set_frame(data, captured_at=captured_at)
                if self.save_frames:
                    replay_name = f"preview_{self.frame_sequence:08d}.jpg"
                    (self.frames_dir / replay_name).write_bytes(data)
                frame_path.unlink()
            except FileNotFoundError:
                continue

    def _update_from_model_meta(self, frame_path):
        try:
            step_text = frame_path.stem.split("_")[-1]
            self.step = int(step_text)
        except Exception:
            self.step += 1

        run_dir = frame_path.parent.parent
        trace_item = self._read_trace_item(run_dir / "trace.jsonl", self.step)
        if trace_item:
            self._apply_trace_item(trace_item)
            return

        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return
        trace = meta.get("trace") or []
        if not trace:
            return
        item = None
        for candidate in trace:
            if int(candidate.get("step", -1)) == self.step:
                item = candidate
                break
        if item:
            self._apply_trace_item(item)

    def _read_trace_item(self, trace_path, step):
        if not trace_path.exists():
            return None
        self._load_trace_cache(trace_path)
        item = self._trace_items_by_step.get(int(step))
        if item is not None:
            return item
        # Fallback for rare out-of-order writes or after a server-side reload.
        try:
            with trace_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if int(item.get("step", -1)) == int(step):
                        return item
        except Exception:
            return None
        return None

    def _load_trace_cache(self, trace_path):
        if self._trace_cache_path != trace_path:
            self._trace_cache_path = trace_path
            self._trace_cache_offset = 0
            self._trace_items_by_step = {}
        try:
            current_size = trace_path.stat().st_size
            if current_size < self._trace_cache_offset:
                self._trace_cache_offset = 0
                self._trace_items_by_step = {}
            with trace_path.open("r", encoding="utf-8") as fp:
                fp.seek(self._trace_cache_offset)
                for line in fp:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    step = item.get("step")
                    if step is not None:
                        self._trace_items_by_step[int(step)] = item
                self._trace_cache_offset = fp.tell()
        except Exception:
            return

    def _refresh_path_from_model_trace(self):
        if self.model_run_id:
            self.model_run_dir = self.model_out_root / self.model_run_id
        if not self.model_run_dir:
            return
        trace_path = self.model_run_dir / "trace.jsonl"
        if not trace_path.exists():
            return
        last_item = None
        try:
            if self._refresh_trace_path != trace_path:
                self._refresh_trace_path = trace_path
                self._refresh_trace_offset = 0
            current_size = trace_path.stat().st_size
            if current_size < self._refresh_trace_offset:
                self._refresh_trace_offset = 0
            with trace_path.open("r", encoding="utf-8") as fp:
                fp.seek(self._refresh_trace_offset)
                for line in fp:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    last_item = item
                    self._apply_trace_item(item)
                self._refresh_trace_offset = fp.tell()
            if len(self.path) > 1200:
                self.path = self.path[-1000:]
            if last_item:
                action_id = last_item.get("action_id")
                self.action = last_item.get("action") or self.ACTION_ID_TO_NAME.get(action_id, self.action)
                self.position = last_item.get("position") or self.position
                self.distance_to_goal = last_item.get("distance_to_goal")
                self.initial_distance = last_item.get("initial_distance", self.initial_distance)
                self.best_distance = last_item.get("best_distance", self.best_distance)
                self.best_distance_step = last_item.get("best_distance_step", self.best_distance_step)
                self.success_distance = last_item.get("success_distance", self.success_distance)
                self.success_20m = bool(last_item.get("success_20m", self.success_20m))
                self.success_20m_with_altitude = bool(
                    last_item.get("success_20m_with_altitude", self.success_20m_with_altitude)
                )
                self.success_20m_with_surface = bool(
                    last_item.get("success_20m_with_surface", self.success_20m_with_surface)
                )
                self.oracle_success_20m = bool(last_item.get("oracle_success_20m", self.oracle_success_20m))
                self.distance_3d_to_goal = last_item.get("distance_3d_to_goal", self.distance_3d_to_goal)
                self.vertical_error_to_goal = last_item.get("vertical_error_to_goal", self.vertical_error_to_goal)
                self.altitude_aligned = bool(last_item.get("altitude_aligned", self.altitude_aligned))
                self.surface_clearance = last_item.get("surface_clearance", self.surface_clearance)
                self.surface_aligned = bool(last_item.get("surface_aligned", self.surface_aligned))
                self.surface_probe_ok = bool(last_item.get("surface_probe_ok", self.surface_probe_ok))
                self.surface_probe_error = last_item.get("surface_probe_error", self.surface_probe_error)
                self.progress_to_goal = last_item.get("progress_to_goal", self.progress_to_goal)
                self.navigation_phase = last_item.get("navigation_phase", self.navigation_phase)
                self.goal_position = last_item.get("goal_position", self.goal_position)
                self.safety_override = last_item.get("safety_override")
                self.collision_rollback_count = int(
                    last_item.get("collision_rollback_count", self.collision_rollback_count) or 0
                )
                self.last_collision = last_item.get("last_collision", self.last_collision)
        except Exception:
            return

    def _apply_trace_item(self, item):
        if item.get("step") is not None:
            self.step = int(item.get("step"))
        action_id = item.get("action_id")
        self.action = item.get("action") or self.ACTION_ID_TO_NAME.get(action_id, self.action)
        self.position = item.get("position")
        self.distance_to_goal = item.get("distance_to_goal")
        self.initial_distance = item.get("initial_distance", self.initial_distance)
        self.best_distance = item.get("best_distance", self.best_distance)
        self.best_distance_step = item.get("best_distance_step", self.best_distance_step)
        self.success_distance = item.get("success_distance", self.success_distance)
        self.success_20m = bool(item.get("success_20m", self.success_20m))
        self.success_20m_with_altitude = bool(
            item.get("success_20m_with_altitude", self.success_20m_with_altitude)
        )
        self.success_20m_with_surface = bool(
            item.get("success_20m_with_surface", self.success_20m_with_surface)
        )
        self.oracle_success_20m = bool(item.get("oracle_success_20m", self.oracle_success_20m))
        self.distance_3d_to_goal = item.get("distance_3d_to_goal", self.distance_3d_to_goal)
        self.vertical_error_to_goal = item.get("vertical_error_to_goal", self.vertical_error_to_goal)
        self.altitude_aligned = bool(item.get("altitude_aligned", self.altitude_aligned))
        self.surface_clearance = item.get("surface_clearance", self.surface_clearance)
        self.surface_aligned = bool(item.get("surface_aligned", self.surface_aligned))
        self.surface_probe_ok = bool(item.get("surface_probe_ok", self.surface_probe_ok))
        self.surface_probe_error = item.get("surface_probe_error", self.surface_probe_error)
        self.progress_to_goal = item.get("progress_to_goal", self.progress_to_goal)
        self.navigation_phase = item.get("navigation_phase", self.navigation_phase)
        self.goal_position = item.get("goal_position", self.goal_position)
        self.safety_override = item.get("safety_override")
        self.collision_rollback_count = int(
            item.get("collision_rollback_count", self.collision_rollback_count) or 0
        )
        self.last_collision = item.get("last_collision", self.last_collision)
        if self.position and len(self.position) >= 2:
            point = {
                "step": item.get("step", self.step),
                "position": self.position,
                "distance_to_goal": self.distance_to_goal,
                "distance_3d_to_goal": self.distance_3d_to_goal,
                "vertical_error_to_goal": self.vertical_error_to_goal,
                "altitude_aligned": self.altitude_aligned,
                "surface_clearance": self.surface_clearance,
                "surface_aligned": self.surface_aligned,
                "surface_probe_ok": self.surface_probe_ok,
                "surface_probe_error": self.surface_probe_error,
                "best_distance": self.best_distance,
                "success_20m": self.success_20m,
                "success_20m_with_altitude": self.success_20m_with_altitude,
                "success_20m_with_surface": self.success_20m_with_surface,
                "oracle_success_20m": self.oracle_success_20m,
                "progress_to_goal": self.progress_to_goal,
                "navigation_phase": self.navigation_phase,
                "goal_position": self.goal_position,
                "action": self.action,
            }
            if not self.path or self.path[-1].get("step") != point["step"]:
                self.path.append(point)
                if len(self.path) > 1200:
                    self.path = self.path[-1000:]

    def _load_final_model_state(self):
        if not self.model_run_dir:
            run_dirs = sorted(
                (path for path in self.model_out_root.glob("*") if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
            )
            if run_dirs:
                self.model_run_dir = run_dirs[-1]
        if not self.model_run_dir:
            return
        self._refresh_path_from_model_trace()
        meta_path = self.model_run_dir / "meta.json"
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return
        trace = meta.get("trace") or []
        if trace:
            self._apply_trace_item(trace[-1])
            self.step = int(trace[-1].get("step", self.step))
        model_stop_reason = meta.get("stop_reason")
        if model_stop_reason:
            self.stop_reason = model_stop_reason
        self.flight_target_telemetry = _flight_target_telemetry_manifest(meta)

    def _write_summary(self):
        payload = self.snapshot()
        payload["path"] = self.path[-1000:]
        payload["started_at"] = self.started_at
        payload["ended_at"] = self.ended_at
        payload["duration_sec"] = round((self.ended_at or time.time()) - (self.started_at or self.created_at), 3)
        (self.session_dir / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def sys_executable():
    import sys

    return sys.executable
