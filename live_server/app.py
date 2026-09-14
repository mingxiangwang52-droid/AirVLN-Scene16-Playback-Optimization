#!/usr/bin/env python3
import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
import traceback
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from flask import Flask, Response, jsonify, request, send_from_directory
import numpy as np
from PIL import Image

from agent import AgentSession

ROOT = Path(__file__).resolve().parent
AIRVLN_ROOT = ROOT.parent
STATIC_DIR = ROOT / "static"
RUNTIME_DIR = ROOT / "runtime"
SESSIONS_DIR = RUNTIME_DIR / "sessions"
MODEL_RUNS_DIR = RUNTIME_DIR / "model_runs"
AGENT_SESSIONS_DIR = RUNTIME_DIR / "agent_sessions"
DATASET_DATA_DIR = Path(os.environ.get("AIRVLN_DATASET_DATA_DIR", str(AIRVLN_ROOT / "DATA" / "data")))
DEFAULT_VERIFIED_SUCCESS_EVAL = RUNTIME_DIR / "semantic_landing_eval_100case_v61_terminal_metric.json"
DEFAULT_CURRENT_VERIFIED_SESSIONS = RUNTIME_DIR / "current_verified_success_sessions.json"
DEFAULT_SCENE16_GENERALIZATION_EVAL = Path(
    os.environ.get(
        "AIRVLN_SCENE16_GENERALIZATION_EVAL",
        str(RUNTIME_DIR / "scene16_generalization_eval_v1.json"),
    )
)
DEFAULT_SCENE16_VARIANT_EVAL = Path(
    os.environ.get(
        "AIRVLN_SCENE16_VARIANT_EVAL",
        str(RUNTIME_DIR / "scene16_variant_eval_v1.json"),
    )
)
DEFAULT_SCENE16_DEMO_STABLE_POOL = Path(
    os.environ.get(
        "AIRVLN_SCENE16_DEMO_STABLE_POOL",
        str(RUNTIME_DIR / "scene16_single_scene_stable_pool_v055_entity_endpoint_stable82_20260822.json"),
    )
)
DEFAULT_SCENE16_PREVIEW_MANIFEST = Path(
    os.environ.get(
        "AIRVLN_SCENE16_PREVIEW_MANIFEST",
        str(RUNTIME_DIR / "scene16_stable82_preview_manifest_20260825.json"),
    )
)
DEFAULT_PHASE_A_SPLIT = "train_scene16_segments_high_conf_1k"
DEFAULT_PHASE_A_EPISODE_ID = "304SM51WAC2LJX66KU06PFQOK2ISBK__seg006"
DEFAULT_PHASE_A_INSTRUCTION = "now go over the buildings and get down towards the road"
DEFAULT_PHASE_A_CKPT = (
    os.environ.get("AIRVLN_PHASE_A_CKPT", "")
)
DEFAULT_SEGMENT_GROUNDING_RIDGE = (
    os.environ.get(
        "AIRVLN_SEGMENT_GROUNDING_RIDGE",
        str(RUNTIME_DIR / "segment_grounding" / "scene16_generalization_ridge_v3_live_compatible" / "segment_grounding_ridge.npz"),
    )
)
DEFAULT_PHASE_A_SIMULATOR_PORT = 30003
BLACKWELL_RUNTIME_MARKER = Path(
    os.environ.get("AIRVLN_BLACKWELL_RUNTIME_MARKER", str(RUNTIME_DIR / "blackwell_probe" / "sm120_ok.json"))
)

for item in (RUNTIME_DIR, SESSIONS_DIR, MODEL_RUNS_DIR, AGENT_SESSIONS_DIR):
    item.mkdir(parents=True, exist_ok=True)

if str(AIRVLN_ROOT) not in sys.path:
    sys.path.insert(0, str(AIRVLN_ROOT))

from grounding import build_grounding, compact_grounding_view
from streaming import resolve_hold_last_frame, resolve_hold_max_age, should_hold_frame

app = Flask(__name__, static_folder=None)
sessions = {}
sessions_lock = threading.Lock()
replay_asset_locks = {}
replay_asset_locks_lock = threading.Lock()
replay_prewarm_jobs = {}
replay_prewarm_jobs_lock = threading.Lock()


def replay_asset_lock(key):
    with replay_asset_locks_lock:
        lock = replay_asset_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            replay_asset_locks[key] = lock
        return lock


def replay_asset_key(kind, session_id, **params):
    items = ",".join(f"{name}={params[name]}" for name in sorted(params))
    return f"{kind}:{session_id}:{items}"


@app.after_request
def add_no_cache_headers(response):
    if request.path == "/" or request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def now_session_id():
    return time.strftime("%Y%m%d-%H%M%S") + "-" + str(int(time.time() * 1000) % 1000).zfill(3)


def optional_imports():
    result = {}
    for name in ("airsim", "PIL", "numpy"):
        try:
            __import__(name)
            result[name] = True
        except Exception as exc:
            result[name] = False
            result[name + "_error"] = str(exc)
    return result


def parse_port(value, default):
    try:
        port = int(value)
    except (TypeError, ValueError):
        return default
    return port if 1 <= port <= 65535 else default


def airsim_endpoint(host=None, port=None):
    default_port = parse_port(os.environ.get("AIRSIM_PORT", "30014"), 30014)
    return {
        "host": host or os.environ.get("AIRSIM_HOST", "127.0.0.1"),
        "port": parse_port(port, default_port),
    }


def tcp_endpoint_reachable(host, port, timeout=0.8):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True, None
    except Exception as exc:
        return False, str(exc)


def airsim_health(timeout=2.0, host=None, port=None):
    endpoint = airsim_endpoint(host=host, port=port)
    tcp_ok, tcp_error = tcp_endpoint_reachable(endpoint["host"], endpoint["port"])
    if not tcp_ok:
        return {"ok": False, "message": tcp_error or "simulator endpoint is not reachable", **endpoint}
    try:
        import airsim

        client = airsim.VehicleClient(
            ip=endpoint["host"],
            port=endpoint["port"],
            timeout_value=timeout,
        )
        client.confirmConnection()
        return {"ok": True, "message": "AirSim connection ok", "standard_airsim_rpc": True, **endpoint}
    except Exception as exc:
        message = str(exc)
        if "getServerVersion" in message or "method not found" in message:
            return {
                "ok": True,
                "message": "AirVLN simulator endpoint is reachable; standard AirSim confirmConnection is not supported by this simulator tool.",
                "standard_airsim_rpc": False,
                **endpoint,
            }
        payload = {"ok": False, "message": str(exc)}
        payload.update(endpoint)
        return payload


def is_port_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


def gpu_inventory():
    try:
        import torch

        gpus = []
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(device_id)
                major, minor = torch.cuda.get_device_capability(device_id)
                compute_cap = float(f"{major}.{minor}")
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
                    memory_total_mb = int(total_bytes // (1024 * 1024))
                    memory_free_mb = int(free_bytes // (1024 * 1024))
                except Exception:
                    memory_total_mb = 0
                    memory_free_mb = 0
                gpu = {
                    "id": device_id,
                    "name": name,
                    "compute_cap": compute_cap,
                    "memory_total_mb": memory_total_mb,
                    "memory_used_mb": max(0, memory_total_mb - memory_free_mb),
                    "memory_free_mb": memory_free_mb,
                    "runtime": "torch",
                }
                compatible, reason = gpu_compatibility(gpu)
                gpu["compatible"] = compatible
                gpu["reason"] = reason
                gpus.append(gpu)
            compatible_gpus = [gpu for gpu in gpus if gpu["compatible"]]
            recommended = recommend_agent_gpu(compatible_gpus)
            return {
                "ok": True,
                "gpus": gpus,
                "recommended_gpu_id": recommended,
                "source": "torch_runtime",
                "torch_version": getattr(torch, "__version__", None),
                "torch_cuda": getattr(torch.version, "cuda", None),
                "torch_arch_list": torch.cuda.get_arch_list(),
            }
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,compute_cap,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "gpus": []}

    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            compute_cap = float(parts[2])
        except ValueError:
            compute_cap = None
        gpu = {
            "id": int(parts[0]),
            "name": parts[1],
            "compute_cap": compute_cap,
            "memory_total_mb": int(parts[3]),
            "memory_used_mb": int(parts[4]),
        }
        gpu["memory_free_mb"] = gpu["memory_total_mb"] - gpu["memory_used_mb"]
        compatible, reason = gpu_compatibility(gpu)
        gpu["compatible"] = compatible
        gpu["reason"] = reason
        gpus.append(gpu)

    compatible_gpus = [gpu for gpu in gpus if gpu["compatible"]]
    recommended = recommend_agent_gpu(compatible_gpus)
    return {"ok": True, "gpus": gpus, "recommended_gpu_id": recommended}


def recommend_agent_gpu(compatible_gpus):
    if not compatible_gpus:
        return None
    stable_runtime_gpus = [gpu for gpu in compatible_gpus if float(gpu.get("compute_cap") or 0.0) < 12.0]
    busy_blackwell_gpus = [
        gpu
        for gpu in compatible_gpus
        if float(gpu.get("compute_cap") or 0.0) >= 12.0
        and int(gpu.get("memory_used_mb") or 0) >= int(os.environ.get("AIRVLN_BUSY_BLACKWELL_USED_MB", "30000"))
    ]
    if os.environ.get("AIRVLN_AVOID_BLACKWELL") == "1" or (busy_blackwell_gpus and stable_runtime_gpus):
        candidates = stable_runtime_gpus
        return max(candidates, key=lambda item: item["memory_free_mb"])["id"]
    return max(
        compatible_gpus,
        key=lambda item: (
            float(item.get("compute_cap") or 0.0),
            int(item.get("memory_free_mb") or 0),
        ),
    )["id"]



def gpu_compatibility(gpu):
    if os.environ.get("AIRVLN_ALLOW_UNSUPPORTED_GPU") == "1":
        return True, "compatibility override enabled"
    compute_cap = gpu.get("compute_cap")
    if compute_cap is None:
        return False, "compute capability is unknown"
    if compute_cap >= 12.0:
        info = blackwell_runtime_info()
        if info.get("ok"):
            return True, f"compatible via Blackwell runtime {info.get('torch', 'torch+cu128')}"
        return (
            False,
            "Blackwell/sm_120 is not supported by the current torch environment; use a 4090 runtime GPU or upgrade PyTorch.",
        )
    return True, "compatible"


def blackwell_runtime_info():
    try:
        if not BLACKWELL_RUNTIME_MARKER.exists():
            return {"ok": False, "reason": "missing Blackwell runtime marker"}
        info = json.loads(BLACKWELL_RUNTIME_MARKER.read_text(encoding="utf-8"))
        python_path = Path(str(info.get("python") or ""))
        if not info.get("ok") or not python_path.exists():
            return {"ok": False, "reason": "Blackwell runtime python missing", **info}
        return info
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def validate_agent_gpu(gpu_id):
    inventory = gpu_inventory()
    if not inventory["ok"]:
        return None, inventory
    for gpu in inventory["gpus"]:
        if gpu["id"] == gpu_id:
            if gpu["compatible"]:
                return None, inventory
            return {
                "error": "selected GPU is not compatible with this AirVLN torch environment",
                "gpu": gpu,
                "recommended_gpu_id": inventory.get("recommended_gpu_id"),
                "gpus": inventory["gpus"],
            }, inventory
    return {
        "error": f"GPU {gpu_id} was not found",
        "recommended_gpu_id": inventory.get("recommended_gpu_id"),
        "gpus": inventory["gpus"],
    }, inventory


def dataset_files(split=None, dataset="aerialvln"):
    roots = []
    if dataset in ("aerialvln", "all"):
        roots.append(DATASET_DATA_DIR / "aerialvln")
    if dataset in ("aerialvln-s", "all"):
        roots.append(DATASET_DATA_DIR / "aerialvln-s")
    names = [split] if split else ["val_unseen", "val_seen", "train", "test"]
    paths = []
    for root in roots:
        for name in names:
            path = root / f"{name}.json"
            if path.exists():
                paths.append((root.name, name, path))
    return paths


def instruction_text(episode):
    instruction = episode.get("instruction", "")
    if isinstance(instruction, dict):
        return str(instruction.get("instruction_text", "")).strip()
    return str(instruction).strip()


def tokenize_query(text):
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def route_cues(text, split=None, scene_id=None, dataset="aerialvln"):
    grounding = build_grounding(text, split=split, scene_id=scene_id, dataset=dataset)
    return compact_grounding_view(grounding)


@lru_cache(maxsize=32)
def cached_dataset_episode_index(split=None, dataset="aerialvln"):
    records = []
    for dataset_name, split_name, path in dataset_files(split=split, dataset=dataset):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        episodes = raw.get("episodes", raw if isinstance(raw, list) else [])
        for index, episode in enumerate(episodes):
            text = instruction_text(episode)
            if not text:
                continue
            records.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "index": index,
                    "episode": episode,
                    "text": text,
                    "text_lower": text.lower(),
                    "tokens": tokenize_query(text),
                    "scene_id": episode.get("scene_id"),
                }
            )
    return tuple(records)


def iter_dataset_episodes(split=None, dataset="aerialvln", scene_id=None):
    for record in cached_dataset_episode_index(split=split, dataset=dataset):
        episode_scene_id = record.get("scene_id")
        if scene_id is not None and int(episode_scene_id if episode_scene_id is not None else -1) != int(scene_id):
            continue
        yield record


def search_dataset_episodes(query, split=None, dataset="aerialvln", scene_id=None, limit=20):
    query = str(query or "").strip()
    query_tokens = tokenize_query(query)
    scored = []
    for record in iter_dataset_episodes(split, dataset, scene_id):
        episode = record["episode"]
        text = record["text"]
        overlap = len(query_tokens & record["tokens"])
        score = overlap
        if query and query.lower() in record["text_lower"]:
            score += 20
        if query and query == str(episode.get("episode_id") or ""):
            score += 100
        if not query:
            score = 1
        if score <= 0:
            continue
        scored.append(
            {
                "score": score,
                "dataset": record["dataset"],
                "split": record["split"],
                "index": record["index"],
                "episode_id": episode.get("episode_id"),
                "trajectory_id": episode.get("trajectory_id"),
                "scene_id": episode.get("scene_id"),
                "instruction": text,
                "start_position": episode.get("start_position"),
                "start_rotation": episode.get("start_rotation"),
            }
        )
    if not query:
        random.shuffle(scored)
    else:
        scored.sort(key=lambda item: item["score"], reverse=True)
    results = scored[: max(1, min(int(limit), 100))]
    for item in results:
        item["route_cues"] = route_cues(
            item["instruction"],
            split=item["split"],
            scene_id=item.get("scene_id"),
            dataset=item["dataset"],
        )
    return results


def parse_bool_arg(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


def verified_success_eval_paths():
    configured = os.environ.get("AIRVLN_VERIFIED_SUCCESS_EVALS", "").strip()
    paths = []
    if configured:
        for item in configured.split(os.pathsep):
            item = item.strip()
            if item:
                paths.append(Path(item))
    paths.append(DEFAULT_VERIFIED_SUCCESS_EVAL)
    paths.extend(sorted(RUNTIME_DIR.glob("semantic_landing_eval_*terminal_metric.json"), reverse=True))
    paths.extend(sorted((RUNTIME_DIR / "grounding_eval").glob("*terminal*.json"), reverse=True))

    seen = set()
    unique = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            unique.append(path)
    return unique


def load_scene_generalization_eval_cases():
    """Map episode_id to the closest result from validated GEN reports."""
    paths = [DEFAULT_SCENE16_GENERALIZATION_EVAL]
    for pattern in (
        "generalization_streamfix_3a0_*.json",
        "generalization_streamfix_3bv8_*rerun.json",
        "generalization_expand_v4_streamsafe_*.json",
        "generalization_v5_v4tail_*.json",
    ):
        paths.extend(sorted(RUNTIME_DIR.glob(pattern)))
    paths.extend(
        sorted(
            path
            for path in RUNTIME_DIR.glob("generalization_v*.json")
            if "seen_seed" not in path.name
            and "label_memory" not in path.name
            and "prefilter" not in path.name
        )
    )
    def strict_report_key(path):
        match = re.search(r"_v(\d+)(?:_|\.json$)", path.name)
        return (int(match.group(1)) if match else -1, path.name)

    strict_paths = sorted(
        RUNTIME_DIR.glob("generalization_expand_strict_*.json"),
        key=strict_report_key,
    )
    by_episode = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cases = payload.get("cases") if isinstance(payload, dict) else None
        if not isinstance(cases, list):
            cases = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(cases, list):
            cases = payload if isinstance(payload, list) else []
        for case in cases:
            if not isinstance(case, dict):
                continue
            episode_id = str(case.get("episode_id") or "").strip()
            if not episode_id:
                continue
            current = by_episode.get(episode_id)
            if current is None:
                by_episode[episode_id] = case
                continue
            if generalization_case_rank(case) < generalization_case_rank(current):
                by_episode[episode_id] = case

    for path in strict_paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cases = payload.get("cases") if isinstance(payload, dict) else None
        if not isinstance(cases, list):
            cases = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(cases, list):
            cases = payload if isinstance(payload, list) else []
        for case in cases:
            if not isinstance(case, dict):
                continue
            episode_id = str(case.get("episode_id") or "").strip()
            if episode_id:
                by_episode[episode_id] = case
    return by_episode


def generalization_case_rank(case):
    """Prefer display-stable showcase runs before using distance as a tiebreaker."""
    if not isinstance(case, dict):
        return (1, 1, float("inf"), float("inf"))

    def numeric(key, default=float("inf")):
        try:
            return float(case.get(key))
        except (TypeError, ValueError):
            return default

    return (
        0 if case.get("showcase_ready") is True else 1,
        int(case.get("stream_sample_over_0_5s") or 0),
        numeric("stream_sample_max_gap"),
        numeric("best_distance", numeric("distance_to_goal")),
    )


WATER_LANDING_RE = re.compile(
    r"\b(?:land|landed|landing|descend|drop down|get down|go down|lower|stop|stay)\b"
    r"[^.]{0,52}\b(?:in|on|into|onto|at|to|towards?|near|next to|beside|by|middle of|center of)?\s*"
    r"(?:the\s+)?(?:water|lake|sea|pond|river|body of water)\b"
    r"|"
    r"\b(?:water|lake|sea|pond|river|body of water)\b"
    r"[^.]{0,52}\b(?:land|landed|landing|descend|drop down|get down|go down|lower|stop|stay)\b",
    re.IGNORECASE,
)

WATER_PASSAGE_RE = re.compile(r"\b(?:water|lake|sea|pond|river|body of water)\b", re.IGNORECASE)
SHOWCASE_COMPLEX_RE = re.compile(
    r"\b(?:billboard|telephone|phone|booth|street light|traffic light|stoplight|sign|"
    r"drink ice cold|bolt cola|restaurant|shop|store|bar|statue|fountain|look down|"
    r"turn back|90 degrees|between two buildings|very tall building|skyscraper|sky scraper)\b",
    re.IGNORECASE,
)


def has_water_landing_semantics(item):
    if bool(item.get("candidate_water_landing")):
        return True
    return bool(WATER_LANDING_RE.search(str(item.get("instruction") or "")))


def instruction_showcase_penalty(item):
    """Lower scores are better for first-screen demos; keep risky-but-proven cases selectable."""
    text = str(item.get("instruction") or "")
    penalty = 0.0
    if has_water_landing_semantics(item):
        penalty += 3.0
    elif WATER_PASSAGE_RE.search(text):
        penalty += 0.45
    penalty += min(1.6, 0.18 * len(SHOWCASE_COMPLEX_RE.findall(text)))
    try:
        raw_words = len(re.findall(r"[a-z0-9]+", text.lower()))
    except Exception:
        raw_words = 0
    if raw_words > 80:
        penalty += 0.45
    if raw_words > 110:
        penalty += 0.7
    try:
        risk = float(item.get("candidate_risky_phrase_penalty") or 0.0)
    except (TypeError, ValueError):
        risk = 0.0
    penalty += risk * 5.0
    return round(penalty, 4)


def showcase_numeric(item, names, default=None):
    if isinstance(names, str):
        names = (names,)
    for name in names:
        try:
            value = float(item.get(name))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return default


def showcase_quality_fields(item, tier):
    score = {
        "frontline": 90.0,
        "stable": 80.0,
        "caution": 62.0,
        "unverified": 35.0,
    }.get(tier, 35.0)
    evidence = []
    if item.get("verification_tier") == "current_smooth":
        score += 6.0
        evidence.append("current-smooth")
    elif item.get("variant_verified_success"):
        score += 4.0
        evidence.append("variant-flight-ok")
    elif item.get("verified_success"):
        score += 3.0
        evidence.append("verified-flight-ok")
    elif item.get("gen_eval_label") == "GEN-SAFE":
        score += 1.5
        evidence.append("gen-safe")

    best_distance = showcase_numeric(
        item,
        ("variant_eval_distance", "variant_eval_best_distance", "eval_best_distance", "final_distance"),
    )
    if best_distance is not None:
        evidence.append(f"best {best_distance:.1f}m")
        if best_distance <= 12:
            score += 2.0
        elif best_distance > 19.5:
            score -= 5.0
        elif best_distance > 18:
            score -= 2.0

    surface = showcase_numeric(item, ("variant_eval_surface_clearance", "eval_surface_clearance", "surface_clearance"))
    if surface is not None:
        evidence.append(f"surface {surface:.1f}m")
        if surface <= 0.05:
            score -= 8.0
        elif surface <= 1.2:
            score += 1.0
        elif surface > 3.0:
            score -= 3.0

    collision_count = showcase_numeric(
        item,
        ("variant_eval_collision_rollback_count", "eval_collision_rollback_count", "collision_rollback_count"),
        default=0.0,
    )
    evidence.append(f"collision {int(collision_count or 0)}")
    score -= min(24.0, float(collision_count or 0) * 1.2)

    fps = showcase_numeric(item, ("eval_stream_average_fps", "stream_average_fps"))
    if fps is not None:
        evidence.append(f"fps {fps:.1f}")
        if fps >= 23:
            score += 1.0
        elif fps < 18:
            score -= 6.0
        elif fps < 21:
            score -= 3.0

    stream_gap = showcase_numeric(item, ("eval_stream_sample_max_gap", "stream_sample_max_gap"))
    if stream_gap is not None:
        evidence.append(f"gap {stream_gap:.3f}s")
        if stream_gap > 0.2:
            score -= 8.0
        elif stream_gap > 0.08:
            score -= 3.0

    route_length = showcase_numeric(item, ("candidate_reference_path_length", "reference_path_length"))
    if route_length is not None:
        if route_length > 2200:
            score -= 8.0
        elif route_length > 1500:
            score -= 4.0

    if item.get("display_water_landing_risk"):
        score -= 15.0
        evidence.append("water-stop-risk")
    penalty = showcase_numeric(item, "display_showcase_penalty", default=0.0) or 0.0
    if penalty:
        score -= min(18.0, penalty * 4.0)

    score = max(0.0, min(100.0, score))
    if score >= 90:
        label = "demo-ready"
    elif score >= 80:
        label = "strong"
    elif score >= 70:
        label = "usable"
    elif score >= 55:
        label = "caution"
    else:
        label = "risky"
    return round(score, 1), label, evidence[:8]


def assign_showcase_fields(item):
    penalty = instruction_showcase_penalty(item)
    item["display_showcase_penalty"] = penalty
    item["display_water_landing_risk"] = has_water_landing_semantics(item)
    if item.get("verification_tier") == "current_smooth":
        tier = "frontline"
    elif item.get("showcase_ready") and item.get("strict_collision_free") and item.get("stream_sample_safe") and penalty <= 0.8:
        tier = "frontline"
    elif item.get("showcase_ready") and item.get("strict_collision_free") and item.get("stream_safe") and penalty <= 1.6:
        tier = "stable"
    elif item.get("verified_success") or item.get("showcase_ready") or item.get("gen_eval_label") == "GEN-SAFE":
        tier = "caution"
    else:
        tier = "unverified"
    item["display_showcase_tier"] = tier
    score, label, evidence = showcase_quality_fields(item, tier)
    item["display_quality_score"] = score
    item["display_quality_label"] = label
    item["display_evidence"] = evidence
    return item


def normalize_generalization_eval_case(case):
    if not isinstance(case, dict):
        return {}
    label = str(case.get("gen_eval_label") or case.get("quality_label") or "").strip() or "GEN-RAW"
    return {
        "gen_eval_label": label,
        "display_safe": bool(case.get("display_safe", False)),
        "success_20m": bool(case.get("success_20m", False)),
        "success_20m_with_surface": bool(case.get("success_20m_with_surface", False)),
        "surface_aligned": bool(case.get("surface_aligned", False)),
        "collision_free": bool(case.get("collision_free", False)),
        "strict_collision_free": bool(case.get("strict_collision_free", False)),
        "stream_sample_safe": bool(case.get("stream_sample_safe", False)),
        "stream_safe": bool(case.get("stream_safe", False)),
        "eval_status": case.get("status"),
        "showcase_ready": bool(case.get("showcase_ready", False)),
        "final_distance": first_finite_number(
            case.get("final_distance"),
            case.get("distance_to_goal"),
            case.get("best_distance"),
        ),
        "final_step": case.get("final_step", case.get("step")),
        "stop_reason": case.get("stop_reason"),
        "best_distance": case.get("best_distance"),
        "distance_to_goal": case.get("distance_to_goal"),
        "surface_clearance": case.get("surface_clearance"),
        "collision_rollback_count": case.get("collision_rollback_count"),
        "stream_average_fps": case.get("stream_average_fps"),
        "stream_sample_max_gap": case.get("stream_sample_max_gap"),
        "stream_sample_over_0_5s": case.get("stream_sample_over_0_5s"),
        "stream_sample_over_1s": case.get("stream_sample_over_1s"),
        "eval_best_distance": case.get("best_distance"),
        "eval_distance_to_goal": case.get("distance_to_goal"),
        "eval_surface_clearance": case.get("surface_clearance"),
        "eval_collision_rollback_count": case.get("collision_rollback_count"),
        "eval_stream_average_fps": case.get("stream_average_fps"),
        "eval_stream_sample_max_gap": case.get("stream_sample_max_gap"),
        "eval_stream_sample_over_0_5s": case.get("stream_sample_over_0_5s"),
        "eval_stream_sample_over_1s": case.get("stream_sample_over_1s"),
        "eval_step": case.get("step"),
        "eval_stop_reason": case.get("stop_reason"),
        "eval_session_id": case.get("session_id"),
        "candidate_verified_similarity": case.get("candidate_verified_similarity"),
        "candidate_prior_score": case.get("candidate_prior_score"),
        "candidate_reference_path_length": case.get("candidate_reference_path_length"),
        "candidate_start_goal_distance": case.get("candidate_start_goal_distance"),
        "candidate_reference_path_points": case.get("candidate_reference_path_points"),
        "candidate_risky_phrase_penalty": case.get("candidate_risky_phrase_penalty"),
        "candidate_complexity_penalty": case.get("candidate_complexity_penalty"),
        "candidate_water_landing": case.get("candidate_water_landing"),
    }


def first_finite_number(*values):
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def is_verified_success_case(case):
    if not isinstance(case, dict):
        return False
    if case.get("success_20m_with_surface") is not True:
        return False
    if case.get("too_fast") is True:
        return False
    if case.get("returncode") not in (None, 0):
        return False
    if case.get("collision_free") is False:
        return False

    def case_numeric(*names):
        for name in names:
            try:
                value = float(case.get(name))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
        return None

    final_distance = case_numeric("final_distance", "distance_to_goal", "best_distance")
    if final_distance is not None and final_distance > 20.0:
        return False
    return bool(case.get("episode_id"))


def normalize_current_success_case(case, source):
    if not isinstance(case, dict):
        return None
    if case.get("success_20m_with_surface") is not True:
        return None
    if case.get("status") not in (None, "done"):
        return None
    if case.get("stream_sample_error") not in (None, ""):
        return None
    if int(case.get("stream_sample_over_0_5s") or 0) > 0:
        return None
    if int(case.get("stream_sample_over_1s") or 0) > 0:
        return None
    if int(case.get("black_frame_drop_count") or 0) > 5:
        return None
    if int(case.get("collision_rollback_count") or 0) > 25:
        return None
    try:
        final_distance = float(case.get("distance_to_goal", case.get("final_distance")))
    except (TypeError, ValueError):
        return None
    if final_distance > 20.0:
        return None
    episode_id = str(case.get("episode_id") or "").strip()
    if not episode_id:
        return None
    normalized = {
        "episode_id": episode_id,
        "verified_success": True,
        "verification_tier": "current_smooth",
        "verified_source": source,
        "final_distance": final_distance,
        "surface_clearance": case.get("surface_clearance"),
        "final_step": case.get("step", case.get("final_step")),
        "stop_reason": case.get("stop_reason"),
        "stream_average_fps": case.get("stream_average_fps"),
        "stream_sample_max_gap": case.get("stream_sample_max_gap"),
        "black_frame_drop_count": case.get("black_frame_drop_count"),
        "collision_rollback_count": case.get("collision_rollback_count"),
    }
    session_id = str(case.get("session_id") or case.get("source_session_id") or case.get("agent_session_id") or "").strip()
    if agent_session_has_replay_frames(session_id):
        normalized["demo_source_session_id"] = session_id
    return normalized


def add_best_verified_case(verified, case):
    if not case:
        return
    current = verified.get(case["episode_id"])
    current_distance = current.get("final_distance", float("inf")) if current else float("inf")
    candidate_distance = case.get("final_distance", float("inf"))
    if current is None or candidate_distance < current_distance:
        verified[case["episode_id"]] = case


@lru_cache(maxsize=8)
def cached_current_smooth_success_cases():
    verified = {}
    for path in sorted(RUNTIME_DIR.glob("verified_smooth_batch_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for case in payload.get("results") or []:
            add_best_verified_case(verified, normalize_current_success_case(case, path.name))

    session_ids = []
    if DEFAULT_CURRENT_VERIFIED_SESSIONS.exists():
        try:
            payload = json.loads(DEFAULT_CURRENT_VERIFIED_SESSIONS.read_text(encoding="utf-8"))
            session_ids.extend(str(item) for item in payload.get("session_ids") or [])
        except Exception:
            pass
    configured_sessions = os.environ.get("AIRVLN_CURRENT_VERIFIED_SESSION_IDS", "").strip()
    if configured_sessions:
        session_ids.extend(item.strip() for item in configured_sessions.split(",") if item.strip())

    for session_id in session_ids:
        summary_path = AGENT_SESSIONS_DIR / session_id / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        case = normalize_current_success_case(summary, f"agent_session:{session_id}")
        if case:
            if agent_session_has_replay_frames(session_id):
                case["demo_source_session_id"] = session_id
            add_best_verified_case(verified, case)
    return verified


@lru_cache(maxsize=8)
def cached_verified_success_cases():
    """Map episode_id to verified runs, preferring current smooth evidence."""
    verified = {}
    for path in verified_success_eval_paths():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cases = payload.get("cases") if isinstance(payload, dict) else None
        if not isinstance(cases, list):
            continue
        for case in cases:
            if not is_verified_success_case(case):
                continue
            episode_id = str(case.get("episode_id") or "").strip()
            if not episode_id:
                continue
            final_distance = first_finite_number(
                case.get("final_distance"),
                case.get("distance_to_goal"),
                case.get("best_distance"),
            )
            final_step = case.get("final_step", case.get("step"))
            current = verified.get(episode_id)
            current_distance = current.get("final_distance", float("inf")) if current else float("inf")
            candidate_distance = final_distance if final_distance is not None else float("inf")
            if current is None or candidate_distance < current_distance:
                add_best_verified_case(verified, {
                    "episode_id": episode_id,
                    "verified_success": True,
                    "verification_tier": "historical_eval",
                    "verified_source": path.name,
                    "demo_source_session_id": demo_source_session_id(str(path), episode_id),
                    "final_distance": final_distance,
                    "surface_clearance": case.get("surface_clearance"),
                    "final_step": final_step,
                    "segment_count": case.get("segment_count"),
                    "stop_reason": case.get("stop_reason"),
                    "mode": payload.get("aggregate", {}).get("mode") if isinstance(payload, dict) else None,
                })

    # Keep the historical catalog available, but let current-version smooth
    # checks override duplicate episodes and sort to the front in the selector.
    for episode_id, case in cached_current_smooth_success_cases().items():
        verified[episode_id] = case
    return verified


def verified_instruction_sort_key(item):
    tier = item.get("verification_tier")
    tier_rank = 0 if tier == "current_smooth" else 1
    try:
        distance = float(item.get("final_distance"))
    except (TypeError, ValueError):
        distance = float("inf")
    return (tier_rank, distance, str(item.get("episode_id") or ""))


def generalized_instruction_sort_key(item):
    showcase_tier_rank = {
        "frontline": 0,
        "stable": 1,
        "caution": 2,
        "unverified": 3,
    }.get(item.get("display_showcase_tier"), 3)
    if item.get("verification_tier") == "current_smooth":
        tier_rank = 0
    elif item.get("showcase_ready") or (item.get("gen_eval_label") == "GEN-SAFE" and item.get("stream_sample_safe")):
        tier_rank = 1
    elif item.get("verified_success"):
        tier_rank = 2
    else:
        label = item.get("gen_eval_label") or "GEN-RAW"
        tier_rank = {
            "GEN-SAFE": 3,
            "GEN-CLOSE": 4,
            "GEN-RISK": 5,
            "GEN-RAW": 6,
        }.get(label, 7)
    try:
        distance = float(item.get("eval_best_distance", item.get("final_distance")))
    except (TypeError, ValueError):
        distance = float("inf")
    def numeric(name, default=float("inf")):
        try:
            value = float(item.get(name))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    stream_gap = numeric("eval_stream_sample_max_gap")
    stream_over = int(numeric("eval_stream_sample_over_0_5s", 99))
    route_length = numeric("candidate_reference_path_length", numeric("reference_path_length"))
    start_goal = numeric("candidate_start_goal_distance", numeric("start_goal_distance"))
    route_points = numeric("candidate_reference_path_points", numeric("reference_path_points"))
    similarity = numeric("candidate_verified_similarity", 0.0)
    steps = numeric("eval_step", numeric("final_step"))
    route_score = (
        route_length / 1200.0
        + start_goal / 850.0
        + route_points / 700.0
        + steps / 700.0
        + stream_over * 3.0
        + stream_gap * 1.5
        + numeric("display_showcase_penalty", 0.0)
        - similarity * 0.5
    )
    quality_score = numeric("display_quality_score", 0.0)
    return (showcase_tier_rank, -quality_score, tier_rank, route_score, distance, str(item.get("episode_id") or ""))


def normalize_reference_point(point):
    if isinstance(point, dict):
        values = [point.get("x"), point.get("y"), point.get("z")]
    elif isinstance(point, (list, tuple)):
        values = list(point[:3])
    else:
        return None
    if len(values) < 3:
        return None
    try:
        return (float(values[0]), float(values[1]), float(values[2]))
    except (TypeError, ValueError):
        return None


def reference_path_stats(raw_path):
    if not isinstance(raw_path, list):
        return {}
    points = [point for point in (normalize_reference_point(item) for item in raw_path) if point is not None]
    if len(points) < 2:
        return {"reference_path_points": len(points)}
    path_length = 0.0
    for left, right in zip(points, points[1:]):
        path_length += math.dist(left, right)
    return {
        "reference_path_points": len(points),
        "reference_path_length": round(path_length, 3),
        "start_goal_distance": round(math.dist(points[0], points[-1]), 3),
    }


def list_dataset_instructions(split=None, dataset="aerialvln", scene_id=None, verified_only=False):
    """Return a lightweight instruction catalog for the selector."""
    verified_cases = cached_verified_success_cases()
    generalization_eval = load_scene_generalization_eval_cases()
    results = []
    for record in iter_dataset_episodes(split, dataset, scene_id):
        episode = record["episode"]
        episode_id = str(episode.get("episode_id") or "")
        verified_case = verified_cases.get(episode_id)
        if verified_only and not verified_case:
            continue
        generalization_case = normalize_generalization_eval_case(generalization_eval.get(episode_id))
        if not verified_case and not generalization_case:
            generalization_case = {"gen_eval_label": "GEN-RAW"}
        item = {
            "dataset": record["dataset"],
            "split": record["split"],
            "index": record["index"],
            "episode_id": episode_id,
            "trajectory_id": episode.get("trajectory_id"),
            "scene_id": episode.get("scene_id"),
            "instruction": record["text"],
            **reference_path_stats(episode.get("reference_path")),
            **generalization_case,
            **(verified_case or {}),
        }
        results.append(assign_showcase_fields(item))
    if verified_only:
        results.sort(key=verified_instruction_sort_key)
    else:
        results.sort(key=generalized_instruction_sort_key)
    return results


def is_safe_display_instruction(item):
    return bool(
        item.get("verified_success")
        or item.get("showcase_ready")
        or item.get("gen_eval_label") == "GEN-SAFE"
    )


def filter_showcase_items(items, showcase_filter="all"):
    mode = str(showcase_filter or "all").lower()
    if mode in ("", "all", "safe"):
        return list(items)
    if mode in ("live", "live-ok", "current", "current_smooth"):
        return [item for item in items if item.get("verification_tier") == "current_smooth"]
    if mode in ("stable", "frontline_stable", "frontline+stable"):
        allowed = {"frontline", "stable"}
    else:
        allowed = {"frontline"}
    return [item for item in items if item.get("display_showcase_tier") in allowed]


def filter_quality_items(items, min_quality_score=0.0):
    try:
        threshold = float(min_quality_score or 0.0)
    except (TypeError, ValueError):
        threshold = 0.0
    if threshold <= 0:
        return list(items)
    return [
        item for item in items
        if showcase_numeric(item, "display_quality_score", default=0.0) >= threshold
    ]


def dedupe_instructions_by_episode(items):
    primary = {}
    extras = []
    for item in sorted(items, key=generalized_instruction_sort_key):
        episode_id = str(item.get("episode_id") or "")
        if episode_id and episode_id not in primary:
            primary[episode_id] = item
        else:
            extras.append(item)
    return list(primary.values()) + extras


VARIANT_RISK_RE = re.compile(
    r"\b("
    r"bridge|ocean|sea|lake|water|river|billboard|sign|booth|phone|"
    r"skyscraper|sky scraper|restaurant|shop|store|banner|board|sky|sun|"
    r"cover|corner|shot|half of|platform|tower|street light|traffic light|"
    r"ember|soap|vehicle|vehicles|athena|zebra|crossing|coated|view|"
    r"extreme end|shy over|get up top|statue|90 degrees|look down|"
    r"keep position|drink ice cold|bolt cola|cola|bedroom|window|"
    r"staircase|telephone|telephones|bench|chair|cafeteria|bar|"
    r"peacock|wheat color|leftward|dive down|mailbox|mail box|"
    r"chocolate|fountain|look left|look right|news|sport|sports|"
    r"car wash|american|ash building|land in the air|land in the tree|"
    r"cam|camera|tilt|pause|"
    r"descend up|patio table|baseboard"
    r")\b",
    re.IGNORECASE,
)


VARIANT_HARD_RISK_RE = re.compile(
    r"\b("
    r"billboard|sign|booth|phone|restaurant|shop|store|banner|board|poster|"
    r"cover|corner|shot|half of|platform|street light|traffic light|"
    r"ember|soap|vehicle|vehicles|athena|zebra|crossing|coated|view|"
    r"extreme end|shy over|get up top|90 degrees|look down|"
    r"keep position|drink ice cold|bolt cola|cola|bedroom|window|"
    r"staircase|telephone|telephones|bench|chair|cafeteria|bar|"
    r"peacock|wheat color|leftward|dive down|mailbox|mail box|"
    r"chocolate|fountain|statue|stoplight|look left|look right|look at|look and|news|sport|sports|"
    r"car wash|american|ash building|land in the air|land in the tree|"
    r"cam|camera|tilt|pause|"
    r"descend up|patio table|baseboard|sky|sun"
    r")\b",
    re.IGNORECASE,
)


VARIANT_BAD_PHRASE_RE = re.compile(
    r"\b("
    r"to in right|top off|beside over|and down towards|"
    r"and down to|to urn|to and to|move to top|move down up|down up|"
    r"move down intersection road|move down and move towards|"
    r"descend intersection road|slightly down at|"
    r"pass over to|and over the|head forward move|move ahead move|"
    r"reach the and|and for car|take of|sky over|"
    r"rising and lowering|lowering down|land it over there|"
    r"fly down the [a-z ]*building|focus down|go straight down the buildings|"
    r"inter section|into the air|move towards to|between to|top it|"
    r"forward ans|and onto|it over there|"
    r"goes down|goes up|comes to|casino|terrace till|"
    r"move over to it|near to the building|land in building|land in the building|"
    r"start by lifting off and .* start by lifting off"
    r")\b",
    re.IGNORECASE,
)


def cleanup_variant_text(text):
    variant = str(text or "").strip()
    variant = re.sub(r"\s+", " ", variant)
    variant = re.sub(r"\bans\b", "and", variant, flags=re.IGNORECASE)
    variant = re.sub(r"\bnear by\b", "nearby", variant, flags=re.IGNORECASE)
    variant = re.sub(r"\bmake a a turn\b", "make a turn", variant, flags=re.IGNORECASE)
    variant = re.sub(r"\b(and)\s+\1\b", r"\1", variant, flags=re.IGNORECASE)
    variant = re.sub(r"\b(now|then|next)\s+\1\b", r"\1", variant, flags=re.IGNORECASE)
    variant = re.sub(r"\s+([.,])", r"\1", variant)
    variant = re.sub(r"\.{2,}", ".", variant)
    return variant.strip()


def apply_variant_replacements(text, replacements):
    variant = str(text or "").strip()
    for pattern, replacement in replacements:
        variant = re.sub(pattern, replacement, variant, flags=re.IGNORECASE)
    return cleanup_variant_text(variant)


def truncate_variant_by_sentence(text, max_words=120):
    original = cleanup_variant_text(text)
    words = original.split()
    if len(words) <= max_words:
        return None
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", original) if part.strip()]
    kept = []
    total = 0
    for sentence in sentences:
        count = len(sentence.split())
        if kept and total + count > max_words:
            break
        if not kept and count > max_words:
            return " ".join(sentence.split()[:max_words]).rstrip(" ,;") + "."
        kept.append(sentence)
        total += count
    if not kept:
        return None
    candidate = cleanup_variant_text(" ".join(kept))
    if candidate == original or len(candidate.split()) < 25:
        return None
    return candidate


def conservative_instruction_variants(text, max_variants=1):
    """Create display-safe paraphrases without changing landmarks or route order."""
    original = str(text or "").strip()
    if not original or VARIANT_HARD_RISK_RE.search(original):
        return []
    replacement_sets = (
        (
            (r"\btake off\b", "lift off"),
            (r"\bgo forward\b", "continue forward"),
            (r"\bmove forward\b", "continue forward"),
            (r"\bfly forward\b", "continue flying forward"),
            (r"\bget down\s+towards\b", "descend toward"),
            (r"\bgo down\s+towards\b", "descend toward"),
            (r"\bget down\s+to\b", "descend to"),
            (r"\bgo down\s+to\b", "descend to"),
            (r"\bget down\s+(onto|on|at)\b", r"descend \1"),
            (r"\bfly up\b", "climb"),
            (r"\bgo up\b", "climb"),
            (r"\bstop at\b", "stop near"),
            (r"\bstay there\b", "hold position there"),
        ),
        (
            (r"\btake off\b", "start by lifting off"),
            (r"\bgo forward\b", "proceed forward"),
            (r"\bmove forward\b", "proceed forward"),
            (r"\bfly forward\b", "proceed forward"),
            (r"\bget down\s+towards\b", "descend toward"),
            (r"\bgo down\s+towards\b", "descend toward"),
            (r"\bget down\s+to\b", "descend to"),
            (r"\bgo down\s+to\b", "descend to"),
            (r"\bget down\s+(onto|on|at)\b", r"descend \1"),
            (r"\bfly up\b", "rise"),
            (r"\bgo up\b", "rise"),
            (r"\breach\b", "arrive at"),
            (r"\bstay there\b", "remain there"),
        ),
        (
            (r"\btake off\b", "lift off"),
            (r"\bnow\b", "then"),
            (r"\bgo over\b", "fly over"),
            (r"\bmove over\b", "fly over"),
            (r"\bgo forward\b", "keep moving forward"),
            (r"\bmove forward\b", "keep moving forward"),
            (r"\bget down\s+towards\b", "descend toward"),
            (r"\bgo down\s+towards\b", "descend toward"),
            (r"\bget down\s+to\b", "descend to"),
            (r"\bgo down\s+to\b", "descend to"),
            (r"\bget down\s+(onto|on|at)\b", r"descend \1"),
        ),
        (
            (r"\btake off\b", "lift off"),
            (r"\bnow\b", "next"),
            (r"\bgo over\b", "pass over"),
            (r"\bmove over\b", "pass over"),
            (r"\bgo forward\b", "head forward"),
            (r"\bmove forward\b", "head forward"),
            (r"\bfly forward\b", "head forward"),
            (r"\bget up\b", "ascend"),
            (r"\bgo up\b", "ascend"),
            (r"\bfly up\b", "ascend"),
            (r"\bget down\s+towards\b", "descend toward"),
            (r"\bgo down\s+towards\b", "descend toward"),
            (r"\bget down\s+to\b", "descend to"),
            (r"\bgo down\s+to\b", "descend to"),
            (r"\bget down\s+(onto|on|at)\b", r"descend \1"),
            (r"\bstay there\b", "stay at that spot"),
        ),
        (
            (r"\btake off\b", "begin by lifting off"),
            (r"\bnow\b", "after that"),
            (r"\bgo over\b", "fly over"),
            (r"\bmove over\b", "fly over"),
            (r"\bgo forward\b", "move ahead"),
            (r"\bmove forward\b", "move ahead"),
            (r"\bfly forward\b", "move ahead"),
            (r"\bturn back\b", "turn around"),
            (r"\bget up\b", "climb"),
            (r"\bgo up\b", "climb"),
            (r"\bfly up\b", "climb"),
            (r"\bget down\s+towards\b", "descend toward"),
            (r"\bgo down\s+towards\b", "descend toward"),
            (r"\bget down\s+to\b", "descend to"),
            (r"\bgo down\s+to\b", "descend to"),
            (r"\bget down\s+(onto|on|at)\b", r"descend \1"),
            (r"\bstay there\b", "remain at that spot"),
        ),
    )
    variants = []
    seen = {original.lower()}
    truncated = truncate_variant_by_sentence(original)
    if truncated:
        key = truncated.lower()
        if not VARIANT_HARD_RISK_RE.search(truncated) and not VARIANT_BAD_PHRASE_RE.search(truncated):
            variants.append(truncated)
            seen.add(key)
    for replacements in replacement_sets:
        if len(variants) >= max_variants:
            break
        variant = apply_variant_replacements(original, replacements)
        key = variant.lower()
        if not variant or key in seen or VARIANT_HARD_RISK_RE.search(variant) or VARIANT_BAD_PHRASE_RE.search(variant):
            continue
        variants.append(variant)
        seen.add(key)
    return variants[:max_variants]


def instruction_variant_key(item):
    raw = f"{item.get('episode_id')}\n{item.get('instruction') or ''}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_scene16_variant_eval_cases():
    path = DEFAULT_SCENE16_VARIANT_EVAL
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    cases = payload.get("cases") if isinstance(payload, dict) else []
    by_key = {}
    for case in cases or []:
        if not isinstance(case, dict):
            continue
        key = str(case.get("variant_key") or "").strip()
        if not key:
            continue
        by_key[key] = case
    return by_key


def add_instruction_variants(items, max_variants=1, variant_eval=None):
    if max_variants <= 0:
        return items
    results = list(items)
    generated = []
    variant_eval = variant_eval or {}
    for item in items:
        if not is_safe_display_instruction(item):
            continue
        variant_texts = conservative_instruction_variants(item.get("instruction"), max_variants=max_variants)
        for variant_index, variant_text in enumerate(variant_texts):
            variant = dict(item)
            variant["instruction"] = variant_text
            variant["original_instruction"] = item.get("instruction")
            variant["instruction_variant"] = True
            variant["variant_index"] = variant_index
            variant["variant_source_episode_id"] = item.get("episode_id")
            variant["variant_note"] = f"conservative same-episode paraphrase v{variant_index + 1}"
            key = instruction_variant_key(variant)
            variant["variant_key"] = key
            evidence = variant_eval.get(key) or {}
            if evidence:
                variant["variant_verified_success"] = bool(evidence.get("variant_verified_success"))
                variant["variant_eval_session_id"] = evidence.get("session_id")
                variant["variant_eval_distance"] = evidence.get("distance_to_goal")
                variant["variant_eval_best_distance"] = evidence.get("best_distance")
                variant["variant_eval_surface_clearance"] = evidence.get("surface_clearance")
                variant["variant_eval_collision_rollback_count"] = evidence.get("collision_rollback_count")
                variant["variant_eval_black_frame_drop_count"] = evidence.get("black_frame_drop_count")
                variant["variant_eval_status"] = evidence.get("status")
                variant["variant_eval_stop_reason"] = evidence.get("stop_reason")
            generated.append(assign_showcase_fields(variant))
    generated.sort(
        key=lambda item: (
            generalized_instruction_sort_key(item),
            0 if item.get("variant_verified_success") else 1,
            float(item.get("variant_eval_distance") or item.get("final_distance") or item.get("eval_best_distance") or 9999),
            str(item.get("episode_id") or ""),
        )
    )
    originals = [item for item in results if not item.get("instruction_variant")]
    verified_variants = [item for item in generated if item.get("variant_verified_success")]
    primary_verified = []
    extra_verified = []
    seen_episodes = set()
    for variant in verified_variants:
        episode_id = str(variant.get("episode_id") or "")
        if episode_id and episode_id not in seen_episodes:
            primary_verified.append(variant)
            seen_episodes.add(episode_id)
        else:
            extra_verified.append(variant)
    pending_variants = [item for item in generated if not item.get("variant_verified_success")]
    primary_candidates = primary_verified + originals
    primary_by_episode = {}
    extra_same_episode = []
    for item in sorted(primary_candidates, key=generalized_instruction_sort_key):
        episode_id = str(item.get("episode_id") or "")
        if episode_id and episode_id not in primary_by_episode:
            primary_by_episode[episode_id] = item
        else:
            extra_same_episode.append(item)
    primary_items = list(primary_by_episode.values())
    primary_items.sort(key=generalized_instruction_sort_key)
    extra_same_episode.sort(key=generalized_instruction_sort_key)
    extra_verified.sort(key=generalized_instruction_sort_key)
    pending_variants.sort(key=generalized_instruction_sort_key)
    return primary_items + extra_same_episode + extra_verified + pending_variants


DEMO_BUILDER_FAMILIES = (
    ("road_landing", "道路/路口降落", re.compile(r"\b(?:road|street|intersection|corner|crossroad)\b", re.IGNORECASE)),
    ("building_landing", "建筑/屋顶目标", re.compile(r"\b(?:building|roof|rooftop|top of|structure)\b", re.IGNORECASE)),
    ("object_landmark", "实体地标目标", re.compile(
        r"\b(?:mailbox|box|sign|bottle|cap|telephone|phone|shop|store|restaurant|cola|drink|board)\b",
        re.IGNORECASE,
    )),
    ("park_tree", "树木/公园边缘", re.compile(r"\b(?:tree|trees|park|grass|field|pond)\b", re.IGNORECASE)),
)


def scene16_demo_family_for_instruction(instruction):
    text = str(instruction or "")
    sentences = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    endpoint_text = sentences[-1] if sentences else text
    for family_id, label, pattern in DEMO_BUILDER_FAMILIES:
        if pattern.search(endpoint_text):
            return family_id, label
    for family_id, label, pattern in DEMO_BUILDER_FAMILIES:
        if pattern.search(text):
            return family_id, label
    return "mixed_route", "混合路线"


def normalize_text_key(text):
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def scene16_demo_builder_variant_options(item, max_variants=2):
    variants = [
        {
            "id": "original",
            "label": "已验证原始长指令",
            "instruction": item.get("instruction") or "",
            "verified": True,
            "variant_kind": "stable_seed_original",
            "risk_note": "strict stable seed",
        }
    ]
    segments = [str(segment).strip() for segment in (item.get("segments") or []) if str(segment).strip()]
    if len(segments) >= 2:
        stepwise = cleanup_variant_text(
            " ".join(
                (
                    f"first, {segments[0]}.",
                    *[f"then, {segment}." for segment in segments[1:-1]],
                    f"finally, {segments[-1]}.",
                )
            )
        )
        if stepwise and normalize_text_key(stepwise) != normalize_text_key(item.get("instruction")):
            variants.append(
                {
                    "id": "segment_restatement",
                    "label": "分段重述保守改写",
                    "instruction": stepwise,
                    "verified": False,
                    "variant_kind": "stable_seed_segment_restatement",
                    "risk_note": "same extracted segments; not separately promoted",
                }
            )
    for index, text in enumerate(conservative_instruction_variants(item.get("instruction"), max_variants=max_variants)):
        if any(normalize_text_key(text) == normalize_text_key(variant.get("instruction")) for variant in variants):
            continue
        variants.append(
            {
                "id": f"conservative_{index + 1}",
                "label": f"保守同义改写 {index + 1}",
                "instruction": text,
                "verified": False,
                "variant_kind": "stable_seed_paraphrase",
                "risk_note": "same landmarks and route order; not separately promoted",
            }
        )
    return variants


def load_scene16_demo_stable_cases():
    if not DEFAULT_SCENE16_DEMO_STABLE_POOL.exists():
        return [], None, {"error": f"stable pool not found: {DEFAULT_SCENE16_DEMO_STABLE_POOL}"}
    try:
        payload = json.loads(DEFAULT_SCENE16_DEMO_STABLE_POOL.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], DEFAULT_SCENE16_DEMO_STABLE_POOL.name, {"error": str(exc)}
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        return [], DEFAULT_SCENE16_DEMO_STABLE_POOL.name, {"error": "stable pool has no cases list"}
    return cases, DEFAULT_SCENE16_DEMO_STABLE_POOL.name, {
        "stable_success_count": payload.get("stable_success_count", len(cases)),
        "source_stable_pool_sha256": payload.get("source_stable_pool_sha256"),
    }


@lru_cache(maxsize=512)
def agent_session_has_replay_frames(session_id):
    session_id = str(session_id or "").strip()
    if not session_id:
        return False
    session_dir = AGENT_SESSIONS_DIR / session_id
    candidate_dirs = [session_dir / "frames"]
    model_runs_dir = session_dir / "model_runs"
    if model_runs_dir.is_dir():
        try:
            candidate_dirs.extend(run_dir / "frames" for run_dir in model_runs_dir.iterdir() if run_dir.is_dir())
        except OSError:
            pass
    for frames_dir in candidate_dirs:
        try:
            if frames_dir.is_dir() and any(frames_dir.glob("*.jpg")):
                return True
        except OSError:
            continue
    return False


@lru_cache(maxsize=512)
def source_report_session_id(source_report, episode_id):
    source_report = str(source_report or "").strip()
    episode_id = str(episode_id or "").strip()
    if not source_report or not episode_id:
        return None
    report_path = Path(source_report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    try:
        report_path = report_path.resolve()
    except OSError:
        return None
    try:
        root_path = ROOT.resolve()
        runtime_path = RUNTIME_DIR.resolve()
    except OSError:
        return None
    if root_path not in report_path.parents and runtime_path not in report_path.parents:
        return None
    if not report_path.exists() or not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    cases = payload.get("cases") or payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        return None
    for case in cases:
        if not isinstance(case, dict):
            continue
        if str(case.get("episode_id") or "").strip() != episode_id:
            continue
        session_id = str(case.get("session_id") or "").strip()
        if agent_session_has_replay_frames(session_id):
            return session_id
    return None


@lru_cache(maxsize=4)
def scene16_report_session_index():
    index = {}
    patterns = (
        "scene16_single_scene_batch_eval*.json",
        "scene16_*eval*.json",
        "generalization_v*.json",
    )
    seen = set()
    for pattern in patterns:
        for report_path in sorted(RUNTIME_DIR.glob(pattern), reverse=True):
            if report_path in seen:
                continue
            seen.add(report_path)
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            cases = payload.get("cases") or payload.get("results") if isinstance(payload, dict) else None
            if not isinstance(cases, list):
                continue
            for case in cases:
                if not isinstance(case, dict):
                    continue
                episode_id = str(case.get("episode_id") or "").strip()
                if not episode_id or episode_id in index:
                    continue
                session_id = str(case.get("session_id") or "").strip()
                if agent_session_has_replay_frames(session_id):
                    index[episode_id] = session_id
    return index


@lru_cache(maxsize=4)
def agent_session_episode_index():
    index = {}
    try:
        session_dirs = sorted(
            (path for path in AGENT_SESSIONS_DIR.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return index
    for session_dir in session_dirs:
        session_id = session_dir.name
        if not agent_session_has_replay_frames(session_id):
            continue
        summary_path = session_dir / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        episode_id = str(summary.get("episode_id") or "").strip()
        if not episode_id or episode_id in index:
            continue
        success = bool(summary.get("success_20m_with_surface") or summary.get("verified_success"))
        showcase = bool(summary.get("showcase_ready") or summary.get("strict_collision_free"))
        if success or showcase:
            index[episode_id] = session_id
    return index


@lru_cache(maxsize=4096)
def scene16_any_report_session_id(episode_id):
    episode_id = str(episode_id or "").strip()
    if not episode_id:
        return None
    return scene16_report_session_index().get(episode_id) or agent_session_episode_index().get(episode_id)


def demo_source_session_id(source_report, episode_id):
    preview_session = scene16_preview_manifest_session_id(episode_id)
    return preview_session or source_report_session_id(source_report, episode_id) or scene16_any_report_session_id(episode_id)


def scene16_preview_manifest_index():
    """Index completed high-resolution presentation replays by episode id."""
    path = DEFAULT_SCENE16_PREVIEW_MANIFEST
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return {}
    required_profile = str(
        os.environ.get("AIRVLN_SCENE16_PREVIEW_RUNTIME_PROFILE", "")
    ).strip()
    index = {}
    for entry in entries:
        metadata = entry.get("video_metadata") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or not entry.get("ok_for_frontend")
            or entry.get("success_20m_with_surface") is not True
            or (required_profile and entry.get("runtime_profile") != required_profile)
            or not isinstance(metadata, dict)
            # Presentation replays use the recorded poses and native AirSim
            # frames.  Flow interpolation introduces visible building warps.
            or str(metadata.get("interp") or "").lower() != "hold"
            or str(metadata.get("codec") or "").lower() != "h264"
            or int(metadata.get("width") or 0) < 960
            or int(metadata.get("height") or 0) < 540
        ):
            continue
        episode_id = str(entry.get("episode_id") or "").strip()
        session_id = str(entry.get("replay_session_id") or "").strip()
        if episode_id and session_id and agent_session_has_replay_frames(session_id):
            index[episode_id] = entry
    return index


def scene16_preview_manifest_session_id(episode_id):
    entry = scene16_preview_manifest_index().get(str(episode_id or "").strip()) or {}
    return str(entry.get("replay_session_id") or "").strip() or None


def build_scene16_demo_builder_payload(max_seeds_per_family=24):
    stable_cases, source_name, meta = load_scene16_demo_stable_cases()
    # The stable pool is a candidate source, not the frontend allow-list.
    # Hide entries that are pending or failed under the active presentation
    # profile until they pass the strict manifest gate.
    preview_index = scene16_preview_manifest_index()
    replay_ready_cases = [
        item
        for item in list_dataset_instructions(split="train", dataset="aerialvln", scene_id=16, verified_only=False)
        if item.get("verified_success") and item.get("demo_source_session_id")
    ]
    cases = stable_cases
    replay_ready_count = 0
    family_map = {
        family_id: {"id": family_id, "label": label, "count": 0, "seeds": []}
        for family_id, label, _pattern in DEMO_BUILDER_FAMILIES
    }
    family_map["mixed_route"] = {"id": "mixed_route", "label": "混合路线", "count": 0, "seeds": []}
    for case in cases:
        if not isinstance(case, dict):
            continue
        instruction = str(case.get("instruction") or "").strip()
        episode_id = str(case.get("episode_id") or "").strip()
        if not instruction or not episode_id:
            continue
        preview_entry = preview_index.get(episode_id)
        if not preview_entry:
            continue
        family_id, family_label = scene16_demo_family_for_instruction(instruction)
        source_report = case.get("source_report") or case.get("verified_source") or source_name
        source_session_id = str(preview_entry.get("replay_session_id") or "").strip()
        replay_available = bool(source_session_id and agent_session_has_replay_frames(source_session_id))
        if replay_available:
            replay_ready_count += 1
        regenerated_preview = bool(replay_available)
        item = assign_showcase_fields({
            "dataset": "aerialvln",
            "split": "train",
            "index": len(family_map.get(family_id, {}).get("seeds", [])),
            "episode_id": episode_id,
            "scene_id": int(case.get("scene_id") or 16),
            "instruction": instruction,
            "verified_success": True,
            "verification_tier": "stable82",
            "verified_source": source_report,
            "demo_source_session_id": source_session_id,
            "demo_replay_available": replay_available,
            "demo_playback_mode": "regenerated_preview" if regenerated_preview else ("cached_replay" if replay_available else "live_regenerate"),
            "demo_replay_quality": (
                f"preview_0_{int(preview_entry['video_metadata'].get('width') or 0)}x"
                f"{int(preview_entry['video_metadata'].get('height') or 0)}"
                if regenerated_preview
                else ("historical" if replay_available else "pending")
            ),
            "demo_video_url": preview_entry.get("video_path") if regenerated_preview else None,
            "demo_video_metadata": preview_entry.get("video_metadata") if regenerated_preview else None,
            "gen_eval_label": case.get("gen_eval_label") or "GEN-SAFE",
            "showcase_ready": bool(case.get("showcase_ready", True)),
            "strict_collision_free": int(case.get("collision_rollback_count") or 0) == 0,
            "stream_safe": int(case.get("black_frame_drop_count") or 0) == 0,
            "stream_sample_safe": int(case.get("stream_sample_over_0_5s") or 0) == 0,
            "final_distance": first_finite_number(case.get("distance_to_goal"), case.get("final_distance")),
            "surface_clearance": case.get("surface_clearance"),
            "collision_rollback_count": case.get("collision_rollback_count"),
            "black_frame_drop_count": case.get("black_frame_drop_count"),
            "stream_sample_max_gap": case.get("stream_sample_max_gap"),
            "stop_reason": case.get("stop_reason"),
            "segment_count": len(case.get("segments") or []),
            "segments": case.get("segments") if isinstance(case.get("segments"), list) else [],
            "demo_family": family_id,
            "demo_family_label": family_label,
            "non_oracle_verified": bool(case.get("non_oracle_verified", True)),
        })
        item["demo_variants"] = scene16_demo_builder_variant_options(item)
        family_map[family_id]["count"] += 1
        family_map[family_id]["seeds"].append(item)
    for family in family_map.values():
        family["seeds"].sort(key=generalized_instruction_sort_key)
        family["seeds"] = family["seeds"][:max(1, int(max_seeds_per_family))]
    families = [family for family in family_map.values() if family["count"] > 0]
    families.sort(key=lambda family: (-int(family["count"]), family["label"]))
    return {
        "scene_id": 16,
        "mode": "scene16_demo_builder",
        "stable_count": sum(
            1
            for case in stable_cases
            if isinstance(case, dict) and str(case.get("episode_id") or "").strip() in preview_index
        ),
        "candidate_stable_count": len(stable_cases),
        "dataset_replay_ready_seed_count": len(replay_ready_cases),
        "replay_ready_seed_count": replay_ready_count,
        "replay_ready_only": True,
        "source_stable_pool": source_name,
        **meta,
        "supported_scope": "Scene 16 single-scene stable demo from all strict successes; cached replay is used when historical frames are available, otherwise the selected stable instruction can be regenerated online.",
        "families": families,
        "family_count": len(families),
        "seed_count": sum(len(family["seeds"]) for family in families),
    }


class LiveSession:
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

    ACTION_NAME_TO_ID = {value: key for key, value in ACTION_ID_TO_NAME.items()}

    def __init__(self, session_id, instruction, scene_id, max_steps, mode, action_sequence=None):
        self.session_id = session_id
        self.instruction = instruction
        self.scene_id = scene_id
        self.max_steps = max_steps
        self.mode = mode
        self.action_sequence = action_sequence or []
        self.status = "initializing"
        self.error = None
        self.step = 0
        self.action = "INIT"
        self.position = None
        self.path = []
        self.collision_rollback_count = 0
        self.last_collision = None
        self.last_frame = None
        self.last_frame_at = 0.0
        self.stop_event = threading.Event()
        self.frame_condition = threading.Condition()
        self.created_at = time.time()
        self.session_dir = SESSIONS_DIR / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self.run, name=f"live-session-{session_id}", daemon=True)

    def start(self):
        self.write_meta()
        self.thread.start()

    def write_meta(self):
        meta = {
            "session_id": self.session_id,
            "instruction": self.instruction,
            "scene_id": self.scene_id,
            "max_steps": self.max_steps,
            "mode": self.mode,
            "planned_actions": self.action_sequence,
            "created_at": self.created_at,
        }
        (self.session_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_trace(self):
        item = {
            "time": time.time(),
            "step": self.step,
            "action": self.action,
            "position": self.position,
            "status": self.status,
            "error": self.error,
        }
        with (self.session_dir / "trajectory.jsonl").open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")

    def snapshot(self):
        return {
            "session_id": self.session_id,
            "instruction": self.instruction,
            "scene_id": self.scene_id,
            "max_steps": self.max_steps,
            "mode": self.mode,
            "planned_actions": self.action_sequence,
            "status": self.status,
            "error": self.error,
            "step": self.step,
            "action": self.action,
            "position": self.position,
            "path": self.path[-1000:],
            "collision_rollback_count": self.collision_rollback_count,
            "last_collision": self.last_collision,
            "last_frame_at": self.last_frame_at,
            "age_sec": round(time.time() - self.created_at, 1),
        }

    def set_frame(self, data):
        with self.frame_condition:
            self.last_frame = data
            self.last_frame_at = time.time()
            self.frame_condition.notify_all()

    def get_frame(self, timeout=5.0):
        with self.frame_condition:
            if self.last_frame is None and self.status not in ("done", "error", "stopped"):
                self.frame_condition.wait(timeout=timeout)
            return self.last_frame

    def run(self):
        try:
            if self.mode == "camera":
                self.run_camera_only()
            else:
                self.run_scripted_flight()
        except Exception as exc:
            self.status = "error"
            self.error = str(exc)
            (self.session_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            self.append_trace()

    def connect_client(self):
        import airsim

        endpoint = airsim_endpoint()
        client = airsim.MultirotorClient(
            ip=endpoint["host"],
            port=endpoint["port"],
            timeout_value=5,
        )
        client.confirmConnection()
        return client

    def get_jpeg(self, client):
        import airsim
        from PIL import Image
        import io
        import numpy as np

        preview_camera = str(os.environ.get("AIRVLN_LIVE_PREVIEW_CAMERA", "preview_0") or "preview_0").strip()
        camera_names = [preview_camera] if preview_camera else []
        if "front_0" not in camera_names:
            camera_names.append("front_0")
        responses = []
        for camera_name in camera_names:
            try:
                candidate = client.simGetImages(
                    [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=False)],
                    vehicle_name="Drone_1",
                )
            except Exception:
                candidate = []
            if candidate and candidate[0].height > 0 and candidate[0].width > 0 and candidate[0].image_data_uint8:
                responses = candidate
                break
        if not responses:
            raise RuntimeError("AirSim returned no image response")
        response = responses[0]
        if response.height <= 0 or response.width <= 0 or not response.image_data_uint8:
            raise RuntimeError("AirSim returned an empty image")
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img = img1d.reshape(response.height, response.width, 3)
        buf = io.BytesIO()
        quality = min(95, max(60, int(os.environ.get("AIRVLN_LIVE_JPEG_QUALITY", "90"))))
        Image.fromarray(img).save(buf, format="JPEG", quality=quality, optimize=False)
        return buf.getvalue()

    def get_position(self, pose):
        return [
            float(pose.position.x_val),
            float(pose.position.y_val),
            float(pose.position.z_val),
        ]

    def remember_position(self):
        if self.position and len(self.position) >= 2:
            point = {
                "step": self.step,
                "position": self.position,
                "action": self.action,
            }
            if not self.path or self.path[-1].get("step") != point["step"]:
                self.path.append(point)
                if len(self.path) > 1200:
                    self.path = self.path[-1000:]

    def run_camera_only(self):
        client = self.connect_client()
        self.status = "running"
        self.action = "CAMERA"
        while not self.stop_event.is_set() and self.step < self.max_steps:
            pose = client.simGetVehiclePose(vehicle_name="Drone_1")
            self.position = self.get_position(pose)
            self.remember_position()
            self.set_frame(self.get_jpeg(client))
            self.append_trace()
            self.step += 1
            time.sleep(0.15)
        self.status = "stopped" if self.stop_event.is_set() else "done"
        self.append_trace()

    @classmethod
    def normalize_action(cls, action):
        if isinstance(action, int):
            if action not in cls.ACTION_ID_TO_NAME:
                raise ValueError(f"unknown action id: {action}")
            return cls.ACTION_ID_TO_NAME[action]
        text = str(action).strip().upper()
        if text.isdigit():
            return cls.normalize_action(int(text))
        aliases = {
            "FORWARD": "MOVE_FORWARD",
            "LEFT": "TURN_LEFT",
            "RIGHT": "TURN_RIGHT",
            "UP": "GO_UP",
            "DOWN": "GO_DOWN",
            "ASCEND": "GO_UP",
            "DESCEND": "GO_DOWN",
            "STRAFE_LEFT": "MOVE_LEFT",
            "STRAFE_RIGHT": "MOVE_RIGHT",
        }
        text = aliases.get(text, text)
        if text not in cls.ACTION_NAME_TO_ID:
            raise ValueError(f"unknown action name: {action}")
        return text

    @classmethod
    def parse_actions_from_text(cls, text):
        match = re.search(r"actions\s*:\s*(\[[^\]]+\])", text, flags=re.IGNORECASE)
        if not match:
            match = re.search(r"\[[\s\d,]+\]", text)
        if not match:
            return []
        raw = ast.literal_eval(match.group(1) if match.lastindex else match.group(0))
        if not isinstance(raw, list):
            raise ValueError("actions must be a list")
        return [cls.normalize_action(item) for item in raw]

    @staticmethod
    def degree_to_turns(value, default=4):
        if value is None:
            return default
        return max(1, min(48, int(round(float(value) / 15.0))))

    @staticmethod
    def distance_to_moves(value, default=4):
        if value is None:
            return default
        return max(1, min(80, int(round(float(value) / 5.0))))

    @classmethod
    def plan_actions_from_instruction(cls, text):
        lower = text.lower()
        actions = cls.parse_actions_from_text(text)
        if actions:
            return actions

        patterns = [
            (r"(take off|ascend|go up|rise)(?:[^.;,]*?(?:first floor|floor height|roof height|height))?", "GO_UP", 2),
            (r"(turn right|right)(?:\s+by)?\s*(\d+)?\s*(?:degrees|degree)?", "TURN_RIGHT", None),
            (r"(turn left|left)(?:\s+by)?\s*(\d+)?\s*(?:degrees|degree)?", "TURN_LEFT", None),
            (r"(move right|strafe right)", "MOVE_RIGHT", 4),
            (r"(move left|strafe left)", "MOVE_LEFT", 4),
            (r"(proceed|move forward|fly forward|go forward|forward|straight|ahead)(?:[^.;,]*?(\d+)\s*(?:m|meter|meters))?", "MOVE_FORWARD", None),
            (r"(descend|go down|land|lower)", "GO_DOWN", 2),
            (r"\bstop\b", "STOP", 1),
        ]

        matches = []
        for pattern, action, default_repeat in patterns:
            for match in re.finditer(pattern, lower):
                repeat = default_repeat
                if action in ("TURN_LEFT", "TURN_RIGHT"):
                    repeat = cls.degree_to_turns(match.group(2) if match.lastindex and match.group(2) else None)
                elif action == "MOVE_FORWARD":
                    repeat = cls.distance_to_moves(match.group(2) if match.lastindex and match.group(2) else None)
                matches.append((match.start(), action, repeat))

        matches.sort(key=lambda item: item[0])
        planned = []
        for _, action, repeat in matches:
            if action == "STOP":
                planned.append(action)
            else:
                planned.extend([action] * int(repeat))
        if not planned:
            planned = ["MOVE_FORWARD"] * 6
        if planned[-1] != "STOP":
            planned.append("STOP")
        return planned

    def plan_actions(self):
        if self.action_sequence:
            actions = [self.normalize_action(item) for item in self.action_sequence]
        else:
            actions = self.plan_actions_from_instruction(self.instruction)
        return actions[: self.max_steps]

    def apply_action(self, client, action):
        import airsim
        import math

        def clone_pose(source):
            return airsim.Pose(
                airsim.Vector3r(
                    float(source.position.x_val),
                    float(source.position.y_val),
                    float(source.position.z_val),
                ),
                airsim.Quaternionr(
                    float(source.orientation.x_val),
                    float(source.orientation.y_val),
                    float(source.orientation.z_val),
                    float(source.orientation.w_val),
                ),
            )

        start_pose = client.simGetVehiclePose(vehicle_name="Drone_1")
        pose = clone_pose(start_pose)
        pitch, roll, yaw = airsim.to_eularian_angles(start_pose.orientation)
        dx = 0.0
        dy = 0.0
        dz = 0.0
        if action == "MOVE_FORWARD":
            dx = math.cos(yaw) * 5.0
            dy = math.sin(yaw) * 5.0
        elif action == "MOVE_LEFT":
            dx = math.cos(yaw - math.pi / 2) * 5.0
            dy = math.sin(yaw - math.pi / 2) * 5.0
        elif action == "MOVE_RIGHT":
            dx = math.cos(yaw + math.pi / 2) * 5.0
            dy = math.sin(yaw + math.pi / 2) * 5.0
        elif action == "GO_UP":
            dz = -2.0
        elif action == "GO_DOWN":
            dz = 2.0
        elif action == "TURN_LEFT":
            yaw -= math.radians(15)
        elif action == "TURN_RIGHT":
            yaw += math.radians(15)
        elif action == "STOP":
            return pose

        pose.position.x_val += dx
        pose.position.y_val += dy
        pose.position.z_val += dz
        pose.orientation = airsim.to_quaternion(pitch, roll, yaw)
        moving = abs(dx) + abs(dy) + abs(dz) > 1e-6
        tween_steps = 5 if moving else 1
        safe_pose = clone_pose(start_pose)
        try:
            collision_before = client.simGetCollisionInfo(vehicle_name="Drone_1")
            collision_before_time = int(getattr(collision_before, "time_stamp", 0) or 0)
        except Exception:
            collision_before_time = 0

        for tween_index in range(1, tween_steps + 1):
            alpha = tween_index / tween_steps
            tween_pose = clone_pose(pose)
            tween_pose.position.x_val = start_pose.position.x_val + dx * alpha
            tween_pose.position.y_val = start_pose.position.y_val + dy * alpha
            tween_pose.position.z_val = start_pose.position.z_val + dz * alpha
            client.simSetVehiclePose(tween_pose, ignore_collision=False, vehicle_name="Drone_1")
            actual_pose = client.simGetVehiclePose(vehicle_name="Drone_1")
            try:
                collision_after = client.simGetCollisionInfo(vehicle_name="Drone_1")
            except Exception:
                collision_after = None
            collision_after_time = int(getattr(collision_after, "time_stamp", 0) or 0)
            new_collision = bool(
                collision_after is not None
                and getattr(collision_after, "has_collided", False)
                and collision_after_time > collision_before_time
            )
            position_error = math.sqrt(
                (float(actual_pose.position.x_val) - float(tween_pose.position.x_val)) ** 2
                + (float(actual_pose.position.y_val) - float(tween_pose.position.y_val)) ** 2
                + (float(actual_pose.position.z_val) - float(tween_pose.position.z_val)) ** 2
            )
            if new_collision or (moving and position_error > 0.35):
                client.simSetVehiclePose(safe_pose, ignore_collision=True, vehicle_name="Drone_1")
                self.collision_rollback_count += 1
                self.last_collision = {
                    "step": self.step,
                    "action": action,
                    "reason": "airsim_collision" if new_collision else "pose_sweep_blocked",
                    "object_name": str(getattr(collision_after, "object_name", "") or ""),
                    "safe_position": self.get_position(safe_pose),
                }
                return safe_pose
            safe_pose = clone_pose(actual_pose)
            collision_before_time = max(collision_before_time, collision_after_time)
            time.sleep(0.03)
        return safe_pose

    def run_scripted_flight(self):
        client = self.connect_client()
        actions = self.plan_actions()
        self.status = "running"
        for action in actions:
            if self.stop_event.is_set():
                self.status = "stopped"
                break
            self.action = action
            pose = self.apply_action(client, action)
            self.position = self.get_position(pose)
            self.remember_position()
            self.set_frame(self.get_jpeg(client))
            self.append_trace()
            self.step += 1
            if action == "STOP":
                self.status = "done"
                break
            time.sleep(0.25)
        if self.status == "running":
            self.status = "done"
        self.append_trace()


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:name>")
def static_file(name):
    return send_from_directory(STATIC_DIR, name)


# Also expose the two root-relative assets used when index.html is opened
# directly from the static directory during local visual inspection.
@app.route("/style.css")
def root_style_file():
    return send_from_directory(STATIC_DIR, "style.css")


@app.route("/app.js")
def root_app_file():
    return send_from_directory(STATIC_DIR, "app.js")


@app.route("/api/health")
def health():
    include_airsim = request.args.get("airsim", "1") != "0"
    airsim_host = request.args.get("airsim_host") or None
    airsim_port = request.args.get("airsim_port") or None
    payload = {
        "ok": True,
        "host": socket.gethostname(),
        "cwd": str(AIRVLN_ROOT),
        "runtime_dir": str(RUNTIME_DIR),
        "imports": optional_imports(),
        "sessions": len(sessions),
    }
    if include_airsim:
        payload["airsim"] = airsim_health(host=airsim_host, port=airsim_port)
    return jsonify(payload)


@app.route("/api/scenes")
def scenes():
    return jsonify({"scenes": list(range(1, 26))})


@app.route("/api/gpus")
def gpus():
    inventory = gpu_inventory()
    status = 200 if inventory["ok"] else 503
    return jsonify(inventory), status


@app.route("/api/dataset/search")
def dataset_search():
    query = request.args.get("q", "")
    split = request.args.get("split") or None
    dataset = request.args.get("dataset", "aerialvln")
    scene_id = request.args.get("scene_id")
    try:
        scene_id = int(scene_id) if scene_id not in (None, "", "all") else None
        limit = int(request.args.get("limit", "20"))
    except (TypeError, ValueError):
        return jsonify({"error": "scene_id and limit must be integers"}), 400
    matches = search_dataset_episodes(
        query=query,
        split=split,
        dataset=dataset,
        scene_id=scene_id,
        limit=limit,
    )
    return jsonify({"matches": matches, "count": len(matches)})


@app.route("/api/dataset/instructions")
def dataset_instructions():
    split = request.args.get("split") or None
    dataset = request.args.get("dataset", "aerialvln")
    scene_id = request.args.get("scene_id")
    try:
        scene_id = int(scene_id) if scene_id not in (None, "", "all") else None
    except (TypeError, ValueError):
        return jsonify({"error": "scene_id must be an integer"}), 400
    verified_only = parse_bool_arg(request.args.get("verified_only"), default=True)
    safe_only = parse_bool_arg(request.args.get("safe_only"), default=False)
    include_variants = parse_bool_arg(request.args.get("include_variants"), default=False)
    try:
        variants_per_safe = max(0, min(5, int(request.args.get("variants_per_safe") or 1)))
    except (TypeError, ValueError):
        return jsonify({"error": "variants_per_safe must be an integer"}), 400
    showcase_filter = str(request.args.get("showcase_filter") or "all").lower()
    try:
        min_quality_score = float(request.args.get("min_quality_score") or 0.0)
    except (TypeError, ValueError):
        min_quality_score = 0.0
    dedupe_episode = parse_bool_arg(request.args.get("dedupe_episode"), default=False)
    # The LIVE-OK registry can be updated by offline evaluation jobs while the
    # service is running, so refresh these lightweight catalogs per request.
    cached_current_smooth_success_cases.cache_clear()
    cached_verified_success_cases.cache_clear()
    all_instructions = list_dataset_instructions(
        split=split,
        dataset=dataset,
        scene_id=scene_id,
        verified_only=verified_only,
    )
    variant_eval = load_scene16_variant_eval_cases()
    safe_instructions = [item for item in all_instructions if is_safe_display_instruction(item)] if safe_only else list(all_instructions)

    def build_showcase_pool(filter_mode):
        items = filter_showcase_items(safe_instructions, showcase_filter=filter_mode)
        if dedupe_episode:
            items = dedupe_instructions_by_episode(items)
        if include_variants:
            items = add_instruction_variants(items, max_variants=variants_per_safe, variant_eval=variant_eval)
            items = filter_showcase_items(items, showcase_filter=filter_mode)
            if dedupe_episode:
                items = dedupe_instructions_by_episode(items)
        return items

    display_pool_unfiltered = build_showcase_pool("all")
    display_pool = sorted(
        filter_quality_items(display_pool_unfiltered, min_quality_score=min_quality_score),
        key=generalized_instruction_sort_key,
    )
    instructions = sorted(
        filter_quality_items(build_showcase_pool(showcase_filter), min_quality_score=min_quality_score),
        key=generalized_instruction_sort_key,
    )
    original_returned_count = sum(1 for item in instructions if not item.get("instruction_variant"))
    variant_count = sum(1 for item in instructions if item.get("instruction_variant"))
    variant_ok_count = sum(1 for item in instructions if item.get("instruction_variant") and item.get("variant_verified_success"))
    showcase_tier_counts = {
        "frontline": sum(1 for item in instructions if item.get("display_showcase_tier") == "frontline"),
        "stable": sum(1 for item in instructions if item.get("display_showcase_tier") == "stable"),
        "caution": sum(1 for item in instructions if item.get("display_showcase_tier") == "caution"),
        "unverified": sum(1 for item in instructions if item.get("display_showcase_tier") == "unverified"),
    }
    display_pool_tier_counts = {
        "frontline": sum(1 for item in display_pool if item.get("display_showcase_tier") == "frontline"),
        "stable": sum(1 for item in display_pool if item.get("display_showcase_tier") == "stable"),
        "caution": sum(1 for item in display_pool if item.get("display_showcase_tier") == "caution"),
        "unverified": sum(1 for item in display_pool if item.get("display_showcase_tier") == "unverified"),
    }
    def quality_counts(items):
        return {
            "demo_ready": sum(1 for item in items if showcase_numeric(item, "display_quality_score", 0.0) >= 90),
            "strong": sum(1 for item in items if 80 <= showcase_numeric(item, "display_quality_score", 0.0) < 90),
            "usable": sum(1 for item in items if 70 <= showcase_numeric(item, "display_quality_score", 0.0) < 80),
            "caution": sum(1 for item in items if 55 <= showcase_numeric(item, "display_quality_score", 0.0) < 70),
            "risky": sum(1 for item in items if showcase_numeric(item, "display_quality_score", 0.0) < 55),
        }
    def unique_episode_count(items, predicate=None):
        episodes = set()
        for item in items:
            if predicate is not None and not predicate(item):
                continue
            episode_id = str(item.get("episode_id") or "").strip()
            if episode_id:
                episodes.add(episode_id)
        return len(episodes)
    def is_demo_ready(item):
        return showcase_numeric(item, "display_quality_score", 0.0) >= 90
    def is_current_smooth(item):
        return item.get("verification_tier") == "current_smooth"
    return jsonify(
        {
            "instructions": instructions,
            "count": len(instructions),
            "unique_episode_count": unique_episode_count(instructions),
            "demo_ready_unique_episode_count": unique_episode_count(instructions, is_demo_ready),
            "current_smooth_unique_episode_count": unique_episode_count(instructions, is_current_smooth),
            "demo_ready_current_smooth_unique_episode_count": unique_episode_count(
                instructions,
                lambda item: is_demo_ready(item) and is_current_smooth(item),
            ),
            "original_count": original_returned_count,
            "variant_count": variant_count,
            "variant_ok_count": variant_ok_count,
            "showcase_tier_counts": showcase_tier_counts,
            "display_pool_count": len(display_pool),
            "display_pool_unique_episode_count": unique_episode_count(display_pool),
            "display_pool_demo_ready_unique_episode_count": unique_episode_count(display_pool, is_demo_ready),
            "display_pool_current_smooth_unique_episode_count": unique_episode_count(display_pool, is_current_smooth),
            "display_pool_demo_ready_current_smooth_unique_episode_count": unique_episode_count(
                display_pool,
                lambda item: is_demo_ready(item) and is_current_smooth(item),
            ),
            "display_pool_unfiltered_count": len(display_pool_unfiltered),
            "display_pool_original_count": sum(1 for item in display_pool if not item.get("instruction_variant")),
            "display_pool_variant_count": sum(1 for item in display_pool if item.get("instruction_variant")),
            "display_pool_variant_ok_count": sum(
                1 for item in display_pool if item.get("instruction_variant") and item.get("variant_verified_success")
            ),
            "display_pool_tier_counts": display_pool_tier_counts,
            "display_pool_quality_counts": quality_counts(display_pool),
            "display_pool_quality_counts_unfiltered": quality_counts(display_pool_unfiltered),
            "active_quality_counts": quality_counts(instructions),
            "showcase_filter": showcase_filter,
            "min_quality_score": min_quality_score,
            "dedupe_episode": dedupe_episode,
            "all_count": len(all_instructions),
            "safe_only": safe_only,
            "include_variants": include_variants,
            "verified_only": verified_only,
            "verified_source_count": len(cached_verified_success_cases()) if verified_only else None,
            "safe_display_count": sum(1 for item in all_instructions if is_safe_display_instruction(item)),
            "safe_unfiltered_count": len(safe_instructions),
            "verified_success_count": sum(1 for item in all_instructions if item.get("verified_success")),
            "gen_safe_count": sum(1 for item in all_instructions if item.get("gen_eval_label") == "GEN-SAFE"),
            "gen_risk_count": sum(1 for item in all_instructions if item.get("gen_eval_label") == "GEN-RISK"),
            "gen_raw_count": sum(1 for item in all_instructions if item.get("gen_eval_label") == "GEN-RAW"),
        }
    )


@app.route("/api/scene16/demo-builder")
def scene16_demo_builder():
    try:
        max_seeds_per_family = max(1, min(120, int(request.args.get("max_seeds_per_family") or 120)))
    except (TypeError, ValueError):
        return jsonify({"error": "max_seeds_per_family must be an integer"}), 400
    return jsonify(build_scene16_demo_builder_payload(max_seeds_per_family=max_seeds_per_family))


@app.route("/api/grounding", methods=["POST"])
def grounding_preview():
    data, error_response = request_json_object()
    if error_response is not None:
        return error_response
    instruction = str(data.get("instruction", "")).strip()
    if not instruction:
        return jsonify({"error": "instruction is required"}), 400
    split = str(data.get("split") or DEFAULT_PHASE_A_SPLIT)
    dataset = str(data.get("dataset") or "aerialvln")
    scene_id = data.get("scene_id")
    if scene_id in (None, "", "all"):
        scene_id = None
    elif isinstance(scene_id, bool) or not isinstance(scene_id, int):
        return jsonify({"error": "scene_id must be an integer"}), 400
    grounding = build_grounding(instruction, split=split, scene_id=scene_id, dataset=dataset)
    return jsonify({
        "grounding": grounding,
        "summary": compact_grounding_view(grounding),
    })


@app.route("/api/plan", methods=["POST"])
def plan():
    data, error_response = request_json_object()
    if error_response is not None:
        return error_response
    instruction = str(data.get("instruction", "")).strip()
    action_sequence = data.get("actions", [])
    if not isinstance(action_sequence, list):
        return jsonify({"error": "actions must be a list"}), 400
    if action_sequence:
        try:
            actions = [LiveSession.normalize_action(item) for item in action_sequence]
        except (TypeError, ValueError, KeyError):
            return jsonify({"error": "actions contains an invalid action"}), 400
    else:
        actions = LiveSession.plan_actions_from_instruction(instruction)
    max_steps, error_response = request_integer(data, "max_steps", 120)
    if error_response is not None:
        return error_response, 400
    max_steps = max(1, min(max_steps, 500))
    actions = actions[:max_steps]
    return jsonify({
        "actions": actions,
        "action_ids": [LiveSession.ACTION_NAME_TO_ID[item] for item in actions],
        "count": len(actions),
    })


@app.route("/api/sessions")
def list_sessions():
    with sessions_lock:
        return jsonify({"sessions": [session.snapshot() for session in sessions.values()]})


def summarize_model_run(run_dir):
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    frames = sorted((run_dir / "frames").glob("*.jpg"))
    trace = meta.get("trace") or []
    last = trace[-1] if trace else None
    return {
        "run_id": run_dir.name,
        "num_frames": len(frames),
        "episode_id": meta.get("episode_id"),
        "scene_id": meta.get("scene_id"),
        "instruction": meta.get("instruction"),
        "ckpt": meta.get("ckpt"),
        "last": last,
        "trace": trace[-1000:],
        "frames": [f"/model-runs/{run_dir.name}/frames/{frame.name}" for frame in frames],
    }


@app.route("/api/model-runs")
def model_runs():
    runs = []
    for run_dir in sorted(MODEL_RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        summary = summarize_model_run(run_dir)
        if summary is not None:
            runs.append(summary)
    return jsonify({"runs": runs})


@app.route("/api/model-runs/latest")
def latest_model_run():
    for run_dir in sorted(MODEL_RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        summary = summarize_model_run(run_dir)
        if summary is not None:
            return jsonify(summary)
    return jsonify({"error": "no model runs"}), 404


def is_active(session):
    return session.status not in ("done", "error", "stopped")


def active_session_snapshot():
    with sessions_lock:
        for session in sessions.values():
            if is_active(session):
                return session.snapshot()
    return None


def register_session_if_available(session_id, session_factory, availability_error=None):
    """Atomically reject competing starts before registering a session."""
    with sessions_lock:
        for existing in sessions.values():
            if is_active(existing):
                return None, {
                    "error": "another session is running",
                    "running": existing.snapshot(),
                }
        if availability_error is not None:
            error = availability_error()
            if error is not None:
                return None, error
        session = session_factory()
        sessions[session_id] = session
        try:
            session.start()
        except Exception:
            sessions.pop(session_id, None)
            raise
        return session, None


def request_json_object():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({"error": "request body must be a JSON object"}), 400)
    return data, None


def request_integer(data, name, default):
    if name not in data:
        try:
            return int(default), None
        except (TypeError, ValueError):
            return None, jsonify({"error": f"{name} must be an integer"})
    value = data[name]
    if isinstance(value, bool) or not isinstance(value, int):
        return None, jsonify({"error": f"{name} must be an integer"})
    return value, None


def request_boolean(data, name, default):
    value = data.get(name, default)
    if not isinstance(value, bool):
        return None, jsonify({"error": f"{name} must be a boolean"})
    return value, None


def request_number(data, name, default, integer=False):
    value = data.get(name, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return None, jsonify({"error": f"{name} must be a number"})
    if integer and not isinstance(value, int):
        return None, jsonify({"error": f"{name} must be an integer"})
    return value, None


_V014_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_V014_ID_RE = re.compile(r"^v014-[0-9a-f]{32}$")
_V014_AUTHORIZATION_FORMAT = "scene16_right_constraint_recovery_authorization_v1"
_V014_LOCK_FORMAT = "scene16_right_constraint_recovery_bundle_lock_v1"
_V014_INPUT_PATHS = {
    "controller": ("live_server/model_live_once.py", "controller/model_live_once.py"),
    "wrapper": ("scripts/run_scene16_single_scene_batch_eval.py", "wrapper/run_scene16_single_scene_batch_eval.py"),
    "evaluator": ("scripts/run_scene16_gen_eval.py", "evaluator/run_scene16_gen_eval.py"),
    "source_selector": (
        "live_server/runtime/scene16_single_scene_selector_v008_matched_20260816.json",
        "source/scene16_single_scene_selector_v008_matched_20260816.json",
    ),
    "v014_selector": (
        "live_server/runtime/scene16_single_scene_selector_v014_right_constraint_recovery_telemetry_20260816.json",
        "selector/scene16_single_scene_selector_v014_right_constraint_recovery_telemetry_20260816.json",
    ),
    "ranker": (
        "live_server/runtime/segment_grounding/scene16_single_scene_ranker_v006_full_stable_pool_20260815",
        "ranker/scene16_single_scene_ranker_v006_full_stable_pool_20260815",
    ),
}
_V014_CONTROLLER_CONFIG = {
    "max_steps": 900,
    "target_radius": 12,
    "learned_max_age": 34,
    "z_clamp": 6,
    "min_xy_step": 24,
    "max_xy_step": 54,
    "resample_worse_streak": 4,
    "resample_regression": 55,
    "goal_regression_worse_streak": 4,
    "goal_regression": 55.0,
    "goal_regression_min_age": 8,
    "cruise_altitude": 42,
    "descent_altitude": 18,
    "segment_switch_confidence": 0.48,
    "segment_min_steps": 3,
    "segment_progress_distance": 22,
    "disable_oracle_goal_homing": True,
    "disable_oracle_path_homing": True,
    "max_black_drops": 0,
    "max_collision_rollbacks": 0,
}
_V014_ASSERTION_FIELDS = {
    "authorization_path",
    "authorization_sha256",
    "agent_session_id",
    "model_run_id",
    "expected_controller_sha256",
    "controller_config",
}
_V014_UNPROVEN_CONTROLLER_FIELDS = {
    "segment_grounding_ridge",
    "segment_grounding_final_ridge",
    "learned_segment_target_radius",
    "learned_segment_max_age",
    "learned_segment_z_clamp",
    "learned_segment_min_xy_step",
    "learned_segment_max_xy_step",
    "learned_segment_conservative_max_xy_step",
    "learned_segment_resample_worse_streak",
    "learned_segment_resample_regression",
    "learned_segment_goal_regression_worse_streak",
    "learned_segment_goal_regression",
    "learned_segment_goal_regression_min_age",
    "learned_segment_cruise_altitude",
    "learned_segment_descent_altitude",
    "segment_switch_confidence",
    "segment_min_steps",
    "segment_progress_distance",
}


def _canonical_v014_json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _v014_error(message):
    return {"error": f"v014 assertion rejected: {message}"}


def _v014_entry_kind(path):
    info = os.lstat(path)
    if _is_reparse_or_link(info.st_mode, getattr(info, "st_file_attributes", 0)):
        raise ValueError("filesystem entry is a symlink or reparse point")
    if stat.S_ISREG(info.st_mode):
        return "file"
    if stat.S_ISDIR(info.st_mode):
        return "directory"
    raise ValueError("filesystem entry is not a regular file or directory")


def _v014_validate_ancestry(root, components):
    path = Path(root)
    if _v014_entry_kind(path) != "directory":
        raise ValueError("runtime root is not a directory")
    for component in components:
        path /= component
        if _v014_entry_kind(path) != "directory":
            raise ValueError("authorization ancestor is not a directory")
    return path


def _v014_walk_tree(root):
    if _v014_entry_kind(root) != "directory":
        raise ValueError("bundle root is not a directory")
    files = {}
    directories = set()

    def visit(directory, relative):
        if relative:
            directories.add(relative)
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            child = Path(entry.path)
            child_relative = f"{relative}/{entry.name}" if relative else entry.name
            kind = _v014_entry_kind(child)
            if kind == "directory":
                visit(child, child_relative)
            else:
                files[child_relative] = child

    visit(Path(root), "")
    return files, directories


def _v014_framed_digest(entries):
    framed = bytearray()
    for relative, content in sorted(entries):
        framed.extend(relative.encode("utf-8"))
        framed.extend(b"\0")
        framed.extend(content)
        framed.extend(b"\0")
    return hashlib.sha256(framed).hexdigest()


def _v014_validate_bundle(bundle_root, authorization, digest, agent_session_id, model_run_id):
    files, directories = _v014_walk_tree(bundle_root)
    expected_files = {"authorization.json", "bundle.lock"}
    ranker_prefix = _V014_INPUT_PATHS["ranker"][1]
    for key, (_source, bundled) in _V014_INPUT_PATHS.items():
        if key != "ranker":
            expected_files.add(bundled)
    ranker_files = {path for path in files if path.startswith(ranker_prefix + "/")}
    if not ranker_files or set(files) != expected_files | ranker_files:
        raise ValueError("bundle file set mismatch")
    expected_directories = set()
    for relative in files:
        parent = Path(relative).parent
        while str(parent) != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if directories != expected_directories:
        raise ValueError("bundle directory set mismatch")
    for key, (_source, bundled) in _V014_INPUT_PATHS.items():
        if key == "ranker":
            digest_entries = [
                (path[len(ranker_prefix) + 1:], files[path].read_bytes())
                for path in ranker_files
            ]
            actual_digest = _v014_framed_digest(digest_entries)
        else:
            actual_digest = hashlib.sha256(files[bundled].read_bytes()).hexdigest()
        if actual_digest != authorization["inputs"][key]["sha256"]:
            raise ValueError(f"bundle input digest mismatch: {key}")
    try:
        lock_bytes = files["bundle.lock"].read_bytes()
        lock = json.loads(lock_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("bundle lock cannot be read") from exc
    expected_lock_keys = {"format", "authorization_sha256", "bundle_sha256", "agent_session_id", "model_run_id"}
    if (
        not isinstance(lock, dict)
        or set(lock) != expected_lock_keys
        or _canonical_v014_json(lock) != lock_bytes
        or lock.get("format") != _V014_LOCK_FORMAT
        or lock.get("authorization_sha256") != digest
        or lock.get("agent_session_id") != agent_session_id
        or lock.get("model_run_id") != model_run_id
        or not isinstance(lock.get("bundle_sha256"), str)
        or not _V014_SHA256_RE.fullmatch(lock["bundle_sha256"])
    ):
        raise ValueError("bundle lock disagreement")
    content_digest = _v014_framed_digest(
        (relative, path.read_bytes()) for relative, path in files.items() if relative != "bundle.lock"
    )
    if content_digest != lock["bundle_sha256"]:
        raise ValueError("bundle digest mismatch")


def validate_v014_start_assertions(data):
    """Validate the optional immutable start assertion before session allocation."""
    present = _V014_ASSERTION_FIELDS.intersection(data)
    if not present:
        return None, None
    if present != _V014_ASSERTION_FIELDS:
        return None, _v014_error("all assertion fields are required")

    digest = data["authorization_sha256"]
    controller_digest = data["expected_controller_sha256"]
    agent_session_id = data["agent_session_id"]
    model_run_id = data["model_run_id"]
    if (
        not isinstance(digest, str)
        or not _V014_SHA256_RE.fullmatch(digest)
        or not isinstance(controller_digest, str)
        or not _V014_SHA256_RE.fullmatch(controller_digest)
        or not isinstance(agent_session_id, str)
        or not _V014_ID_RE.fullmatch(agent_session_id)
        or not isinstance(model_run_id, str)
        or not _V014_ID_RE.fullmatch(model_run_id)
    ):
        return None, _v014_error("digest or ID format is invalid")

    authorization_path = data["authorization_path"]
    expected_relative_path = f"live_server/runtime/v014_execution_bundles/{digest}/authorization.json"
    if not isinstance(authorization_path, str) or not authorization_path.isascii() or authorization_path != expected_relative_path:
        return None, _v014_error("authorization path is not permitted")
    authorization_components = authorization_path.split("/")
    try:
        authorization_parent = _v014_validate_ancestry(AIRVLN_ROOT, authorization_components[:-1])
        authorization_file = authorization_parent / authorization_components[-1]
        if _v014_entry_kind(authorization_file) != "file":
            return None, _v014_error("authorization cannot be read")
        authorization_bytes = authorization_file.read_bytes()
        authorization = json.loads(authorization_bytes.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None, _v014_error("authorization cannot be read")
    if _canonical_v014_json(authorization) != authorization_bytes or hashlib.sha256(authorization_bytes).hexdigest() != digest:
        return None, _v014_error("authorization bytes or digest mismatch")

    expected_auth_keys = {"format", "remote_worktree", "source_revision", "selected_id", "controller_config", "inputs"}
    if (
        not isinstance(authorization, dict)
        or set(authorization) != expected_auth_keys
        or authorization["format"] != _V014_AUTHORIZATION_FORMAT
    ):
        return None, _v014_error("authorization schema mismatch")
    if not all(isinstance(authorization[key], str) and authorization[key].isascii() and authorization[key] for key in ("remote_worktree", "source_revision", "selected_id")):
        return None, _v014_error("authorization identity is invalid")
    config = authorization["controller_config"]
    if (
        config != _V014_CONTROLLER_CONFIG
        or not isinstance(data["controller_config"], dict)
        or data["controller_config"] != config
        or any(type(config[key]) is not type(value) for key, value in _V014_CONTROLLER_CONFIG.items())
    ):
        return None, _v014_error("controller config mismatch")
    inputs = authorization["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != set(_V014_INPUT_PATHS):
        return None, _v014_error("authorization inputs mismatch")
    for key, (source_path, bundled_path) in _V014_INPUT_PATHS.items():
        record = inputs[key]
        if (
            not isinstance(record, dict)
            or set(record) != {"source_path", "bundled_path", "sha256"}
            or record.get("source_path") != source_path
            or record.get("bundled_path") != bundled_path
            or not isinstance(record.get("sha256"), str)
            or not _V014_SHA256_RE.fullmatch(record["sha256"])
        ):
            return None, _v014_error(f"authorization input mismatch: {key}")
    actual_controller_digest = hashlib.sha256((ROOT / "model_live_once.py").read_bytes()).hexdigest()
    if inputs["controller"]["sha256"] != actual_controller_digest or controller_digest != actual_controller_digest:
        return None, _v014_error("deployed controller digest mismatch")
    expected_agent_session_id = "v014-" + hashlib.sha256(
        f"scene16-v014-session\0{digest}\0{authorization['selected_id']}".encode("utf-8")
    ).hexdigest()[:32]
    expected_model_run_id = "v014-" + hashlib.sha256(
        f"scene16-v014-run\0{digest}\0{authorization['selected_id']}".encode("utf-8")
    ).hexdigest()[:32]
    if (agent_session_id, model_run_id) != (expected_agent_session_id, expected_model_run_id):
        return None, _v014_error("derived IDs mismatch")
    if (
        data.get("max_steps", 900) != 900
        or data.get("enable_oracle_goal_homing", False) is not False
        or data.get("enable_oracle_path_homing", False) is not False
        or data.get("enable_learned_segment_homing", False) is not False
        or data.get("enable_scene16_support_ranker", False) is not True
        or data.get("scene16_support_ranker") != inputs["ranker"]["bundled_path"]
    ):
        return None, _v014_error("strict runtime settings mismatch")
    if _V014_UNPROVEN_CONTROLLER_FIELDS.intersection(data):
        return None, _v014_error("unproven learned-homing input")
    if data.get("episode_id") != authorization["selected_id"]:
        return None, _v014_error("selected episode mismatch")
    try:
        bundle_root = authorization_file.parent
        _v014_validate_bundle(bundle_root, authorization, digest, agent_session_id, model_run_id)
    except (OSError, ValueError):
        return None, _v014_error("bundle contents or lock mismatch")
    return {
        "authorization_sha256": digest,
        "agent_session_id": agent_session_id,
        "model_run_id": model_run_id,
        "controller_sha256": actual_controller_digest,
        "controller_config": config,
        "scene16_support_ranker": str(bundle_root / inputs["ranker"]["bundled_path"]),
        "trace_contract": {
            "agent_session_id": agent_session_id,
            "model_run_id": model_run_id,
            "trace_path": f"live_server/runtime/agent_sessions/{agent_session_id}/model_runs/{model_run_id}/trace.jsonl",
        },
    }, None


def _is_reparse_or_link(mode, attributes):
    return stat.S_ISLNK(mode) or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _v014_directory_identity(path):
    info = os.lstat(path)
    if _is_reparse_or_link(info.st_mode, getattr(info, "st_file_attributes", 0)) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("v014 namespace contains a redirected or non-directory path")
    return (info.st_dev, info.st_ino)


def reserve_v014_trace_namespace(agent_session_id, model_run_id):
    """Atomically allocate the complete v014 runtime namespace before session construction."""
    for path in (RUNTIME_DIR, AGENT_SESSIONS_DIR):
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        _v014_directory_identity(path)
    session_dir = AGENT_SESSIONS_DIR / agent_session_id
    try:
        os.mkdir(session_dir)
    except FileExistsError as exc:
        raise FileExistsError("v014 trace namespace already exists") from exc
    model_out_root = session_dir / "model_runs"
    model_run_dir = model_out_root / model_run_id
    paths = (session_dir, model_out_root, model_run_dir, session_dir / "frames", session_dir / "preview_frames", session_dir / "logs")
    identities = {}
    for index, path in enumerate(paths):
        try:
            if index:
                os.mkdir(path)
        except FileNotFoundError:
            raise ValueError("v014 namespace parent disappeared during allocation")
        identities[str(path)] = _v014_directory_identity(path)
    namespace_proof = hashlib.sha256(
        _canonical_v014_json({
            "agent_session_id": agent_session_id,
            "model_run_id": model_run_id,
            "identities": identities,
        })
    ).hexdigest()
    return {
        "agent_session_id": agent_session_id,
        "model_run_id": model_run_id,
        "session_dir": session_dir,
        "model_out_root": model_out_root,
        "model_run_dir": model_run_dir,
        "identities": identities,
        "namespace_proof": namespace_proof,
    }


def cleanup_v014_trace_namespace(namespace, agent_session_id):
    """Remove only an unchanged, unregistered v014 reservation with rmdir."""
    try:
        model_run_id = namespace["model_run_id"]
        session_dir = namespace["session_dir"]
        model_out_root = namespace["model_out_root"]
        model_run_dir = namespace["model_run_dir"]
        identities = namespace["identities"]
        if (
            namespace.get("agent_session_id") != agent_session_id
            or not _V014_ID_RE.fullmatch(agent_session_id)
            or not isinstance(model_run_id, str)
            or not _V014_ID_RE.fullmatch(model_run_id)
            or session_dir != AGENT_SESSIONS_DIR / agent_session_id
            or model_out_root != session_dir / "model_runs"
            or model_run_dir != model_out_root / model_run_id
        ):
            return False
        paths = (
            session_dir,
            model_out_root,
            model_run_dir,
            session_dir / "frames",
            session_dir / "preview_frames",
            session_dir / "logs",
        )
        if set(identities) != {str(path) for path in paths}:
            return False
        proof = hashlib.sha256(_canonical_v014_json({
            "agent_session_id": agent_session_id,
            "model_run_id": model_run_id,
            "identities": identities,
        })).hexdigest()
        if namespace.get("namespace_proof") != proof:
            return False
        # Keep registration and deletion mutually exclusive for this session ID.
        with sessions_lock:
            if agent_session_id in sessions:
                return False
            for path in paths:
                if _v014_directory_identity(path) != identities[str(path)]:
                    return False
            expected_children = {
                session_dir: {"model_runs", "frames", "preview_frames", "logs"},
                model_out_root: {model_run_id},
                model_run_dir: set(),
                session_dir / "frames": set(),
                session_dir / "preview_frames": set(),
                session_dir / "logs": set(),
            }
            for path, expected in expected_children.items():
                if {entry.name for entry in os.scandir(path)} != expected:
                    return False
            for path in (
                model_run_dir,
                session_dir / "frames",
                session_dir / "preview_frames",
                session_dir / "logs",
                model_out_root,
                session_dir,
            ):
                if _v014_directory_identity(path) != identities[str(path)]:
                    return False
                os.rmdir(path)
        return True
    except (KeyError, OSError, TypeError, ValueError):
        return False


def validate_v014_trace_namespace(agent_session_id, model_run_id):
    """Reject existing or redirected v014 trace paths before session allocation."""
    session_dir = AGENT_SESSIONS_DIR / agent_session_id
    run_parent = session_dir / "model_runs"
    target = run_parent / model_run_id
    for path in (RUNTIME_DIR, AGENT_SESSIONS_DIR, session_dir, run_parent):
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            continue
        except OSError:
            return _v014_error("trace namespace cannot be inspected")
        if _is_reparse_or_link(info.st_mode, getattr(info, "st_file_attributes", 0)) or not stat.S_ISDIR(info.st_mode):
            return _v014_error("trace namespace parent is malformed")
    try:
        os.lstat(target)
    except FileNotFoundError:
        return None
    except OSError:
        return _v014_error("trace namespace cannot be inspected")
    return _v014_error("trace namespace already exists")


def read_agent_locks():
    locks_dir = RUNTIME_DIR / "locks"
    if not locks_dir.exists():
        return []
    locks = []
    for lock_path in sorted(locks_dir.glob("*.lock")):
        item = {"name": lock_path.name, "path": str(lock_path)}
        try:
            item["data"] = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception as exc:
            item["error"] = str(exc)
        locks.append(item)
    return locks


@app.route("/model-runs/<run_id>/frames/<filename>")
def model_run_frame(run_id, filename):
    safe_dir = MODEL_RUNS_DIR / run_id / "frames"
    return send_from_directory(safe_dir, filename)


def agent_session_frame_dirs(session_id):
    session_dir = AGENT_SESSIONS_DIR / session_id
    dirs = [session_dir / "frames"]
    meta_path = session_dir / "meta.json"
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        model_run_id = str(metadata.get("model_run_id") or "").strip()
        if model_run_id:
            dirs.append(session_dir / "model_runs" / model_run_id / "frames")
            dirs.append(MODEL_RUNS_DIR / model_run_id / "frames")
    except Exception:
        pass
    model_runs_dir = session_dir / "model_runs"
    if model_runs_dir.is_dir():
        try:
            for run_dir in sorted(model_runs_dir.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
                if run_dir.is_dir():
                    dirs.append(run_dir / "frames")
        except OSError:
            pass
    unique_dirs = []
    seen = set()
    for frames_dir in dirs:
        try:
            key = frames_dir.resolve()
        except OSError:
            key = frames_dir
        if key in seen:
            continue
        seen.add(key)
        unique_dirs.append(frames_dir)
    return unique_dirs


def agent_session_frame_paths_all(session_id):
    for frames_dir in agent_session_frame_dirs(session_id):
        if not frames_dir.is_dir():
            continue
        try:
            frames = [frame for frame in sorted(frames_dir.glob("*.jpg")) if frame.is_file()]
        except OSError:
            continue
        if frames:
            return frames, frames_dir
    return [], None


def agent_session_frame_items(session_id, limit=600):
    frames, frames_dir = agent_session_frame_paths_all(session_id)
    if not frames:
        return []
    if limit and limit > 0:
        frames = frames[-int(limit):]
    session_dir = AGENT_SESSIONS_DIR / session_id
    return [
        {
            "name": frame.name,
            "url": (
                f"/agent-sessions/{session_id}/frames/{frame.name}"
                if frames_dir == session_dir / "frames"
                else f"/agent-sessions/{session_id}/replay-source-frame?path={quote(str(frame.relative_to(session_dir)))}"
            ),
            "mtime": frame.stat().st_mtime,
            "size": frame.stat().st_size,
        }
        for frame in frames
        if frame.is_file()
    ]


@app.route("/api/agent/sessions/<session_id>/frames")
def get_agent_session_frames(session_id):
    try:
        limit = max(1, min(10000, int(request.args.get("limit", "600"))))
    except (TypeError, ValueError):
        limit = 600
    return jsonify({
        "session_id": session_id,
        "frames": agent_session_frame_items(session_id, limit=limit),
    })


@app.route("/agent-sessions/<session_id>/frames/<filename>")
def agent_session_frame(session_id, filename):
    safe_dir = AGENT_SESSIONS_DIR / session_id / "frames"
    return send_from_directory(safe_dir, filename)


@app.route("/agent-sessions/<session_id>/replay-source-frame")
def agent_session_replay_source_frame(session_id):
    session_dir = AGENT_SESSIONS_DIR / session_id
    rel_path = request.args.get("path", "")
    frame_path = (session_dir / rel_path).resolve()
    try:
        if session_dir.resolve() not in frame_path.parents:
            return jsonify({"error": "invalid frame path"}), 400
    except OSError:
        return jsonify({"error": "invalid frame path"}), 400
    return send_from_directory(frame_path.parent, frame_path.name)


@app.route("/api/agent/sessions/<session_id>/replay.mjpg")
def replay_agent_session_frames(session_id):
    frames, _frames_dir = agent_session_frame_paths_all(session_id)
    if not frames:
        return jsonify({"error": "session frames not found"}), 404
    try:
        target_fps = min(24.0, max(8.0, float(request.args.get("fps", "18"))))
    except (TypeError, ValueError):
        target_fps = 18.0
    try:
        max_frames = max(1, min(10000, int(request.args.get("limit", str(len(frames))))))
    except (TypeError, ValueError):
        max_frames = len(frames)
    frames = frames[:max_frames]

    def generate():
        boundary = b"--frame\r\n"
        frame_period = 1.0 / target_fps
        next_frame_at = time.monotonic()
        for index, frame_path in enumerate(frames, start=1):
            try:
                frame = frame_path.read_bytes()
            except OSError:
                continue
            now = time.monotonic()
            delay = next_frame_at - now
            if 0.0 < delay <= 0.5:
                time.sleep(delay)
            elif delay > 0.5 or now - next_frame_at > 0.5:
                next_frame_at = now
            yield boundary
            yield b"Content-Type: image/jpeg\r\n"
            yield f"X-Frame-Sequence: {index}\r\n".encode("ascii")
            yield f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
            yield frame
            yield b"\r\n"
            next_frame_at = max(next_frame_at + frame_period, time.monotonic())

    response = Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["X-Display-FPS"] = f"{target_fps:.1f}"
    response.headers["X-Replay-Frame-Count"] = str(len(frames))
    response.headers["X-Replay-Mode"] = "saved-agent-frames"
    return response


def build_agent_session_replay_video(
    session_id,
    fps=18.0,
    limit=10000,
    fmt="webm",
    smooth=False,
    smooth_multiplier=1.5,
    max_width=None,
    interp="blend",
    stride=1,
    video_codec="auto",
    upscale=False,
    sharpen=0.0,
):
    session_dir = AGENT_SESSIONS_DIR / session_id
    frames, _frames_dir = agent_session_frame_paths_all(session_id)
    if not frames:
        raise FileNotFoundError("session has no replay frames")
    frames = frames[: max(1, min(int(limit), 10000))]
    try:
        stride = min(8, max(1, int(stride)))
    except (TypeError, ValueError):
        stride = 1
    original_source_count = len(frames)
    interp = str(interp or "blend").lower()
    try:
        smooth_multiplier = float(smooth_multiplier)
    except (TypeError, ValueError):
        smooth_multiplier = 1.5
    try:
        max_width = int(max_width) if max_width else None
    except (TypeError, ValueError):
        max_width = None
    # Showcase replays requested with flow preserve their requested cadence
    # and output resolution. A hold fallback visibly stutters on long flights.
    if stride > 1:
        frames = frames[::stride]
    fps = min(24.0, max(8.0, float(fps)))
    smooth = bool(smooth)
    try:
        smooth_multiplier = min(3.0, max(1.0, float(smooth_multiplier)))
    except (TypeError, ValueError):
        smooth_multiplier = 1.5
    try:
        max_width = int(max_width) if max_width else None
    except (TypeError, ValueError):
        max_width = None
    if max_width is not None:
        max_width = min(1280, max(320, max_width))
    try:
        sharpen = min(1.2, max(0.0, float(sharpen or 0.0)))
    except (TypeError, ValueError):
        sharpen = 0.0
    upscale = bool(upscale)
    interp = str(interp or "blend").lower()
    if interp not in ("blend", "flow", "hold"):
        interp = "blend"
    replay_dir = session_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    fmt = str(fmt or "webm").lower()
    if fmt not in ("webm", "mp4"):
        fmt = "webm"
    requested_video_codec = str(video_codec or "auto").lower()
    if requested_video_codec not in ("auto", "mp4v", "h264"):
        requested_video_codec = "auto"
    ffmpeg_executable = find_ffmpeg_executable()
    use_h264 = fmt == "mp4" and requested_video_codec == "h264" and ffmpeg_executable
    if fmt == "webm":
        codec, extension, mimetype = "VP80", "webm", "video/webm"
    elif use_h264:
        codec, extension, mimetype = "h264", "mp4", "video/mp4"
    else:
        codec, extension, mimetype = "mp4v", "mp4", "video/mp4"
    source_count = len(frames)
    output_count = source_count
    mode = "raw" if stride <= 1 else f"raw_s{stride}"
    if smooth and source_count > 1:
        output_count = min(10000, max(source_count, int(round(source_count * smooth_multiplier))))
        mode = f"smooth{smooth_multiplier:.2f}_{interp}" if stride <= 1 else f"smooth{smooth_multiplier:.2f}_{interp}_s{stride}"
        mode = mode.replace(".", "p")
    # Flow interpolation now includes a forward/backward consistency check.
    # Version the cache name so older videos with unguarded flow are not reused.
    if smooth and interp == "flow":
        mode = f"{mode}_guardedv3"
    if sharpen > 0.0:
        mode = f"{mode}_sharp{sharpen:.2f}".replace(".", "p")

    import cv2

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError("could not read first replay frame")
    height, width = first.shape[:2]
    original_width, original_height = width, height
    if max_width is not None and (width > max_width or (upscale and width < max_width)):
        scaled_height = max(1, int(round(height * (max_width / width))))
        width, height = max_width, scaled_height
    # H.264 4:2:0 requires even dimensions. Rounding the scaled height down
    # avoids a late encoder failure on 16:9 source frames (for example 720x405).
    width = max(2, width - (width % 2))
    height = max(2, height - (height % 2))
    scale_tag = f"w{width}"
    codec_tag = "h264_" if codec == "h264" else ""
    video_path = replay_dir / f"replay_{codec_tag}{mode}_{scale_tag}_{int(round(fps))}fps_{source_count}src_{output_count}frames.{extension}"
    meta_path = replay_dir / f"{video_path.stem}.json"
    if video_path.exists() and video_path.stat().st_size > 0:
        cached_metadata = {}
        try:
            loaded_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded_metadata, dict):
                cached_metadata = loaded_metadata
        except (OSError, ValueError, TypeError):
            pass
        metadata = {
            "cached": True,
            "codec": codec,
            "requested_video_codec": requested_video_codec,
            "h264_available": bool(ffmpeg_executable),
            "ffmpeg_executable": str(ffmpeg_executable) if ffmpeg_executable else None,
            "format": fmt,
            "mimetype": mimetype,
            "fps": fps,
            "frame_count": output_count,
            "source_frame_count": source_count,
            "original_source_frame_count": original_source_count,
            "stride": stride,
            "smooth": smooth,
            "smooth_multiplier": smooth_multiplier,
            "interp": interp,
            "width": width,
            "height": height,
            "max_width": max_width,
            "source_width": original_width,
            "source_height": original_height,
            "upscale": upscale,
            "sharpen": sharpen,
            "size": video_path.stat().st_size,
        }
        # Preserve encoder-specific fields such as the guarded-flow version and
        # fallback count when serving a previously generated presentation file.
        metadata.update(cached_metadata)
        metadata["cached"] = True
        metadata["size"] = video_path.stat().st_size
        return video_path, metadata
    ffmpeg_process = None
    writer = None
    if codec == "h264":
        ffmpeg_cmd = [
            str(ffmpeg_executable),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "faster",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ]
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if ffmpeg_process.stdin is None:
            raise RuntimeError("could not open ffmpeg h264 stdin")
    else:
        # Prefer browser-playable cached video. If this codec is unavailable,
        # the frontend can request another profile or fall back to WebP/MJPEG.
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("could not open video writer")
    def sharpen_frame(frame):
        if sharpen <= 0.0:
            return frame
        blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)
        return cv2.addWeighted(frame, 1.0 + sharpen, blurred, -sharpen, 0.0)

    def read_resized(frame_path):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return None
        if frame.shape[1] != width or frame.shape[0] != height:
            interpolation = cv2.INTER_LANCZOS4 if (width > frame.shape[1] or height > frame.shape[0]) else cv2.INTER_AREA
            frame = cv2.resize(frame, (width, height), interpolation=interpolation)
        return sharpen_frame(frame)

    flow_cache_key = None
    flow_cache = None
    flow_cache_hits = 0
    flow_cache_misses = 0
    flow_fallback_pairs = 0

    def flow_interpolate(low_frame, high_frame, alpha, pair_key=None):
        nonlocal flow_cache_key, flow_cache, flow_cache_hits, flow_cache_misses, flow_fallback_pairs
        if alpha <= 0.001:
            return low_frame
        if alpha >= 0.999:
            return high_frame
        try:
            if pair_key is None or pair_key != flow_cache_key or flow_cache is None:
                flow_cache_misses += 1
                # Estimate motion on a presentation-sized proxy.  The previous
                # 320px proxy was too coarse for turns near buildings and could
                # create warped edges after the frame was enlarged to 960px.
                flow_width = min(width, 480)
                flow_height = max(1, int(round(height * (flow_width / width))))
                low_proxy = cv2.resize(low_frame, (flow_width, flow_height), interpolation=cv2.INTER_AREA)
                high_proxy = cv2.resize(high_frame, (flow_width, flow_height), interpolation=cv2.INTER_AREA)
                low_gray = cv2.cvtColor(low_proxy, cv2.COLOR_BGR2GRAY)
                high_gray = cv2.cvtColor(high_proxy, cv2.COLOR_BGR2GRAY)
                forward = cv2.calcOpticalFlowFarneback(
                    low_gray, high_gray, None, 0.5, 4, 21, 5, 7, 1.4, 0,
                )
                backward = cv2.calcOpticalFlowFarneback(
                    high_gray, low_gray, None, 0.5, 4, 21, 5, 7, 1.4, 0,
                )
                proxy_x, proxy_y = np.meshgrid(
                    np.arange(flow_width, dtype=np.float32),
                    np.arange(flow_height, dtype=np.float32),
                )
                # A valid flow vector should return close to its starting point
                # after following the backward flow at its destination.  Camera
                # cuts, occlusions, and large perspective changes fail this test.
                backward_at_forward = cv2.remap(
                    backward,
                    proxy_x + forward[..., 0],
                    proxy_y + forward[..., 1],
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                cycle_error = np.linalg.norm(forward + backward_at_forward, axis=2)
                motion = np.linalg.norm(forward, axis=2)
                in_bounds = (
                    (proxy_x + forward[..., 0] >= 0)
                    & (proxy_x + forward[..., 0] < flow_width - 1)
                    & (proxy_y + forward[..., 1] >= 0)
                    & (proxy_y + forward[..., 1] < flow_height - 1)
                )
                # Evaluate consistency only where the source has a matching
                # destination. A valid camera pan naturally moves a border out
                # of frame; treating that border as a failed vector made the v2
                # guard discard otherwise smooth flights.
                in_bounds_ratio = float(np.mean(in_bounds))
                if np.any(in_bounds):
                    consistent_ratio = float(np.mean(
                        cycle_error[in_bounds] <= (1.75 + (motion[in_bounds] * 0.12))
                    ))
                else:
                    consistent_ratio = 0.0
                flow_reliable = in_bounds_ratio >= 0.55 and consistent_ratio >= 0.68
                forward = cv2.resize(forward, (width, height), interpolation=cv2.INTER_LINEAR)
                backward = cv2.resize(backward, (width, height), interpolation=cv2.INTER_LINEAR)
                scale_x = width / float(flow_width)
                scale_y = height / float(flow_height)
                forward[..., 0] *= scale_x
                forward[..., 1] *= scale_y
                backward[..., 0] *= scale_x
                backward[..., 1] *= scale_y
                flow_cache_key = pair_key
                flow_cache = (forward, backward, flow_reliable)
            else:
                flow_cache_hits += 1
            forward, backward, flow_reliable = flow_cache
            if not flow_reliable:
                flow_fallback_pairs += 1
                return low_frame if alpha < 0.5 else high_frame
            grid_x, grid_y = np.meshgrid(
                np.arange(width, dtype=np.float32),
                np.arange(height, dtype=np.float32),
            )
            low_map_x = grid_x - (forward[..., 0] * alpha)
            low_map_y = grid_y - (forward[..., 1] * alpha)
            high_map_x = grid_x - (backward[..., 0] * (1.0 - alpha))
            high_map_y = grid_y - (backward[..., 1] * (1.0 - alpha))
            low_warp = cv2.remap(low_frame, low_map_x, low_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            high_warp = cv2.remap(high_frame, high_map_x, high_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            # Occlusions and large viewpoint changes make a two-sided blend
            # visibly double edges. Keep the nearer warped frame in that case.
            motion_error = float(np.mean(cv2.absdiff(low_warp, high_warp)))
            if not np.isfinite(motion_error) or motion_error > 22.0:
                return low_warp if alpha < 0.5 else high_warp
            return cv2.addWeighted(low_warp, 1.0 - alpha, high_warp, alpha, 0.0)
        except Exception:
            return low_frame if alpha < 0.5 else high_frame

    written = 0
    def write_frame(frame):
        if ffmpeg_process is not None:
            try:
                ffmpeg_process.stdin.write(frame.tobytes())
            except BrokenPipeError as exc:
                raise RuntimeError("ffmpeg h264 encoder pipe closed") from exc
        else:
            writer.write(frame)

    try:
        if smooth and output_count > source_count and source_count > 1:
            last_low_index = None
            low_frame = None
            high_frame = None
            for output_index in range(output_count):
                source_pos = (output_index * (source_count - 1)) / max(1, output_count - 1)
                low_index = int(math.floor(source_pos))
                high_index = min(source_count - 1, low_index + 1)
                alpha = float(source_pos - low_index)
                if low_index != last_low_index:
                    low_frame = read_resized(frames[low_index])
                    high_frame = read_resized(frames[high_index])
                    last_low_index = low_index
                elif high_frame is None and high_index != low_index:
                    high_frame = read_resized(frames[high_index])
                if low_frame is None:
                    continue
                if high_frame is None or high_index == low_index or alpha <= 0.001:
                    frame = low_frame
                elif interp == "flow":
                    frame = flow_interpolate(low_frame, high_frame, alpha, low_index)
                elif interp == "hold":
                    # Preserve a real camera frame for each display slot.  This
                    # is the deterministic fallback when motion estimation is
                    # too expensive or unreliable; it never creates double
                    # edges by mixing two viewpoints.
                    frame = low_frame if alpha < 0.5 else high_frame
                else:
                    frame = cv2.addWeighted(low_frame, 1.0 - alpha, high_frame, alpha, 0.0)
                write_frame(frame)
                written += 1
        else:
            for frame_path in frames:
                frame = read_resized(frame_path)
                if frame is None:
                    continue
                write_frame(frame)
                written += 1
    finally:
        if ffmpeg_process is not None:
            try:
                ffmpeg_process.stdin.close()
            except Exception:
                pass
            stderr = b""
            try:
                stderr = ffmpeg_process.stderr.read() if ffmpeg_process.stderr else b""
            except Exception:
                stderr = b""
            ffmpeg_process.wait()
            if ffmpeg_process.returncode != 0:
                try:
                    video_path.unlink(missing_ok=True)
                except OSError:
                    pass
                message = stderr.decode("utf-8", errors="ignore").strip() if stderr else "ffmpeg h264 encoder failed"
                raise RuntimeError(message or "ffmpeg h264 encoder failed")
        else:
            writer.release()
    if written <= 0 or not video_path.exists() or video_path.stat().st_size <= 0:
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError("video writer produced no playable output")
    metadata = {
        "cached": False,
        "codec": codec,
        "requested_video_codec": requested_video_codec,
        "h264_available": bool(ffmpeg_executable),
        "ffmpeg_executable": str(ffmpeg_executable) if ffmpeg_executable else None,
        "format": fmt,
        "mimetype": mimetype,
        "fps": fps,
        "frame_count": written,
        "width": width,
        "height": height,
        "size": video_path.stat().st_size,
        "source_frame_count": len(frames),
        "original_source_frame_count": original_source_count,
        "stride": stride,
        "smooth": smooth,
        "smooth_multiplier": smooth_multiplier,
        "interp": interp,
        "max_width": max_width,
        "source_width": original_width,
        "source_height": original_height,
        "upscale": upscale,
        "sharpen": sharpen,
        "flow_cache_hits": flow_cache_hits,
        "flow_cache_misses": flow_cache_misses,
        "flow_fallback_pairs": flow_fallback_pairs,
        "flow_guard": "forward_backward_v3" if interp == "flow" else None,
        "created_at": time.time(),
    }
    try:
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    metadata["mimetype"] = mimetype
    return video_path, metadata


def build_agent_session_replay_animation(
    session_id,
    fps=20.0,
    limit=10000,
    max_width=420,
    stride=4,
    quality=40,
):
    session_dir = AGENT_SESSIONS_DIR / session_id
    frames, _frames_dir = agent_session_frame_paths_all(session_id)
    if not frames:
        raise FileNotFoundError("session has no replay frames")
    frames = frames[: max(1, min(int(limit), 10000))]
    try:
        stride = min(8, max(1, int(stride)))
    except (TypeError, ValueError):
        stride = 4
    try:
        fps = min(24.0, max(8.0, float(fps)))
    except (TypeError, ValueError):
        fps = 20.0
    try:
        max_width = min(1280, max(320, int(max_width or 420)))
    except (TypeError, ValueError):
        max_width = 420
    try:
        quality = min(85, max(35, int(quality)))
    except (TypeError, ValueError):
        quality = 40

    original_source_count = len(frames)
    frames = frames[::stride]
    replay_dir = session_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)

    first = Image.open(frames[0]).convert("RGB")
    width, height = first.size
    if width > max_width:
        height = max(1, int(round(height * (max_width / width))))
        width = max_width
    first.close()
    anim_path = replay_dir / (
        f"replay_anim_s{stride}_w{width}_{int(round(fps))}fps_"
        f"q{quality}_{len(frames)}frames.webp"
    )
    meta_path = replay_dir / f"{anim_path.stem}.json"
    if anim_path.exists() and anim_path.stat().st_size > 0:
        return anim_path, {
            "cached": True,
            "format": "webp",
            "mimetype": "image/webp",
            "fps": fps,
            "frame_count": len(frames),
            "source_frame_count": len(frames),
            "original_source_frame_count": original_source_count,
            "stride": stride,
            "width": width,
            "height": height,
            "max_width": max_width,
            "quality": quality,
            "size": anim_path.stat().st_size,
        }

    images = []
    try:
        for frame_path in frames:
            image = Image.open(frame_path).convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.BILINEAR)
            images.append(image)
        duration_ms = max(1, int(round(1000.0 / fps)))
        images[0].save(
            anim_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            quality=quality,
            method=4,
        )
    finally:
        for image in images:
            try:
                image.close()
            except Exception:
                pass
    if not anim_path.exists() or anim_path.stat().st_size <= 0:
        raise RuntimeError("animated replay writer produced no output")
    metadata = {
        "cached": False,
        "format": "webp",
        "mimetype": "image/webp",
        "fps": fps,
        "frame_count": len(frames),
        "source_frame_count": len(frames),
        "original_source_frame_count": original_source_count,
        "stride": stride,
        "width": width,
        "height": height,
        "max_width": max_width,
        "quality": quality,
        "size": anim_path.stat().st_size,
        "created_at": time.time(),
    }
    try:
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return anim_path, metadata


def agent_session_replay_frame_paths(session_id, limit=10000, stride=4):
    frames, _frames_dir = agent_session_frame_paths_all(session_id)
    if not frames:
        raise FileNotFoundError("session has no replay frames")
    frames = frames[: max(1, min(int(limit), 10000))]
    try:
        stride = min(8, max(1, int(stride)))
    except (TypeError, ValueError):
        stride = 4
    original_source_count = len(frames)
    return frames[::stride], original_source_count, stride


def find_ffmpeg_executable():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    try:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    if ffmpeg_path and Path(ffmpeg_path).exists():
        return ffmpeg_path
    return None


def resolve_animation_params(fps=20.0, max_width=420, quality=40):
    try:
        fps = min(24.0, max(8.0, float(fps)))
    except (TypeError, ValueError):
        fps = 20.0
    try:
        max_width = min(1280, max(320, int(max_width or 420)))
    except (TypeError, ValueError):
        max_width = 420
    try:
        quality = min(85, max(35, int(quality)))
    except (TypeError, ValueError):
        quality = 40
    return fps, max_width, quality


def replay_animation_dimensions(frames, max_width):
    first = Image.open(frames[0]).convert("RGB")
    width, height = first.size
    if width > max_width:
        height = max(1, int(round(height * (max_width / width))))
        width = max_width
    first.close()
    return width, height


def write_webp_animation(frames, output_path, width, height, fps, quality):
    images = []
    try:
        for frame_path in frames:
            image = Image.open(frame_path).convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.BILINEAR)
            images.append(image)
        duration_ms = max(1, int(round(1000.0 / fps)))
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            quality=quality,
            method=4,
        )
    finally:
        for image in images:
            try:
                image.close()
            except Exception:
                pass


def build_agent_session_replay_animation_chunk(
    session_id,
    chunk_index=0,
    chunk_frames=120,
    fps=20.0,
    limit=10000,
    max_width=420,
    stride=4,
    quality=40,
):
    frames, original_source_count, stride = agent_session_replay_frame_paths(session_id, limit=limit, stride=stride)
    fps, max_width, quality = resolve_animation_params(fps=fps, max_width=max_width, quality=quality)
    try:
        chunk_frames = min(480, max(30, int(chunk_frames)))
    except (TypeError, ValueError):
        chunk_frames = 120
    try:
        chunk_index = max(0, int(chunk_index))
    except (TypeError, ValueError):
        chunk_index = 0
    chunk_count = int(math.ceil(len(frames) / float(chunk_frames)))
    if chunk_index >= chunk_count:
        raise FileNotFoundError("replay animation chunk not found")
    start = chunk_index * chunk_frames
    end = min(len(frames), start + chunk_frames)
    chunk = frames[start:end]
    width, height = replay_animation_dimensions(frames, max_width)
    replay_dir = AGENT_SESSIONS_DIR / session_id / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    anim_path = replay_dir / (
        f"replay_chunk_s{stride}_w{width}_{int(round(fps))}fps_q{quality}_"
        f"c{chunk_frames}_{chunk_index:03d}_{len(chunk)}frames.webp"
    )
    cached = anim_path.exists() and anim_path.stat().st_size > 0
    if not cached:
        write_webp_animation(chunk, anim_path, width, height, fps, quality)
    if not anim_path.exists() or anim_path.stat().st_size <= 0:
        raise RuntimeError("animated replay chunk writer produced no output")
    return anim_path, {
        "cached": cached,
        "format": "webp",
        "mimetype": "image/webp",
        "fps": fps,
        "frame_count": len(chunk),
        "total_frame_count": len(frames),
        "source_frame_count": len(frames),
        "original_source_frame_count": original_source_count,
        "stride": stride,
        "width": width,
        "height": height,
        "max_width": max_width,
        "quality": quality,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "chunk_frames": chunk_frames,
        "duration_ms": int(round((len(chunk) / fps) * 1000.0)),
        "size": anim_path.stat().st_size,
    }


def build_agent_session_replay_sprite_chunk(
    session_id,
    chunk_index=0,
    chunk_frames=120,
    fps=20.0,
    limit=10000,
    max_width=420,
    stride=4,
    quality=40,
    columns=10,
):
    frames, original_source_count, stride = agent_session_replay_frame_paths(session_id, limit=limit, stride=stride)
    fps, max_width, quality = resolve_animation_params(fps=fps, max_width=max_width, quality=quality)
    try:
        chunk_frames = min(480, max(30, int(chunk_frames)))
    except (TypeError, ValueError):
        chunk_frames = 120
    try:
        chunk_index = max(0, int(chunk_index))
    except (TypeError, ValueError):
        chunk_index = 0
    try:
        columns = min(20, max(4, int(columns)))
    except (TypeError, ValueError):
        columns = 10
    chunk_count = int(math.ceil(len(frames) / float(chunk_frames)))
    if chunk_index >= chunk_count:
        raise FileNotFoundError("replay sprite chunk not found")
    start = chunk_index * chunk_frames
    end = min(len(frames), start + chunk_frames)
    chunk = frames[start:end]
    frame_width, frame_height = replay_animation_dimensions(frames, max_width)
    rows = int(math.ceil(len(chunk) / float(columns)))
    replay_dir = AGENT_SESSIONS_DIR / session_id / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    sprite_path = replay_dir / (
        f"replay_sprite_s{stride}_w{frame_width}_{int(round(fps))}fps_q{quality}_"
        f"c{chunk_frames}_cols{columns}_{chunk_index:03d}_{len(chunk)}frames.webp"
    )
    cached = sprite_path.exists() and sprite_path.stat().st_size > 0
    if not cached:
        sprite = Image.new("RGB", (frame_width * columns, frame_height * rows))
        images = []
        try:
            for frame_number, frame_path in enumerate(chunk):
                image = Image.open(frame_path).convert("RGB")
                if image.size != (frame_width, frame_height):
                    image = image.resize((frame_width, frame_height), Image.Resampling.BILINEAR)
                images.append(image)
                x = (frame_number % columns) * frame_width
                y = (frame_number // columns) * frame_height
                sprite.paste(image, (x, y))
            sprite.save(sprite_path, quality=quality, method=4)
        finally:
            for image in images:
                try:
                    image.close()
                except Exception:
                    pass
            try:
                sprite.close()
            except Exception:
                pass
    if not sprite_path.exists() or sprite_path.stat().st_size <= 0:
        raise RuntimeError("sprite replay chunk writer produced no output")
    return sprite_path, {
        "cached": cached,
        "format": "webp-sprite",
        "mimetype": "image/webp",
        "fps": fps,
        "frame_count": len(chunk),
        "total_frame_count": len(frames),
        "source_frame_count": len(frames),
        "original_source_frame_count": original_source_count,
        "stride": stride,
        "width": frame_width * columns,
        "height": frame_height * rows,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "columns": columns,
        "rows": rows,
        "max_width": max_width,
        "quality": quality,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "chunk_frames": chunk_frames,
        "duration_ms": int(round((len(chunk) / fps) * 1000.0)),
        "size": sprite_path.stat().st_size,
    }


@app.route("/api/agent/sessions/<session_id>/replay-video")
def get_agent_session_replay_video(session_id):
    try:
        fps = float(request.args.get("fps", "18"))
    except (TypeError, ValueError):
        fps = 18.0
    try:
        limit = int(request.args.get("limit", "10000"))
    except (TypeError, ValueError):
        limit = 10000
    fmt = request.args.get("format", "webm")
    smooth = str(request.args.get("smooth", "0")).lower() not in ("0", "false", "no")
    try:
        smooth_multiplier = float(request.args.get("smooth_multiplier", "1.5"))
    except (TypeError, ValueError):
        smooth_multiplier = 1.5
    try:
        max_width = int(request.args.get("max_width", "0")) or None
    except (TypeError, ValueError):
        max_width = None
    interp = request.args.get("interp", "blend")
    video_codec = request.args.get("video_codec", "auto")
    upscale = str(request.args.get("upscale", "0")).lower() not in ("0", "false", "no")
    try:
        sharpen = float(request.args.get("sharpen", "0"))
    except (TypeError, ValueError):
        sharpen = 0.0
    try:
        stride = int(request.args.get("stride", "1"))
    except (TypeError, ValueError):
        stride = 1
    try:
        key = replay_asset_key(
            "video",
            session_id,
            fps=fps,
            limit=limit,
            fmt=fmt,
            smooth=smooth,
            smooth_multiplier=smooth_multiplier,
            max_width=max_width,
            interp=interp,
            stride=stride,
            video_codec=video_codec,
            upscale=upscale,
            sharpen=sharpen,
        )
        with replay_asset_lock(key):
            video_path, metadata = build_agent_session_replay_video(
                session_id,
                fps=fps,
                limit=limit,
                fmt=fmt,
                smooth=smooth,
                smooth_multiplier=smooth_multiplier,
                max_width=max_width,
                interp=interp,
                stride=stride,
                video_codec=video_codec,
                upscale=upscale,
                sharpen=sharpen,
            )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        if str(fmt).lower() != "mp4":
            try:
                key = replay_asset_key(
                    "video",
                    session_id,
                    fps=fps,
                    limit=limit,
                    fmt="mp4",
                    smooth=smooth,
                    smooth_multiplier=smooth_multiplier,
                    max_width=max_width,
                    interp=interp,
                    stride=stride,
                    video_codec=video_codec,
                    upscale=upscale,
                    sharpen=sharpen,
                )
                with replay_asset_lock(key):
                    video_path, metadata = build_agent_session_replay_video(
                        session_id,
                        fps=fps,
                        limit=limit,
                        fmt="mp4",
                        smooth=smooth,
                        smooth_multiplier=smooth_multiplier,
                        max_width=max_width,
                        interp=interp,
                        stride=stride,
                        video_codec=video_codec,
                        upscale=upscale,
                        sharpen=sharpen,
                    )
            except Exception:
                return jsonify({"error": str(exc), "fallback": "mjpeg"}), 500
        else:
            return jsonify({"error": str(exc), "fallback": "mjpeg"}), 500
    return jsonify({
        "ok": True,
        "session_id": session_id,
        "url": f"/agent-sessions/{session_id}/replay/{video_path.name}",
        "metadata": metadata,
    })


@app.route("/api/agent/sessions/<session_id>/replay-animation")
def get_agent_session_replay_animation(session_id):
    try:
        fps = float(request.args.get("fps", "20"))
    except (TypeError, ValueError):
        fps = 20.0
    try:
        limit = int(request.args.get("limit", "10000"))
    except (TypeError, ValueError):
        limit = 10000
    try:
        max_width = int(request.args.get("max_width", "420"))
    except (TypeError, ValueError):
        max_width = 420
    try:
        stride = int(request.args.get("stride", "4"))
    except (TypeError, ValueError):
        stride = 4
    try:
        quality = int(request.args.get("quality", "40"))
    except (TypeError, ValueError):
        quality = 40
    try:
        anim_path, metadata = build_agent_session_replay_animation(
            session_id,
            fps=fps,
            limit=limit,
            max_width=max_width,
            stride=stride,
            quality=quality,
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc), "fallback": "video"}), 500
    return jsonify({
        "ok": True,
        "session_id": session_id,
        "url": f"/agent-sessions/{session_id}/replay/{anim_path.name}",
        "metadata": metadata,
    })


@app.route("/api/agent/sessions/<session_id>/replay-animation-manifest")
def get_agent_session_replay_animation_manifest(session_id):
    try:
        fps = float(request.args.get("fps", "20"))
    except (TypeError, ValueError):
        fps = 20.0
    try:
        limit = int(request.args.get("limit", "10000"))
    except (TypeError, ValueError):
        limit = 10000
    try:
        max_width = int(request.args.get("max_width", "420"))
    except (TypeError, ValueError):
        max_width = 420
    try:
        stride = int(request.args.get("stride", "4"))
    except (TypeError, ValueError):
        stride = 4
    try:
        quality = int(request.args.get("quality", "40"))
    except (TypeError, ValueError):
        quality = 40
    try:
        chunk_frames = min(480, max(30, int(request.args.get("chunk_frames", "120"))))
    except (TypeError, ValueError):
        chunk_frames = 120
    try:
        frames, original_source_count, stride = agent_session_replay_frame_paths(session_id, limit=limit, stride=stride)
        fps, max_width, quality = resolve_animation_params(fps=fps, max_width=max_width, quality=quality)
        width, height = replay_animation_dimensions(frames, max_width)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc), "fallback": "animation"}), 500
    chunk_count = int(math.ceil(len(frames) / float(chunk_frames)))
    chunks = []
    for index in range(chunk_count):
        start = index * chunk_frames
        frame_count = min(chunk_frames, max(0, len(frames) - start))
        chunks.append({
            "index": index,
            "frame_count": frame_count,
            "duration_ms": int(round((frame_count / fps) * 1000.0)),
            "url": (
                f"/api/agent/sessions/{session_id}/replay-animation-chunk"
                f"?chunk={index}&chunk_frames={chunk_frames}&stride={stride}"
                f"&fps={fps:g}&limit={limit}&max_width={max_width}&quality={quality}"
            ),
        })
    return jsonify({
        "ok": True,
        "session_id": session_id,
        "metadata": {
            "format": "webp-chunks",
            "mimetype": "image/webp",
            "fps": fps,
            "frame_count": len(frames),
            "source_frame_count": len(frames),
            "original_source_frame_count": original_source_count,
            "stride": stride,
            "width": width,
            "height": height,
            "max_width": max_width,
            "quality": quality,
            "chunk_count": chunk_count,
            "chunk_frames": chunk_frames,
        },
        "chunks": chunks,
    })


@app.route("/api/agent/sessions/<session_id>/replay-animation-chunk")
def get_agent_session_replay_animation_chunk(session_id):
    try:
        chunk_index = int(request.args.get("chunk", "0"))
    except (TypeError, ValueError):
        chunk_index = 0
    try:
        chunk_frames = int(request.args.get("chunk_frames", "120"))
    except (TypeError, ValueError):
        chunk_frames = 120
    try:
        fps = float(request.args.get("fps", "20"))
    except (TypeError, ValueError):
        fps = 20.0
    try:
        limit = int(request.args.get("limit", "10000"))
    except (TypeError, ValueError):
        limit = 10000
    try:
        max_width = int(request.args.get("max_width", "420"))
    except (TypeError, ValueError):
        max_width = 420
    try:
        stride = int(request.args.get("stride", "4"))
    except (TypeError, ValueError):
        stride = 4
    try:
        quality = int(request.args.get("quality", "40"))
    except (TypeError, ValueError):
        quality = 40
    try:
        anim_path, metadata = build_agent_session_replay_animation_chunk(
            session_id,
            chunk_index=chunk_index,
            chunk_frames=chunk_frames,
            fps=fps,
            limit=limit,
            max_width=max_width,
            stride=stride,
            quality=quality,
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc), "fallback": "animation"}), 500
    return jsonify({
        "ok": True,
        "session_id": session_id,
        "url": f"/agent-sessions/{session_id}/replay/{anim_path.name}",
        "metadata": metadata,
    })


@app.route("/api/agent/sessions/<session_id>/replay-sprite-manifest")
def get_agent_session_replay_sprite_manifest(session_id):
    try:
        fps = float(request.args.get("fps", "20"))
    except (TypeError, ValueError):
        fps = 20.0
    try:
        limit = int(request.args.get("limit", "10000"))
    except (TypeError, ValueError):
        limit = 10000
    try:
        max_width = int(request.args.get("max_width", "420"))
    except (TypeError, ValueError):
        max_width = 420
    try:
        stride = int(request.args.get("stride", "4"))
    except (TypeError, ValueError):
        stride = 4
    try:
        quality = int(request.args.get("quality", "40"))
    except (TypeError, ValueError):
        quality = 40
    try:
        chunk_frames = min(480, max(30, int(request.args.get("chunk_frames", "120"))))
    except (TypeError, ValueError):
        chunk_frames = 120
    try:
        columns = min(20, max(4, int(request.args.get("columns", "10"))))
    except (TypeError, ValueError):
        columns = 10
    try:
        frames, original_source_count, stride = agent_session_replay_frame_paths(session_id, limit=limit, stride=stride)
        fps, max_width, quality = resolve_animation_params(fps=fps, max_width=max_width, quality=quality)
        frame_width, frame_height = replay_animation_dimensions(frames, max_width)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc), "fallback": "animation"}), 500
    chunk_count = int(math.ceil(len(frames) / float(chunk_frames)))
    chunks = []
    for index in range(chunk_count):
        start = index * chunk_frames
        frame_count = min(chunk_frames, max(0, len(frames) - start))
        rows = int(math.ceil(frame_count / float(columns)))
        chunks.append({
            "index": index,
            "frame_count": frame_count,
            "duration_ms": int(round((frame_count / fps) * 1000.0)),
            "columns": columns,
            "rows": rows,
            "url": (
                f"/api/agent/sessions/{session_id}/replay-sprite-chunk"
                f"?chunk={index}&chunk_frames={chunk_frames}&stride={stride}"
                f"&fps={fps:g}&limit={limit}&max_width={max_width}&quality={quality}&columns={columns}"
            ),
        })
    return jsonify({
        "ok": True,
        "session_id": session_id,
        "metadata": {
            "format": "webp-sprite-chunks",
            "mimetype": "image/webp",
            "fps": fps,
            "frame_count": len(frames),
            "source_frame_count": len(frames),
            "original_source_frame_count": original_source_count,
            "stride": stride,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "max_width": max_width,
            "quality": quality,
            "chunk_count": chunk_count,
            "chunk_frames": chunk_frames,
            "columns": columns,
        },
        "chunks": chunks,
    })


@app.route("/api/agent/sessions/<session_id>/replay-sprite-chunk")
def get_agent_session_replay_sprite_chunk(session_id):
    try:
        chunk_index = int(request.args.get("chunk", "0"))
    except (TypeError, ValueError):
        chunk_index = 0
    try:
        chunk_frames = int(request.args.get("chunk_frames", "120"))
    except (TypeError, ValueError):
        chunk_frames = 120
    try:
        fps = float(request.args.get("fps", "20"))
    except (TypeError, ValueError):
        fps = 20.0
    try:
        limit = int(request.args.get("limit", "10000"))
    except (TypeError, ValueError):
        limit = 10000
    try:
        max_width = int(request.args.get("max_width", "420"))
    except (TypeError, ValueError):
        max_width = 420
    try:
        stride = int(request.args.get("stride", "4"))
    except (TypeError, ValueError):
        stride = 4
    try:
        quality = int(request.args.get("quality", "40"))
    except (TypeError, ValueError):
        quality = 40
    try:
        columns = int(request.args.get("columns", "10"))
    except (TypeError, ValueError):
        columns = 10
    try:
        key = replay_asset_key(
            "sprite",
            session_id,
            chunk_index=chunk_index,
            chunk_frames=chunk_frames,
            fps=fps,
            limit=limit,
            max_width=max_width,
            stride=stride,
            quality=quality,
            columns=columns,
        )
        with replay_asset_lock(key):
            sprite_path, metadata = build_agent_session_replay_sprite_chunk(
                session_id,
                chunk_index=chunk_index,
                chunk_frames=chunk_frames,
                fps=fps,
                limit=limit,
                max_width=max_width,
                stride=stride,
                quality=quality,
                columns=columns,
            )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc), "fallback": "animation"}), 500
    return jsonify({
        "ok": True,
        "session_id": session_id,
        "url": f"/agent-sessions/{session_id}/replay/{sprite_path.name}",
        "metadata": metadata,
    })


def replay_prewarm_worker(job_id, session_id, params):
    with replay_prewarm_jobs_lock:
        replay_prewarm_jobs[job_id] = {
            "status": "running",
            "session_id": session_id,
            "started_at": time.time(),
            "finished_at": None,
            "video": None,
            "sprite_chunks": [],
            "error": None,
        }
    try:
        frames, _, stride = agent_session_replay_frame_paths(
            session_id,
            limit=params["limit"],
            stride=params["stride"],
        )
        chunk_count = int(math.ceil(len(frames) / float(params["chunk_frames"])))
        end_chunk = min(chunk_count, params["start_chunk"] + params["max_chunks"])
        for chunk_index in range(params["start_chunk"], end_chunk):
            if chunk_index < 0:
                continue
            sprite_key = replay_asset_key(
                "sprite",
                session_id,
                chunk_index=chunk_index,
                chunk_frames=params["chunk_frames"],
                fps=params["fps"],
                limit=params["limit"],
                max_width=params["max_width"],
                stride=stride,
                quality=params["quality"],
                columns=params["columns"],
            )
            with replay_asset_lock(sprite_key):
                sprite_path, sprite_meta = build_agent_session_replay_sprite_chunk(
                    session_id,
                    chunk_index=chunk_index,
                    chunk_frames=params["chunk_frames"],
                    fps=params["fps"],
                    limit=params["limit"],
                    max_width=params["max_width"],
                    stride=stride,
                    quality=params["quality"],
                    columns=params["columns"],
                )
            with replay_prewarm_jobs_lock:
                replay_prewarm_jobs[job_id]["sprite_chunks"].append({
                    "index": chunk_index,
                    "name": sprite_path.name,
                    "cached": sprite_meta.get("cached"),
                    "size": sprite_meta.get("size"),
                })

        if params.get("include_video"):
            video_key = replay_asset_key(
                "video",
                session_id,
                fps=params["fps"],
                limit=params["limit"],
                fmt="mp4",
                smooth=False,
                smooth_multiplier=1.0,
                max_width=params["max_width"],
                interp="blend",
                stride=params["stride"],
            )
            with replay_asset_lock(video_key):
                video_path, video_meta = build_agent_session_replay_video(
                    session_id,
                    fps=params["fps"],
                    limit=params["limit"],
                    fmt="mp4",
                    smooth=False,
                    smooth_multiplier=1.0,
                    max_width=params["max_width"],
                    interp="blend",
                    stride=params["stride"],
                )
            with replay_prewarm_jobs_lock:
                replay_prewarm_jobs[job_id]["video"] = {
                    "name": video_path.name,
                    "cached": video_meta.get("cached"),
                    "size": video_meta.get("size"),
                }
        else:
            with replay_prewarm_jobs_lock:
                replay_prewarm_jobs[job_id]["video"] = {"skipped": True}
        with replay_prewarm_jobs_lock:
            replay_prewarm_jobs[job_id]["status"] = "done"
            replay_prewarm_jobs[job_id]["finished_at"] = time.time()
    except Exception as exc:
        with replay_prewarm_jobs_lock:
            replay_prewarm_jobs[job_id]["status"] = "error"
            replay_prewarm_jobs[job_id]["finished_at"] = time.time()
            replay_prewarm_jobs[job_id]["error"] = str(exc)


def demo_video_prewarm_worker(job_id, session_id, profiles, stable_animation=None):
    with replay_prewarm_jobs_lock:
        replay_prewarm_jobs[job_id] = {
            "status": "running",
            "session_id": session_id,
            "started_at": time.time(),
            "finished_at": None,
            "video": None,
            "videos": [],
            "animations": [],
            "sprite_chunks": [],
            "error": None,
        }
    try:
        for profile in profiles:
            video_key = replay_asset_key(
                "demo-video",
                session_id,
                profile=profile["name"],
                fps=profile["fps"],
                limit=profile["limit"],
                fmt=profile.get("fmt", "mp4"),
                smooth=bool(profile.get("smooth", False)),
                smooth_multiplier=float(profile.get("smooth_multiplier", 1.0)),
                max_width=profile["max_width"],
                interp=profile.get("interp", "blend"),
                stride=profile["stride"],
                video_codec=profile.get("video_codec", "auto"),
                upscale=bool(profile.get("upscale", False)),
                sharpen=float(profile.get("sharpen", 0.0)),
            )
            with replay_asset_lock(video_key):
                video_path, video_meta = build_agent_session_replay_video(
                    session_id,
                    fps=profile["fps"],
                    limit=profile["limit"],
                    fmt=profile.get("fmt", "mp4"),
                    smooth=bool(profile.get("smooth", False)),
                    smooth_multiplier=float(profile.get("smooth_multiplier", 1.0)),
                    max_width=profile["max_width"],
                    interp=profile.get("interp", "blend"),
                    stride=profile["stride"],
                    video_codec=profile.get("video_codec", "auto"),
                    upscale=bool(profile.get("upscale", False)),
                    sharpen=float(profile.get("sharpen", 0.0)),
                )
            with replay_prewarm_jobs_lock:
                replay_prewarm_jobs[job_id]["videos"].append({
                    "profile": profile["name"],
                    "name": video_path.name,
                    "cached": video_meta.get("cached"),
                    "size": video_meta.get("size"),
                    "fps": video_meta.get("fps"),
                    "width": video_meta.get("width"),
                    "stride": video_meta.get("stride"),
                    "smooth": video_meta.get("smooth"),
                    "smooth_multiplier": video_meta.get("smooth_multiplier"),
                    "codec": video_meta.get("codec"),
                    "requested_video_codec": video_meta.get("requested_video_codec"),
                    "upscale": video_meta.get("upscale"),
                })
        if stable_animation:
            animation_key = replay_asset_key(
                "demo-animation",
                session_id,
                profile=stable_animation["name"],
                fps=stable_animation["fps"],
                limit=stable_animation["limit"],
                max_width=stable_animation["max_width"],
                stride=stable_animation["stride"],
                quality=stable_animation["quality"],
            )
            with replay_asset_lock(animation_key):
                anim_path, anim_meta = build_agent_session_replay_animation(
                    session_id,
                    fps=stable_animation["fps"],
                    limit=stable_animation["limit"],
                    max_width=stable_animation["max_width"],
                    stride=stable_animation["stride"],
                    quality=stable_animation["quality"],
                )
            with replay_prewarm_jobs_lock:
                replay_prewarm_jobs[job_id]["animations"].append({
                    "profile": stable_animation["name"],
                    "name": anim_path.name,
                    "cached": anim_meta.get("cached"),
                    "size": anim_meta.get("size"),
                    "fps": anim_meta.get("fps"),
                    "width": anim_meta.get("width"),
                    "stride": anim_meta.get("stride"),
                    "quality": anim_meta.get("quality"),
                })
        with replay_prewarm_jobs_lock:
            replay_prewarm_jobs[job_id]["status"] = "done"
            replay_prewarm_jobs[job_id]["finished_at"] = time.time()
    except Exception as exc:
        with replay_prewarm_jobs_lock:
            replay_prewarm_jobs[job_id]["status"] = "error"
            replay_prewarm_jobs[job_id]["finished_at"] = time.time()
            replay_prewarm_jobs[job_id]["error"] = str(exc)


@app.route("/api/agent/sessions/<session_id>/replay-prewarm", methods=["POST"])
def start_agent_session_replay_prewarm(session_id):
    def int_arg(name, default, lower, upper):
        try:
            return min(upper, max(lower, int(request.args.get(name, default))))
        except (TypeError, ValueError):
            return default

    def float_arg(name, default, lower, upper):
        try:
            return min(upper, max(lower, float(request.args.get(name, default))))
        except (TypeError, ValueError):
            return default

    def bool_arg(name, default=False):
        value = request.args.get(name)
        if value is None:
            return default
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    params = {
        "fps": float_arg("fps", 20.0, 8.0, 24.0),
        "limit": int_arg("limit", 10000, 1, 10000),
        "max_width": int_arg("max_width", 420, 320, 1280),
        "stride": int_arg("stride", 4, 1, 8),
        "quality": int_arg("quality", 40, 20, 90),
        "chunk_frames": int_arg("chunk_frames", 60, 30, 480),
        "columns": int_arg("columns", 6, 4, 20),
        "start_chunk": int_arg("start_chunk", 2, 0, 10000),
        "max_chunks": int_arg("max_chunks", 96, 1, 256),
        "include_video": bool_arg("include_video", False),
    }
    job_id = replay_asset_key("prewarm", session_id, **params)
    with replay_prewarm_jobs_lock:
        existing = replay_prewarm_jobs.get(job_id)
        if existing and existing.get("status") in ("queued", "running", "done"):
            return jsonify({"ok": True, "job_id": job_id, "status": existing.get("status"), "deduped": True})
        replay_prewarm_jobs[job_id] = {
            "status": "queued",
            "session_id": session_id,
            "started_at": None,
            "finished_at": None,
            "video": None,
            "sprite_chunks": [],
            "error": None,
        }
    thread = threading.Thread(
        target=replay_prewarm_worker,
        args=(job_id, session_id, params),
        name=f"replay-prewarm-{session_id}",
        daemon=True,
    )
    thread.start()
    return jsonify({"ok": True, "job_id": job_id, "status": "queued", "deduped": False})


@app.route("/api/agent/sessions/<session_id>/demo-video-prewarm", methods=["POST"])
def start_agent_session_demo_video_prewarm(session_id):
    session_dir = AGENT_SESSIONS_DIR / session_id
    if not session_dir.is_dir():
        return jsonify({"error": "agent session not found"}), 404
    if not agent_session_has_replay_frames(session_id):
        return jsonify({"error": "agent session has no replay frames"}), 404
    profiles = [
        {"name": "demo-smooth", "fps": 8.0, "stride": 1, "max_width": 768, "limit": 10000, "smooth": False, "smooth_multiplier": 1.0, "interp": "hold", "video_codec": "h264", "upscale": False, "sharpen": 0.0},
    ]
    stable_animation = None
    job_id = replay_asset_key("demo-video-prewarm", session_id, profiles=profiles, stable_animation=stable_animation)
    with replay_prewarm_jobs_lock:
        existing = replay_prewarm_jobs.get(job_id)
        if existing and existing.get("status") in ("queued", "running", "done"):
            return jsonify({"ok": True, "job_id": job_id, "status": existing.get("status"), "deduped": True})
        replay_prewarm_jobs[job_id] = {
            "status": "queued",
            "session_id": session_id,
            "started_at": None,
            "finished_at": None,
            "video": None,
            "videos": [],
            "animations": [],
            "sprite_chunks": [],
            "error": None,
        }
    thread = threading.Thread(
        target=demo_video_prewarm_worker,
        args=(job_id, session_id, profiles, stable_animation),
        name=f"demo-video-prewarm-{session_id}",
        daemon=True,
    )
    thread.start()
    return jsonify({"ok": True, "job_id": job_id, "status": "queued", "deduped": False})


@app.route("/agent-sessions/<session_id>/replay/<filename>")
def agent_session_replay_file(session_id, filename):
    safe_dir = AGENT_SESSIONS_DIR / session_id / "replay"
    lower_name = str(filename).lower()
    if lower_name.endswith(".webp"):
        mimetype = "image/webp"
    elif lower_name.endswith(".webm"):
        mimetype = "video/webm"
    else:
        mimetype = "video/mp4"
    return send_from_directory(safe_dir, filename, mimetype=mimetype)


@app.route("/api/sessions/<session_id>")
def get_session(session_id):
    with sessions_lock:
        session = sessions.get(session_id)
    if session is None:
        return jsonify({"error": "session not found"}), 404
    return jsonify(session.snapshot())


@app.route("/api/agent/sessions")
def list_agent_sessions():
    with sessions_lock:
        current = [session.snapshot() for session in sessions.values() if getattr(session, "mode", None) == "agent"]

    archived = []
    for summary_path in sorted(AGENT_SESSIONS_DIR.glob("*/summary.json"), reverse=True):
        try:
            archived.append(json.loads(summary_path.read_text(encoding="utf-8")))
        except Exception:
            continue
        if len(archived) >= 50:
            break
    return jsonify({"sessions": current, "archived": archived})


@app.route("/api/agent/locks")
def agent_locks():
    return jsonify({"locks": read_agent_locks()})


@app.route("/api/agent/sessions/latest")
def latest_agent_session():
    candidates = sorted(AGENT_SESSIONS_DIR.glob("*/summary.json"), reverse=True)
    for summary_path in candidates:
        try:
            return jsonify(json.loads(summary_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return jsonify({"error": "no agent sessions"}), 404


@app.route("/api/agent/sessions/active")
def active_agent_session():
    with sessions_lock:
        for session in sessions.values():
            if getattr(session, "mode", None) == "agent" and is_active(session):
                return jsonify(session.snapshot())
    return jsonify({"error": "no active agent session"}), 404


@app.route("/api/agent/sessions/<session_id>")
def get_agent_session(session_id):
    with sessions_lock:
        session = sessions.get(session_id)
    if session is not None and getattr(session, "mode", None) == "agent":
        return jsonify(session.snapshot())

    summary_path = AGENT_SESSIONS_DIR / session_id / "summary.json"
    if summary_path.exists():
        return jsonify(json.loads(summary_path.read_text(encoding="utf-8")))
    return jsonify({"error": "agent session not found"}), 404


def read_agent_trace_points(session_id, limit=2000, model_run_id=None):
    session_dir = AGENT_SESSIONS_DIR / session_id
    if not session_dir.exists() or session_dir.parent != AGENT_SESSIONS_DIR:
        return []

    if model_run_id is not None:
        trace_paths = [session_dir / "model_runs" / model_run_id / "trace.jsonl"]
    else:
        trace_paths = sorted(session_dir.glob("model_runs/*/trace.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
        trace_paths.append(session_dir / "trajectory.jsonl")

    for trace_path in trace_paths:
        if not trace_path.exists():
            continue
        points = []
        try:
            with trace_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    position = item.get("position")
                    if not position or len(position) < 2:
                        continue
                    points.append(
                        {
                            "step": item.get("step"),
                            "position": position,
                            "distance_to_goal": item.get("distance_to_goal"),
                            "distance_3d_to_goal": item.get("distance_3d_to_goal"),
                            "vertical_error_to_goal": item.get("vertical_error_to_goal"),
                            "altitude_aligned": item.get("altitude_aligned"),
                            "surface_clearance": item.get("surface_clearance"),
                            "surface_aligned": item.get("surface_aligned"),
                            "surface_probe_ok": item.get("surface_probe_ok"),
                            "surface_probe_error": item.get("surface_probe_error"),
                            "best_distance": item.get("best_distance"),
                            "best_distance_step": item.get("best_distance_step"),
                            "initial_distance": item.get("initial_distance"),
                            "progress_to_goal": item.get("progress_to_goal"),
                            "success_distance": item.get("success_distance"),
                            "success_20m": item.get("success_20m"),
                            "success_20m_with_altitude": item.get("success_20m_with_altitude"),
                            "success_20m_with_surface": item.get("success_20m_with_surface"),
                            "oracle_success_20m": item.get("oracle_success_20m"),
                            "navigation_phase": item.get("navigation_phase"),
                            "goal_position": item.get("goal_position"),
                            "action": item.get("action"),
                            "safety_override": item.get("safety_override"),
                        }
                    )
            if points:
                return points[-limit:]
        except Exception:
            continue

    summary_path = session_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            return (summary.get("path") or [])[-limit:]
        except Exception:
            return []
    return []


@app.route("/api/agent/sessions/<session_id>/trace")
def get_agent_session_trace(session_id):
    model_run_id = None
    with sessions_lock:
        session = sessions.get(session_id)
        if session is not None:
            model_run_id = getattr(session, "model_run_id", None)
    if model_run_id is None:
        try:
            metadata = json.loads((AGENT_SESSIONS_DIR / session_id / "meta.json").read_text(encoding="utf-8"))
            model_run_id = metadata.get("model_run_id")
        except Exception:
            pass
    points = read_agent_trace_points(session_id, model_run_id=model_run_id)
    latest = points[-1] if points else None
    return jsonify({"session_id": session_id, "path": points, "latest": latest, "count": len(points)})


@app.route("/api/agent/start", methods=["POST"])
def start_agent():
    data, error_response = request_json_object()
    if error_response is not None:
        return error_response
    instruction = str(data.get("instruction", "")).strip()
    if not instruction:
        return jsonify({"error": "instruction is required"}), 400
    if len(instruction) > 2000:
        return jsonify({"error": "instruction is too long"}), 400
    confirm_control, error_response = request_boolean(data, "confirm_control", False)
    if error_response is not None:
        return error_response, 400
    if not confirm_control:
        return jsonify({"error": "agent flight requires confirm_control=true"}), 400
    v014_handshake, v014_error = validate_v014_start_assertions(data)
    if v014_error is not None:
        return jsonify(v014_error), 400

    max_steps, error_response = request_integer(data, "max_steps", 900)
    if error_response is not None:
        return error_response, 400
    gpu_id, error_response = request_integer(
        data,
        "gpu_id",
        os.environ.get("AIRVLN_AGENT_GPU_ID", "0"),
    )
    if error_response is not None:
        return error_response, 400
    scene_id, error_response = request_integer(data, "scene_id", 16)
    if error_response is not None:
        return error_response, 400
    simulator_tool_port, error_response = request_integer(
        data,
        "simulator_tool_port",
        os.environ.get("SIMULATOR_TOOL_PORT", str(DEFAULT_PHASE_A_SIMULATOR_PORT)),
    )
    if error_response is not None:
        return error_response, 400
    max_steps = max(1, min(max_steps, 1200))
    gpu_error, gpu_inventory_data = validate_agent_gpu(gpu_id)
    if gpu_error:
        return jsonify(gpu_error), 400
    allowed_model_extra_args = {
        "learned_segment_target_radius",
        "learned_segment_max_age",
        "learned_segment_z_clamp",
        "learned_segment_min_xy_step",
        "learned_segment_max_xy_step",
        "learned_segment_conservative_max_xy_step",
        "learned_segment_resample_worse_streak",
        "learned_segment_resample_regression",
        "learned_segment_goal_regression",
        "learned_segment_goal_regression_worse_streak",
        "learned_segment_goal_regression",
        "learned_segment_goal_regression_min_age",
        "final_goal_progress_tolerance",
        "final_goal_progress_max_retries",
        "learned_segment_cruise_altitude",
        "learned_segment_descent_altitude",
        "segment_grounding_final_ridge",
        "segment_switch_confidence",
        "segment_min_steps",
        "segment_progress_distance",
    }
    model_extra_args = {
        key: data.get(key)
        for key in allowed_model_extra_args
        if key in data and data.get(key) is not None
    }
    tuning_numeric_fields = {
        "learned_segment_target_radius",
        "learned_segment_z_clamp",
        "learned_segment_min_xy_step",
        "learned_segment_max_xy_step",
        "learned_segment_conservative_max_xy_step",
        "learned_segment_resample_regression",
        "learned_segment_cruise_altitude",
        "learned_segment_descent_altitude",
        "segment_switch_confidence",
        "segment_progress_distance",
    }
    tuning_integer_fields = {
        "learned_segment_max_age",
        "learned_segment_resample_worse_streak",
        "learned_segment_goal_regression_worse_streak",
        "learned_segment_goal_regression_min_age",
        "final_goal_progress_max_retries",
        "segment_min_steps",
    }
    for key in model_extra_args:
        if key in tuning_numeric_fields or key in tuning_integer_fields:
            value, error_response = request_number(
                data,
                key,
                None,
                integer=key in tuning_integer_fields,
            )
            if error_response is not None:
                return error_response, 400
            model_extra_args[key] = value
    boolean_fields = {
        "save_frames": True,
        "enable_oracle_goal_homing": False,
        "enable_oracle_path_homing": False,
        "enable_learned_segment_homing": False,
        "enable_final_goal_progress_gate": False,
        "enable_scene16_support_ranker": False,
        "enable_scene16_route_prior": False,
    }
    boolean_values = {}
    for key, default in boolean_fields.items():
        value, error_response = request_boolean(data, key, default)
        if error_response is not None:
            return error_response, 400
        boolean_values[key] = value
    if boolean_values["enable_final_goal_progress_gate"]:
        model_extra_args["enable_final_goal_progress_gate"] = True

    session_id = v014_handshake["agent_session_id"] if v014_handshake is not None else now_session_id()
    agent_session_id = session_id
    v014_namespace = None
    if v014_handshake is not None:
        try:
            v014_namespace = reserve_v014_trace_namespace(
                agent_session_id,
                v014_handshake["model_run_id"],
            )
        except (FileExistsError, OSError, ValueError) as exc:
            return jsonify(_v014_error(str(exc))), 409
        v014_handshake["namespace_proof"] = v014_namespace["namespace_proof"]
        v014_handshake["trace_contract"]["namespace_proof"] = v014_namespace["namespace_proof"]
    if v014_handshake is not None:
        config = v014_handshake["controller_config"]
        model_extra_args.update({
            "learned_segment_target_radius": config["target_radius"],
            "learned_segment_max_age": config["learned_max_age"],
            "learned_segment_z_clamp": config["z_clamp"],
            "learned_segment_min_xy_step": config["min_xy_step"],
            "learned_segment_max_xy_step": config["max_xy_step"],
            "learned_segment_resample_worse_streak": config["resample_worse_streak"],
            "learned_segment_resample_regression": config["resample_regression"],
            "learned_segment_goal_regression_worse_streak": config["goal_regression_worse_streak"],
            "learned_segment_goal_regression": config["goal_regression"],
            "learned_segment_goal_regression_min_age": config["goal_regression_min_age"],
            "final_goal_progress_tolerance": config.get("final_goal_progress_tolerance", 8.0),
            "final_goal_progress_max_retries": config.get("final_goal_progress_max_retries", 2),
            "learned_segment_cruise_altitude": config["cruise_altitude"],
            "learned_segment_descent_altitude": config["descent_altitude"],
            "segment_switch_confidence": config["segment_switch_confidence"],
            "segment_min_steps": config["segment_min_steps"],
            "segment_progress_distance": config["segment_progress_distance"],
        })
    session = AgentSession(
        session_id=agent_session_id,
        instruction=instruction,
        scene_id=scene_id,
        max_steps=max_steps,
        runtime_dir=RUNTIME_DIR,
        model_script=ROOT / "model_live_once.py",
        episode_id=str(data.get("episode_id") or DEFAULT_PHASE_A_EPISODE_ID),
        split=str(data.get("split") or DEFAULT_PHASE_A_SPLIT),
        checkpoint=str(data.get("checkpoint") or DEFAULT_PHASE_A_CKPT),
        simulator_tool_port=simulator_tool_port,
        save_frames=boolean_values["save_frames"],
        gpu_id=gpu_id,
        enable_oracle_goal_homing=boolean_values["enable_oracle_goal_homing"],
        enable_oracle_path_homing=boolean_values["enable_oracle_path_homing"],
        enable_learned_segment_homing=boolean_values["enable_learned_segment_homing"],
        enable_scene16_support_ranker=boolean_values["enable_scene16_support_ranker"],
        scene16_support_ranker=(
            v014_handshake["scene16_support_ranker"] if v014_handshake is not None
            else str(data.get("scene16_support_ranker") or "")
        ),
        enable_scene16_route_prior=boolean_values["enable_scene16_route_prior"],
        scene16_route_prior=str(data.get("scene16_route_prior") or ""),
        segment_grounding_ridge=(
            "" if v014_handshake is not None
            else str(data.get("segment_grounding_ridge") or DEFAULT_SEGMENT_GROUNDING_RIDGE)
        ),
        model_run_id=v014_handshake["model_run_id"] if v014_handshake is not None else None,
        model_extra_args=model_extra_args,
        v014_namespace=v014_namespace,
    )
    session, error = register_session_if_available(
        session_id,
        lambda: session,
        availability_error=lambda: (
            {"error": "agent resource lock exists", "locks": locks}
            if (locks := read_agent_locks())
            else None
        ),
    )
    if error is not None:
        if v014_namespace is not None:
            cleanup_v014_trace_namespace(v014_namespace, agent_session_id)
        return jsonify(error), 409
    response = {
        "ok": True,
        "session_id": session_id,
        "stream": f"/stream/{session_id}",
        "status_url": f"/api/agent/sessions/{session_id}",
    }
    if v014_handshake is not None:
        response.update(v014_handshake)
    return jsonify(response)


@app.route("/api/agent/stop/<session_id>", methods=["POST"])
def stop_agent(session_id):
    with sessions_lock:
        session = sessions.get(session_id)
    if session is None or getattr(session, "mode", None) != "agent":
        return jsonify({"error": "agent session not found"}), 404
    session.request_stop()
    return jsonify({"ok": True, "session": session.snapshot()})


@app.route("/api/agent/stop-active", methods=["POST"])
def stop_active_agent():
    with sessions_lock:
        active = [
            session
            for session in sessions.values()
            if getattr(session, "mode", None) == "agent" and is_active(session)
        ]
    if not active:
        return jsonify({"error": "no active agent session"}), 404
    stopped = []
    for session in active:
        session.request_stop()
        stopped.append(session.snapshot())
    return jsonify({"ok": True, "sessions": stopped})


@app.route("/api/start", methods=["POST"])
def start():
    data, error_response = request_json_object()
    if error_response is not None:
        return error_response
    instruction = str(data.get("instruction", "")).strip()
    if not instruction:
        return jsonify({"error": "instruction is required"}), 400
    if len(instruction) > 1000:
        return jsonify({"error": "instruction is too long"}), 400
    confirmed, error_response = request_boolean(data, "confirm_control", False)
    if error_response is not None:
        return error_response, 400
    mode = str(data.get("mode", "scripted"))
    if mode not in ("scripted", "camera"):
        return jsonify({"error": "mode must be scripted or camera"}), 400
    if mode == "scripted" and not confirmed:
        return jsonify({"error": "scripted flight requires confirm_control=true"}), 400

    scene_id, error_response = request_integer(data, "scene_id", 4)
    if error_response is not None:
        return error_response, 400
    max_steps, error_response = request_integer(data, "max_steps", 120)
    if error_response is not None:
        return error_response, 400
    action_sequence = data.get("actions", [])
    if not isinstance(action_sequence, list):
        return jsonify({"error": "actions must be a list"}), 400
    try:
        [LiveSession.normalize_action(item) for item in action_sequence]
    except (TypeError, ValueError, KeyError):
        return jsonify({"error": "actions contains an invalid action"}), 400

    session_id = now_session_id()
    session = LiveSession(
        session_id=session_id,
        instruction=instruction,
        scene_id=scene_id,
        max_steps=max(1, min(max_steps, 500)),
        mode=mode,
        action_sequence=action_sequence,
    )
    session, error = register_session_if_available(session_id, lambda: session)
    if error is not None:
        return jsonify(error), 409
    return jsonify({"session_id": session_id, "stream": f"/stream/{session_id}"})


@app.route("/api/stop/<session_id>", methods=["POST"])
def stop(session_id):
    with sessions_lock:
        session = sessions.get(session_id)
    if session is None:
        return jsonify({"error": "session not found"}), 404
    session.stop_event.set()
    return jsonify({"ok": True, "session": session.snapshot()})


def blend_jpeg_frames(previous, current, alpha, quality=82):
    """Create a lightweight display-only transition between two camera frames."""
    try:
        previous_image = Image.open(BytesIO(previous)).convert("RGB")
        current_image = Image.open(BytesIO(current)).convert("RGB")
        if previous_image.size != current_image.size:
            resampling = getattr(Image, "Resampling", Image)
            previous_image = previous_image.resize(current_image.size, resampling.BILINEAR)
        alpha = min(1.0, max(0.0, float(alpha)))
        eased_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        blended = Image.blend(previous_image, current_image, eased_alpha)
        output = BytesIO()
        blended.save(output, format="JPEG", quality=quality, optimize=False)
        return output.getvalue()
    except Exception:
        return current


@app.route("/stream/<session_id>")
def stream(session_id):
    with sessions_lock:
        session = sessions.get(session_id)
    if session is None:
        return jsonify({"error": "session not found"}), 404

    try:
        target_fps = min(30.0, max(8.0, float(request.args.get("fps", "24"))))
    except (TypeError, ValueError):
        target_fps = 24.0

    cinematic = request.args.get("cinematic", "0").lower() not in ("0", "false", "no")
    turbo = request.args.get("turbo", "0").lower() not in ("0", "false", "no")
    # Cinematic mode deliberately trades a small delay for stable playback when
    # model/AirSim step latency spikes late in the route.
    interpolate_default = "1" if cinematic else "0"
    interpolate = request.args.get("interpolate", interpolate_default).lower() not in ("0", "false", "no")
    try:
        default_buffer = "0.10" if turbo else ("2.50" if cinematic else "0.35")
        max_buffer = 4.0 if cinematic else 0.9
        playback_buffer = min(max_buffer, max(0.0, float(request.args.get("buffer", default_buffer))))
    except (TypeError, ValueError):
        playback_buffer = 2.5 if cinematic else 0.35
    hold_last_frame = resolve_hold_last_frame(request.args, turbo)
    hold_max_age = resolve_hold_max_age(request.args, cinematic=cinematic)
    blend_default = "0" if cinematic else "1"
    server_blend = request.args.get("blend", blend_default).lower() not in ("0", "false", "no")
    latest_only = request.args.get("latest", "1" if turbo else "0").lower() not in ("0", "false", "no")
    try:
        playback_speed = min(3.0, max(0.5, float(request.args.get("speed", "1.0"))))
    except (TypeError, ValueError):
        playback_speed = 1.0
    try:
        transition_cap = int(request.args.get("transition_cap", "10" if cinematic else "8"))
    except (TypeError, ValueError):
        transition_cap = 10 if cinematic else 8
    transition_cap_limit = 36 if cinematic and not turbo else 18
    transition_cap = min(transition_cap_limit, max(1, transition_cap))

    def generate():
        boundary = b"--frame\r\n"
        last_at = 0.0
        last_sequence = 0
        frame_period = 1.0 / target_fps
        next_frame_at = time.monotonic()
        previous_frame = None
        previous_frame_at = None
        last_good_frame_at = None
        playback_started = False
        transition_credit = 0.0

        def emit_frame(frame, sequence):
            nonlocal next_frame_at
            now = time.monotonic()
            delay = next_frame_at - now
            if 0.0 < delay <= 0.25:
                time.sleep(delay)
            elif delay > 0.25 or now - next_frame_at > 0.5:
                next_frame_at = now
            yield boundary
            yield b"Content-Type: image/jpeg\r\n"
            yield f"X-Step: {session.step}\r\n".encode("ascii")
            yield f"X-Action: {session.action}\r\n".encode("ascii")
            if sequence:
                yield f"X-Frame-Sequence: {sequence}\r\n".encode("ascii")
            yield f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
            yield frame
            yield b"\r\n"
            next_frame_at = max(next_frame_at + frame_period, time.monotonic())

        while True:
            if hasattr(session, "get_frame_packet"):
                wait_timeout = min(frame_period * 1.25, 0.08) if previous_frame is not None and hold_last_frame else 2.0
                packet = session.get_frame_packet(after_sequence=last_sequence, timeout=wait_timeout, latest_only=latest_only)
                if packet is None:
                    frame = None
                else:
                    if len(packet) == 3:
                        last_sequence, frame_at, frame = packet
                    else:
                        last_sequence, frame = packet
                        frame_at = session.last_frame_at
            else:
                frame = session.get_frame(timeout=2.0)
                frame_at = session.last_frame_at
            is_new_frame = frame is not None and (
                last_sequence > 0 or session.last_frame_at != last_at
            )
            if is_new_frame:
                if not playback_started and interpolate and playback_buffer > 0:
                    deadline = time.monotonic() + playback_buffer
                    while time.monotonic() < deadline and session.status not in ("done", "error", "stopped"):
                        time.sleep(min(0.02, deadline - time.monotonic()))
                    playback_started = True
                last_at = session.last_frame_at
                if previous_frame is None or not interpolate:
                    yield from emit_frame(frame, last_sequence)
                else:
                    interval_cap = 0.35 if turbo else (0.75 if cinematic else 0.35)
                    max_transition_steps = 4 if turbo else transition_cap
                    source_interval = max(
                        frame_period,
                        min(interval_cap, float(frame_at) - float(previous_frame_at)) / playback_speed,
                    )
                    transition_credit += source_interval * target_fps
                    transition_steps = max(1, min(max_transition_steps, int(transition_credit)))
                    transition_credit = max(0.0, transition_credit - transition_steps)
                    for transition_index in range(1, transition_steps + 1):
                        if transition_index == transition_steps:
                            display_frame = frame
                        elif not server_blend:
                            # Let the browser canvas do temporal smoothing. PIL JPEG blending here
                            # can become the late-route bottleneck when source frames arrive slowly.
                            display_frame = previous_frame
                        else:
                            display_frame = blend_jpeg_frames(
                                previous_frame,
                                frame,
                                transition_index / transition_steps,
                            )
                        yield from emit_frame(display_frame, last_sequence)
                previous_frame = frame
                previous_frame_at = frame_at
                last_good_frame_at = frame_at
            elif hold_last_frame and session.status not in ("done", "error", "stopped") and should_hold_frame(
                previous_frame,
                last_good_frame_at,
                time.time(),
                max_age_sec=hold_max_age,
            ):
                yield from emit_frame(previous_frame, last_sequence)
            queue_drained = not hasattr(session, "frame_sequence") or last_sequence >= session.frame_sequence
            if session.status in ("done", "error", "stopped") and session.last_frame_at == last_at and queue_drained:
                break

    response = Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["X-Display-FPS"] = str(int(target_fps))
    if not interpolate:
        response.headers["X-Frame-Interpolation"] = "off"
    else:
        response.headers["X-Frame-Interpolation"] = "client-temporal" if not server_blend else "linear-jpeg"
    response.headers["X-Cinematic-Stream"] = "on" if cinematic else "off"
    response.headers["X-Turbo-Stream"] = "on" if turbo else "off"
    response.headers["X-Latest-Only"] = "on" if latest_only else "off"
    response.headers["X-Playback-Buffer"] = f"{playback_buffer:.2f}"
    response.headers["X-Playback-Speed"] = f"{playback_speed:.2f}"
    response.headers["X-Transition-Cap"] = str(transition_cap)
    response.headers["X-Frame-Hold"] = "on" if hold_last_frame else "off"
    response.headers["X-Frame-Hold-Age"] = f"{hold_max_age:.2f}"
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    if args.host in ("127.0.0.1", "localhost") and not is_port_free(args.host, args.port):
        raise SystemExit(f"Port {args.port} is already in use on {args.host}")

    print(f"AirVLN live server: http://{args.host}:{args.port}", flush=True)
    print(f"Runtime dir: {RUNTIME_DIR}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
