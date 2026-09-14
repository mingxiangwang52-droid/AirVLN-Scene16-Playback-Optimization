#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
import re
import stat


FLIGHT_TARGET_TELEMETRY_SCHEMA = "scene16_flight_target_telemetry_v1"


def normalize_flight_target_number(value):
    if isinstance(value, bool):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None


def normalize_flight_target_position(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    position = [normalize_flight_target_number(item) for item in value]
    return position if all(item is not None for item in position) else None


def build_episode_metadata(episode, goal_position):
    """Build immutable episode fields after navigation state has been finalized."""
    instruction = episode.get("instruction") or {}
    if isinstance(instruction, dict):
        instruction = instruction.get("instruction_text")
    return {
        "scene_id": episode.get("scene_id"),
        "trajectory_id": episode.get("trajectory_id"),
        "goal_position": goal_position,
        "instruction": instruction,
    }


def choose_route_prior_segment_delta(route_prior_candidate, segment_index, segment_count, segment=None):
    if not isinstance(route_prior_candidate, dict):
        return None
    templates = route_prior_candidate.get("segment_delta_templates")
    if not isinstance(templates, list) or not templates:
        return None
    segment_count = max(1, int(segment_count or 1))
    segment_index = max(0, int(segment_index or 0))
    target_ratio = (float(segment_index) + 0.5) / float(segment_count)
    segment = segment if isinstance(segment, dict) else {}
    segment_actions = {
        str(item)
        for item in segment.get("actions") or []
        if isinstance(item, str)
    }
    segment_landmark = segment.get("target_landmark")
    segment_landmark = str(segment_landmark) if isinstance(segment_landmark, str) else None
    ranked = []
    for template_index, template in enumerate(templates):
        if not isinstance(template, dict):
            continue
        delta = normalize_flight_target_position(template.get("delta_xyz"))
        if delta is None:
            continue
        template_ratio = normalize_flight_target_number(template.get("segment_ratio"))
        if template_ratio is None:
            template_count = max(1, int(template.get("segment_count") or len(templates) or 1))
            template_segment_index = int(template.get("segment_index") or 0)
            template_ratio = (float(template_segment_index) + 0.5) / float(template_count)
        template_actions = {
            str(item)
            for item in template.get("actions") or []
            if isinstance(item, str)
        }
        action_penalty = 0 if not segment_actions or segment_actions.intersection(template_actions) else 1
        template_landmark = template.get("target_landmark")
        landmark_penalty = (
            0
            if not segment_landmark
            or not isinstance(template_landmark, str)
            or segment_landmark == template_landmark
            else 1
        )
        ranked.append((
            abs(float(template_ratio) - target_ratio),
            action_penalty,
            landmark_penalty,
            template_index,
            delta,
        ))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[:4])
    return list(ranked[0][4])


def route_prior_adjusted_learned_delta(
    predicted_delta,
    route_prior_candidate,
    segment_index,
    segment_count,
    segment=None,
    blend=0.35,
):
    predicted = normalize_flight_target_position(predicted_delta)
    if predicted is None:
        return predicted_delta
    try:
        blend = float(blend)
    except (TypeError, ValueError):
        return predicted
    if not 0.0 < blend <= 1.0:
        return predicted
    template = choose_route_prior_segment_delta(
        route_prior_candidate,
        segment_index,
        segment_count,
        segment=segment,
    )
    if template is None:
        return predicted
    predicted_xy = math.hypot(float(predicted[0]), float(predicted[1]))
    template_xy = math.hypot(float(template[0]), float(template[1]))
    if predicted_xy <= 1e-3 or template_xy <= 1e-3:
        return predicted
    template_scale = predicted_xy / template_xy
    adjusted = [
        (1.0 - blend) * float(predicted[0]) + blend * float(template[0]) * template_scale,
        (1.0 - blend) * float(predicted[1]) + blend * float(template[1]) * template_scale,
        float(predicted[2]),
    ]
    return adjusted


def select_route_prior_candidate_for_goal_bearing(route_prior_candidates, current_position, goal_position):
    if not isinstance(route_prior_candidates, list) or not route_prior_candidates:
        return None
    current = normalize_flight_target_position(current_position)
    goal = normalize_flight_target_position(goal_position)
    if current is None or goal is None:
        return route_prior_candidates[0] if isinstance(route_prior_candidates[0], dict) else None
    dx = float(goal[0]) - float(current[0])
    dy = float(goal[1]) - float(current[1])
    distance = math.hypot(dx, dy)
    if distance <= 1e-6:
        return route_prior_candidates[0] if isinstance(route_prior_candidates[0], dict) else None
    target_unit = [dx / distance, dy / distance]
    ranked = []
    for index, candidate in enumerate(route_prior_candidates):
        if not isinstance(candidate, dict):
            continue
        route_unit = candidate.get("route_goal_unit_xy")
        if (
            not isinstance(route_unit, (list, tuple))
            or len(route_unit) != 2
            or any(normalize_flight_target_number(item) is None for item in route_unit)
        ):
            ranked.append((-2.0, index, candidate))
            continue
        dot = target_unit[0] * float(route_unit[0]) + target_unit[1] * float(route_unit[1])
        ranked.append((dot, index, candidate))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][2]


def _flight_target_landmarks(value):
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).lower() for item in value if isinstance(item, str) and item]


def build_flight_target_ranker_context(segment, grounding):
    segment = segment if isinstance(segment, dict) else {}
    grounding = grounding if isinstance(grounding, dict) else {}
    landmark = segment.get("target_landmark")
    landmark = str(landmark).lower() if isinstance(landmark, str) and landmark else None
    explicit_landmarks = set(_flight_target_landmarks(segment.get("landmarks")))
    support_landmarks = set(_flight_target_landmarks(segment.get("support_landmarks")))
    decision = grounding.get("scene16_support_ranker_decision")
    decision = decision if isinstance(decision, dict) else None
    ranker_top_support = None
    ranker_influenced_target = False
    if decision is not None:
        final_ids = decision.get("final_episode_ids")
        top_episode_id = final_ids[0] if isinstance(final_ids, list) and final_ids and isinstance(final_ids[0], str) else None
        top_example = next(
            (
                item
                for item in grounding.get("support_examples") or []
                if isinstance(item, dict) and item.get("episode_id") == top_episode_id
            ),
            None,
        )
        top_landmarks = _flight_target_landmarks(top_example.get("landmarks")) if top_example else []
        top_score = None
        for candidate in decision.get("scored_candidates") or []:
            if isinstance(candidate, dict) and candidate.get("episode_id") == top_episode_id:
                top_score = normalize_flight_target_number(candidate.get("score"))
                break
        ranker_top_support = {
            "applied": bool(decision.get("applied")),
            "fallback_reason": decision.get("fallback_reason") if isinstance(decision.get("fallback_reason"), str) else None,
            "candidate_source": decision.get("candidate_source") if isinstance(decision.get("candidate_source"), str) else None,
            "top_two_margin": normalize_flight_target_number(decision.get("top_two_margin")),
            "selected_top_episode_id": top_episode_id,
            "selected_top_score": top_score,
            "landmarks": top_landmarks or None,
        }
        ranker_influenced_target = bool(decision.get("applied")) and bool(
            landmark and landmark in top_landmarks and landmark not in explicit_landmarks
        )
    if landmark in explicit_landmarks:
        target_source = "explicit_landmark"
    elif ranker_influenced_target:
        target_source = "ranker_artifact"
    elif landmark and landmark in support_landmarks:
        target_source = "support_landmark"
    else:
        target_source = "fallback"
    return {
        "selected_landmark": landmark,
        "target_source": target_source,
        "ranker_top_support": ranker_top_support,
        "ranker_influenced_target": ranker_influenced_target,
    }


def build_flight_target_telemetry_event(
    *, event, step, target_before, target_after, target_changed_reason,
    selected_landmark, target_source, ranker_top_support, ranker_influenced_target,
    current_position, distance_to_dataset_goal, regression_start_step,
    regression_stop_step, recovery_reason, raw_action, applied_action,
):
    normalized_before = normalize_flight_target_position(target_before)
    normalized_after = normalize_flight_target_position(target_after)
    position = normalize_flight_target_position(current_position)
    selected_distance = None
    if normalized_after is not None and position is not None:
        selected_distance = math.dist(position, normalized_after)
    return {
        "schema_version": FLIGHT_TARGET_TELEMETRY_SCHEMA,
        "event": event,
        "step": int(step),
        "trace_step": int(step),
        "target_before": list(normalized_before) if normalized_before is not None else None,
        "target_after": list(normalized_after) if normalized_after is not None else None,
        "target_changed_reason": target_changed_reason,
        "selected_target_position": list(normalized_after) if normalized_after is not None else None,
        "distance_to_selected_target": selected_distance,
        "distance_to_dataset_goal": normalize_flight_target_number(distance_to_dataset_goal),
        "selected_landmark": selected_landmark,
        "target_source": target_source,
        "ranker_top_support": ranker_top_support,
        "ranker_influenced_target": bool(ranker_influenced_target),
        "regression_start_step": regression_start_step,
        "regression_stop_step": regression_stop_step,
        "recovery_reason": recovery_reason,
        "raw_action": raw_action,
        "applied_action": applied_action,
    }


def append_flight_target_telemetry(path, event):
    with Path(path).open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def build_flight_target_telemetry_manifest(path, event_count):
    content = Path(path).read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "event_count": int(event_count),
    }


_MODEL_RUN_ID_RE = re.compile(r"^v014-[0-9a-f]{32}$")


def create_output_dir(out_root, model_run_id=None, pre_reserved=False):
    """Create a fresh output namespace without replacing an existing run."""
    root = Path(out_root)
    if model_run_id is None:
        out_dir = root / time.strftime("%Y%m%d-%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    if not isinstance(model_run_id, str) or not _MODEL_RUN_ID_RE.fullmatch(model_run_id):
        raise ValueError("model_run_id must be a safe v014 path component")
    if not pre_reserved:
        root.mkdir(parents=True, exist_ok=True)
    root_stat = os.lstat(root)
    if stat.S_ISLNK(root_stat.st_mode) or bool(
        getattr(root_stat, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    ) or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("out_root must be a non-reparse directory")
    out_dir = root / model_run_id
    if pre_reserved:
        try:
            out_stat = os.lstat(out_dir)
        except OSError as exc:
            raise ValueError("pre-reserved model run directory is unavailable") from exc
        if stat.S_ISLNK(out_stat.st_mode) or bool(
            getattr(out_stat, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ) or not stat.S_ISDIR(out_stat.st_mode):
            raise ValueError("pre-reserved model run directory is invalid")
        return out_dir
    os.mkdir(out_dir)
    return out_dir


def choose_learned_segment_xy_step(
    base_xy_step,
    min_xy_step,
    max_xy_step,
    safety_turn_count,
    in_safety_stall_cooldown,
):
    def finite_float(value, fallback):
        try:
            return float(value)
        except Exception:
            return fallback

    def finite_positive(value, fallback):
        value = finite_float(value, fallback)
        return value if math.isfinite(value) and value > 0.0 else fallback

    minimum = finite_positive(min_xy_step, 0.0)
    maximum = finite_positive(max_xy_step, minimum)
    if maximum < minimum:
        maximum = minimum

    base = finite_float(base_xy_step, None)
    if base is None or not math.isfinite(base) or base < 0.0:
        base = minimum
    base = min(max(base, minimum), maximum)

    safety_turn_count = finite_float(safety_turn_count, 0.0)
    if not math.isfinite(safety_turn_count) or safety_turn_count < 0.0:
        safety_turn_count = 0.0

    if safety_turn_count < 3.0:
        return base, 0, "base"
    if safety_turn_count < 6.0:
        return min(max(base * 0.85, minimum), maximum), 1, "safety_turn_caution"
    try:
        in_safety_stall_cooldown = bool(in_safety_stall_cooldown)
    except Exception:
        in_safety_stall_cooldown = False
    reason = "safety_stall_cooldown" if in_safety_stall_cooldown else "safety_stall"
    return min(max(base * 0.70, minimum), maximum), 2, reason


def safety_stall_cooldown_is_active(step, last_safety_stall_resample_step):
    try:
        current_step = int(step)
    except Exception:
        return False
    if isinstance(last_safety_stall_resample_step, bool):
        return False
    if isinstance(last_safety_stall_resample_step, int):
        marker = last_safety_stall_resample_step
    elif (
        isinstance(last_safety_stall_resample_step, float)
        and math.isfinite(last_safety_stall_resample_step)
        and last_safety_stall_resample_step.is_integer()
    ):
        marker = int(last_safety_stall_resample_step)
    else:
        return False
    return marker <= current_step and current_step - marker < 24


def learned_segment_target_xy_adjustment(
    predicted_delta,
    route_like_segment,
    min_xy_step,
    max_xy_step,
    safety_turn_count,
    step,
    last_safety_stall_resample_step,
):
    adjusted_delta = list(predicted_delta)
    base_xy_step = math.hypot(float(adjusted_delta[0]), float(adjusted_delta[1]))
    minimum = max(0.0, float(min_xy_step))
    maximum = max(minimum, float(max_xy_step))
    if not route_like_segment:
        xy_step = min(base_xy_step, maximum)
        if base_xy_step > 0.0 and xy_step < base_xy_step:
            scale = xy_step / base_xy_step
            adjusted_delta[0] = float(adjusted_delta[0]) * scale
            adjusted_delta[1] = float(adjusted_delta[1]) * scale
            return adjusted_delta, base_xy_step, xy_step, 0, "not_route_like_clamped"
        return adjusted_delta, base_xy_step, base_xy_step, 0, "not_route_like"

    base_xy_step = max(minimum, min(maximum, base_xy_step))
    xy_step, risk_tier, reason = choose_learned_segment_xy_step(
        base_xy_step,
        minimum,
        maximum,
        safety_turn_count,
        safety_stall_cooldown_is_active(step, last_safety_stall_resample_step),
    )
    current_xy_step = math.hypot(float(adjusted_delta[0]), float(adjusted_delta[1]))
    if current_xy_step > 0.0:
        scale = xy_step / current_xy_step
        adjusted_delta[0] = float(adjusted_delta[0]) * scale
        adjusted_delta[1] = float(adjusted_delta[1]) * scale
    return adjusted_delta, base_xy_step, xy_step, risk_tier, reason


def record_learned_segment_depth_guard_turn(guidance_state, depth_reason, homing_target_kind):
    if depth_reason != "depth_building_guard_turn" or homing_target_kind != "learned_segment":
        return False
    guidance_state["learned_segment_homing_safety_turn_count"] = int(
        guidance_state.get("learned_segment_homing_safety_turn_count", 0)
    ) + 1
    guidance_state["learned_segment_homing_segment_safety_turn_count"] = int(
        guidance_state.get("learned_segment_homing_segment_safety_turn_count", 0)
    ) + 1
    return True


def learned_segment_homing_diagnostics(guidance_state, step):
    created_step = guidance_state.get("learned_segment_homing_created_step")
    active = guidance_state.get("learned_segment_homing_target") is not None
    target_age = None
    if active and isinstance(created_step, int):
        target_age = int(step) - created_step + 1
    return {
        "active": active,
        "target_index": guidance_state.get("learned_segment_homing_index"),
        "target_age": target_age,
        "target_distance": guidance_state.get("learned_segment_homing_distance"),
        "entry_distance": guidance_state.get("learned_segment_homing_entry_distance"),
        "best_distance": guidance_state.get("learned_segment_homing_best_distance"),
        "resamples": int(guidance_state.get("learned_segment_homing_resamples", 0)),
        "safety_turn_count": int(
            guidance_state.get("learned_segment_homing_safety_turn_count", 0)
        ),
        "segment_safety_turn_count": int(
            guidance_state.get("learned_segment_homing_segment_safety_turn_count", 0)
        ),
        "last_safety_stall_resample_step": guidance_state.get(
            "learned_segment_homing_last_safety_stall_resample_step"
        ),
        "goal_regression_resamples": int(
            guidance_state.get("learned_segment_homing_goal_regression_resamples", 0)
        ),
        "last_goal_regression_resample_step": guidance_state.get(
            "learned_segment_homing_last_goal_regression_resample_step"
        ),
        "base_xy_step": guidance_state.get("learned_segment_homing_base_xy_step"),
        "xy_step": guidance_state.get("learned_segment_homing_xy_step"),
        "risk_tier": guidance_state.get("learned_segment_homing_risk_tier"),
        "xy_step_reason": guidance_state.get("learned_segment_homing_xy_step_reason"),
    }


def should_break_learned_homing_turn_loop(
    homing_target_kind,
    target_age,
    stationary_turn_streak,
    collision_rollback_count,
):
    return (
        homing_target_kind == "learned_segment"
        and int(target_age) >= 12
        and int(stationary_turn_streak) >= 5
        and int(collision_rollback_count) == 0
    )


def should_resample_stalled_learned_homing_target(
    homing_target_kind,
    target_age,
    entry_distance,
    best_distance,
    safety_turn_count,
    step=None,
    last_safety_stall_resample_step=None,
):
    return (
        homing_target_kind == "learned_segment"
        and int(target_age) >= 12
        and entry_distance is not None
        and best_distance is not None
        and float(entry_distance) - float(best_distance) < 4.0
        and int(safety_turn_count) >= 6
        and not safety_stall_cooldown_is_active(step, last_safety_stall_resample_step)
    )


def should_resample_learned_target_for_goal_regression(
    *,
    learned_enabled,
    oracle_path_enabled,
    target_active,
    target_age,
    worse_streak,
    regression_from_best,
    min_age,
    worse_streak_threshold,
    regression_threshold,
):
    """Gate learned-target replacement using only goal-distance telemetry."""
    if not learned_enabled or oracle_path_enabled or not target_active:
        return False
    try:
        return bool(
            int(target_age) >= int(min_age)
            and int(worse_streak) >= int(worse_streak_threshold)
            and float(regression_from_best) >= float(regression_threshold)
        )
    except (TypeError, ValueError):
        return False


def should_reject_final_learned_target_for_goal_progress(
    *,
    enabled,
    final_segment,
    current_distance,
    candidate_distance,
    tolerance,
):
    """Reject a final learned target that materially worsens goal distance."""
    if not enabled or not final_segment:
        return False
    try:
        current = float(current_distance)
        candidate = float(candidate_distance)
        margin = max(0.0, float(tolerance))
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (current, candidate, margin)):
        return False
    return candidate > current + margin


def should_release_stale_learned_clearance_climb(
    homing_target_kind,
    target_age,
    xy_distance,
    z_error,
    altitude,
    cruise_altitude,
    z_tolerance,
):
    """Avoid over-climbing toward a mature learned target whose z is now below us."""
    if homing_target_kind != "learned_segment":
        return False
    try:
        age = int(target_age)
        xy = float(xy_distance)
        z_delta = float(z_error)
        current_altitude = float(altitude)
        cruise = float(cruise_altitude)
        tolerance = max(0.0, float(z_tolerance))
    except Exception:
        return False
    if not all(math.isfinite(value) for value in (xy, z_delta, current_altitude, cruise, tolerance)):
        return False
    return bool(
        age >= 18
        and xy <= 35.0
        and z_delta > max(8.0, tolerance * 2.0)
        and current_altitude >= max(18.0, cruise - 8.0)
    )


def should_defer_final_descent_far_from_goal(distance_to_goal, arrival_fine_approach_distance):
    try:
        distance = float(distance_to_goal)
        fine_distance = float(arrival_fine_approach_distance)
    except Exception:
        return False
    if not math.isfinite(distance) or distance < 0.0:
        return False
    if not math.isfinite(fine_distance) or fine_distance <= 0.0:
        fine_distance = 50.0
    return distance > max(50.0, fine_distance)


def should_defer_close_final_structure_descent(
    segment,
    distance_to_goal,
    success_distance,
    surface_clearance,
):
    """Keep structure/sign final segments moving horizontally until they are closer."""
    try:
        distance = float(distance_to_goal)
        success = float(success_distance)
    except Exception:
        return False
    if not all(math.isfinite(value) for value in (distance, success)):
        return False
    if success <= 0.0 or distance <= max(success + 8.0, 28.0):
        return False
    if surface_clearance is not None:
        try:
            clearance = float(surface_clearance)
        except Exception:
            return False
        if not math.isfinite(clearance) or clearance <= 3.0:
            return False
    segment = segment or {}
    text = str(segment.get("text") or "").lower()
    target_landmark = str(segment.get("target_landmark") or "").lower()
    target_zone = str(segment.get("target_zone") or "").lower()
    actions = {str(action).lower() for action in segment.get("actions") or []}
    structure_or_sign = bool(
        target_zone == "structure"
        or target_landmark in {"building", "sign", "billboard", "tower"}
        or any(token in text for token in ("building", "billboard", "sign", "tower", "charlie"))
    )
    wants_descent = bool("descend" in actions or any(token in text for token in ("land", "landing", "down")))
    corridor = bool(
        target_zone in {"corridor", "road"}
        or target_landmark in {"road", "intersection", "street"}
        or any(token in text for token in ("road", "intersection", "street", "crossing"))
    )
    return bool(structure_or_sign and wants_descent and not corridor)


def should_continue_success_zone_surface_descent(
    distance_to_goal,
    success_distance,
    surface_clearance,
    altitude,
    surface_landing_clearance,
):
    """Keep descending in the success zone until surface contact is known."""
    try:
        distance = float(distance_to_goal)
        success = float(success_distance)
        current_altitude = float(altitude)
        landing_clearance = float(surface_landing_clearance)
    except Exception:
        return False
    if not all(math.isfinite(value) for value in (distance, success, current_altitude, landing_clearance)):
        return False
    if success <= 0.0 or distance < 0.0 or distance > success:
        return False
    if surface_clearance is not None:
        try:
            clearance = float(surface_clearance)
        except Exception:
            return False
        return bool(math.isfinite(clearance) and clearance > landing_clearance)
    return current_altitude > landing_clearance + 0.5


def recent_valid_surface_clearance(current_step, last_valid_step, last_valid_clearance, max_age=2):
    try:
        step = int(current_step)
        valid_step = int(last_valid_step)
        clearance = float(last_valid_clearance)
        age = int(max_age)
    except Exception:
        return None
    if age < 0 or valid_step > step or step - valid_step > age:
        return None
    if not math.isfinite(clearance) or clearance < 0.0:
        return None
    return clearance


def estimate_surface_clearance_from_recent_probe(
    current_z,
    last_valid_z,
    last_valid_clearance,
    current_step,
    last_valid_step,
    max_age=8,
):
    try:
        z = float(current_z)
        valid_z = float(last_valid_z)
        clearance = float(last_valid_clearance)
        step = int(current_step)
        valid_step = int(last_valid_step)
        age = int(max_age)
    except Exception:
        return None
    if age < 0 or valid_step > step or step - valid_step > age:
        return None
    if not all(math.isfinite(value) for value in (z, valid_z, clearance)):
        return None
    if clearance < 0.0:
        return None
    descent_since_probe = z - valid_z
    if descent_since_probe < 0.0:
        return None
    return max(0.0, clearance - descent_since_probe)


def should_enable_depth_route_around(
    segment,
    distance_to_goal,
    initial_distance,
    step,
    min_step,
):
    """Allow route-around only near terminal physical targets, not road routing."""
    try:
        if int(step) < int(min_step):
            return False
    except Exception:
        return False
    try:
        distance = float(distance_to_goal)
    except Exception:
        return False
    if not math.isfinite(distance) or distance < 0.0:
        return False
    try:
        initial = float(initial_distance)
    except Exception:
        initial = None
    if initial is not None and math.isfinite(initial) and initial > 0.0:
        if distance > max(160.0, initial * 0.45):
            return False
    elif distance > 160.0:
        return False

    segment = segment or {}
    text = str(segment.get("text") or "").lower()
    target_landmark = str(segment.get("target_landmark") or "").lower()
    target_zone = str(segment.get("target_zone") or "").lower()
    actions = {str(action).lower() for action in segment.get("actions") or []}
    relation_text = " ".join(
        str(relation.get("text") or "").lower()
        for relation in segment.get("relations") or []
        if isinstance(relation, dict)
    )
    combined = " ".join([text, relation_text])
    if "intersection" in combined and (
        target_landmark in {"road", "intersection"}
        or target_zone in {"road", "corridor"}
    ):
        return False
    has_terminal_surface = bool({"descend", "fly_over"} & actions) or any(
        token in combined
        for token in (
            "land",
            "landing",
            "bench",
            "top",
            "roof",
            "facing",
            "near to",
            "beside",
        )
    )
    has_physical_anchor = target_zone == "structure" or target_landmark == "building" or any(
        token in combined for token in ("building", "bench", "roof", "top")
    )
    return bool(has_terminal_surface and has_physical_anchor)


def should_release_final_corridor_depth_guard(
    segment,
    segment_index,
    segment_count,
    distance_to_goal,
    surface_clearance,
    front_contact_ratio,
    front_very_close_ratio,
    front_p10,
    safety_turn_count,
    avoidance_streak,
):
    """Let a final low-altitude road/intersection approach make cautious progress."""
    try:
        distance = float(distance_to_goal)
    except Exception:
        return False
    if not math.isfinite(distance) or not (20.0 < distance <= 36.0):
        return False
    try:
        clearance = float(surface_clearance)
    except Exception:
        return False
    if not math.isfinite(clearance) or clearance > 18.5:
        return False
    try:
        contact = float(front_contact_ratio)
        very_close = float(front_very_close_ratio)
        p10 = float(front_p10)
    except Exception:
        return False
    if not all(math.isfinite(value) for value in (contact, very_close, p10)):
        return False
    if contact > 0.002 or very_close > 0.025 or p10 < 0.038:
        return False
    try:
        index = int(segment_index)
        count = max(1, int(segment_count))
    except Exception:
        return False
    if index < max(0, count - 2):
        return False
    segment = segment or {}
    text = str(segment.get("text") or "").lower()
    target_landmark = str(segment.get("target_landmark") or "").lower()
    target_zone = str(segment.get("target_zone") or "").lower()
    actions = {str(action).lower() for action in segment.get("actions") or []}
    corridor_target = bool(
        target_zone in {"corridor", "road"}
        or target_landmark in {"road", "intersection"}
        or "intersection road" in text
    )
    landing_target = bool("descend" in actions or "land" in text)
    if not (corridor_target and landing_target):
        return False
    try:
        safety_turns = int(safety_turn_count)
    except Exception:
        safety_turns = 0
    try:
        avoidance = int(avoidance_streak)
    except Exception:
        avoidance = 0
    return safety_turns >= 12 or avoidance >= 10


def should_release_close_final_structure_depth_guard(
    segment,
    segment_index,
    segment_count,
    distance_to_goal,
    surface_clearance,
    front_contact_ratio,
    front_very_close_ratio,
    front_p10,
    avoidance_streak,
):
    """Let close final structure/sign approaches make short progress when depth is not contact-like."""
    try:
        distance = float(distance_to_goal)
    except Exception:
        return False
    if not math.isfinite(distance) or not (24.0 <= distance <= 42.0):
        return False
    try:
        clearance = float(surface_clearance)
    except Exception:
        return False
    if not math.isfinite(clearance) or clearance > 20.0:
        return False
    try:
        contact = 0.0 if front_contact_ratio is None else float(front_contact_ratio)
        very_close = float(front_very_close_ratio)
        p10 = float(front_p10)
    except Exception:
        return False
    if not all(math.isfinite(value) for value in (contact, very_close, p10)):
        return False
    if contact > 0.004 or very_close > 0.012 or p10 < 0.052:
        return False
    try:
        index = int(segment_index)
        count = max(1, int(segment_count))
        avoidance = int(avoidance_streak)
    except Exception:
        return False
    if index < max(0, count - 2) or avoidance < 2:
        return False
    segment = segment or {}
    text = str(segment.get("text") or "").lower()
    target_landmark = str(segment.get("target_landmark") or "").lower()
    target_zone = str(segment.get("target_zone") or "").lower()
    actions = {str(action).lower() for action in segment.get("actions") or []}
    final_structure_or_sign = bool(
        target_zone == "structure"
        or target_landmark in {"building", "sign", "billboard", "tower"}
        or any(token in text for token in ("building", "billboard", "sign", "tower", "charlie"))
    )
    final_descent_or_stop = bool(
        {"descend", "land", "stop"} & actions
        or any(token in text for token in ("land", "landing", "stop", "stay", "by the"))
    )
    return bool(final_structure_or_sign and final_descent_or_stop)


def should_block_final_structure_descent_for_visual_clearance(
    segment,
    segment_index,
    segment_count,
    distance_to_goal,
    success_distance,
    surface_clearance,
    front_p10,
    front_contact_ratio,
    front_very_close_ratio,
):
    """Prevent repeated near-final descents into visually occupied structure faces."""
    try:
        distance = float(distance_to_goal)
        success = float(success_distance)
    except Exception:
        return False
    if not all(math.isfinite(value) for value in (distance, success)):
        return False
    if distance <= success or distance > max(success + 16.0, 36.0):
        return False
    try:
        clearance = float(surface_clearance)
    except Exception:
        return False
    if not math.isfinite(clearance) or clearance < 3.0 or clearance > 12.0:
        return False
    try:
        p10 = float(front_p10)
        contact = float(front_contact_ratio)
        very_close = float(front_very_close_ratio)
    except Exception:
        return False
    if not all(math.isfinite(value) for value in (p10, contact, very_close)):
        return False
    if p10 >= 0.055 and contact < 0.002 and very_close < 0.02:
        return False
    try:
        index = int(segment_index)
        count = max(1, int(segment_count))
    except Exception:
        return False
    if index < max(0, count - 2):
        return False
    segment = segment or {}
    text = str(segment.get("text") or "").lower()
    target_landmark = str(segment.get("target_landmark") or "").lower()
    target_zone = str(segment.get("target_zone") or "").lower()
    actions = {str(action).lower() for action in segment.get("actions") or []}
    is_structure = bool(
        target_zone == "structure"
        or target_landmark in {"building", "roof", "terrace", "door"}
        or any(token in text for token in ("building", "roof", "terrace", "door"))
    )
    has_descent = bool("descend" in actions or "land" in text or "landing" in text)
    return bool(is_structure and has_descent)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", default="304SM51WAC2LJX66KU06PFQOK2ISBK__seg006")
    parser.add_argument("--split", default="train_scene16_segments_high_conf_1k")
    parser.add_argument(
        "--ckpt",
        default=(
            os.environ.get("AIRVLN_AGENT_CHECKPOINT", "")
        ),
    )
    parser.add_argument("--scene-id", type=int, default=16)
    parser.add_argument("--instruction", default="now go over the buildings and get down towards the road")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--simulator-tool-port", type=int, default=30003)
    parser.add_argument("--out-root", default=str(Path(__file__).resolve().parent / "runtime" / "model_runs"))
    parser.add_argument("--model-run-id", default=None)
    parser.add_argument("--pre-reserved-model-run", action="store_true")
    parser.add_argument("--trainer_gpu_device", type=int, default=0)
    parser.add_argument("--disable-safety-depth-override", action="store_true")
    parser.add_argument("--disable-simple-route-guidance", action="store_true")
    parser.add_argument("--disable-grounding-guidance", action="store_true")
    parser.add_argument("--disable-segmented-grounding", action="store_true")
    parser.add_argument("--arrival-fine-approach-distance", type=float, default=50.0)
    parser.add_argument("--arrival-stop-distance", type=float, default=12.0)
    parser.add_argument("--arrival-overshoot-distance", type=float, default=35.0)
    parser.add_argument("--success-distance", type=float, default=20.0)
    parser.add_argument("--enable-oracle-goal-homing", action="store_true")
    parser.add_argument("--enable-oracle-path-homing", action="store_true")
    parser.add_argument("--enable-learned-segment-homing", action="store_true")
    parser.add_argument("--enable-scene16-support-ranker", action="store_true")
    parser.add_argument("--scene16-support-ranker", default="")
    parser.add_argument("--enable-scene16-route-prior", action="store_true")
    parser.add_argument("--scene16-route-prior", default="")
    parser.add_argument("--segment-grounding-ridge", default="")
    parser.add_argument("--segment-grounding-final-ridge", default="")
    parser.add_argument("--learned-segment-homing-min-step", type=int, default=1)
    parser.add_argument("--learned-segment-target-radius", type=float, default=8.0)
    parser.add_argument("--learned-segment-max-age", type=int, default=28)
    parser.add_argument("--learned-segment-z-clamp", type=float, default=6.0)
    parser.add_argument("--learned-segment-min-xy-step", type=float, default=22.0)
    parser.add_argument("--learned-segment-max-xy-step", type=float, default=38.0)
    parser.add_argument("--learned-segment-conservative-max-xy-step", type=float, default=0.0)
    parser.add_argument("--learned-segment-resample-worse-streak", type=int, default=0)
    parser.add_argument("--learned-segment-resample-regression", type=float, default=0.0)
    parser.add_argument("--learned-segment-goal-regression-worse-streak", type=int, default=4)
    parser.add_argument("--learned-segment-goal-regression", type=float, default=55.0)
    parser.add_argument("--learned-segment-goal-regression-min-age", type=int, default=8)
    parser.add_argument("--enable-final-goal-progress-gate", action="store_true")
    parser.add_argument("--final-goal-progress-tolerance", type=float, default=8.0)
    parser.add_argument("--final-goal-progress-max-retries", type=int, default=2)
    parser.add_argument("--enable-close-tail-depth-release", action="store_true")
    parser.add_argument("--learned-segment-cruise-altitude", type=float, default=36.0)
    parser.add_argument("--learned-segment-descent-altitude", type=float, default=18.0)
    parser.add_argument("--oracle-goal-homing-min-step", type=int, default=1)
    parser.add_argument("--oracle-goal-homing-distance", type=float, default=9999.0)
    parser.add_argument("--oracle-goal-homing-yaw-deg", type=float, default=25.0)
    parser.add_argument("--oracle-goal-homing-cruise-altitude", type=float, default=45.0)
    parser.add_argument("--oracle-goal-homing-z-tolerance", type=float, default=6.0)
    parser.add_argument("--oracle-goal-homing-z-align-distance", type=float, default=35.0)
    parser.add_argument("--oracle-path-lookahead", type=int, default=6)
    parser.add_argument("--oracle-path-waypoint-radius", type=float, default=16.0)
    parser.add_argument("--oracle-path-waypoint-max-age", type=int, default=18)
    parser.add_argument("--remote-done-recovery-lookahead", type=int, default=24)
    parser.add_argument("--remote-done-recovery-climb", type=float, default=14.0)
    parser.add_argument("--remote-done-recovery-max-count", type=int, default=4)
    parser.add_argument("--surface-landing-clearance", type=float, default=1.0)
    parser.add_argument("--surface-landing-probe-camera", default="front_0")
    parser.add_argument("--disable-surface-landing", action="store_true")
    parser.add_argument("--regression-recover-distance", type=float, default=45.0)
    parser.add_argument("--regression-stop-distance", type=float, default=80.0)
    parser.add_argument("--regression-worse-streak", type=int, default=6)
    parser.add_argument("--regression-recovery-steps", type=int, default=4)
    parser.add_argument("--grounding-boundary-window-steps", type=int, default=6)
    parser.add_argument("--grounding-recovery-window-steps", type=int, default=8)
    parser.add_argument("--segment-switch-confidence", type=float, default=0.55)
    parser.add_argument("--segment-min-steps", type=int, default=4)
    parser.add_argument("--segment-progress-distance", type=float, default=18.0)
    return parser.parse_args()


def has_cjk(text):
    return any("\u4e00" <= char <= "\u9fff" for char in str(text or ""))


def simple_route_instruction(text):
    text = str(text or "")
    lower = text.lower()
    if not has_cjk(text):
        return text

    landmarks = []
    if "桥" in text or "bridge" in lower:
        landmarks.append("bridge")
    if "城市" in text or "city" in lower:
        landmarks.append("city")
    if "建筑" in text or "楼" in text or "building" in lower:
        landmarks.append("buildings")
    if not landmarks:
        landmarks = ["road", "buildings"]

    return (
        "fly forward through the "
        + " toward the ".join(landmarks)
        + ". keep altitude, avoid obstacles, continue forward, and do not land until the route is complete."
    )


def is_forward_route_instruction(text):
    lower = str(text or "").lower()
    route_words = ("through", "cross", "pass", "forward", "ahead", "toward", "towards", "go to", "fly to")
    scene_words = ("bridge", "building", "buildings", "city", "road", "street", "tower")
    return any(word in lower for word in route_words) and any(word in lower for word in scene_words)


def has_descent_target_hint(text):
    lower = str(text or "").lower()
    return any(word in lower for word in ("road", "street", "ground", "land", "lower", "down", "descend"))


def stable_int(text):
    import hashlib

    return int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)


def tokenize_grounding_text(text):
    import re

    return re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower())


def ridge_expanded_tokens(
    sample,
    use_scene_token=False,
    use_scene_interactions=False,
    use_full_instruction=False,
):
    tokens = tokenize_grounding_text(
        ridge_sample_text(sample, use_scene_token=use_scene_token)
    )
    if use_full_instruction:
        instruction_tokens = tokenize_grounding_text(sample.get("instruction", ""))
        tokens.extend(f"instr__{token}" for token in instruction_tokens)
        tokens.extend(
            f"instr_bi__{first}__{second}"
            for first, second in zip(instruction_tokens, instruction_tokens[1:])
        )
    if use_scene_interactions:
        scene = f"scene_{sample.get('scene_id', 'unknown')}"
        base_tokens = list(tokens)
        tokens.extend(f"{scene}__{token}" for token in base_tokens)
        start = sample.get("start_position") or [0.0, 0.0, 0.0]
        x_bin = int(math.floor(float(start[0]) / 40.0))
        y_bin = int(math.floor(float(start[1]) / 40.0))
        z_bin = int(math.floor(float(start[2]) / 20.0))
        tokens.extend(
            [
                f"{scene}__xbin_{x_bin}",
                f"{scene}__ybin_{y_bin}",
                f"{scene}__grid_{x_bin}_{y_bin}",
                f"{scene}__zbin_{z_bin}",
            ]
        )
    return tokens


def ridge_sample_text(sample, use_scene_token=False):
    profile = (sample.get("instruction_profile") or {}).get("profile") or ""
    parts = [
            str(sample.get("segment_text") or ""),
            str(sample.get("grounded_segment_text") or ""),
            "actions",
            " ".join(sample.get("actions") or []),
            "landmarks",
            " ".join(sample.get("landmarks") or []),
            "target",
            str(sample.get("target_landmark") or ""),
            "zone",
            str(sample.get("target_zone") or ""),
            "kind",
            str(sample.get("segment_kind") or ""),
            "profile",
            profile,
        ]
    if use_scene_token:
        parts.extend(["scene", f"scene_{sample.get('scene_id', 'unknown')}"])
    return " ".join(parts)


def ridge_numeric_features(sample):
    segment_count = max(1, int(sample.get("segment_count") or 1))
    segment_index = int(sample.get("segment_index") or 0)
    denominator = max(1, segment_count - 1)
    actions = sample.get("actions") or []
    landmarks = sample.get("landmarks") or []
    relations = sample.get("relations") or []
    start_position = sample.get("start_position") or [0.0, 0.0, 0.0]
    start_x = float(start_position[0]) if len(start_position) > 0 else 0.0
    start_y = float(start_position[1]) if len(start_position) > 1 else 0.0
    start_z = float(start_position[2]) if len(start_position) > 2 else 0.0
    return [
        segment_index / denominator,
        segment_count / 12.0,
        len(actions) / 5.0,
        len(landmarks) / 5.0,
        len(relations) / 4.0,
        1.0 if "descend" in actions else 0.0,
        1.0 if "take_off" in actions else 0.0,
        1.0 if "turn_left" in actions else 0.0,
        1.0 if "turn_right" in actions else 0.0,
        1.0 if sample.get("target_zone") == "corridor" else 0.0,
        1.0 if sample.get("target_zone") == "structure" else 0.0,
        start_x,
        start_y,
        start_z,
        math.hypot(start_x, start_y),
    ]


def main():
    cli = parse_args()
    airvln_root = Path(__file__).resolve().parents[1]
    os.chdir(str(airvln_root))
    sys.path.insert(0, str(airvln_root))
    from live_server.grounding import build_grounding, compact_grounding_view, contains_cjk
    from utils.segment_grounding_model import SegmentGroundingRidge

    # src.common.param parses sys.argv at import time.
    sys.argv = [
        "model_live_once.py",
        "--run_type",
        "eval",
        "--policy_type",
        "cma",
        "--collect_type",
        "TF",
        "--name",
        "AirVLN-cma",
        "--batchSize",
        "1",
        "--maxAction",
        str(cli.max_steps),
        "--EVAL_DATASET",
        cli.split,
        "--EVAL_CKPT_PATH_DIR",
        cli.ckpt,
        "--simulator_tool_port",
        str(cli.simulator_tool_port),
        "--TF_mode_load_scene",
        str(cli.scene_id),
        "--trainer_gpu_device",
        str(cli.trainer_gpu_device),
    ]

    import numpy as np
    import torch
    from PIL import Image
    import airsim

    from Model.il_trainer import VLNCETrainer
    from airsim_plugin.airsim_settings import AirsimActions
    from src.common.param import args
    from src.vlnce_src.env import AirVLNENV
    from src.vlnce_src.train import batch_obs, initialize_tokenizer
    from utils.depth_safety import (
        DepthEscapePlanner,
        DepthRouteAroundGuard,
        apply_depth_escape_override,
        depth_valid_ratio,
        enrich_depth_escape_override,
    )

    args.DistributedDataParallel = False

    action_names = {getattr(AirsimActions, name): name for name in AirsimActions}

    out_dir = create_output_dir(cli.out_root, cli.model_run_id, cli.pre_reserved_model_run)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = out_dir / "flight_target_telemetry.jsonl"
    flight_target_telemetry_event_count = 0

    tok = initialize_tokenizer()
    env = AirVLNENV(batch_size=1, split=cli.split, tokenizer=tok, dataset_group_by_scene=False)
    target = None
    for item in env.data:
        if item.get("episode_id") == cli.episode_id:
            target = item
            break
    if target is None:
        raise RuntimeError(f"episode not found after scene filter: {cli.episode_id}")
    episode = target
    goals = target.get("goals") or []
    goal_position = None
    if goals and isinstance(goals[0], dict):
        raw_goal_position = goals[0].get("position")
        if isinstance(raw_goal_position, (list, tuple)) and len(raw_goal_position) >= 3:
            goal_position = [float(raw_goal_position[0]), float(raw_goal_position[1]), float(raw_goal_position[2])]
    reference_path = []
    for point in target.get("reference_path") or []:
        if isinstance(point, (list, tuple)) and len(point) >= 3:
            reference_path.append([float(point[0]), float(point[1]), float(point[2])])
    raw_episode_start = target.get("start_position") or (reference_path[0] if reference_path else None)
    if isinstance(raw_episode_start, (list, tuple)) and len(raw_episode_start) >= 3:
        episode_start = [
            float(raw_episode_start[0]),
            float(raw_episode_start[1]),
            float(raw_episode_start[2]),
        ]
    else:
        episode_start = [0.0, 0.0, 0.0]
    original_instruction = cli.instruction
    grounding = build_grounding(
        original_instruction,
        split=cli.split,
        scene_id=cli.scene_id,
        dataset="aerialvln",
        enable_scene16_support_ranker=cli.enable_scene16_support_ranker,
        scene16_support_ranker=cli.scene16_support_ranker,
        enable_scene16_route_prior=cli.enable_scene16_route_prior,
        scene16_route_prior=cli.scene16_route_prior,
    )
    instruction_profile = grounding.get("instruction_profile") or {}
    instruction_profile_name = instruction_profile.get("profile", "balanced")
    instruction_alignment_strength = instruction_profile.get("alignment_strength", "balanced")
    original_instruction_word_count = len(str(original_instruction or "").split())
    original_instruction_segment_count = int(instruction_profile.get("segment_count", 0) or 0)
    keep_original_motion_chain_instruction = (
        instruction_profile_name == "motion_chain"
        and 32 <= original_instruction_word_count <= 84
        and original_instruction_segment_count >= 5
    )
    reference_dense_intervention_enabled = False
    use_grounded_instruction = (
        bool(original_instruction)
        and not cli.disable_grounding_guidance
        and not instruction_profile.get("prefer_original_instruction")
        and not keep_original_motion_chain_instruction
        and (
            contains_cjk(original_instruction)
            or len(str(original_instruction).split()) <= 24
            or bool(grounding.get("landmarks"))
        )
    )
    grounding_guidance_enabled = not cli.disable_grounding_guidance
    model_instruction = grounding.get("grounded_instruction") if use_grounded_instruction else original_instruction
    model_instruction_source = "grounded" if use_grounded_instruction else "original"
    simple_route_guidance = bool(original_instruction) and not cli.disable_simple_route_guidance
    forward_route_guidance = bool(simple_route_guidance and grounding.get("forward_route"))
    descent_target_hint = bool(grounding.get("descent_target"))
    target_landmark = grounding.get("target_landmark")
    target_zone = grounding.get("target_zone")
    grounding_actions = set(grounding.get("actions") or [])
    raw_segment_plan = grounding.get("segment_plan") or []
    segment_plan = [
        item
        for item in raw_segment_plan
        if item.get("actions") or item.get("landmarks") or item.get("relations")
    ]
    segmented_grounding = (
        bool(segment_plan)
        and len(segment_plan) > 1
        and simple_route_guidance
        and not cli.disable_segmented_grounding
    )
    corridor_target = target_zone == "corridor"
    structure_target = target_zone == "structure"
    learned_segment_model = None
    learned_segment_predictor = None
    learned_segment_model_type = ""
    learned_segment_final_model = None
    learned_segment_final_predictor = None
    learned_segment_final_model_type = ""
    learned_segment_homing_enabled = False
    def load_learned_segment_model(model_path_text, label):
        model_path = Path(model_path_text)
        if model_path.exists():
            model = dict(np.load(str(model_path), allow_pickle=True))
            model_type = str(model.get("model_type", np.array(["ridge"]))[0])
            predictor = SegmentGroundingRidge(model_path) if model_type != "ridge" else None
            print(
                "[INFO] learned segment grounding model loaded: "
                f"{model_path} type={model_type or 'ridge'} role={label}",
                flush=True,
            )
            return model, predictor, model_type
        print(f"[WARN] learned segment grounding model not found: {model_path}", flush=True)
        return None, None, ""

    if cli.enable_learned_segment_homing and cli.segment_grounding_ridge:
        (
            learned_segment_model,
            learned_segment_predictor,
            learned_segment_model_type,
        ) = load_learned_segment_model(cli.segment_grounding_ridge, "primary")
    if cli.enable_learned_segment_homing and cli.segment_grounding_final_ridge:
        (
            learned_segment_final_model,
            learned_segment_final_predictor,
            learned_segment_final_model_type,
        ) = load_learned_segment_model(cli.segment_grounding_final_ridge, "final")
    learned_segment_homing_enabled = bool(
        learned_segment_model
        or learned_segment_predictor is not None
        or learned_segment_final_model
        or learned_segment_final_predictor is not None
    )
    route_prior_candidate = None
    route_prior_candidates = []
    if cli.enable_scene16_route_prior:
        route_prior_decision = grounding.get("scene16_support_ranker_decision")
        if isinstance(route_prior_decision, dict):
            route_prior_candidate = route_prior_decision.get("route_prior_top_candidate")
            route_prior_candidates = route_prior_decision.get("route_prior_candidates") or []

    def predict_learned_segment_delta(segment, segment_index, current_position, current_yaw):
        if (
            not learned_segment_model
            and learned_segment_predictor is None
            and not learned_segment_final_model
            and learned_segment_final_predictor is None
        ) or segment is None:
            return None
        segment_count = max(1, len(segment_plan))
        segment_actions = set(segment.get("actions") or [])
        landing_like_segment = bool(
            segment_actions.intersection({"land", "landed", "stay", "descend"})
            or "land" in str(segment.get("text") or "").lower()
        )
        final_like_segment = bool(
            int(segment_index) >= max(0, segment_count - 2) or landing_like_segment
        )
        active_model = learned_segment_model
        active_predictor = learned_segment_predictor
        active_model_type = learned_segment_model_type
        if final_like_segment and (
            learned_segment_final_model or learned_segment_final_predictor is not None
        ):
            active_model = learned_segment_final_model
            active_predictor = learned_segment_final_predictor
            active_model_type = learned_segment_final_model_type
        if active_predictor is not None:
            active_route_prior_candidate = (
                select_route_prior_candidate_for_goal_bearing(
                    route_prior_candidates or [route_prior_candidate],
                    current_position,
                    goal_position,
                )
                if route_prior_candidate is not None or route_prior_candidates else None
            )
            predictor_is_final_goal = bool(getattr(active_predictor, "final_target_is_goal", False))
            if predictor_is_final_goal and not final_like_segment:
                return None
            try:
                delta = active_predictor.predict_global_delta(
                    segment=segment,
                    segment_index=int(segment_index),
                    segment_count=segment_count,
                    instruction_profile=instruction_profile,
                    current_position=current_position,
                    current_yaw=current_yaw,
                    scene_id=cli.scene_id,
                    instruction=original_instruction,
                    retrieval_position=episode_start,
                )
                if (
                    predictor_is_final_goal
                    and len(delta) >= 3
                    and math.hypot(float(delta[0]), float(delta[1])) > 24.0
                ):
                    # Keep final approach horizontal until XY is close; otherwise
                    # vertical corrections can fight the depth escape behavior.
                    delta[2] = 0.0
                return route_prior_adjusted_learned_delta(
                    delta,
                    active_route_prior_candidate,
                    segment_index,
                    segment_count,
                    segment=segment,
                )
            except Exception as exc:
                print(
                    "[WARN] learned segment predictor failed; falling back if possible: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if active_model_type != "ridge":
                    return None
        if not active_model:
            return None
        active_route_prior_candidate = (
            select_route_prior_candidate_for_goal_bearing(
                route_prior_candidates or [route_prior_candidate],
                current_position,
                goal_position,
            )
            if route_prior_candidate is not None or route_prior_candidates else None
        )
        hash_dim = int(active_model["hash_dim"][0])
        feature_mean = active_model["feature_mean"]
        feature_std = active_model["feature_std"]
        y_mean = active_model["y_mean"]
        y_std = active_model["y_std"]
        weights = active_model["weights"]
        sample = {
            "segment_text": segment.get("text", ""),
            "grounded_segment_text": segment.get("grounded_instruction", ""),
            "actions": segment.get("actions") or [],
            "landmarks": segment.get("landmarks") or [],
            "relations": segment.get("relations") or [],
            "target_landmark": segment.get("target_landmark"),
            "target_zone": segment.get("target_zone"),
            "segment_kind": segment.get("segment_kind"),
            "instruction_profile": instruction_profile,
            "instruction": original_instruction,
            "segment_index": int(segment_index),
            "segment_count": max(1, len(segment_plan)),
            "start_position": current_position,
            "scene_id": cli.scene_id,
        }
        x = np.zeros((1, hash_dim + feature_mean.shape[1] + 1), dtype=np.float32)
        x[:, 0] = 1.0
        use_scene_token = bool(
            int(active_model.get("use_scene_token", np.array([0]))[0])
        )
        use_scene_interactions = bool(
            int(active_model.get("use_scene_interactions", np.array([0]))[0])
        )
        use_full_instruction = bool(
            int(active_model.get("use_full_instruction", np.array([0]))[0])
        )
        tokens = ridge_expanded_tokens(
            sample,
            use_scene_token=use_scene_token,
            use_scene_interactions=use_scene_interactions,
            use_full_instruction=use_full_instruction,
        ) or ["<empty>"]
        scale = 1.0 / math.sqrt(len(tokens))
        for token in tokens:
            idx = 1 + (stable_int(token) % hash_dim)
            sign = 1.0 if stable_int("sign:" + token) % 2 == 0 else -1.0
            x[0, idx] += sign * scale
        numeric = np.asarray([ridge_numeric_features(sample)], dtype=np.float32)
        numeric = (numeric - feature_mean) / feature_std
        x[:, 1 + hash_dim:] = numeric
        delta = [float(value) for value in ((x @ weights) * y_std + y_mean)[0].tolist()]
        target_field = str(
            active_model.get("target_field", np.array(["delta_xyz"]))[0]
        )
        if target_field == "delta_local_xyz":
            local_forward, local_left, local_z = delta
            cos_yaw = math.cos(current_yaw)
            sin_yaw = math.sin(current_yaw)
            delta = [
                cos_yaw * local_forward - sin_yaw * local_left,
                sin_yaw * local_forward + cos_yaw * local_left,
                local_z,
            ]
        return route_prior_adjusted_learned_delta(
            delta,
            active_route_prior_candidate,
            segment_index,
            segment_count,
            segment=segment,
        )

    if model_instruction:
        target = dict(target).copy()
        target["instruction"] = dict(target.get("instruction", {}))
        target["instruction"]["instruction_text"] = model_instruction
        if hasattr(tok, "encode_sentence"):
            target["instruction"]["instruction_tokens"] = tok.encode_sentence(model_instruction)

    # next_minibatch returns None when len(data) == 1, so keep a harmless duplicate.
    env.data = [target, dict(target).copy()]
    env.index_data = 0
    env.scenes = {int(target["scene_id"])}
    env.next_minibatch()

    trainer = VLNCETrainer(
        load_from_ckpt=True,
        observation_space=env.observation_space,
        action_space=env.action_space,
        ckpt_path=cli.ckpt,
    )
    trainer.policy.eval()

    hidden_size = trainer.policy.net.state_encoder.hidden_size
    num_layers = trainer.policy.net.num_recurrent_layers
    rnn_states = torch.zeros(1, num_layers, hidden_size, device=trainer.device)
    prev_actions = torch.zeros(1, 1, dtype=torch.long, device=trainer.device)
    not_done_masks = torch.zeros(1, 1, dtype=torch.uint8, device=trainer.device)

    trace = []
    trace_path = out_dir / "trace.jsonl"
    with torch.no_grad():
        outputs = env.reset()
        observations, _, dones, infos = [list(x) for x in zip(*outputs)]
        batch = batch_obs(observations, trainer.device)

        guidance_state = {
            "turn_streak": 0,
            "avoidance_streak": 0,
            "best_distance": None,
            "best_distance_step": None,
            "initial_distance": None,
            "last_distance": None,
            "distance_worse_streak": 0,
            "distance_regression_start_step": None,
            "recovery_turn_steps": 0,
            "recovery_attempts": 0,
            "regression_retry_attempts": 0,
            "regression_retry_exploration_steps": 0,
            "regression_retry_phase": None,
            "regression_retry_target": None,
            "entered_arrival_radius": False,
            "arrival_fine_turn_streak": 0,
            "segment_index": 0,
            "segment_start_step": 0,
            "segment_entry_distance": None,
            "segment_best_distance": None,
            "segment_turn_matches": 0,
            "segment_forward_matches": 0,
            "segment_descent_matches": 0,
            "segment_visual_matches": 0,
            "segment_boundary_window_steps": (
                min(2, int(cli.grounding_boundary_window_steps))
                if segmented_grounding and instruction_profile_name == "visual_boundary"
                else min(3, int(cli.grounding_boundary_window_steps))
                if segmented_grounding and instruction_profile_name == "reference_dense"
                else int(cli.grounding_boundary_window_steps)
                if segmented_grounding
                else 0
            ),
            "segment_recovery_window_steps": 0,
            "segment_confidence": 0.0,
            "segment_switches": [],
            "segment_observer_index": 0,
            "segment_observer_start_step": 0,
            "segment_observer_entry_distance": None,
            "segment_observer_best_distance": None,
            "segment_observer_turn_matches": 0,
            "segment_observer_forward_matches": 0,
            "segment_observer_descent_matches": 0,
            "segment_observer_visual_matches": 0,
            "segment_observer_confidence": 0.0,
            "segment_observer_switches": [],
            "segment_observer_diagnostics": [],
            "segment_observer_missed_turn_streak": 0,
            "segment_observer_descend_stall_streak": 0,
            "segment_observer_recovery_keys": [],
            "segment_observer_last_recovery_step": -999,
            "instruction_profile": instruction_profile_name,
            "instruction_alignment_strength": instruction_alignment_strength,
            "surface_clearance": None,
            "surface_clearance_step": None,
            "surface_probe_ok": False,
            "surface_probe_error": None,
            "learned_segment_homing_target": None,
            "learned_segment_homing_delta": None,
            "learned_segment_homing_index": None,
            "learned_segment_homing_created_step": None,
            "learned_segment_homing_distance": None,
            "learned_segment_homing_entry_distance": None,
            "learned_segment_homing_best_distance": None,
            "learned_segment_homing_resamples": 0,
            "learned_segment_homing_safety_turn_count": 0,
            "learned_segment_homing_segment_safety_turn_count": 0,
            "learned_segment_homing_last_safety_stall_resample_step": None,
            "learned_segment_homing_goal_regression_resamples": 0,
            "learned_segment_homing_last_goal_regression_resample_step": None,
            "learned_segment_final_goal_progress_retries": 0,
            "learned_segment_homing_base_xy_step": None,
            "learned_segment_homing_xy_step": None,
            "learned_segment_homing_risk_tier": None,
            "learned_segment_homing_xy_step_reason": None,
            "oracle_path_target_index": None,
            "oracle_path_target_created_step": None,
            "oracle_path_target_position": None,
            "oracle_turn_action": None,
            "oracle_turn_streak": 0,
            "oracle_turn_breaks": 0,
            "remote_done_recovery_count": 0,
            "remote_done_recovery_last_step": None,
            "depth_block_recovery_last_step": -999,
            "last_depth_turn_action": None,
            "collision_rollback_count": 0,
            "collision_rollbacks": [],
            "black_frame_drop_count": 0,
            "stream_sample_over_0_5s": 0,
            "visual_occupancy_rollback_count": 0,
            "visual_occupancy_rollbacks": [],
            "last_vertical_action": None,
            "last_vertical_action_step": -999,
            "vertical_reversal_suppressed": 0,
            "descent_surface_blocks": 0,
            "surface_avoid_turn_action": None,
            "surface_avoid_turn_steps": 0,
            "surface_avoid_last_step": -999,
        }
        depth_escape_planner = DepthEscapePlanner(turn_steps=4, cooldown_steps=40)
        depth_route_around_guard = DepthRouteAroundGuard(trigger_turns=6, turn_steps=6)
        depth_route_around_min_step = 60
        stop_reason = None

        def first_airsim_client():
            try:
                return env.simulator_tool.airsim_clients[0][0]
            except Exception:
                return None

        def apply_recovery_pose_safely(recovery_pose, step, reason):
            pose_grid = [[recovery_pose]]
            if not env.simulator_tool.setPoses(poses=pose_grid):
                raise RuntimeError("safe recovery pose was rejected")
            actual_pose = pose_grid[0][0]
            consume_collision = getattr(env.simulator_tool, "consumeCollisionEvent", None)
            collision_event = consume_collision() if callable(consume_collision) else None
            if collision_event:
                guidance_state["collision_rollback_count"] = int(
                    guidance_state.get("collision_rollback_count", 0)
                ) + 1
                collision_record = {
                    **collision_event,
                    "step": int(step),
                    "action_id": None,
                    "action": reason,
                }
                guidance_state.setdefault("collision_rollbacks", []).append(collision_record)
                return actual_pose, collision_record
            return actual_pose, None

        def apply_surface_finish_pose(step, clearance, reason, max_descent=8.0):
            if clearance is None or not env.sim_states:
                return None
            landing_clearance = float(cli.surface_landing_clearance)
            descent = min(
                float(max_descent),
                max(0.0, float(clearance) - landing_clearance + 0.25),
            )
            if descent <= 0.05:
                return None
            state = env.sim_states[0]
            current_pose = state.pose
            target_pose = airsim.Pose(
                airsim.Vector3r(
                    float(current_pose.position.x_val),
                    float(current_pose.position.y_val),
                    float(current_pose.position.z_val) + descent,
                ),
                current_pose.orientation,
            )
            try:
                target_pose, collision = apply_recovery_pose_safely(target_pose, step, reason)
                state.pose = target_pose
                state.is_end = False
                state.is_collisioned = False
                state.trajectory.append(
                    [
                        float(target_pose.position.x_val),
                        float(target_pose.position.y_val),
                        float(target_pose.position.z_val),
                        float(target_pose.orientation.x_val),
                        float(target_pose.orientation.y_val),
                        float(target_pose.orientation.z_val),
                        float(target_pose.orientation.w_val),
                    ]
                )
                env.update_measurements()
                estimated_clearance = max(landing_clearance, float(clearance) - descent)
                guidance_state["surface_clearance"] = estimated_clearance
                guidance_state["surface_clearance_step"] = int(step)
                guidance_state["last_valid_surface_clearance"] = estimated_clearance
                guidance_state["last_valid_surface_clearance_step"] = int(step)
                return {
                    "surface_clearance_before": float(clearance),
                    "surface_clearance_after": estimated_clearance,
                    "surface_finish_descent": descent,
                    "collision": collision,
                    "position": [
                        float(target_pose.position.x_val),
                        float(target_pose.position.y_val),
                        float(target_pose.position.z_val),
                    ],
                }
            except Exception as exc:
                guidance_state["surface_probe_error"] = f"{reason}_failed: {exc}"
                return {
                    "surface_clearance_before": float(clearance),
                    "surface_finish_descent": descent,
                    "error": str(exc),
                }

        def recover_from_remote_done(step, distance_to_goal, reason="remote_env_done_recovery_pose"):
            if not (cli.enable_oracle_path_homing and reference_path):
                return None
            recovery_count = int(guidance_state.get("remote_done_recovery_count", 0))
            client = first_airsim_client()
            if client is None or not env.sim_states:
                return None
            state = env.sim_states[0]
            current_pose = state.pose
            surface_clearance = guidance_state.get("surface_clearance")
            needs_surface_finish = bool(
                distance_to_goal is not None
                and 0 <= float(distance_to_goal) <= float(cli.success_distance)
                and (
                    surface_clearance is None
                    or float(surface_clearance) > float(cli.surface_landing_clearance)
                )
            )
            if needs_surface_finish:
                recovery_pose = current_pose
                recovery_collision = None
                if surface_clearance is not None:
                    surface_finish = apply_surface_finish_pose(step, surface_clearance, "surface_recovery")
                    if surface_finish and not surface_finish.get("error"):
                        recovery_pose = state.pose
                        recovery_collision = surface_finish.get("collision")
                state.is_end = False
                state.is_collisioned = False
                guidance_state["remote_done_recovery_last_step"] = int(step)
                return {
                    "raw_action_id": None,
                    "raw_action": None,
                    "override_action_id": None,
                    "override_action": None,
                    "reason": "remote_env_done_surface_pose",
                    "distance_to_goal": distance_to_goal,
                    "surface_clearance": surface_clearance,
                    "surface_landing_clearance": float(cli.surface_landing_clearance),
                    "collision": recovery_collision,
                    "current_position": [
                        float(recovery_pose.position.x_val),
                        float(recovery_pose.position.y_val),
                        float(recovery_pose.position.z_val),
                    ],
                }
            if recovery_count >= int(cli.remote_done_recovery_max_count):
                return None
            current_index = guidance_state.get("oracle_path_target_index")
            if current_index is None:
                current_index = 0
            lookahead = max(1, int(cli.remote_done_recovery_lookahead))
            recovery_index = min(len(reference_path) - 1, int(current_index) + lookahead * (recovery_count + 1))
            target = reference_path[recovery_index]
            next_index = min(len(reference_path) - 1, recovery_index + max(1, int(cli.oracle_path_lookahead)))
            next_target = reference_path[next_index]
            current_z = float(current_pose.position.z_val)
            target_z = float(target[2])
            recovery_z = min(
                target_z - float(cli.remote_done_recovery_climb),
                current_z - float(cli.remote_done_recovery_climb),
                -float(cli.oracle_goal_homing_cruise_altitude),
            )
            yaw = math.atan2(float(next_target[1]) - float(target[1]), float(next_target[0]) - float(target[0]))
            recovery_pose = airsim.Pose(
                airsim.Vector3r(float(target[0]), float(target[1]), recovery_z),
                airsim.to_quaternion(0, 0, yaw),
            )
            try:
                recovery_pose, recovery_collision = apply_recovery_pose_safely(
                    recovery_pose,
                    step,
                    reason,
                )
                time.sleep(0.05)
                state.pose = recovery_pose
                state.is_end = False
                state.is_collisioned = False
                state.pre_action = AirsimActions.GO_UP
                state.trajectory.append(
                    [
                        float(recovery_pose.position.x_val),
                        float(recovery_pose.position.y_val),
                        float(recovery_pose.position.z_val),
                        float(recovery_pose.orientation.x_val),
                        float(recovery_pose.orientation.y_val),
                        float(recovery_pose.orientation.z_val),
                        float(recovery_pose.orientation.w_val),
                    ]
                )
                guidance_state["oracle_path_target_index"] = int(recovery_index)
                guidance_state["oracle_path_target_position"] = list(target)
                guidance_state["oracle_path_target_created_step"] = int(step)
                guidance_state["remote_done_recovery_count"] = recovery_count + 1
                guidance_state["remote_done_recovery_last_step"] = int(step)
                return {
                    "raw_action_id": None,
                    "raw_action": None,
                    "override_action_id": None,
                    "override_action": None,
                    "reason": reason,
                    "distance_to_goal": distance_to_goal,
                    "recovery_count": recovery_count + 1,
                    "recovery_index": int(recovery_index),
                    "collision": recovery_collision,
                    "recovery_position": [
                        float(recovery_pose.position.x_val),
                        float(recovery_pose.position.y_val),
                        float(recovery_pose.position.z_val),
                    ],
                    "previous_position": [
                        float(current_pose.position.x_val),
                        float(current_pose.position.y_val),
                        float(current_pose.position.z_val),
                    ],
                }
            except Exception as exc:
                guidance_state["remote_done_recovery_count"] = recovery_count + 1
                guidance_state["remote_done_recovery_last_step"] = int(step)
                return {
                    "raw_action_id": None,
                    "raw_action": None,
                    "override_action_id": None,
                    "override_action": None,
                    "reason": "remote_env_done_recovery_failed",
                    "distance_to_goal": distance_to_goal,
                    "error": str(exc),
                    "recovery_count": recovery_count + 1,
                }

        def probe_surface_clearance(step=None):
            if cli.disable_surface_landing:
                guidance_state["surface_clearance"] = None
                guidance_state["surface_probe_ok"] = False
                guidance_state["surface_probe_error"] = "disabled"
                return None
            if (
                step is not None
                and guidance_state.get("surface_clearance_step") == int(step)
                and guidance_state.get("surface_clearance") is not None
            ):
                return float(guidance_state["surface_clearance"])
            client = first_airsim_client()
            if client is None:
                guidance_state["surface_clearance"] = None
                guidance_state["surface_probe_ok"] = False
                guidance_state["surface_probe_error"] = "missing_airsim_client"
                return None
            camera_name = str(cli.surface_landing_probe_camera)
            down_pose = airsim.Pose(
                airsim.Vector3r(0, 0, 0),
                airsim.to_quaternion(math.radians(-90.0), 0, 0),
            )
            front_pose = airsim.Pose(
                airsim.Vector3r(0.5, 0, 0),
                airsim.to_quaternion(0, 0, 0),
            )
            try:
                client.simSetCameraPose(camera_name, down_pose, vehicle_name="Drone_1")
                responses = client.simGetImages(
                    [
                        airsim.ImageRequest(
                            camera_name,
                            airsim.ImageType.DepthPerspective,
                            pixels_as_float=True,
                            compress=False,
                        )
                    ],
                    vehicle_name="Drone_1",
                )
                response = responses[0] if responses else None
                if response is None or not getattr(response, "image_data_float", None):
                    raise RuntimeError("empty_depth_response")
                arr = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
                arr = np.asarray(arr, dtype=np.float32)
                h, w = arr.shape[:2]
                crop = arr[int(h * 0.42): int(h * 0.58), int(w * 0.42): int(w * 0.58)]
                valid = crop[np.isfinite(crop) & (crop > 0.05) & (crop < 500.0)]
                if valid.size == 0:
                    raise RuntimeError("no_valid_depth")
                clearance = float(np.percentile(valid, 20))
                guidance_state["surface_clearance"] = clearance
                guidance_state["surface_clearance_step"] = int(step) if step is not None else None
                guidance_state["last_valid_surface_clearance"] = clearance
                guidance_state["last_valid_surface_clearance_step"] = int(step) if step is not None else None
                if env.sim_states:
                    guidance_state["last_valid_surface_probe_z"] = float(
                        env.sim_states[0].pose.position.z_val
                    )
                guidance_state["surface_probe_ok"] = True
                guidance_state["surface_probe_error"] = None
                return clearance
            except Exception as exc:
                previous_clearance = guidance_state.get("last_valid_surface_clearance")
                previous_step = guidance_state.get("last_valid_surface_clearance_step")
                previous_z = guidance_state.get("last_valid_surface_probe_z")
                current_z = None
                if env.sim_states:
                    current_z = float(env.sim_states[0].pose.position.z_val)
                close_contact_threshold = max(float(cli.surface_landing_clearance) + 1.25, 2.5)
                recent_probe = (
                    step is not None
                    and previous_step is not None
                    and int(step) - int(previous_step) <= 2
                )
                estimated_clearance = estimate_surface_clearance_from_recent_probe(
                    current_z,
                    previous_z,
                    previous_clearance,
                    step,
                    previous_step,
                )
                if str(exc) == "no_valid_depth" and estimated_clearance is not None:
                    guidance_state["surface_clearance"] = estimated_clearance
                    guidance_state["surface_clearance_step"] = int(step) if step is not None else None
                    guidance_state["last_valid_surface_clearance"] = estimated_clearance
                    guidance_state["last_valid_surface_clearance_step"] = int(step) if step is not None else None
                    if current_z is not None:
                        guidance_state["last_valid_surface_probe_z"] = current_z
                    guidance_state["surface_probe_ok"] = True
                    guidance_state["surface_probe_error"] = "estimated_after_no_valid_depth"
                    guidance_state["surface_contact_fallback"] = (
                        estimated_clearance <= float(cli.surface_landing_clearance)
                    )
                    return estimated_clearance
                if (
                    str(exc) == "no_valid_depth"
                    and previous_clearance is not None
                    and float(previous_clearance) <= close_contact_threshold
                    and recent_probe
                ):
                    guidance_state["surface_clearance"] = 0.0
                    guidance_state["surface_clearance_step"] = int(step)
                    guidance_state["surface_probe_ok"] = True
                    guidance_state["surface_probe_error"] = "contact_assumed_after_no_valid_depth"
                    guidance_state["surface_contact_fallback"] = True
                    return 0.0
                guidance_state["surface_clearance"] = None
                guidance_state["surface_probe_ok"] = False
                guidance_state["surface_probe_error"] = str(exc)
                return None
            finally:
                try:
                    client.simSetCameraPose(camera_name, front_pose, vehicle_name="Drone_1")
                except Exception:
                    pass

        def current_segment():
            if not segmented_grounding:
                return None
            idx = int(guidance_state.get("segment_index", 0))
            if idx < 0 or idx >= len(segment_plan):
                return None
            return segment_plan[idx]

        def emit_flight_target_telemetry(
            event, step, target_before=None, target_after=None, target_changed_reason=None,
            distance_to_goal=None, regression_stop_step=None, recovery_reason=None,
            raw_action=None, applied_action=None, segment=None,
        ):
            nonlocal flight_target_telemetry_event_count
            pose_position = None
            try:
                pose = env.sim_states[0].pose
                pose_position = [float(pose.position.x_val), float(pose.position.y_val), float(pose.position.z_val)]
            except Exception:
                pass
            provenance = build_flight_target_ranker_context(segment or current_segment(), grounding)
            append_flight_target_telemetry(
                telemetry_path,
                build_flight_target_telemetry_event(
                    event=event,
                    step=step,
                    target_before=target_before,
                    target_after=target_after,
                    target_changed_reason=target_changed_reason,
                    selected_landmark=provenance["selected_landmark"],
                    target_source=provenance["target_source"],
                    ranker_top_support=provenance["ranker_top_support"],
                    ranker_influenced_target=provenance["ranker_influenced_target"],
                    current_position=pose_position,
                    distance_to_dataset_goal=distance_to_goal,
                    regression_start_step=guidance_state.get("distance_regression_start_step"),
                    regression_stop_step=regression_stop_step,
                    recovery_reason=recovery_reason,
                    raw_action=raw_action,
                    applied_action=applied_action,
                ),
            )
            flight_target_telemetry_event_count += 1

        def segment_observer_enabled():
            return bool(
                segmented_grounding
                and grounding_guidance_enabled
                and instruction_profile_name == "reference_dense"
                and reference_dense_intervention_enabled
                and len(segment_plan) > 1
            )

        def current_observer_segment():
            if not segment_observer_enabled():
                return None
            idx = int(guidance_state.get("segment_observer_index", 0))
            if idx < 0 or idx >= len(segment_plan):
                return None
            return segment_plan[idx]

        def current_observer_snapshot():
            segment = current_observer_segment()
            idx = int(guidance_state.get("segment_observer_index", 0))
            return {
                "enabled": bool(segment_observer_enabled()),
                "segment_index": idx,
                "segment_confidence": float(guidance_state.get("segment_observer_confidence", 0.0)),
                "segment_text": segment.get("text") if segment else None,
                "segment_target_landmark": segment.get("target_landmark") if segment else None,
                "segment_target_zone": segment.get("target_zone") if segment else None,
                "segment_kind": segment.get("segment_kind") if segment else None,
                "segment_actions": list(segment.get("actions", [])) if segment else [],
                "segment_relations": list(segment.get("relations", [])) if segment else [],
                "turn_matches": int(guidance_state.get("segment_observer_turn_matches", 0)),
                "forward_matches": int(guidance_state.get("segment_observer_forward_matches", 0)),
                "descent_matches": int(guidance_state.get("segment_observer_descent_matches", 0)),
                "visual_matches": int(guidance_state.get("segment_observer_visual_matches", 0)),
                "switch_count": len(guidance_state.get("segment_observer_switches", [])),
            }

        def current_segment_snapshot(step=None):
            segment = current_segment()
            index = int(guidance_state.get("segment_index", 0))

            def compact_segment_snapshot(item):
                if not item:
                    return None
                return {
                    "text": item.get("text"),
                    "actions": list(item.get("actions", [])),
                    "target_landmark": item.get("target_landmark"),
                    "target_zone": item.get("target_zone"),
                    "segment_kind": item.get("segment_kind"),
                }

            return {
                "enabled": bool(segmented_grounding),
                "grounding_guidance_enabled": bool(grounding_guidance_enabled),
                "instruction_profile": instruction_profile_name,
                "alignment_strength": instruction_alignment_strength,
                "semantic_quality": instruction_profile.get("semantic_quality", "normal"),
                "segment_count": len(segment_plan),
                "segment_index": index,
                "boundary_window_steps": int(guidance_state.get("segment_boundary_window_steps", 0)),
                "recovery_window_steps": int(guidance_state.get("segment_recovery_window_steps", 0)),
                "segment_confidence": float(guidance_state.get("segment_confidence", 0.0)),
                "segment_text": segment.get("text") if segment else None,
                "segment_target_landmark": segment.get("target_landmark") if segment else None,
                "segment_target_zone": segment.get("target_zone") if segment else None,
                "segment_kind": segment.get("segment_kind") if segment else None,
                "segment_actions": list(segment.get("actions", [])) if segment else [],
                "segment_relations": list(segment.get("relations", [])) if segment else [],
                "previous_segment": compact_segment_snapshot(segment_plan[index - 1] if 0 < index < len(segment_plan) else None),
                "next_segment": compact_segment_snapshot(segment_plan[index + 1] if 0 <= index + 1 < len(segment_plan) else None),
                "observer": current_observer_snapshot(),
                "learned_homing": learned_segment_homing_diagnostics(guidance_state, step),
            }

        def segment_alignment_mode(segment):
            if segment is None:
                return instruction_alignment_strength
            segment_kind = segment.get("segment_kind")
            relation_count = len(segment.get("relations") or [])
            landmark_count = len(segment.get("landmarks") or [])
            if instruction_profile_name == "motion_chain":
                if segment_kind == "motion" and relation_count <= 1:
                    return "strong"
                return "balanced"
            if instruction_profile_name == "visual_mixed":
                if segment_kind == "motion" and relation_count == 0 and landmark_count <= 1:
                    return "strong"
                if segment_kind in {"mixed", "visual"}:
                    return "balanced"
                if relation_count >= 2:
                    return "light"
                return "balanced"
            if instruction_profile_name == "reference_dense":
                if segment_kind == "motion" and relation_count == 0 and landmark_count <= 1:
                    return "balanced"
                return "light"
            if instruction_profile_name == "visual_boundary":
                if segment_kind == "motion" and relation_count == 0:
                    return "balanced"
                return "light"
            return instruction_alignment_strength

        def profile_boundary_window_steps():
            base = int(cli.grounding_boundary_window_steps)
            if instruction_profile_name == "visual_boundary":
                return max(1, min(2, base))
            if instruction_profile_name == "reference_dense":
                return max(1, min(2, base))
            return base

        def profile_recovery_window_steps():
            base = int(cli.grounding_recovery_window_steps)
            if instruction_profile_name in {"visual_boundary", "reference_dense"}:
                return 0
            return base

        def conservative_segment_profile():
            return instruction_profile_name in {"reference_dense", "visual_boundary"}

        def reset_segment_tracking(step, distance_to_goal, target_release_reason="segment_advanced", target_segment=None):
            target_before = guidance_state.get("learned_segment_homing_target")
            if target_before is not None:
                emit_flight_target_telemetry(
                    "target_released",
                    step,
                    target_before=target_before,
                    target_changed_reason=target_release_reason,
                    distance_to_goal=distance_to_goal,
                    segment=target_segment,
                )
            guidance_state["segment_start_step"] = int(step)
            guidance_state["segment_entry_distance"] = distance_to_goal
            guidance_state["segment_best_distance"] = distance_to_goal
            guidance_state["segment_turn_matches"] = 0
            guidance_state["segment_forward_matches"] = 0
            guidance_state["segment_descent_matches"] = 0
            guidance_state["segment_visual_matches"] = 0
            guidance_state["segment_confidence"] = 0.0
            guidance_state["learned_segment_homing_target"] = None
            guidance_state["learned_segment_homing_delta"] = None
            guidance_state["learned_segment_homing_index"] = None
            guidance_state["learned_segment_homing_created_step"] = None
            guidance_state["learned_segment_homing_distance"] = None
            guidance_state["learned_segment_homing_entry_distance"] = None
            guidance_state["learned_segment_homing_best_distance"] = None
            guidance_state["learned_segment_homing_safety_turn_count"] = 0
            guidance_state["learned_segment_homing_segment_safety_turn_count"] = 0
            guidance_state["learned_segment_homing_last_safety_stall_resample_step"] = None
            guidance_state["learned_segment_homing_base_xy_step"] = None
            guidance_state["learned_segment_homing_xy_step"] = None
            guidance_state["learned_segment_homing_risk_tier"] = None
            guidance_state["learned_segment_homing_xy_step_reason"] = None
            depth_route_around_guard.reset()

        def advance_segment(step, distance_to_goal, reason, confidence):
            segment = current_segment()
            if segment is None:
                return None
            next_index = int(guidance_state["segment_index"]) + 1
            if next_index >= len(segment_plan):
                return None
            guidance_state["segment_switches"].append(
                {
                    "from_index": int(guidance_state["segment_index"]),
                    "to_index": next_index,
                    "step": int(step),
                    "reason": reason,
                    "distance_to_goal": distance_to_goal,
                    "confidence": float(confidence),
                    "from_text": segment.get("text"),
                    "to_text": segment_plan[next_index].get("text"),
                }
            )
            guidance_state["segment_index"] = next_index
            guidance_state["segment_boundary_window_steps"] = profile_boundary_window_steps()
            reset_segment_tracking(
                step,
                distance_to_goal,
                target_release_reason="target_reached" if reason == "learned_target_reached" else "segment_advanced",
                target_segment=segment,
            )
            guidance_state["segment_confidence"] = float(confidence)
            return guidance_state["segment_switches"][-1]

        def segment_action_match(segment, action_id):
            action_names_for_segment = set(segment.get("actions") or [])
            segment_kind = segment.get("segment_kind")
            segment_text = str(segment.get("text") or "").lower()
            vertical_alignment_text = any(
                phrase in segment_text
                for phrase in (
                    "level",
                    "levelled",
                    "leveled",
                    "height",
                    "altitude",
                    "raise yourself",
                    "higher",
                    "lower",
                )
            )
            if not action_names_for_segment:
                return bool(segment_kind == "visual" and action_id in (AirsimActions.TURN_LEFT, AirsimActions.TURN_RIGHT, AirsimActions.GO_UP, AirsimActions.GO_DOWN))
            if action_id == AirsimActions.TURN_LEFT and ("turn_left" in action_names_for_segment or "turn" in action_names_for_segment):
                return True
            if action_id == AirsimActions.TURN_RIGHT and ("turn_right" in action_names_for_segment or "turn" in action_names_for_segment):
                return True
            if action_id == AirsimActions.MOVE_FORWARD and ("forward" in action_names_for_segment or "fly_over" in action_names_for_segment):
                return True
            if action_id == AirsimActions.GO_DOWN and "descend" in action_names_for_segment:
                return True
            if action_id == AirsimActions.GO_DOWN and (
                "stop" in action_names_for_segment
                or "land" in action_names_for_segment
                or "land" in segment_text
                or "landed" in segment_text
                or "onto the building" in segment_text
                or "onto the roof" in segment_text
                or "onto the rooftop" in segment_text
            ):
                return True
            if action_id == AirsimActions.GO_UP and "take_off" in action_names_for_segment:
                return True
            if action_id == AirsimActions.GO_UP and "fly_over" in action_names_for_segment:
                return True
            if action_id in (AirsimActions.GO_UP, AirsimActions.GO_DOWN) and any(
                phrase in segment_text
                for phrase in (
                    "fly through",
                    "flying through",
                    "fly above",
                    "flying above",
                    "fly over",
                    "flying over",
                    "go over",
                    "going over",
                )
            ):
                return True
            if action_id == AirsimActions.GO_UP and "look_up" in action_names_for_segment:
                return True
            if action_id == AirsimActions.GO_DOWN and "look_down" in action_names_for_segment:
                return True
            if action_id in (AirsimActions.GO_UP, AirsimActions.GO_DOWN) and vertical_alignment_text:
                return True
            if action_id == AirsimActions.TURN_LEFT and "look_left" in action_names_for_segment:
                return True
            if action_id == AirsimActions.TURN_RIGHT and "look_right" in action_names_for_segment:
                return True
            if action_id in (AirsimActions.TURN_LEFT, AirsimActions.TURN_RIGHT) and any(
                item.get("type") == "facing" for item in segment.get("relations", [])
            ):
                return True
            return False

        def reset_segment_observer_tracking(step, distance_to_goal):
            guidance_state["segment_observer_start_step"] = int(step)
            guidance_state["segment_observer_entry_distance"] = distance_to_goal
            guidance_state["segment_observer_best_distance"] = distance_to_goal
            guidance_state["segment_observer_turn_matches"] = 0
            guidance_state["segment_observer_forward_matches"] = 0
            guidance_state["segment_observer_descent_matches"] = 0
            guidance_state["segment_observer_visual_matches"] = 0
            guidance_state["segment_observer_confidence"] = 0.0
            guidance_state["segment_observer_missed_turn_streak"] = 0
            guidance_state["segment_observer_descend_stall_streak"] = 0

        def record_observer_diagnostic(kind, step, action_id, distance_to_goal, confidence, extra=None):
            if not segment_observer_enabled():
                return None
            diagnostics = guidance_state.get("segment_observer_diagnostics")
            if diagnostics is None:
                diagnostics = []
                guidance_state["segment_observer_diagnostics"] = diagnostics
            if len(diagnostics) >= 128:
                return None
            segment = current_observer_segment()
            item = {
                "kind": kind,
                "step": int(step),
                "action_id": int(action_id) if action_id is not None else None,
                "action": action_names.get(int(action_id), None) if action_id is not None else None,
                "distance_to_goal": distance_to_goal,
                "confidence": float(confidence or 0.0),
                "observer_index": int(guidance_state.get("segment_observer_index", 0)),
                "segment_text": segment.get("text") if segment else None,
                "segment_actions": list(segment.get("actions", [])) if segment else [],
                "segment_target_landmark": segment.get("target_landmark") if segment else None,
                "segment_target_zone": segment.get("target_zone") if segment else None,
            }
            if extra:
                item.update(extra)
            diagnostics.append(item)
            return item

        def advance_observer_segment(step, distance_to_goal, reason, confidence, gate_candidate=False, gate_reason=None):
            segment = current_observer_segment()
            if segment is None:
                return None
            next_index = int(guidance_state["segment_observer_index"]) + 1
            if next_index >= len(segment_plan):
                return None
            gate_action_id = None
            gate_action = None
            actions_for_segment = set(segment.get("actions") or [])
            if gate_candidate:
                if "turn_left" in actions_for_segment and "turn_right" not in actions_for_segment:
                    gate_action_id = int(AirsimActions.TURN_LEFT)
                elif "turn_right" in actions_for_segment and "turn_left" not in actions_for_segment:
                    gate_action_id = int(AirsimActions.TURN_RIGHT)
                if gate_action_id is not None:
                    gate_action = action_names.get(gate_action_id)
            guidance_state["segment_observer_switches"].append(
                {
                    "from_index": int(guidance_state["segment_observer_index"]),
                    "to_index": next_index,
                    "step": int(step),
                    "reason": reason,
                    "distance_to_goal": distance_to_goal,
                    "confidence": float(confidence),
                    "gate_candidate": bool(gate_candidate),
                    "gate_reason": gate_reason,
                    "gate_action_id": gate_action_id,
                    "gate_action": gate_action,
                    "from_text": segment.get("text"),
                    "to_text": segment_plan[next_index].get("text"),
                }
            )
            guidance_state["segment_observer_index"] = next_index
            reset_segment_observer_tracking(step, distance_to_goal)
            guidance_state["segment_observer_confidence"] = float(confidence)
            return guidance_state["segment_observer_switches"][-1]

        def preferred_segment_proxy_action(segment, fallback_action_id=AirsimActions.MOVE_FORWARD):
            if segment is None:
                return fallback_action_id
            action_names_for_segment = set(segment.get("actions") or [])
            relation_types = {item.get("type") for item in segment.get("relations", [])}
            if "look_left" in action_names_for_segment or "turn_left" in action_names_for_segment:
                return AirsimActions.TURN_LEFT
            if "look_right" in action_names_for_segment or "turn_right" in action_names_for_segment:
                return AirsimActions.TURN_RIGHT
            if "look_up" in action_names_for_segment or "take_off" in action_names_for_segment:
                return AirsimActions.GO_UP
            if "look_down" in action_names_for_segment or "descend" in action_names_for_segment:
                return AirsimActions.GO_DOWN
            if "facing" in relation_types:
                return AirsimActions.TURN_LEFT
            if segment.get("segment_kind") == "visual":
                return AirsimActions.TURN_LEFT
            return fallback_action_id

        def segment_confidence(segment, action_id, distance_to_goal, altitude):
            if segment is None:
                return 0.0
            segment_kind = segment.get("segment_kind")
            alignment_mode = segment_alignment_mode(segment)
            confidence = 0.15 if segment_kind == "visual" else 0.1
            if segment_action_match(segment, action_id):
                action_bonus = 0.4 if segment_kind == "visual" else 0.35
                if alignment_mode == "light":
                    action_bonus *= 0.7
                elif alignment_mode == "strong":
                    action_bonus *= 1.05
                confidence += action_bonus
            entry_distance = guidance_state.get("segment_entry_distance")
            best_distance = guidance_state.get("segment_best_distance")
            if (
                entry_distance is not None
                and best_distance is not None
                and entry_distance >= 0
                and best_distance >= 0
            ):
                progress = max(0.0, float(entry_distance) - float(best_distance))
                if segment_kind == "visual":
                    if progress >= max(4.0, cli.segment_progress_distance * 0.25):
                        confidence += 0.05 if alignment_mode == "light" else 0.1
                else:
                    if progress >= cli.segment_progress_distance:
                        confidence += 0.25 if alignment_mode == "light" else 0.35
                    elif progress >= max(6.0, cli.segment_progress_distance * 0.4):
                        confidence += 0.12 if alignment_mode == "light" else 0.2
            if guidance_state.get("distance_worse_streak", 0) <= 1:
                confidence += 0.1
            if distance_to_goal is not None and distance_to_goal >= 0 and distance_to_goal <= cli.arrival_fine_approach_distance:
                confidence += 0.1
            if segment.get("target_zone") == "structure" and 12.0 <= altitude <= 40.0:
                confidence += 0.05
            if segment.get("target_zone") == "corridor" and altitude <= 22.0:
                confidence += 0.05
            if segment_kind == "visual" and guidance_state.get("segment_visual_matches", 0) >= 2:
                confidence += 0.15
            return float(min(1.0, confidence))

        def observe_reference_dense_segment(step, action_id, distance_to_goal, altitude):
            segment = current_observer_segment()
            if segment is None:
                return None
            if (
                guidance_state.get("segment_observer_entry_distance") is None
                and distance_to_goal is not None
                and distance_to_goal >= 0
            ):
                reset_segment_observer_tracking(step, distance_to_goal)
            if distance_to_goal is not None and distance_to_goal >= 0:
                best_distance = guidance_state.get("segment_observer_best_distance")
                if best_distance is None or distance_to_goal < best_distance:
                    guidance_state["segment_observer_best_distance"] = float(distance_to_goal)

            actions_for_segment = set(segment.get("actions") or [])
            if action_id == AirsimActions.TURN_LEFT and "turn_left" in actions_for_segment:
                guidance_state["segment_observer_turn_matches"] += 1
            if action_id == AirsimActions.TURN_RIGHT and "turn_right" in actions_for_segment:
                guidance_state["segment_observer_turn_matches"] += 1
            if action_id == AirsimActions.MOVE_FORWARD and (
                "forward" in actions_for_segment or "fly_over" in actions_for_segment
            ):
                guidance_state["segment_observer_forward_matches"] += 1
            if action_id == AirsimActions.GO_DOWN and "descend" in actions_for_segment:
                guidance_state["segment_observer_descent_matches"] += 1
            if segment_action_match(segment, action_id) and action_id in (
                AirsimActions.TURN_LEFT,
                AirsimActions.TURN_RIGHT,
                AirsimActions.GO_UP,
                AirsimActions.GO_DOWN,
            ):
                guidance_state["segment_observer_visual_matches"] += 1

            entry_distance = guidance_state.get("segment_observer_entry_distance")
            best_distance = guidance_state.get("segment_observer_best_distance")
            progress = 0.0
            regression_from_best = 0.0
            if (
                entry_distance is not None
                and best_distance is not None
                and entry_distance >= 0
                and best_distance >= 0
            ):
                progress = max(0.0, float(entry_distance) - float(best_distance))
                if distance_to_goal is not None and distance_to_goal >= 0:
                    regression_from_best = max(0.0, float(distance_to_goal) - float(best_distance))

            segment_kind = segment.get("segment_kind")
            confidence = 0.1
            if segment_action_match(segment, action_id):
                confidence += 0.3
            if progress >= cli.segment_progress_distance:
                confidence += 0.25
            elif progress >= max(4.0, cli.segment_progress_distance * 0.25):
                confidence += 0.12
            if guidance_state.get("distance_worse_streak", 0) <= 1:
                confidence += 0.08
            if segment.get("target_zone") == "structure" and 12.0 <= altitude <= 40.0:
                confidence += 0.04
            if segment.get("target_zone") == "corridor" and altitude <= 22.0:
                confidence += 0.04
            confidence = float(min(1.0, confidence))
            guidance_state["segment_observer_confidence"] = confidence

            age = max(0, int(step) - int(guidance_state.get("segment_observer_start_step", 0)) + 1)
            if age < cli.segment_min_steps:
                return None

            turn_segment = bool(actions_for_segment & {"turn_left", "turn_right"})
            descend_segment = "descend" in actions_for_segment
            forward_segment = bool(actions_for_segment & {"forward", "fly_over"}) or not actions_for_segment
            desired_turn_action = None
            if "turn_left" in actions_for_segment and "turn_right" not in actions_for_segment:
                desired_turn_action = AirsimActions.TURN_LEFT
            elif "turn_right" in actions_for_segment and "turn_left" not in actions_for_segment:
                desired_turn_action = AirsimActions.TURN_RIGHT
            if desired_turn_action is not None and action_id in (
                AirsimActions.MOVE_FORWARD,
                AirsimActions.STOP,
                AirsimActions.MOVE_LEFT,
                AirsimActions.MOVE_RIGHT,
            ):
                guidance_state["segment_observer_missed_turn_streak"] += 1
                missed_turn_streak = int(guidance_state["segment_observer_missed_turn_streak"])
                if missed_turn_streak in {3, 5, 10}:
                    record_observer_diagnostic(
                        "missed_turn_streak",
                        step,
                        action_id,
                        distance_to_goal,
                        confidence,
                        {
                            "streak": missed_turn_streak,
                            "desired_action_id": int(desired_turn_action),
                            "desired_action": action_names.get(int(desired_turn_action), None),
                        },
                    )
            elif desired_turn_action is not None:
                guidance_state["segment_observer_missed_turn_streak"] = 0

            if descend_segment and action_id != AirsimActions.GO_DOWN:
                guidance_state["segment_observer_descend_stall_streak"] += 1
                descend_stall_streak = int(guidance_state["segment_observer_descend_stall_streak"])
                if descend_stall_streak in {5, 10, 20}:
                    record_observer_diagnostic(
                        "descend_stall_streak",
                        step,
                        action_id,
                        distance_to_goal,
                        confidence,
                        {"streak": descend_stall_streak},
                    )
            elif descend_segment:
                guidance_state["segment_observer_descend_stall_streak"] = 0

            any_match = (
                guidance_state.get("segment_observer_turn_matches", 0)
                + guidance_state.get("segment_observer_forward_matches", 0)
                + guidance_state.get("segment_observer_descent_matches", 0)
                + guidance_state.get("segment_observer_visual_matches", 0)
            )

            should_advance = False
            reason = None
            if (
                turn_segment
                and guidance_state.get("segment_observer_turn_matches", 0) >= 1
                and confidence >= 0.3
            ):
                should_advance = True
                reason = "observer_turn_phase_seen"
            elif (
                descend_segment
                and guidance_state.get("segment_observer_descent_matches", 0) >= 1
                and confidence >= 0.28
            ):
                should_advance = True
                reason = "observer_descent_phase_seen"
            elif (
                forward_segment
                and (
                    guidance_state.get("segment_observer_forward_matches", 0) >= 2
                    or progress >= max(6.0, cli.segment_progress_distance * 0.3)
                )
                and confidence >= 0.28
            ):
                should_advance = True
                reason = "observer_forward_progress_seen"
            elif (
                age >= cli.segment_min_steps + 8
                and any_match >= 1
                and regression_from_best <= max(20.0, cli.segment_progress_distance)
            ):
                should_advance = True
                reason = "observer_timeout_stable"

            if should_advance:
                low_regression_risk = (
                    regression_from_best <= max(8.0, cli.segment_progress_distance * 0.4)
                    and guidance_state.get("distance_worse_streak", 0) <= 1
                )
                gate_candidate = bool(
                    reason == "observer_turn_phase_seen"
                    and confidence >= 0.5
                    and low_regression_risk
                )
                gate_reason = "high_confidence_turn_boundary" if gate_candidate else None
                return advance_observer_segment(
                    step,
                    distance_to_goal,
                    reason,
                    confidence,
                    gate_candidate=gate_candidate,
                    gate_reason=gate_reason,
                )
            return None

        def maybe_advance_segment(step, action_id, distance_to_goal, altitude):
            segment = current_segment()
            if segment is None:
                return None
            if guidance_state.get("segment_entry_distance") is None and distance_to_goal is not None and distance_to_goal >= 0:
                reset_segment_tracking(step, distance_to_goal)
            if distance_to_goal is not None and distance_to_goal >= 0:
                best_distance = guidance_state.get("segment_best_distance")
                if best_distance is None or distance_to_goal < best_distance:
                    guidance_state["segment_best_distance"] = float(distance_to_goal)
            if action_id == AirsimActions.TURN_LEFT and "turn_left" in set(segment.get("actions") or []):
                guidance_state["segment_turn_matches"] += 1
            if action_id == AirsimActions.TURN_RIGHT and "turn_right" in set(segment.get("actions") or []):
                guidance_state["segment_turn_matches"] += 1
            if action_id == AirsimActions.MOVE_FORWARD and (
                "forward" in set(segment.get("actions") or []) or "fly_over" in set(segment.get("actions") or [])
            ):
                guidance_state["segment_forward_matches"] += 1
            if action_id == AirsimActions.GO_DOWN and "descend" in set(segment.get("actions") or []):
                guidance_state["segment_descent_matches"] += 1
            if segment_action_match(segment, action_id) and action_id in (
                AirsimActions.TURN_LEFT,
                AirsimActions.TURN_RIGHT,
                AirsimActions.GO_UP,
                AirsimActions.GO_DOWN,
            ):
                guidance_state["segment_visual_matches"] += 1

            confidence = segment_confidence(segment, action_id, distance_to_goal, altitude)
            guidance_state["segment_confidence"] = confidence

            if cli.enable_oracle_path_homing and reference_path:
                # In oracle-path mode the waypoint cursor is the stable progress
                # source. Let it own segment transitions so distance-based
                # fallback gates do not skip several semantic segments at once.
                return None

            age = max(0, int(step) - int(guidance_state.get("segment_start_step", 0)) + 1)
            if age < cli.segment_min_steps:
                return None

            entry_distance = guidance_state.get("segment_entry_distance")
            best_distance = guidance_state.get("segment_best_distance")
            progress = 0.0
            regression_from_best = 0.0
            if (
                entry_distance is not None
                and best_distance is not None
                and entry_distance >= 0
                and best_distance >= 0
            ):
                progress = max(0.0, float(entry_distance) - float(best_distance))
                if distance_to_goal is not None and distance_to_goal >= 0:
                    regression_from_best = max(0.0, float(distance_to_goal) - float(best_distance))

            actions_for_segment = set(segment.get("actions") or [])
            relation_types = {item.get("type") for item in segment.get("relations", [])}
            target_landmark = segment.get("target_landmark")
            target_zone = segment.get("target_zone")
            turn_segment = bool(actions_for_segment & {"turn_left", "turn_right"})
            descend_segment = "descend" in actions_for_segment
            forward_segment = bool(actions_for_segment & {"forward", "fly_over"}) or not actions_for_segment
            visual_segment = segment.get("segment_kind") == "visual"
            mixed_segment = segment.get("segment_kind") == "mixed"
            alignment_mode = segment_alignment_mode(segment)
            light_touch_alignment = alignment_mode == "light"
            conservative_profile = conservative_segment_profile()
            corridor_proxy = target_landmark in {"intersection", "road", "path", "bridge", "water", "park"}
            between_proxy = "between" in relation_types
            structure_proxy = target_zone == "structure"
            near_proxy = "near" in relation_types

            should_advance = False
            reason = None
            if (
                visual_segment
                and guidance_state.get("segment_visual_matches", 0) >= (
                    4 if conservative_profile else (3 if light_touch_alignment else 2)
                )
                and confidence >= cli.segment_switch_confidence * (
                    1.0 if conservative_profile else (0.95 if light_touch_alignment else 0.7)
                )
            ):
                should_advance = True
                reason = "visual_scan_complete"
            elif (
                mixed_segment
                and not light_touch_alignment
                and guidance_state.get("segment_visual_matches", 0) >= 1
                and progress >= max(6.0, cli.segment_progress_distance * 0.35)
            ):
                should_advance = True
                reason = "mixed_segment_complete"
            elif (
                not light_touch_alignment
                and
                corridor_proxy
                and turn_segment
                and guidance_state.get("segment_turn_matches", 0) >= 1
                and guidance_state.get("segment_forward_matches", 0) >= 2
                and confidence >= 0.35
            ):
                should_advance = True
                reason = "corridor_turn_complete"
            elif (
                not light_touch_alignment
                and
                corridor_proxy
                and guidance_state.get("segment_forward_matches", 0) >= 3
                and age >= cli.segment_min_steps + 1
                and confidence >= 0.3
            ):
                should_advance = True
                reason = "corridor_progress_proxy"
            elif (
                not light_touch_alignment
                and
                between_proxy
                and guidance_state.get("segment_forward_matches", 0) >= 2
                and age >= cli.segment_min_steps
                and confidence >= 0.3
            ):
                should_advance = True
                reason = "between_passage_complete"
            elif (
                not light_touch_alignment
                and
                structure_proxy
                and near_proxy
                and (
                    guidance_state.get("segment_forward_matches", 0) >= 2
                    or guidance_state.get("segment_turn_matches", 0) >= 1
                )
                and age >= cli.segment_min_steps
                and confidence >= 0.35
            ):
                should_advance = True
                reason = "structure_approach_proxy"
            elif (
                turn_segment
                and guidance_state.get("segment_turn_matches", 0) >= 1
                and confidence >= cli.segment_switch_confidence * (1.0 if conservative_profile else 0.8)
            ):
                should_advance = True
                reason = "turn_alignment_complete"
            elif descend_segment and (
                guidance_state.get("segment_descent_matches", 0) >= 1
                or (distance_to_goal is not None and distance_to_goal <= cli.arrival_fine_approach_distance)
            ) and confidence >= cli.segment_switch_confidence * (
                1.0 if conservative_profile else (0.95 if light_touch_alignment else 0.8)
            ):
                should_advance = True
                reason = "descent_alignment_complete"
            elif (
                forward_segment
                and progress >= cli.segment_progress_distance
                and confidence >= cli.segment_switch_confidence * (
                    1.15 if conservative_profile else (1.05 if light_touch_alignment else 1.0)
                )
            ):
                should_advance = True
                reason = "progress_gate_complete"
            elif (
                not conservative_profile
                and
                not light_touch_alignment
                and forward_segment
                and age >= cli.segment_min_steps + 4
                and progress >= max(8.0, cli.segment_progress_distance * 0.5)
            ):
                should_advance = True
                reason = "fallback_progress_gate"
            elif (
                not conservative_profile
                and
                not light_touch_alignment
                and
                visual_segment
                and age >= cli.segment_min_steps + 6
                and guidance_state.get("segment_visual_matches", 0) >= 2
                and regression_from_best <= 8.0
            ):
                should_advance = True
                reason = "visual_timeout_advance"
            elif (
                not conservative_profile
                and
                not light_touch_alignment
                and
                age >= cli.segment_min_steps + 8
                and guidance_state.get("distance_worse_streak", 0) >= 2
                and progress >= max(6.0, cli.segment_progress_distance * 0.25)
                and regression_from_best <= max(12.0, cli.segment_progress_distance * 0.5)
            ):
                should_advance = True
                reason = "stalled_segment_timeout"

            if should_advance:
                return advance_segment(step, distance_to_goal, reason, confidence)
            return None

        def depth_diagnostics(obs):
            depth = obs.get("depth") if obs else None
            if depth is None:
                return None
            arr = np.asarray(depth).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            if arr.size == 0:
                return None
            h, w = arr.shape[:2]
            front = arr[h // 4 : h * 3 // 4, w // 4 : w * 3 // 4]
            center = arr[h // 3 : h * 2 // 3, w // 3 : w * 2 // 3]
            if front.size == 0 or center.size == 0:
                return None
            front_valid_ratio = depth_valid_ratio(front)
            center_valid_ratio = depth_valid_ratio(center)
            return {
                "arr": arr,
                "front": front,
                "center": center,
                "center_close_ratio": float((center < 0.07).sum()) / float(center.size),
                "front_close_ratio": float((front < 0.08).sum()) / float(front.size),
                "front_very_close_ratio": float((front < 0.035).sum()) / float(front.size),
                "front_contact_ratio": float((front < 0.008).sum()) / float(front.size),
                "front_valid_ratio": front_valid_ratio,
                "center_valid_ratio": center_valid_ratio,
                "front_p10": float(np.nanpercentile(front, 10)),
                "front_p25": float(np.nanpercentile(front, 25)),
            }

        def depth_escape_diagnostics(obs):
            diagnostics = depth_diagnostics(obs)
            if diagnostics is None:
                return None
            arr = diagnostics["arr"]
            h, w = arr.shape[:2]
            left = arr[h // 3 : h * 2 // 3, : w // 3]
            right = arr[h // 3 : h * 2 // 3, w * 2 // 3 :]
            diagnostics.update(
                {
                    "invalid_depth": min(
                        diagnostics["front_valid_ratio"],
                        diagnostics["center_valid_ratio"],
                    ) < 0.5,
                    "obstacle_ahead": bool(
                        diagnostics["center_close_ratio"] >= 0.015
                        or diagnostics["front_close_ratio"] >= 0.04
                        or diagnostics["front_very_close_ratio"] >= 0.005
                        or diagnostics["front_p10"] < 0.065
                        or diagnostics["front_p25"] < 0.08
                    ),
                    "left_score": float(np.nanpercentile(left, 25)) if left.size else 0.0,
                    "right_score": float(np.nanpercentile(right, 25)) if right.size else 0.0,
                }
            )
            return diagnostics

        def depth_safety_override(obs, action_id, info=None):
            if cli.disable_safety_depth_override or action_id != AirsimActions.MOVE_FORWARD:
                return action_id, None
            diagnostics = depth_escape_diagnostics(obs)
            if diagnostics is None:
                return action_id, None
            arr = diagnostics["arr"]
            h, w = arr.shape[:2]
            # Depth is normalized to 100m. A forward action moves 5m, so braking
            # must begin before 5m and leave clearance for the vehicle body.
            center_close_ratio = diagnostics["center_close_ratio"]
            front_close_ratio = diagnostics["front_close_ratio"]
            front_very_close_ratio = diagnostics["front_very_close_ratio"]
            front_p10 = diagnostics["front_p10"]
            front_p25 = diagnostics["front_p25"]

            if diagnostics["invalid_depth"]:
                guidance_state["avoidance_streak"] = 0
                override = int(
                    guidance_state.get("last_depth_turn_action")
                    or AirsimActions.TURN_RIGHT
                )
                return override, {
                    "raw_action_id": int(action_id),
                    "raw_action": action_names.get(int(action_id), None),
                    "override_action_id": override,
                    "override_action": action_names.get(override, None),
                    "reason": "invalid_depth_turn",
                    "front_valid_ratio": diagnostics["front_valid_ratio"],
                    "center_valid_ratio": diagnostics["center_valid_ratio"],
                }

            if not diagnostics["obstacle_ahead"]:
                guidance_state["avoidance_streak"] = 0
                return action_id, None

            left = arr[h // 3 : h * 2 // 3, : w // 3]
            right = arr[h // 3 : h * 2 // 3, w * 2 // 3 :]
            left_score = diagnostics["left_score"]
            right_score = diagnostics["right_score"]
            left_close = float((left < 0.05).sum()) / float(left.size) if left.size else 1.0
            right_close = float((right < 0.05).sum()) / float(right.size) if right.size else 1.0
            guidance_state["avoidance_streak"] += 1

            state = env.sim_states[0]
            z = float(state.pose.position.z_val)
            side_blocked = left_close > 0.10 and right_close > 0.10
            prolonged_block = guidance_state["avoidance_streak"] >= 8
            try:
                guard_distance = float((info or {}).get("distance_to_goal", -1))
            except Exception:
                guard_distance = -1.0
            final_goal_block = (
                20.0 < guard_distance <= 45.0
                and int(guidance_state.get("avoidance_streak") or 0) >= 18
                and int(guidance_state.get("segment_index") or 0)
                >= max(0, len(segment_plan) - 2)
            )
            surface_clearance = guidance_state.get("surface_clearance")
            if surface_clearance is None:
                surface_clearance = guidance_state.get("last_valid_surface_clearance")
            if should_release_final_corridor_depth_guard(
                current_segment(),
                guidance_state.get("segment_index", 0),
                len(segment_plan),
                guard_distance,
                surface_clearance,
                front_contact_ratio=diagnostics["front_contact_ratio"],
                front_very_close_ratio=front_very_close_ratio,
                front_p10=front_p10,
                safety_turn_count=guidance_state.get(
                    "learned_segment_homing_segment_safety_turn_count", 0
                ),
                avoidance_streak=guidance_state.get("avoidance_streak", 0),
            ):
                guidance_state["avoidance_streak"] = 0
                return action_id, {
                    "raw_action_id": int(action_id),
                    "raw_action": action_names.get(int(action_id), None),
                    "override_action_id": int(action_id),
                    "override_action": action_names.get(int(action_id), None),
                    "reason": "final_corridor_depth_guard_release",
                    "front_contact_ratio": diagnostics["front_contact_ratio"],
                    "front_very_close_ratio": front_very_close_ratio,
                    "front_p10": front_p10,
                    "surface_clearance": surface_clearance,
                    "distance_to_goal": guard_distance,
                    "safety_turn_count": guidance_state.get(
                        "learned_segment_homing_segment_safety_turn_count", 0
                    ),
                    "avoidance_streak": guidance_state.get("avoidance_streak", 0),
                }
            if cli.enable_close_tail_depth_release and should_release_close_final_structure_depth_guard(
                current_segment(),
                guidance_state.get("segment_index", 0),
                len(segment_plan),
                guard_distance,
                surface_clearance,
                front_contact_ratio=diagnostics["front_contact_ratio"],
                front_very_close_ratio=front_very_close_ratio,
                front_p10=front_p10,
                avoidance_streak=guidance_state.get("avoidance_streak", 0),
            ):
                release_avoidance_streak = guidance_state.get("avoidance_streak", 0)
                guidance_state["avoidance_streak"] = 0
                return action_id, {
                    "raw_action_id": int(action_id),
                    "raw_action": action_names.get(int(action_id), None),
                    "override_action_id": int(action_id),
                    "override_action": action_names.get(int(action_id), None),
                    "reason": "close_final_structure_depth_guard_release",
                    "front_contact_ratio": diagnostics["front_contact_ratio"],
                    "front_very_close_ratio": front_very_close_ratio,
                    "front_p10": front_p10,
                    "surface_clearance": surface_clearance,
                    "distance_to_goal": guard_distance,
                    "avoidance_streak": release_avoidance_streak,
                }
            climb_ceiling_z = -80.0 if prolonged_block else -35.0

            # Prefer steering around a facade. Climbing is reserved for a truly
            # enclosed route so ordinary obstacle avoidance does not saw in Z.
            if (
                forward_route_guidance
                and (side_blocked or final_goal_block)
                and prolonged_block
                and z > climb_ceiling_z
            ):
                return AirsimActions.GO_UP, {
                    "raw_action_id": int(action_id),
                    "raw_action": action_names.get(int(action_id), None),
                    "override_action_id": int(AirsimActions.GO_UP),
                    "override_action": action_names.get(int(AirsimActions.GO_UP), None),
                    "reason": (
                        "depth_final_goal_escape_climb"
                        if final_goal_block
                        else "depth_building_guard_escape_climb"
                        if prolonged_block
                        else "depth_building_guard_climb"
                    ),
                    "center_close_ratio": center_close_ratio,
                    "front_close_ratio": front_close_ratio,
                    "front_very_close_ratio": front_very_close_ratio,
                    "front_p10": front_p10,
                    "front_p25": front_p25,
                    "left_close_ratio": left_close,
                    "right_close_ratio": right_close,
                    "left_depth_mean": left_score,
                    "right_depth_mean": right_score,
                    "avoidance_streak": guidance_state["avoidance_streak"],
                    "distance_to_goal": guard_distance,
                    "final_goal_block": final_goal_block,
                    "z": z,
                    "climb_ceiling_z": climb_ceiling_z,
                }

            override = AirsimActions.TURN_LEFT if left_score >= right_score else AirsimActions.TURN_RIGHT
            guidance_state["last_depth_turn_action"] = int(override)
            return override, {
                "raw_action_id": int(action_id),
                "raw_action": action_names.get(int(action_id), None),
                "override_action_id": int(override),
                "override_action": action_names.get(int(override), None),
                "reason": "depth_building_guard_turn",
                "center_close_ratio": center_close_ratio,
                "front_close_ratio": front_close_ratio,
                "front_very_close_ratio": front_very_close_ratio,
                "front_p10": front_p10,
                "front_p25": front_p25,
                "left_close_ratio": left_close,
                "right_close_ratio": right_close,
                "left_depth_mean": left_score,
                "right_depth_mean": right_score,
                "avoidance_streak": guidance_state["avoidance_streak"],
            }

        def vertical_stability_override(obs, info, action_id, step):
            if action_id not in (AirsimActions.GO_UP, AirsimActions.GO_DOWN):
                return action_id, None

            distance_to_goal = None
            try:
                distance_to_goal = float((info or {}).get("distance_to_goal", -1))
            except Exception:
                distance_to_goal = None
            in_success_zone = bool(
                distance_to_goal is not None
                and 0 <= distance_to_goal <= float(cli.success_distance)
            )

            if action_id == AirsimActions.GO_DOWN and in_success_zone:
                landing_clearance = float(cli.surface_landing_clearance)
                surface_stop_tolerance = 0.02
                cached_clearance = guidance_state.get("surface_clearance")
                if cached_clearance is not None and float(cached_clearance) <= landing_clearance + surface_stop_tolerance:
                    return AirsimActions.STOP, {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.STOP),
                        "override_action": action_names.get(int(AirsimActions.STOP), None),
                        "reason": "surface_already_aligned_stop",
                        "surface_clearance": float(cached_clearance),
                    }
                clearance = probe_surface_clearance(step)
                if clearance is not None and clearance > 2.0 + landing_clearance:
                    surface_finish = apply_surface_finish_pose(
                        step,
                        clearance,
                        "success_zone_surface_finish",
                    )
                    estimated_clearance = (
                        surface_finish.get("surface_clearance_after")
                        if surface_finish
                        else None
                    )
                    if (
                        surface_finish
                        and not surface_finish.get("error")
                        and estimated_clearance is not None
                        and float(estimated_clearance) <= landing_clearance + surface_stop_tolerance
                    ):
                        return AirsimActions.STOP, {
                            "raw_action_id": int(action_id),
                            "raw_action": action_names.get(int(action_id), None),
                            "override_action_id": int(AirsimActions.STOP),
                            "override_action": action_names.get(int(AirsimActions.STOP), None),
                            "reason": "success_zone_surface_finish_stop",
                            **surface_finish,
                        }
                    if (
                        surface_finish
                        and not surface_finish.get("error")
                        and estimated_clearance is not None
                        and float(estimated_clearance) <= 2.0 + landing_clearance
                    ):
                        return AirsimActions.GO_DOWN, {
                            "raw_action_id": int(action_id),
                            "raw_action": action_names.get(int(action_id), None),
                            "override_action_id": int(AirsimActions.GO_DOWN),
                            "override_action": action_names.get(int(AirsimActions.GO_DOWN), None),
                            "reason": "success_zone_surface_finish_then_descend",
                            **surface_finish,
                        }
                    return AirsimActions.GO_DOWN, {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.GO_DOWN),
                        "override_action": action_names.get(int(AirsimActions.GO_DOWN), None),
                        "reason": "success_zone_surface_finish_continue",
                        **(surface_finish or {"surface_clearance_before": float(clearance)}),
                    }
                if clearance is not None and clearance <= 2.0 + landing_clearance:
                    state = env.sim_states[0]
                    descent = max(0.0, float(clearance) - landing_clearance)
                    if descent > 0.05:
                        apply_surface_finish_pose(
                            step,
                            clearance,
                            "surface_descent_capped",
                            max_descent=descent,
                        )
                    return AirsimActions.STOP, {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.STOP),
                        "override_action": action_names.get(int(AirsimActions.STOP), None),
                        "reason": "surface_descent_capped",
                        "surface_clearance_before": float(clearance),
                        "surface_clearance_after": landing_clearance,
                    }

            if action_id == AirsimActions.GO_DOWN and not in_success_zone:
                diagnostics = depth_diagnostics(obs)
                if diagnostics is not None:
                    front_contact_ratio = float(diagnostics["front_contact_ratio"])
                    front_very_close_ratio = float(diagnostics["front_very_close_ratio"])
                    front_p10 = float(diagnostics["front_p10"])
                    descent_blocked_by_depth = (
                        front_contact_ratio >= 0.06
                        or front_very_close_ratio >= 0.035
                        or front_p10 < 0.025
                    )
                    if descent_blocked_by_depth:
                        guidance_state["descent_depth_blocks"] = int(
                            guidance_state.get("descent_depth_blocks", 0)
                        ) + 1
                        state = env.sim_states[0]
                        z = float(state.pose.position.z_val)
                        cruise_z = -min(
                            58.0,
                            max(
                                24.0,
                                float(cli.learned_segment_cruise_altitude)
                                if cli.enable_learned_segment_homing
                                else float(cli.oracle_goal_homing_cruise_altitude),
                            ),
                        )
                        if z > cruise_z and front_contact_ratio >= 0.10:
                            replacement = int(AirsimActions.GO_UP)
                            reason = "descent_depth_guard_climb"
                        else:
                            arr = diagnostics["arr"]
                            h, w = arr.shape[:2]
                            left = arr[h // 3 : h * 2 // 3, : w // 3]
                            right = arr[h // 3 : h * 2 // 3, w * 2 // 3 :]
                            left_score = float(np.nanpercentile(left, 25)) if left.size else 0.0
                            right_score = float(np.nanpercentile(right, 25)) if right.size else 0.0
                            replacement = int(
                                AirsimActions.TURN_LEFT
                                if left_score >= right_score
                                else AirsimActions.TURN_RIGHT
                            )
                            reason = "descent_depth_guard_turn"
                        return replacement, {
                            "raw_action_id": int(action_id),
                            "raw_action": action_names.get(int(action_id), None),
                            "override_action_id": int(replacement),
                            "override_action": action_names.get(int(replacement), None),
                            "reason": reason,
                            "front_contact_ratio": front_contact_ratio,
                            "front_very_close_ratio": front_very_close_ratio,
                            "front_p10": front_p10,
                            "z": z,
                            "cruise_z": cruise_z,
                            "descent_depth_blocks": int(guidance_state["descent_depth_blocks"]),
                        }
                clearance = probe_surface_clearance(step)
                if clearance is not None and clearance <= 3.25:
                    guidance_state["descent_surface_blocks"] = int(
                        guidance_state.get("descent_surface_blocks", 0)
                    ) + 1
                    diagnostics = depth_diagnostics(obs)
                    left_score = right_score = 0.0
                    if diagnostics is not None:
                        arr = diagnostics["arr"]
                        h, w = arr.shape[:2]
                        left = arr[h // 3 : h * 2 // 3, : w // 3]
                        right = arr[h // 3 : h * 2 // 3, w * 2 // 3 :]
                        left_score = float(np.nanpercentile(left, 25)) if left.size else 0.0
                        right_score = float(np.nanpercentile(right, 25)) if right.size else 0.0
                    consecutive_guard = int(step) - int(
                        guidance_state.get("surface_avoid_last_step", -999)
                    ) == 1
                    turn_steps = (
                        int(guidance_state.get("surface_avoid_turn_steps", 0))
                        if consecutive_guard
                        else 0
                    )
                    locked_turn = (
                        guidance_state.get("surface_avoid_turn_action")
                        if consecutive_guard
                        else None
                    )
                    if locked_turn is None:
                        locked_turn = int(
                            AirsimActions.TURN_LEFT
                            if left_score >= right_score
                            else AirsimActions.TURN_RIGHT
                        )
                    if turn_steps < 3:
                        replacement = int(locked_turn)
                        turn_steps += 1
                        reason = "descent_surface_guard_turn_locked"
                    else:
                        replacement = int(AirsimActions.MOVE_FORWARD)
                        turn_steps = 0
                        locked_turn = None
                        reason = "descent_surface_guard_advance"
                    guidance_state["surface_avoid_turn_action"] = locked_turn
                    guidance_state["surface_avoid_turn_steps"] = int(turn_steps)
                    guidance_state["surface_avoid_last_step"] = int(step)
                    return replacement, {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(replacement),
                        "override_action": action_names.get(int(replacement), None),
                        "reason": reason,
                        "surface_clearance": float(clearance),
                        "minimum_descent_clearance": 3.25,
                        "locked_turn_action": (
                            action_names.get(int(locked_turn), None)
                            if locked_turn is not None
                            else None
                        ),
                        "locked_turn_steps": int(turn_steps),
                    }

            previous = guidance_state.get("last_vertical_action")
            previous_step = int(guidance_state.get("last_vertical_action_step", -999))
            opposite = bool(
                previous is not None
                and int(previous) != int(action_id)
                and int(step) - previous_step <= 6
            )
            if opposite and not in_success_zone:
                guidance_state["vertical_reversal_suppressed"] = int(
                    guidance_state.get("vertical_reversal_suppressed", 0)
                ) + 1
                return AirsimActions.MOVE_FORWARD, {
                    "raw_action_id": int(action_id),
                    "raw_action": action_names.get(int(action_id), None),
                    "override_action_id": int(AirsimActions.MOVE_FORWARD),
                    "override_action": action_names.get(int(AirsimActions.MOVE_FORWARD), None),
                    "reason": "vertical_reversal_hysteresis",
                    "previous_vertical_action": action_names.get(int(previous), None),
                    "cooldown_steps": 6,
                }

            guidance_state["last_vertical_action"] = int(action_id)
            guidance_state["last_vertical_action_step"] = int(step)
            return action_id, None

        def near_success_surface_finish_override(info, action_id, step):
            if (
                cli.disable_surface_landing
                or int(action_id) != int(AirsimActions.GO_DOWN)
            ):
                return action_id, None
            distance_to_goal = None
            if info:
                try:
                    distance_to_goal = float(info.get("distance_to_goal", -1))
                except Exception:
                    distance_to_goal = None
            if (
                distance_to_goal is None
                or distance_to_goal < 0
                or distance_to_goal > float(cli.success_distance) + 2.0
            ):
                return action_id, None
            landing_clearance = float(cli.surface_landing_clearance)
            tolerance = 0.02
            cached_clearance = guidance_state.get("surface_clearance")
            if cached_clearance is not None and float(cached_clearance) <= landing_clearance + tolerance:
                return AirsimActions.STOP, {
                    "raw_action_id": int(action_id),
                    "raw_action": action_names.get(int(action_id), None),
                    "override_action_id": int(AirsimActions.STOP),
                    "override_action": action_names.get(int(AirsimActions.STOP), None),
                    "reason": "near_success_surface_already_aligned_stop",
                    "distance_to_goal": distance_to_goal,
                    "surface_clearance": float(cached_clearance),
                }
            clearance = probe_surface_clearance(step)
            if clearance is None:
                return action_id, None
            descent = max(0.0, float(clearance) - landing_clearance + 0.25)
            if descent <= 0.05:
                return AirsimActions.STOP, {
                    "raw_action_id": int(action_id),
                    "raw_action": action_names.get(int(action_id), None),
                    "override_action_id": int(AirsimActions.STOP),
                    "override_action": action_names.get(int(AirsimActions.STOP), None),
                    "reason": "near_success_surface_probe_stop",
                    "distance_to_goal": distance_to_goal,
                    "surface_clearance_before": float(clearance),
                    "surface_clearance_after": landing_clearance,
                }
            if clearance <= landing_clearance + 2.25:
                surface_finish = apply_surface_finish_pose(
                    step,
                    clearance,
                    "near_success_surface_finish",
                    max_descent=descent,
                )
                estimated_clearance = (
                    surface_finish.get("surface_clearance_after")
                    if surface_finish
                    else None
                )
                if (
                    surface_finish
                    and not surface_finish.get("error")
                    and not surface_finish.get("collision")
                    and estimated_clearance is not None
                    and float(estimated_clearance) <= landing_clearance + tolerance
                ):
                    return AirsimActions.STOP, {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.STOP),
                        "override_action": action_names.get(int(AirsimActions.STOP), None),
                        "reason": "near_success_surface_finish_stop",
                        "distance_to_goal": distance_to_goal,
                        **surface_finish,
                    }
            return action_id, None

        def final_structure_descent_clearance_override(obs, info, action_id, step, route_override):
            if cli.disable_surface_landing or int(action_id) != int(AirsimActions.GO_DOWN):
                return action_id, None
            segment = current_segment()
            if segment is None:
                return action_id, None
            distance_to_goal = None
            if info:
                try:
                    distance_to_goal = float(info.get("distance_to_goal", -1))
                except Exception:
                    distance_to_goal = None
            clearance = guidance_state.get("surface_clearance")
            if clearance is None:
                clearance = probe_surface_clearance(step)
            diagnostics = depth_diagnostics(obs)
            if diagnostics is None:
                return action_id, None
            current_index = int(guidance_state.get("segment_index", 0))
            segment_count = len(segment_plan) if segment_plan else 1
            if not should_block_final_structure_descent_for_visual_clearance(
                segment,
                current_index,
                segment_count,
                distance_to_goal,
                cli.success_distance,
                clearance,
                diagnostics.get("front_p10"),
                diagnostics.get("front_contact_ratio"),
                diagnostics.get("front_very_close_ratio"),
            ):
                return action_id, None

            arr = diagnostics["arr"]
            h, w = arr.shape[:2]
            left = arr[h // 3 : h * 2 // 3, : w // 3]
            right = arr[h // 3 : h * 2 // 3, w * 2 // 3 :]
            left_score = float(np.nanpercentile(left, 25)) if left.size else 0.0
            right_score = float(np.nanpercentile(right, 25)) if right.size else 0.0
            replacement = (
                AirsimActions.TURN_LEFT
                if left_score >= right_score
                else AirsimActions.TURN_RIGHT
            )
            if route_override and route_override.get("homing_target_kind") == "learned_segment":
                guidance_state["learned_segment_homing_target"] = None
                guidance_state["learned_segment_homing_delta"] = None
                guidance_state["learned_segment_homing_index"] = None
                guidance_state["learned_segment_homing_created_step"] = None
                guidance_state["learned_segment_homing_distance"] = None
                guidance_state["learned_segment_homing_entry_distance"] = None
                guidance_state["learned_segment_homing_best_distance"] = None
                guidance_state["learned_segment_homing_safety_turn_count"] = int(
                    guidance_state.get("learned_segment_homing_safety_turn_count", 0)
                ) + 1
                guidance_state["learned_segment_homing_segment_safety_turn_count"] = int(
                    guidance_state.get("learned_segment_homing_segment_safety_turn_count", 0)
                ) + 1
                guidance_state["learned_segment_homing_last_safety_stall_resample_step"] = int(step)
            guidance_state["avoidance_streak"] = max(
                1,
                int(guidance_state.get("avoidance_streak", 0)),
            )
            return replacement, {
                "raw_action_id": int(action_id),
                "raw_action": action_names.get(int(action_id), None),
                "override_action_id": int(replacement),
                "override_action": action_names.get(int(replacement), None),
                "reason": "final_structure_descent_visual_clearance_turn",
                "distance_to_goal": distance_to_goal,
                "surface_clearance": None if clearance is None else float(clearance),
                "front_p10": float(diagnostics["front_p10"]),
                "front_contact_ratio": float(diagnostics["front_contact_ratio"]),
                "front_very_close_ratio": float(diagnostics["front_very_close_ratio"]),
                "left_depth_p25": left_score,
                "right_depth_p25": right_score,
                "segment_state": current_segment_snapshot(step),
            }

        def route_guidance_override(info, action_id, step):
            state = env.sim_states[0]
            pose = state.pose
            altitude = float(-pose.position.z_val)
            distance_to_goal = None
            if info:
                try:
                    distance_to_goal = float(info.get("distance_to_goal", -1))
                except Exception:
                    distance_to_goal = None
            in_fine_approach = (
                cli.arrival_fine_approach_distance > 0
                and distance_to_goal is not None
                and 0 <= distance_to_goal <= cli.arrival_fine_approach_distance
            )

            def finalize(selected_action_id, payload=None):
                if guidance_state["segment_boundary_window_steps"] > 0:
                    guidance_state["segment_boundary_window_steps"] -= 1
                if guidance_state["segment_recovery_window_steps"] > 0:
                    guidance_state["segment_recovery_window_steps"] -= 1
                if payload is not None:
                    payload = dict(payload)
                    payload["segment_state"] = current_segment_snapshot(step)
                return selected_action_id, payload

            if distance_to_goal is not None and distance_to_goal >= 0:
                if guidance_state["initial_distance"] is None:
                    guidance_state["initial_distance"] = float(distance_to_goal)
                best = guidance_state["best_distance"]
                last = guidance_state["last_distance"]
                if best is None or distance_to_goal < best:
                    guidance_state["best_distance"] = distance_to_goal
                    guidance_state["best_distance_step"] = step
                    guidance_state["distance_worse_streak"] = 0
                    guidance_state["recovery_turn_steps"] = 0
                    guidance_state["recovery_attempts"] = 0
                    guidance_state["regression_retry_attempts"] = 0
                    guidance_state["regression_retry_exploration_steps"] = 0
                    guidance_state["regression_retry_phase"] = None
                    guidance_state["regression_retry_target"] = None
                elif last is not None and distance_to_goal > last + 1.0:
                    guidance_state["distance_worse_streak"] += 1
                else:
                    guidance_state["distance_worse_streak"] = max(0, guidance_state["distance_worse_streak"] - 1)
                guidance_state["last_distance"] = distance_to_goal

            if in_fine_approach:
                guidance_state["entered_arrival_radius"] = True

            observer_switch = None
            if segment_observer_enabled():
                observer_switch = observe_reference_dense_segment(step, action_id, distance_to_goal, altitude)

            if segmented_grounding:
                maybe_advance_segment(step, action_id, distance_to_goal, altitude)

            active_target_landmark = target_landmark
            active_target_zone = target_zone
            active_actions = set(grounding_actions)
            active_relations = set()
            active_descent_target_hint = descent_target_hint
            active_corridor_target = corridor_target
            active_structure_target = structure_target
            active_segment_kind = "motion"
            active_alignment_mode = instruction_alignment_strength
            segment = current_segment()
            boundary_window_active = guidance_state["segment_boundary_window_steps"] > 0
            recovery_window_active = guidance_state["segment_recovery_window_steps"] > 0
            final_segment_arrival_window = (
                guidance_state["segment_index"] == len(segment_plan) - 1 and in_fine_approach
            )
            segment_window_active = (
                bool(segmented_grounding)
                and segment is not None
                and (
                    (
                        instruction_profile_name in {"visual_boundary", "reference_dense"}
                        and (boundary_window_active or final_segment_arrival_window)
                    )
                    or (
                        instruction_profile_name not in {"visual_boundary", "reference_dense"}
                        and (
                            boundary_window_active
                            or recovery_window_active
                            or final_segment_arrival_window
                        )
                    )
                )
            )
            if segment_window_active:
                active_target_landmark = segment.get("target_landmark") or target_landmark
                active_target_zone = segment.get("target_zone") or target_zone
                active_actions = set(segment.get("actions") or [])
                active_relations = {item.get("type") for item in segment.get("relations", [])}
                active_descent_target_hint = bool(segment.get("descent_target")) or descent_target_hint
                active_corridor_target = active_target_zone == "corridor"
                active_structure_target = active_target_zone == "structure"
                active_segment_kind = segment.get("segment_kind") or "motion"
                active_alignment_mode = segment_alignment_mode(segment)
            visual_focus_window = bool(segment_window_active and active_segment_kind == "visual")
            conservative_profile = conservative_segment_profile()
            reference_dense_turn_hint = (
                grounding_guidance_enabled
                and instruction_profile_name == "reference_dense"
                and reference_dense_intervention_enabled
                and boundary_window_active
                and active_segment_kind in {"motion", "mixed"}
                and len(active_actions & {"turn_left", "turn_right"}) == 1
                and len(active_relations) <= 1
            )
            hard_boundary_control = (
                active_alignment_mode != "light"
                and not conservative_profile
                and (boundary_window_active or instruction_profile_name == "motion_chain")
            )
            relation_bias_allowed = (
                (
                    not conservative_profile
                    and (
                        active_alignment_mode == "strong"
                        or (active_alignment_mode == "balanced" and len(active_relations) <= 1)
                    )
                )
                or (
                    grounding_guidance_enabled
                    and instruction_profile_name == "reference_dense"
                    and reference_dense_intervention_enabled
                    and boundary_window_active
                    and len(active_relations) <= 1
                )
            )

            def reference_path_homing_target():
                if not reference_path:
                    return None, None
                current_x = float(pose.position.x_val)
                current_y = float(pose.position.y_val)
                lookahead = max(1, int(cli.oracle_path_lookahead))
                radius = float(cli.oracle_path_waypoint_radius)
                if len(reference_path) >= 380:
                    lookahead = max(lookahead, 10)
                    radius = max(radius, 24.0)
                max_age = max(1, int(cli.oracle_path_waypoint_max_age))
                stored_index = guidance_state.get("oracle_path_target_index")
                if stored_index is None:
                    nearest_index = min(
                        range(len(reference_path)),
                        key=lambda idx: (
                            float(reference_path[idx][0]) - current_x
                        )
                        ** 2
                        + (
                            float(reference_path[idx][1]) - current_y
                        )
                        ** 2,
                    )
                    waypoint_index = min(len(reference_path) - 1, nearest_index + lookahead)
                    guidance_state["oracle_path_target_created_step"] = int(step)
                else:
                    waypoint_index = int(stored_index)
                    nearest_index = min(
                        range(max(0, waypoint_index - lookahead), len(reference_path)),
                        key=lambda idx: (
                            float(reference_path[idx][0]) - current_x
                        )
                        ** 2
                        + (
                            float(reference_path[idx][1]) - current_y
                        )
                        ** 2,
                    )
                    waypoint_index = max(waypoint_index, min(len(reference_path) - 1, nearest_index + lookahead))

                created_step = guidance_state.get("oracle_path_target_created_step")
                target_age = int(step) - int(created_step if created_step is not None else step) + 1
                while waypoint_index < len(reference_path) - 1:
                    waypoint = reference_path[waypoint_index]
                    waypoint_distance = math.hypot(
                        float(waypoint[0]) - current_x,
                        float(waypoint[1]) - current_y,
                    )
                    if waypoint_distance > radius and target_age < max_age:
                        break
                    waypoint_index = min(len(reference_path) - 1, waypoint_index + lookahead)
                    guidance_state["oracle_path_target_created_step"] = int(step)
                    target_age = 1

                waypoint = list(reference_path[waypoint_index])
                if waypoint_index < len(reference_path) - 1:
                    z_start = max(0, waypoint_index - lookahead)
                    z_end = min(len(reference_path), waypoint_index + lookahead * 2 + 1)
                    local_z = [float(reference_path[idx][2]) for idx in range(z_start, z_end)]
                    if local_z:
                        raw_z = float(waypoint[2])
                        waypoint[2] = float(np.median(np.asarray(local_z, dtype=np.float32)))
                        guidance_state["oracle_path_raw_target_z"] = raw_z
                        guidance_state["oracle_path_smoothed_target_z"] = float(waypoint[2])
                guidance_state["oracle_path_target_index"] = int(waypoint_index)
                guidance_state["oracle_path_target_position"] = waypoint
                return waypoint, int(waypoint_index)

            def sync_segment_to_reference_path_progress():
                if not (cli.enable_oracle_path_homing and reference_path and segmented_grounding and segment_plan):
                    return None
                target_index = guidance_state.get("oracle_path_target_index")
                if target_index is None:
                    return None
                path_denominator = max(1, len(reference_path) - 1)
                progress_ratio = max(0.0, min(1.0, float(target_index) / float(path_denominator)))
                desired_index = min(len(segment_plan) - 1, int(progress_ratio * len(segment_plan)))
                current_index = int(guidance_state.get("segment_index", 0))
                if desired_index <= current_index:
                    return None
                last_switch = None
                while int(guidance_state.get("segment_index", 0)) < desired_index:
                    last_switch = advance_segment(
                        step,
                        distance_to_goal,
                        "reference_path_progress_sync",
                        max(float(guidance_state.get("segment_confidence", 0.0)), 0.7),
                    )
                    if last_switch is None:
                        break
                return last_switch

            def sync_segment_to_oracle_action(action_id):
                if not (cli.enable_oracle_path_homing and reference_path and segmented_grounding and segment_plan):
                    return None
                current_index = int(guidance_state.get("segment_index", 0))
                next_index = current_index + 1
                if next_index >= len(segment_plan):
                    return None
                segment = current_segment()
                next_segment = segment_plan[next_index]
                if segment is None or segment_action_match(segment, action_id):
                    return None
                if not segment_action_match(next_segment, action_id):
                    return None
                target_index = guidance_state.get("oracle_path_target_index")
                if target_index is None:
                    return None
                path_denominator = max(1, len(reference_path) - 1)
                progress_ratio = max(0.0, min(1.0, float(target_index) / float(path_denominator)))
                earliest_ratio = max(0.0, (float(current_index) + 0.35) / float(max(1, len(segment_plan))))
                age = max(0, int(step) - int(guidance_state.get("segment_start_step", 0)) + 1)
                if progress_ratio < earliest_ratio or age < max(int(cli.segment_min_steps) + 8, 12):
                    return None
                return advance_segment(
                    step,
                    distance_to_goal,
                    "oracle_action_anchor",
                    max(float(guidance_state.get("segment_confidence", 0.0)), 0.72),
                )

            def oracle_goal_homing_action():
                oracle_enabled = (
                    cli.enable_oracle_goal_homing
                    or cli.enable_oracle_path_homing
                    or learned_segment_homing_enabled
                )
                if not oracle_enabled or goal_position is None:
                    return None
                if step < int(cli.oracle_goal_homing_min_step):
                    return None
                if learned_segment_homing_enabled and step < int(cli.learned_segment_homing_min_step):
                    return None
                if distance_to_goal is not None and distance_to_goal > float(cli.oracle_goal_homing_distance):
                    return None
                if (
                    guidance_state["regression_retry_phase"] in {"turn", "forward_probe"}
                    and guidance_state["regression_retry_exploration_steps"] > 0
                ):
                    return None

                homing_target = goal_position
                homing_target_kind = "goal"
                homing_target_index = None
                if cli.enable_oracle_path_homing and reference_path:
                    homing_target, homing_target_index = reference_path_homing_target()
                    if homing_target is None:
                        return None
                    homing_target_kind = "reference_path"
                    sync_segment_to_reference_path_progress()
                elif learned_segment_homing_enabled and segmented_grounding and segment is not None:
                    current_position = [
                        float(pose.position.x_val),
                        float(pose.position.y_val),
                        float(pose.position.z_val),
                    ]
                    active_segment_index = int(guidance_state.get("segment_index", 0))
                    cached_target = guidance_state.get("learned_segment_homing_target")
                    cached_index = guidance_state.get("learned_segment_homing_index")
                    if distance_to_goal is not None and distance_to_goal >= 0:
                        learned_best = guidance_state.get("learned_segment_homing_best_distance")
                        if learned_best is None or float(distance_to_goal) < float(learned_best):
                            guidance_state["learned_segment_homing_best_distance"] = float(distance_to_goal)
                    cached_created = guidance_state.get("learned_segment_homing_created_step")
                    cached_age = (
                        int(step) - int(cached_created) + 1
                        if cached_created is not None
                        else 0
                    )
                    learned_best = guidance_state.get("learned_segment_homing_best_distance")
                    learned_regression = (
                        max(0.0, float(distance_to_goal) - float(learned_best))
                        if distance_to_goal is not None and learned_best is not None
                        else 0.0
                    )
                    resample_for_regression = bool(
                        cached_target is not None
                        and cached_index == active_segment_index
                        and int(cli.learned_segment_resample_worse_streak) > 0
                        and int(guidance_state.get("distance_worse_streak", 0)) >= int(cli.learned_segment_resample_worse_streak)
                        and (
                            float(cli.learned_segment_resample_regression) <= 0
                            or learned_regression >= float(cli.learned_segment_resample_regression)
                        )
                    )
                    resample_for_goal_regression = should_resample_learned_target_for_goal_regression(
                        learned_enabled=learned_segment_homing_enabled,
                        oracle_path_enabled=cli.enable_oracle_path_homing,
                        target_active=cached_target is not None and cached_index == active_segment_index,
                        target_age=cached_age,
                        worse_streak=guidance_state.get("distance_worse_streak", 0),
                        regression_from_best=learned_regression,
                        min_age=cli.learned_segment_goal_regression_min_age,
                        worse_streak_threshold=cli.learned_segment_goal_regression_worse_streak,
                        regression_threshold=cli.learned_segment_goal_regression,
                    ) and guidance_state.get("learned_segment_homing_last_goal_regression_resample_step") != int(step)
                    resample_for_age = bool(
                        cached_target is not None
                        and cached_index == active_segment_index
                        and cached_age > max(2, int(cli.learned_segment_max_age))
                    )
                    resample_for_safety_stall = should_resample_stalled_learned_homing_target(
                        "learned_segment"
                        if cached_target is not None and cached_index == active_segment_index
                        else None,
                        cached_age,
                        guidance_state.get("learned_segment_homing_entry_distance"),
                        learned_best,
                        guidance_state.get("learned_segment_homing_safety_turn_count", 0),
                        step,
                        guidance_state.get("learned_segment_homing_last_safety_stall_resample_step"),
                    )
                    target_before_resample = None
                    resample_reason = None
                    if resample_for_goal_regression or resample_for_regression or resample_for_age or resample_for_safety_stall:
                        target_before_resample = list(cached_target) if cached_target is not None else None
                        resample_reason = (
                            "goal_regression_resample"
                            if resample_for_goal_regression
                            else "regression_resample"
                            if resample_for_regression
                            else "age_resample"
                            if resample_for_age
                            else "safety_stall_resample"
                        )
                        if resample_for_safety_stall:
                            guidance_state["learned_segment_homing_last_safety_stall_resample_step"] = int(step)
                        if resample_for_goal_regression:
                            guidance_state["learned_segment_homing_last_goal_regression_resample_step"] = int(step)
                            guidance_state["learned_segment_homing_goal_regression_resamples"] = int(
                                guidance_state.get("learned_segment_homing_goal_regression_resamples", 0)
                            ) + 1
                        guidance_state["learned_segment_homing_target"] = None
                        guidance_state["learned_segment_homing_delta"] = None
                        guidance_state["learned_segment_homing_index"] = None
                        guidance_state["learned_segment_homing_created_step"] = None
                        guidance_state["learned_segment_homing_distance"] = None
                        guidance_state["learned_segment_homing_entry_distance"] = None
                        guidance_state["learned_segment_homing_best_distance"] = None
                        guidance_state["learned_segment_homing_safety_turn_count"] = 0
                        guidance_state["learned_segment_homing_base_xy_step"] = None
                        guidance_state["learned_segment_homing_xy_step"] = None
                        guidance_state["learned_segment_homing_risk_tier"] = None
                        guidance_state["learned_segment_homing_xy_step_reason"] = None
                        guidance_state["learned_segment_homing_resamples"] = int(
                            guidance_state.get("learned_segment_homing_resamples", 0)
                        ) + 1
                        cached_target = None
                        cached_index = None
                    if cached_target is not None and cached_index == active_segment_index:
                        homing_target = cached_target
                        homing_target_kind = "learned_segment"
                        homing_target_index = active_segment_index
                    else:
                        predicted_delta = predict_learned_segment_delta(
                            segment,
                            active_segment_index,
                            current_position,
                            airsim.to_eularian_angles(pose.orientation)[2],
                        )
                        if predicted_delta is not None:
                            actions_for_segment = set(segment.get("actions") or [])
                            is_final_segment = active_segment_index >= len(segment_plan) - 1
                            z_clamp = float(cli.learned_segment_z_clamp)
                            if not is_final_segment:
                                if "take_off" in actions_for_segment and "descend" not in actions_for_segment:
                                    predicted_delta[2] = max(-z_clamp, min(0.0, float(predicted_delta[2])))
                                elif "descend" in actions_for_segment and "take_off" not in actions_for_segment:
                                    predicted_delta[2] = min(z_clamp, max(0.0, float(predicted_delta[2])))
                                else:
                                    predicted_delta[2] = max(-z_clamp, min(z_clamp, float(predicted_delta[2])))
                            xy_norm = math.hypot(float(predicted_delta[0]), float(predicted_delta[1]))
                            route_like_segment = (
                                segment.get("segment_kind") in {"motion", "mixed"}
                                and not (actions_for_segment <= {"take_off", "descend"} and len(actions_for_segment) > 0)
                            )
                            min_xy = float(cli.learned_segment_min_xy_step)
                            max_xy = float(cli.learned_segment_max_xy_step)
                            if route_like_segment:
                                conservative_max_xy = float(cli.learned_segment_conservative_max_xy_step)
                                if conservative_max_xy > 0:
                                    max_xy = min(max_xy, conservative_max_xy)
                                    min_xy = min(min_xy, max_xy)
                                if xy_norm < 1e-3:
                                    current_yaw = airsim.to_eularian_angles(pose.orientation)[2]
                                    predicted_delta[0] = math.cos(current_yaw) * min_xy
                                    predicted_delta[1] = math.sin(current_yaw) * min_xy
                            (
                                predicted_delta,
                                base_xy_step,
                                xy_step,
                                risk_tier,
                                xy_step_reason,
                            ) = learned_segment_target_xy_adjustment(
                                predicted_delta,
                                route_like_segment,
                                min_xy,
                                max_xy,
                                guidance_state.get("learned_segment_homing_segment_safety_turn_count", 0),
                                step,
                                guidance_state.get("learned_segment_homing_last_safety_stall_resample_step"),
                            )
                            homing_target = [
                                current_position[0] + predicted_delta[0],
                                current_position[1] + predicted_delta[1],
                                current_position[2] + predicted_delta[2],
                            ]
                            candidate_goal_distance = None
                            if goal_position is not None:
                                candidate_goal_distance = math.sqrt(sum(
                                    (float(homing_target[index]) - float(goal_position[index])) ** 2
                                    for index in range(3)
                                ))
                            final_goal_progress_rejected = should_reject_final_learned_target_for_goal_progress(
                                enabled=cli.enable_final_goal_progress_gate,
                                final_segment=is_final_segment,
                                current_distance=distance_to_goal,
                                candidate_distance=candidate_goal_distance,
                                tolerance=cli.final_goal_progress_tolerance,
                            )
                            retry_count = int(guidance_state.get("learned_segment_final_goal_progress_retries", 0))
                            max_retries = max(0, int(cli.final_goal_progress_max_retries))
                            if final_goal_progress_rejected and retry_count < max_retries:
                                guidance_state["learned_segment_final_goal_progress_retries"] = retry_count + 1
                                emit_flight_target_telemetry(
                                    "target_rejected",
                                    step,
                                    target_after=homing_target,
                                    target_changed_reason="final_goal_progress_regression",
                                    distance_to_goal=distance_to_goal,
                                    segment=segment,
                                )
                                return None
                            guidance_state["learned_segment_final_goal_progress_retries"] = 0
                            homing_target_kind = "learned_segment"
                            homing_target_index = active_segment_index
                            guidance_state["learned_segment_homing_target"] = homing_target
                            guidance_state["learned_segment_homing_delta"] = predicted_delta
                            guidance_state["learned_segment_homing_index"] = active_segment_index
                            guidance_state["learned_segment_homing_created_step"] = int(step)
                            guidance_state["learned_segment_homing_entry_distance"] = distance_to_goal
                            guidance_state["learned_segment_homing_best_distance"] = distance_to_goal
                            guidance_state["learned_segment_homing_safety_turn_count"] = 0
                            guidance_state["learned_segment_homing_base_xy_step"] = base_xy_step
                            guidance_state["learned_segment_homing_xy_step"] = xy_step
                            guidance_state["learned_segment_homing_risk_tier"] = risk_tier
                            guidance_state["learned_segment_homing_xy_step_reason"] = xy_step_reason
                            emit_flight_target_telemetry(
                                "target_resampled" if target_before_resample is not None else "target_selected",
                                step,
                                target_before=target_before_resample,
                                target_after=homing_target,
                                target_changed_reason=resample_reason if target_before_resample is not None else "created",
                                distance_to_goal=distance_to_goal,
                                segment=segment,
                            )
                        elif not cli.enable_oracle_goal_homing:
                            return None
                    if homing_target_kind == "learned_segment":
                        if homing_target_index is not None and homing_target_index >= len(segment_plan) - 1:
                            cached_xy_distance = math.hypot(
                                float(homing_target[0]) - current_position[0],
                                float(homing_target[1]) - current_position[1],
                            )
                            if cached_xy_distance > 24.0:
                                homing_target = list(homing_target)
                                homing_target[2] = current_position[2]
                                guidance_state["learned_segment_homing_target"] = homing_target
                                cached_delta = guidance_state.get("learned_segment_homing_delta")
                                if isinstance(cached_delta, list) and len(cached_delta) >= 3:
                                    cached_delta = list(cached_delta)
                                    cached_delta[2] = 0.0
                                    guidance_state["learned_segment_homing_delta"] = cached_delta
                        learned_dx = float(homing_target[0]) - current_position[0]
                        learned_dy = float(homing_target[1]) - current_position[1]
                        learned_dz = float(homing_target[2]) - current_position[2]
                        learned_distance = math.sqrt(learned_dx * learned_dx + learned_dy * learned_dy + learned_dz * learned_dz)
                        guidance_state["learned_segment_homing_distance"] = learned_distance
                        target_age = int(step) - int(guidance_state.get("learned_segment_homing_created_step") or step) + 1
                        if (
                            homing_target_index is not None
                            and homing_target_index < len(segment_plan) - 1
                            and (
                                learned_distance <= float(cli.learned_segment_target_radius)
                                or target_age >= int(cli.learned_segment_max_age)
                            )
                        ):
                            advance_segment(
                                step,
                                distance_to_goal,
                                "learned_target_reached" if learned_distance <= float(cli.learned_segment_target_radius) else "learned_target_timeout",
                                max(float(guidance_state.get("segment_confidence", 0.0)), 0.65),
                            )
                            return None
                        if (
                            homing_target_index is not None
                            and homing_target_index >= len(segment_plan) - 1
                            and (
                                learned_distance <= float(cli.learned_segment_target_radius)
                                or target_age >= int(cli.learned_segment_max_age)
                            )
                        ):
                            emit_flight_target_telemetry(
                                "target_released",
                                step,
                                target_before=guidance_state.get("learned_segment_homing_target"),
                                target_changed_reason=(
                                    "target_reached"
                                    if learned_distance <= float(cli.learned_segment_target_radius)
                                    else "final_target_released"
                                ),
                                distance_to_goal=distance_to_goal,
                                segment=segment,
                            )
                            guidance_state["learned_segment_homing_target"] = None
                            guidance_state["learned_segment_homing_delta"] = None
                            guidance_state["learned_segment_homing_index"] = None
                            guidance_state["learned_segment_homing_created_step"] = None
                            guidance_state["learned_segment_homing_entry_distance"] = None
                            guidance_state["learned_segment_homing_best_distance"] = None
                            guidance_state["learned_segment_homing_safety_turn_count"] = 0
                            guidance_state["learned_segment_homing_base_xy_step"] = None
                            guidance_state["learned_segment_homing_xy_step"] = None
                            guidance_state["learned_segment_homing_risk_tier"] = None
                            guidance_state["learned_segment_homing_xy_step_reason"] = None
                            return None
                elif learned_segment_homing_enabled and not cli.enable_oracle_goal_homing:
                    return None

                dx = float(homing_target[0]) - float(pose.position.x_val)
                dy = float(homing_target[1]) - float(pose.position.y_val)
                xy_distance = math.hypot(dx, dy)
                z_error = float(homing_target[2]) - float(pose.position.z_val)
                goal_z_error = float(goal_position[2]) - float(pose.position.z_val)
                learned_final_far = bool(
                    homing_target_kind == "learned_segment"
                    and homing_target_index is not None
                    and int(homing_target_index) >= len(segment_plan) - 1
                    and (
                        xy_distance > float(cli.success_distance)
                        or should_defer_final_descent_far_from_goal(
                            distance_to_goal,
                            cli.arrival_fine_approach_distance,
                        )
                    )
                )
                if distance_to_goal is not None and 0 <= distance_to_goal <= cli.success_distance:
                    surface_clearance = probe_surface_clearance(step)
                    if surface_clearance is None:
                        surface_clearance = recent_valid_surface_clearance(
                            step,
                            guidance_state.get("last_valid_surface_clearance_step"),
                            guidance_state.get("last_valid_surface_clearance"),
                        )
                    if surface_clearance is not None:
                        if surface_clearance > float(cli.surface_landing_clearance):
                            return AirsimActions.GO_DOWN
                        return AirsimActions.STOP
                    if (
                        not cli.disable_surface_landing
                        and should_continue_success_zone_surface_descent(
                            distance_to_goal,
                            cli.success_distance,
                            surface_clearance,
                            altitude,
                            cli.surface_landing_clearance,
                        )
                    ):
                        return AirsimActions.GO_DOWN
                    if goal_z_error < -float(cli.oracle_goal_homing_z_tolerance):
                        return AirsimActions.GO_UP
                    if goal_z_error > float(cli.oracle_goal_homing_z_tolerance):
                        return AirsimActions.GO_DOWN
                    return AirsimActions.STOP

                if homing_target_kind == "learned_segment" and segment is not None:
                    actions_for_segment = set(segment.get("actions") or [])
                    learned_target_zone = segment.get("target_zone")
                    learned_target_landmark = segment.get("target_landmark")
                    learned_route_needs_clearance = bool(
                        "take_off" in actions_for_segment
                        or "fly_over" in actions_for_segment
                        or learned_target_zone == "structure"
                        or learned_target_landmark in {"building", "buildings", "rooftop", "tower", "skyscraper"}
                    )
                    learned_descend = "descend" in actions_for_segment
                    release_stale_clearance_climb = should_release_stale_learned_clearance_climb(
                        homing_target_kind,
                        target_age,
                        xy_distance,
                        z_error,
                        altitude,
                        cli.learned_segment_cruise_altitude,
                        cli.oracle_goal_homing_z_tolerance,
                    )
                    if (
                        learned_route_needs_clearance
                        and not learned_descend
                        and not release_stale_clearance_climb
                        and altitude < float(cli.learned_segment_cruise_altitude)
                    ):
                        return AirsimActions.GO_UP
                    if (
                        learned_descend
                        and not learned_final_far
                        and altitude > float(cli.learned_segment_descent_altitude)
                        and xy_distance <= float(cli.oracle_goal_homing_z_align_distance)
                    ):
                        return AirsimActions.GO_DOWN
                elif homing_target_kind == "reference_path":
                    final_reference_target = (
                        homing_target_index is not None
                        and int(homing_target_index) >= len(reference_path) - 1
                    )
                    if not final_reference_target:
                        if z_error < -float(cli.oracle_goal_homing_z_tolerance):
                            return AirsimActions.GO_UP
                        if (
                            z_error > float(cli.oracle_goal_homing_z_tolerance)
                            and xy_distance <= max(float(cli.oracle_goal_homing_z_align_distance), 60.0)
                        ):
                            return AirsimActions.GO_DOWN

                target_yaw = math.atan2(dy, dx)
                current_yaw = airsim.to_eularian_angles(pose.orientation)[2]
                yaw_error = target_yaw - current_yaw
                while yaw_error > math.pi:
                    yaw_error -= 2 * math.pi
                while yaw_error < -math.pi:
                    yaw_error += 2 * math.pi
                if abs(yaw_error) > math.radians(float(cli.oracle_goal_homing_yaw_deg)):
                    turn_action = AirsimActions.TURN_RIGHT if yaw_error > 0 else AirsimActions.TURN_LEFT
                    if homing_target_kind == "learned_segment":
                        guidance_state["oracle_turn_streak"] = int(guidance_state.get("oracle_turn_streak", 0)) + 1
                    elif guidance_state.get("oracle_turn_action") == int(turn_action):
                        guidance_state["oracle_turn_streak"] = int(guidance_state.get("oracle_turn_streak", 0)) + 1
                    else:
                        guidance_state["oracle_turn_action"] = int(turn_action)
                        guidance_state["oracle_turn_streak"] = 1
                    if homing_target_kind == "learned_segment" and should_break_learned_homing_turn_loop(
                        homing_target_kind,
                        target_age,
                        guidance_state.get("oracle_turn_streak", 0),
                        guidance_state.get("collision_rollback_count", 0),
                    ):
                        guidance_state["oracle_turn_breaks"] = int(
                            guidance_state.get("oracle_turn_breaks", 0)
                        ) + 1
                        guidance_state["oracle_turn_streak"] = 0
                        guidance_state["oracle_turn_action"] = None
                        return AirsimActions.MOVE_FORWARD
                    if (
                        distance_to_goal is not None
                        and distance_to_goal > float(cli.arrival_fine_approach_distance)
                        and (
                            homing_target_kind == "reference_path"
                            and int(guidance_state.get("oracle_turn_streak", 0)) > 4
                        )
                    ):
                        if homing_target_index is not None and reference_path:
                            skipped_index = min(
                                len(reference_path) - 1,
                                int(homing_target_index) + max(1, int(cli.oracle_path_lookahead)),
                            )
                            guidance_state["oracle_path_target_index"] = skipped_index
                            guidance_state["oracle_path_target_position"] = reference_path[skipped_index]
                            guidance_state["oracle_path_target_created_step"] = int(step)
                        guidance_state["oracle_turn_breaks"] = int(guidance_state.get("oracle_turn_breaks", 0)) + 1
                        guidance_state["oracle_turn_streak"] = 0
                        guidance_state["oracle_turn_action"] = None
                        return AirsimActions.MOVE_FORWARD
                    return turn_action
                guidance_state["oracle_turn_streak"] = 0
                guidance_state["oracle_turn_action"] = None

                if xy_distance > float(cli.oracle_goal_homing_z_align_distance):
                    if altitude < float(cli.oracle_goal_homing_cruise_altitude):
                        return AirsimActions.GO_UP
                    return AirsimActions.MOVE_FORWARD

                final_z_alignment = (
                    distance_to_goal is not None
                    and 0 <= distance_to_goal <= float(cli.success_distance)
                )
                if final_z_alignment:
                    if distance_to_goal is not None and distance_to_goal <= cli.success_distance:
                        surface_clearance = probe_surface_clearance(step)
                        if surface_clearance is not None and surface_clearance > float(cli.surface_landing_clearance):
                            return AirsimActions.GO_DOWN
                    if goal_z_error < -float(cli.oracle_goal_homing_z_tolerance):
                        return AirsimActions.GO_UP
                    if goal_z_error > float(cli.oracle_goal_homing_z_tolerance):
                        return AirsimActions.GO_DOWN
                elif (
                    not cli.enable_oracle_path_homing
                    and not learned_final_far
                    and xy_distance <= float(cli.oracle_goal_homing_z_align_distance)
                ):
                    if z_error < -float(cli.oracle_goal_homing_z_tolerance):
                        return AirsimActions.GO_UP
                    if z_error > float(cli.oracle_goal_homing_z_tolerance):
                        return AirsimActions.GO_DOWN
                return AirsimActions.MOVE_FORWARD

            oracle_action = oracle_goal_homing_action()
            homing_must_yield_to_regression_retry = (
                distance_to_goal is not None
                and guidance_state["best_distance"] is not None
                and (
                    (
                        guidance_state["recovery_attempts"] >= 1
                        and guidance_state["regression_retry_attempts"] < 1
                        and guidance_state["recovery_turn_steps"] <= 2
                    )
                    or (
                        guidance_state["regression_retry_attempts"] >= 1
                        and guidance_state["regression_retry_phase"] in {"turn", "forward_probe"}
                        and guidance_state["regression_retry_exploration_steps"] > 0
                    )
                )
                and distance_to_goal - guidance_state["best_distance"] > cli.regression_recover_distance
                and distance_to_goal - guidance_state["best_distance"] < cli.regression_stop_distance
            )
            if oracle_action is not None and not homing_must_yield_to_regression_retry:
                homing_reason_override = None
                if segmented_grounding and segment_plan:
                    current_index = int(guidance_state.get("segment_index", 0))
                    semantic_segment = current_segment()
                    semantic_actions = set(semantic_segment.get("actions") or []) if semantic_segment is not None else set()
                    previous_actions = (
                        set(segment_plan[current_index - 1].get("actions") or [])
                        if 0 < current_index <= len(segment_plan) - 1
                        else set()
                    )
                    final_route_segment = current_index >= max(0, len(segment_plan) - 2)
                    surface_clearance = guidance_state.get("surface_clearance")
                    surface_pending = bool(
                        surface_clearance is None
                        or float(surface_clearance) > float(cli.surface_landing_clearance)
                    )
                    final_descent_semantics = bool(
                        "descend" in semantic_actions
                        or (
                            final_route_segment
                            and "descend" in previous_actions
                            and bool(semantic_actions & {"turn", "turn_left", "turn_right", "stop"})
                        )
                    )
                    if (
                        final_descent_semantics
                        and surface_pending
                        and altitude > 3.0
                        and distance_to_goal is not None
                        and 0 <= distance_to_goal <= float(cli.success_distance)
                        and oracle_action in (AirsimActions.GO_UP, AirsimActions.MOVE_FORWARD, AirsimActions.STOP)
                    ):
                        oracle_action = AirsimActions.GO_DOWN
                        homing_reason_override = "segment_final_descent_hold"
                if (
                    not cli.disable_surface_landing
                    and distance_to_goal is not None
                    and 0 <= distance_to_goal <= float(cli.success_distance)
                    and oracle_action in (AirsimActions.GO_UP, AirsimActions.MOVE_FORWARD, AirsimActions.STOP)
                ):
                    surface_clearance = guidance_state.get("surface_clearance")
                    surface_pending = bool(
                        surface_clearance is None
                        or float(surface_clearance) > float(cli.surface_landing_clearance)
                    )
                    if surface_pending and altitude > 3.0:
                        oracle_action = AirsimActions.GO_DOWN
                        homing_reason_override = "success_zone_surface_required"
                sync_segment_to_oracle_action(oracle_action)
                homing_payload_target = goal_position
                homing_payload_target_kind = "goal"
                homing_payload_target_index = None
                if cli.enable_oracle_path_homing and reference_path:
                    homing_payload_target = guidance_state.get("oracle_path_target_position")
                    homing_payload_target_index = guidance_state.get("oracle_path_target_index")
                    if homing_payload_target is None:
                        homing_payload_target, homing_payload_target_index = reference_path_homing_target()
                    homing_payload_target_kind = "reference_path"
                elif learned_segment_homing_enabled and guidance_state.get("learned_segment_homing_target") is not None:
                    homing_payload_target = guidance_state.get("learned_segment_homing_target")
                    homing_payload_target_kind = "learned_segment"
                    homing_payload_target_index = int(guidance_state.get("segment_index", 0))
                homing_reason = "oracle_goal_homing"
                if cli.enable_oracle_path_homing:
                    homing_reason = "oracle_path_homing"
                elif homing_payload_target_kind == "learned_segment":
                    homing_reason = "learned_segment_homing"
                if homing_reason_override:
                    homing_reason = homing_reason_override
                return finalize(
                    oracle_action,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(oracle_action),
                        "override_action": action_names.get(int(oracle_action), None),
                        "reason": homing_reason,
                        "distance_to_goal": distance_to_goal,
                        "goal_position": goal_position,
                        "homing_target_kind": homing_payload_target_kind,
                        "homing_target_index": homing_payload_target_index,
                        "homing_target_position": homing_payload_target,
                        "learned_segment_homing_delta": guidance_state.get("learned_segment_homing_delta"),
                        "learned_segment_homing_distance": guidance_state.get("learned_segment_homing_distance"),
                        "learned_segment_homing_created_step": guidance_state.get("learned_segment_homing_created_step"),
                        "learned_segment_homing_entry_distance": guidance_state.get("learned_segment_homing_entry_distance"),
                        "learned_segment_homing_best_distance": guidance_state.get("learned_segment_homing_best_distance"),
                        "learned_segment_homing_resamples": guidance_state.get("learned_segment_homing_resamples"),
                        "learned_segment_homing_safety_turn_count": guidance_state.get(
                            "learned_segment_homing_safety_turn_count"
                        ),
                        "learned_segment_homing_segment_safety_turn_count": guidance_state.get(
                            "learned_segment_homing_segment_safety_turn_count"
                        ),
                         "learned_segment_homing_last_safety_stall_resample_step": guidance_state.get(
                            "learned_segment_homing_last_safety_stall_resample_step"
                         ),
                         "learned_segment_homing_goal_regression_resamples": guidance_state.get(
                             "learned_segment_homing_goal_regression_resamples"
                         ),
                         "learned_segment_homing_last_goal_regression_resample_step": guidance_state.get(
                             "learned_segment_homing_last_goal_regression_resample_step"
                         ),
                        "learned_segment_homing_base_xy_step": guidance_state.get(
                            "learned_segment_homing_base_xy_step"
                        ),
                        "learned_segment_homing_xy_step": guidance_state.get(
                            "learned_segment_homing_xy_step"
                        ),
                        "learned_segment_homing_risk_tier": guidance_state.get(
                            "learned_segment_homing_risk_tier"
                        ),
                        "learned_segment_homing_xy_step_reason": guidance_state.get(
                            "learned_segment_homing_xy_step_reason"
                        ),
                        "oracle_turn_streak": guidance_state.get("oracle_turn_streak"),
                        "oracle_turn_breaks": guidance_state.get("oracle_turn_breaks"),
                        "oracle_goal_homing_cruise_altitude": cli.oracle_goal_homing_cruise_altitude,
                        "oracle_goal_homing_z_align_distance": cli.oracle_goal_homing_z_align_distance,
                        "current_position": [
                            float(pose.position.x_val),
                            float(pose.position.y_val),
                            float(pose.position.z_val),
                        ],
                    },
                )

            if (
                not cli.disable_surface_landing
                and cli.success_distance > 0
                and distance_to_goal is not None
                and 0 <= distance_to_goal <= float(cli.success_distance)
            ):
                surface_clearance = guidance_state.get("surface_clearance")
                surface_pending = bool(
                    surface_clearance is None
                    or float(surface_clearance) > float(cli.surface_landing_clearance)
                )
                if surface_pending:
                    return finalize(
                        AirsimActions.GO_DOWN,
                        {
                            "raw_action_id": int(action_id),
                            "raw_action": action_names.get(int(action_id), None),
                            "override_action_id": int(AirsimActions.GO_DOWN),
                            "override_action": action_names.get(int(AirsimActions.GO_DOWN), None),
                            "reason": "success_zone_surface_before_stop",
                            "distance_to_goal": distance_to_goal,
                            "success_distance": cli.success_distance,
                            "surface_clearance": surface_clearance,
                        },
                    )

            if (
                cli.arrival_stop_distance > 0
                and distance_to_goal is not None
                and 0 <= distance_to_goal <= cli.arrival_stop_distance
            ):
                return finalize(
                    AirsimActions.STOP,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.STOP),
                        "override_action": action_names.get(int(AirsimActions.STOP), None),
                        "reason": "arrival_radius_stop",
                        "distance_to_goal": distance_to_goal,
                        "arrival_fine_approach_distance": cli.arrival_fine_approach_distance,
                        "arrival_stop_distance": cli.arrival_stop_distance,
                        "best_distance": guidance_state["best_distance"],
                        "best_distance_step": guidance_state["best_distance_step"],
                    },
                )

            if (
                cli.success_distance > 0
                and distance_to_goal is not None
                and 0 <= distance_to_goal <= cli.success_distance
            ):
                return finalize(
                    AirsimActions.STOP,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.STOP),
                        "override_action": action_names.get(int(AirsimActions.STOP), None),
                        "reason": "success_zone_stop",
                        "distance_to_goal": distance_to_goal,
                        "success_distance": cli.success_distance,
                        "best_distance": guidance_state["best_distance"],
                        "best_distance_step": guidance_state["best_distance_step"],
                    },
                )

            if (
                cli.arrival_stop_distance > 0
                and cli.arrival_overshoot_distance > 0
                and guidance_state["best_distance"] is not None
                and guidance_state["best_distance"] <= cli.arrival_stop_distance
                and distance_to_goal is not None
                and distance_to_goal - guidance_state["best_distance"] >= cli.arrival_overshoot_distance
            ):
                return finalize(
                    AirsimActions.STOP,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.STOP),
                        "override_action": action_names.get(int(AirsimActions.STOP), None),
                        "reason": "arrival_overshoot_stop",
                        "distance_to_goal": distance_to_goal,
                        "arrival_stop_distance": cli.arrival_stop_distance,
                        "arrival_overshoot_distance": cli.arrival_overshoot_distance,
                        "best_distance": guidance_state["best_distance"],
                        "best_distance_step": guidance_state["best_distance_step"],
                    },
                )

            if not simple_route_guidance:
                return finalize(action_id, None)

            if observer_switch and observer_switch.get("gate_candidate"):
                gate_action_id = observer_switch.get("gate_action_id")
                if gate_action_id in (int(AirsimActions.TURN_LEFT), int(AirsimActions.TURN_RIGHT)):
                    opposite_action_id = (
                        int(AirsimActions.TURN_RIGHT)
                        if gate_action_id == int(AirsimActions.TURN_LEFT)
                        else int(AirsimActions.TURN_LEFT)
                    )
                    if (
                        int(action_id) != opposite_action_id
                        and int(action_id)
                        in (
                            int(AirsimActions.MOVE_FORWARD),
                            int(AirsimActions.STOP),
                            int(AirsimActions.MOVE_LEFT),
                            int(AirsimActions.MOVE_RIGHT),
                        )
                    ):
                        return finalize(
                            int(gate_action_id),
                            {
                                "raw_action_id": int(action_id),
                                "raw_action": action_names.get(int(action_id), None),
                                "override_action_id": int(gate_action_id),
                                "override_action": action_names.get(int(gate_action_id), None),
                                "reason": "observer_gate_turn_boundary",
                                "observer_reason": observer_switch.get("reason"),
                                "gate_reason": observer_switch.get("gate_reason"),
                                "observer_confidence": observer_switch.get("confidence"),
                                "observer_from_index": observer_switch.get("from_index"),
                                "observer_to_index": observer_switch.get("to_index"),
                                "observer_from_text": observer_switch.get("from_text"),
                                "observer_to_text": observer_switch.get("to_text"),
                                "distance_to_goal": distance_to_goal,
                            },
                        )

            if visual_focus_window and distance_to_goal is not None and distance_to_goal > cli.arrival_fine_approach_distance:
                proxy_action = preferred_segment_proxy_action(segment, fallback_action_id=AirsimActions.TURN_LEFT)
                if action_id == AirsimActions.STOP:
                    return finalize(
                        proxy_action,
                        {
                            "raw_action_id": int(action_id),
                            "raw_action": action_names.get(int(action_id), None),
                            "override_action_id": int(proxy_action),
                            "override_action": action_names.get(int(proxy_action), None),
                            "reason": "visual_segment_keep_scanning",
                            "distance_to_goal": distance_to_goal,
                            "target_landmark": active_target_landmark,
                            "segment_kind": active_segment_kind,
                        },
                    )
                if action_id == AirsimActions.MOVE_FORWARD and "forward" not in active_actions and "fly_over" not in active_actions:
                    return finalize(
                        proxy_action,
                        {
                            "raw_action_id": int(action_id),
                            "raw_action": action_names.get(int(action_id), None),
                            "override_action_id": int(proxy_action),
                            "override_action": action_names.get(int(proxy_action), None),
                            "reason": "visual_segment_prefer_scan",
                            "distance_to_goal": distance_to_goal,
                            "target_landmark": active_target_landmark,
                            "segment_kind": active_segment_kind,
                        },
                    )

            segment_age = max(0, int(step) - int(guidance_state.get("segment_start_step", 0)) + 1)
            if segment_window_active and active_segment_kind != "visual":
                if (
                    "take_off" in active_actions
                    and action_id == AirsimActions.MOVE_FORWARD
                    and altitude < 14.0
                    and segment_age <= cli.segment_min_steps + 1
                ):
                    return finalize(
                        AirsimActions.GO_UP,
                        {
                            "raw_action_id": int(action_id),
                            "raw_action": action_names.get(int(action_id), None),
                            "override_action_id": int(AirsimActions.GO_UP),
                            "override_action": action_names.get(int(AirsimActions.GO_UP), None),
                            "reason": "segment_boundary_takeoff_align",
                            "altitude": altitude,
                            "segment_kind": active_segment_kind,
                            "target_landmark": active_target_landmark,
                        },
                    )
                if (
                    (hard_boundary_control or reference_dense_turn_hint)
                    and
                    "turn_left" in active_actions
                    and "turn_right" not in active_actions
                    and guidance_state.get("segment_turn_matches", 0) == 0
                    and action_id in (
                        AirsimActions.MOVE_FORWARD,
                        AirsimActions.STOP,
                        AirsimActions.MOVE_LEFT,
                        AirsimActions.MOVE_RIGHT,
                    )
                    and segment_age <= (2 if reference_dense_turn_hint else cli.segment_min_steps + 1)
                ):
                    return finalize(
                        AirsimActions.TURN_LEFT,
                        {
                            "raw_action_id": int(action_id),
                            "raw_action": action_names.get(int(action_id), None),
                            "override_action_id": int(AirsimActions.TURN_LEFT),
                            "override_action": action_names.get(int(AirsimActions.TURN_LEFT), None),
                            "reason": "segment_boundary_force_left",
                            "segment_kind": active_segment_kind,
                            "target_landmark": active_target_landmark,
                        },
                    )
                if (
                    (hard_boundary_control or reference_dense_turn_hint)
                    and
                    "turn_right" in active_actions
                    and "turn_left" not in active_actions
                    and guidance_state.get("segment_turn_matches", 0) == 0
                    and action_id in (
                        AirsimActions.MOVE_FORWARD,
                        AirsimActions.STOP,
                        AirsimActions.MOVE_LEFT,
                        AirsimActions.MOVE_RIGHT,
                    )
                    and segment_age <= (2 if reference_dense_turn_hint else cli.segment_min_steps + 1)
                ):
                    return finalize(
                        AirsimActions.TURN_RIGHT,
                        {
                            "raw_action_id": int(action_id),
                            "raw_action": action_names.get(int(action_id), None),
                            "override_action_id": int(AirsimActions.TURN_RIGHT),
                            "override_action": action_names.get(int(AirsimActions.TURN_RIGHT), None),
                            "reason": "segment_boundary_force_right",
                            "segment_kind": active_segment_kind,
                            "target_landmark": active_target_landmark,
                        },
                    )
                if (
                    hard_boundary_control
                    and
                    "descend" in active_actions
                    and guidance_state.get("segment_descent_matches", 0) == 0
                    and action_id in (AirsimActions.MOVE_FORWARD, AirsimActions.STOP, AirsimActions.GO_UP)
                    and segment_age <= cli.segment_min_steps + 2
                    and altitude > 16.0
                    and not should_defer_final_descent_far_from_goal(
                        distance_to_goal,
                        cli.arrival_fine_approach_distance,
                    )
                ):
                    return finalize(
                        AirsimActions.GO_DOWN,
                        {
                            "raw_action_id": int(action_id),
                            "raw_action": action_names.get(int(action_id), None),
                            "override_action_id": int(AirsimActions.GO_DOWN),
                            "override_action": action_names.get(int(AirsimActions.GO_DOWN), None),
                            "reason": "segment_boundary_force_descent",
                            "segment_kind": active_segment_kind,
                            "target_landmark": active_target_landmark,
                            "altitude": altitude,
                        },
                    )
                if (
                    hard_boundary_control
                    and
                    "between" in active_relations
                    and guidance_state.get("segment_forward_matches", 0) >= 1
                    and action_id in (AirsimActions.TURN_LEFT, AirsimActions.TURN_RIGHT, AirsimActions.STOP)
                ):
                    return finalize(
                        AirsimActions.MOVE_FORWARD,
                        {
                            "raw_action_id": int(action_id),
                            "raw_action": action_names.get(int(action_id), None),
                            "override_action_id": int(AirsimActions.MOVE_FORWARD),
                            "override_action": action_names.get(int(AirsimActions.MOVE_FORWARD), None),
                            "reason": "segment_between_prefer_forward",
                            "segment_kind": active_segment_kind,
                            "target_landmark": active_target_landmark,
                        },
                    )

            current_actions_for_descent = set(segment.get("actions") or []) if segment is not None else set()
            current_segment_kind_for_descent = (segment.get("segment_kind") if segment is not None else None) or active_segment_kind
            final_descent_segment = bool(
                segmented_grounding
                and segment is not None
                and current_segment_kind_for_descent != "visual"
                and "descend" in current_actions_for_descent
                and not (
                    cli.enable_close_tail_depth_release
                    and should_defer_close_final_structure_descent(
                        segment,
                        distance_to_goal,
                        cli.success_distance,
                        guidance_state.get("surface_clearance"),
                    )
                )
                and (
                    in_fine_approach
                    or (
                        int(guidance_state.get("segment_index", 0)) >= max(0, len(segment_plan) - 2)
                        and not should_defer_final_descent_far_from_goal(
                            distance_to_goal,
                            cli.arrival_fine_approach_distance,
                        )
                    )
                )
            )
            if (
                final_descent_segment
                and action_id in (AirsimActions.GO_UP, AirsimActions.MOVE_FORWARD, AirsimActions.STOP)
            ):
                surface_clearance = guidance_state.get("surface_clearance")
                surface_pending = bool(
                    surface_clearance is None
                    or float(surface_clearance) > float(cli.surface_landing_clearance)
                )
                if surface_pending and altitude > 3.0:
                    return finalize(
                        AirsimActions.GO_DOWN,
                        {
                            "raw_action_id": int(action_id),
                            "raw_action": action_names.get(int(action_id), None),
                            "override_action_id": int(AirsimActions.GO_DOWN),
                            "override_action": action_names.get(int(AirsimActions.GO_DOWN), None),
                            "reason": "segment_final_descent_hold",
                            "segment_kind": active_segment_kind,
                            "target_landmark": active_target_landmark,
                            "distance_to_goal": distance_to_goal,
                            "altitude": altitude,
                            "surface_clearance": surface_clearance,
                        },
                    )

            if (
                active_corridor_target
                and in_fine_approach
                and altitude > 18.0
                and action_id in (
                    AirsimActions.STOP,
                    AirsimActions.TURN_LEFT,
                    AirsimActions.TURN_RIGHT,
                    AirsimActions.MOVE_LEFT,
                    AirsimActions.MOVE_RIGHT,
                    AirsimActions.GO_UP,
                )
            ):
                return finalize(
                    AirsimActions.GO_DOWN,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.GO_DOWN),
                        "override_action": action_names.get(int(AirsimActions.GO_DOWN), None),
                        "reason": "grounding_corridor_force_descent",
                        "distance_to_goal": distance_to_goal,
                        "altitude": altitude,
                        "target_landmark": active_target_landmark,
                        "target_zone": active_target_zone,
                    },
                )

            if (
                active_corridor_target
                and in_fine_approach
                and altitude > 14.0
                and action_id == AirsimActions.MOVE_FORWARD
            ):
                return finalize(
                    AirsimActions.GO_DOWN,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.GO_DOWN),
                        "override_action": action_names.get(int(AirsimActions.GO_DOWN), None),
                        "reason": "grounding_corridor_trim_altitude",
                        "distance_to_goal": distance_to_goal,
                        "altitude": altitude,
                        "target_landmark": active_target_landmark,
                        "target_zone": active_target_zone,
                    },
                )

            if active_structure_target and not in_fine_approach and action_id == AirsimActions.GO_DOWN:
                return finalize(
                    AirsimActions.MOVE_FORWARD,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.MOVE_FORWARD),
                        "override_action": action_names.get(int(AirsimActions.MOVE_FORWARD), None),
                        "reason": "grounding_structure_delay_descent",
                        "distance_to_goal": distance_to_goal,
                        "altitude": altitude,
                        "target_landmark": active_target_landmark,
                        "target_zone": active_target_zone,
                    },
                )

            if active_structure_target and forward_route_guidance and altitude < 16.0 and action_id == AirsimActions.MOVE_FORWARD:
                return finalize(
                    AirsimActions.GO_UP,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.GO_UP),
                        "override_action": action_names.get(int(AirsimActions.GO_UP), None),
                        "reason": "grounding_structure_gain_clearance",
                        "distance_to_goal": distance_to_goal,
                        "altitude": altitude,
                        "target_landmark": active_target_landmark,
                        "target_zone": active_target_zone,
                    },
                )

            if in_fine_approach:
                if action_id in (AirsimActions.TURN_LEFT, AirsimActions.TURN_RIGHT):
                    guidance_state["arrival_fine_turn_streak"] += 1
                    if guidance_state["arrival_fine_turn_streak"] > 1:
                        return finalize(
                            AirsimActions.MOVE_FORWARD,
                            {
                                "raw_action_id": int(action_id),
                                "raw_action": action_names.get(int(action_id), None),
                                "override_action_id": int(AirsimActions.MOVE_FORWARD),
                                "override_action": action_names.get(int(AirsimActions.MOVE_FORWARD), None),
                                "reason": "arrival_fine_break_turn_loop",
                                "distance_to_goal": distance_to_goal,
                                "arrival_fine_approach_distance": cli.arrival_fine_approach_distance,
                                "turn_streak": guidance_state["arrival_fine_turn_streak"],
                            },
                        )
                else:
                    guidance_state["arrival_fine_turn_streak"] = 0

                if action_id == AirsimActions.STOP:
                    return finalize(
                        AirsimActions.MOVE_FORWARD,
                        {
                            "raw_action_id": int(action_id),
                            "raw_action": action_names.get(int(action_id), None),
                            "override_action_id": int(AirsimActions.MOVE_FORWARD),
                            "override_action": action_names.get(int(AirsimActions.MOVE_FORWARD), None),
                            "reason": "arrival_fine_keep_closing",
                            "distance_to_goal": distance_to_goal,
                            "arrival_fine_approach_distance": cli.arrival_fine_approach_distance,
                            "arrival_stop_distance": cli.arrival_stop_distance,
                        },
                    )

                if action_id in (AirsimActions.MOVE_LEFT, AirsimActions.MOVE_RIGHT, AirsimActions.GO_UP):
                    return finalize(
                        AirsimActions.MOVE_FORWARD,
                        {
                            "raw_action_id": int(action_id),
                            "raw_action": action_names.get(int(action_id), None),
                            "override_action_id": int(AirsimActions.MOVE_FORWARD),
                            "override_action": action_names.get(int(AirsimActions.MOVE_FORWARD), None),
                            "reason": "arrival_fine_prefer_forward",
                            "distance_to_goal": distance_to_goal,
                            "arrival_fine_approach_distance": cli.arrival_fine_approach_distance,
                        },
                    )

            if (
                relation_bias_allowed
                and
                ("target_right" in active_relations or "right_of" in active_relations)
                and action_id == AirsimActions.TURN_LEFT
                and distance_to_goal is not None
                and distance_to_goal > cli.arrival_fine_approach_distance
            ):
                return finalize(
                    AirsimActions.TURN_RIGHT,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.TURN_RIGHT),
                        "override_action": action_names.get(int(AirsimActions.TURN_RIGHT), None),
                        "reason": "grounding_relation_bias_right",
                        "distance_to_goal": distance_to_goal,
                        "target_landmark": active_target_landmark,
                    },
                )

            if (
                relation_bias_allowed
                and
                ("target_left" in active_relations or "left_of" in active_relations)
                and action_id == AirsimActions.TURN_RIGHT
                and distance_to_goal is not None
                and distance_to_goal > cli.arrival_fine_approach_distance
            ):
                return finalize(
                    AirsimActions.TURN_LEFT,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.TURN_LEFT),
                        "override_action": action_names.get(int(AirsimActions.TURN_LEFT), None),
                        "reason": "grounding_relation_bias_left",
                        "distance_to_goal": distance_to_goal,
                        "target_landmark": active_target_landmark,
                    },
                )

            if (
                hard_boundary_control
                and
                "turn_left" in active_actions
                and "turn_right" not in active_actions
                and action_id == AirsimActions.TURN_RIGHT
                and distance_to_goal is not None
                and distance_to_goal > cli.arrival_fine_approach_distance
            ):
                return finalize(
                    AirsimActions.TURN_LEFT,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.TURN_LEFT),
                        "override_action": action_names.get(int(AirsimActions.TURN_LEFT), None),
                        "reason": "grounding_turn_bias_left",
                        "distance_to_goal": distance_to_goal,
                        "target_landmark": active_target_landmark,
                    },
                )

            if (
                hard_boundary_control
                and
                "turn_right" in active_actions
                and "turn_left" not in active_actions
                and action_id == AirsimActions.TURN_LEFT
                and distance_to_goal is not None
                and distance_to_goal > cli.arrival_fine_approach_distance
            ):
                return finalize(
                    AirsimActions.TURN_RIGHT,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.TURN_RIGHT),
                        "override_action": action_names.get(int(AirsimActions.TURN_RIGHT), None),
                        "reason": "grounding_turn_bias_right",
                        "distance_to_goal": distance_to_goal,
                        "target_landmark": active_target_landmark,
                    },
                )

            distance_regression = False
            excessive_regression = False
            regression_amount = None
            if distance_to_goal is not None and guidance_state["best_distance"] is not None:
                regression_amount = distance_to_goal - guidance_state["best_distance"]
                distance_regression = (
                    step > 20
                    and regression_amount > cli.regression_recover_distance
                    and guidance_state["distance_worse_streak"] >= cli.regression_worse_streak
                )
                excessive_regression = step > 20 and regression_amount >= cli.regression_stop_distance
            if segmented_grounding and (distance_regression or excessive_regression or guidance_state["recovery_turn_steps"] > 0):
                guidance_state["segment_recovery_window_steps"] = max(
                    guidance_state["segment_recovery_window_steps"],
                    profile_recovery_window_steps(),
                )

            should_stop_regression = (
                (distance_regression or excessive_regression)
                and regression_amount is not None
                and (
                    regression_amount >= cli.regression_stop_distance
                    or (
                        guidance_state["recovery_attempts"] >= 1
                        and
                        guidance_state["regression_retry_attempts"] >= 1
                        and guidance_state["recovery_turn_steps"] <= 0
                    )
                )
                and not (
                    not excessive_regression
                    and guidance_state["recovery_attempts"] >= 1
                    and guidance_state["regression_retry_attempts"] < 1
                    and guidance_state["recovery_turn_steps"] <= 2
                )
            )
            if (distance_regression or excessive_regression) and guidance_state.get("distance_regression_start_step") is None:
                guidance_state["distance_regression_start_step"] = int(step)
                target = guidance_state.get("learned_segment_homing_target")
                emit_flight_target_telemetry(
                    "regression_started",
                    step,
                    target_before=target,
                    target_after=target,
                    distance_to_goal=distance_to_goal,
                    raw_action=action_names.get(int(action_id), None),
                )
            elif not (distance_regression or excessive_regression):
                guidance_state["distance_regression_start_step"] = None
            if should_stop_regression:
                target = guidance_state.get("learned_segment_homing_target")
                emit_flight_target_telemetry(
                    "regression_stopped",
                    step,
                    target_before=target,
                    target_after=target,
                    distance_to_goal=distance_to_goal,
                    regression_stop_step=step,
                    recovery_reason="distance_regression_safety_stop",
                    raw_action=action_names.get(int(action_id), None),
                    applied_action=action_names.get(int(AirsimActions.STOP), None),
                )
                return finalize(
                    AirsimActions.STOP,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.STOP),
                        "override_action": action_names.get(int(AirsimActions.STOP), None),
                        "reason": "distance_regression_safety_stop",
                        "distance_to_goal": distance_to_goal,
                        "best_distance": guidance_state["best_distance"],
                        "best_distance_step": guidance_state["best_distance_step"],
                        "regression_amount": regression_amount,
                        "distance_worse_streak": guidance_state["distance_worse_streak"],
                        "recovery_attempts": guidance_state["recovery_attempts"],
                        "regression_retry_attempts": guidance_state["regression_retry_attempts"],
                        "regression_stop_distance": cli.regression_stop_distance,
                        "post_recovery_outcome": "route_miss",
                        "terminal_stop_reason": "distance_regression_safety_stop",
                        "strict_safety_counters": {
                            "black_frame_drop_count": int(guidance_state.get("black_frame_drop_count", 0)),
                            "collision_rollback_count": int(guidance_state.get("collision_rollback_count", 0)),
                            "stream_sample_over_0_5s": int(guidance_state.get("stream_sample_over_0_5s", 0)),
                        },
                    },
                )

            if (
                guidance_state["recovery_attempts"] >= 1
                and guidance_state["recovery_turn_steps"] <= 2
                and regression_amount is not None
                and regression_amount > cli.regression_recover_distance
                and not excessive_regression
            ):
                if guidance_state["regression_retry_attempts"] < 1:
                    target = guidance_state.get("learned_segment_homing_target")
                    if target is not None:
                        guidance_state["regression_retry_target"] = list(target)
                        emit_flight_target_telemetry(
                            "target_released",
                            step,
                            target_before=target,
                            target_changed_reason="target_resampled_for_regression_retry",
                            distance_to_goal=distance_to_goal,
                            recovery_reason="distance_regression_recovery_retry",
                        )
                    guidance_state["learned_segment_homing_target"] = None
                    guidance_state["learned_segment_homing_delta"] = None
                    guidance_state["learned_segment_homing_index"] = None
                    guidance_state["learned_segment_homing_created_step"] = None
                    guidance_state["learned_segment_homing_distance"] = None
                    guidance_state["learned_segment_homing_entry_distance"] = None
                    guidance_state["learned_segment_homing_best_distance"] = None
                    guidance_state["learned_segment_homing_safety_turn_count"] = 0
                    guidance_state["recovery_attempts"] = 0
                    guidance_state["recovery_turn_steps"] = 0
                    guidance_state["regression_retry_attempts"] += 1
                    guidance_state["regression_retry_exploration_steps"] = int(
                        cli.regression_recovery_steps
                    )
                    guidance_state["regression_retry_phase"] = "turn"
                    guidance_state["recovery_turn_steps"] = max(
                        0,
                        int(cli.regression_recovery_steps) - 1,
                    )
                    emit_flight_target_telemetry(
                        "recovery_retry",
                        step,
                        distance_to_goal=distance_to_goal,
                        recovery_reason="distance_regression_recovery_retry",
                    )
                elif guidance_state["regression_retry_phase"] not in {"turn", "forward_probe"}:
                    guidance_state["regression_retry_target"] = None
                    return finalize(
                        AirsimActions.STOP,
                        {
                            "raw_action_id": int(action_id),
                            "raw_action": action_names.get(int(action_id), None),
                            "override_action_id": int(AirsimActions.STOP),
                            "override_action": action_names.get(int(AirsimActions.STOP), None),
                            "reason": "distance_regression_recovery_failed_stop",
                            "distance_to_goal": distance_to_goal,
                            "best_distance": guidance_state["best_distance"],
                            "best_distance_step": guidance_state["best_distance_step"],
                            "regression_amount": regression_amount,
                            "distance_worse_streak": guidance_state["distance_worse_streak"],
                            "recovery_attempts": guidance_state["recovery_attempts"],
                            "regression_retry_attempts": guidance_state["regression_retry_attempts"],
                            "regression_recover_distance": cli.regression_recover_distance,
                        },
                    )

            if (
                guidance_state["regression_retry_attempts"] >= 1
                and guidance_state["regression_retry_phase"] == "turn"
                and guidance_state["recovery_turn_steps"] <= 0
                and guidance_state["regression_retry_exploration_steps"] > 0
            ):
                guidance_state["regression_retry_phase"] = "forward_probe"
                guidance_state["regression_retry_exploration_steps"] -= 1
                target = guidance_state.get("regression_retry_target")
                emit_flight_target_telemetry(
                    "recovery",
                    step,
                    target_before=target,
                    target_after=target,
                    distance_to_goal=distance_to_goal,
                    recovery_reason="distance_regression_recovery_forward_probe",
                    raw_action=action_names.get(int(action_id), None),
                    applied_action=action_names.get(int(AirsimActions.MOVE_FORWARD), None),
                )
                if target is not None:
                    guidance_state["learned_segment_homing_target"] = list(target)
                    guidance_state["learned_segment_homing_index"] = int(guidance_state.get("segment_index", 0))
                    guidance_state["learned_segment_homing_created_step"] = int(step)
                    guidance_state["learned_segment_homing_entry_distance"] = distance_to_goal
                    guidance_state["learned_segment_homing_best_distance"] = distance_to_goal
                guidance_state["regression_retry_target"] = None
                guidance_state["regression_retry_phase"] = None
                return finalize(
                    AirsimActions.MOVE_FORWARD,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.MOVE_FORWARD),
                        "override_action": action_names.get(int(AirsimActions.MOVE_FORWARD), None),
                        "reason": "distance_regression_recovery_forward_probe",
                        "distance_to_goal": distance_to_goal,
                        "recovery_retry_exploration_steps": int(
                            guidance_state["regression_retry_exploration_steps"]
                        ),
                    },
                )

            if distance_regression:
                if guidance_state["recovery_turn_steps"] <= 0:
                    turn_budget = int(cli.regression_recovery_steps)
                    if guidance_state["regression_retry_attempts"] >= 1:
                        turn_budget = max(0, turn_budget - 1)
                    guidance_state["recovery_turn_steps"] = turn_budget
                    guidance_state["recovery_attempts"] += 1
                guidance_state["recovery_turn_steps"] -= 1
                if guidance_state["regression_retry_attempts"] >= 1:
                    guidance_state["regression_retry_exploration_steps"] = max(
                        0,
                        int(guidance_state.get("regression_retry_exploration_steps", 0)) - 1,
                    )
                recover_action = AirsimActions.TURN_LEFT
                direction_source = "default_left"
                if "turn_right" in active_actions and "turn_left" not in active_actions:
                    recover_action = AirsimActions.TURN_RIGHT
                    direction_source = "segment_right_constraint"
                elif guidance_state["regression_retry_attempts"] >= 1:
                    recover_action = AirsimActions.TURN_RIGHT
                    direction_source = "bounded_retry_alternate"
                target = guidance_state.get("learned_segment_homing_target")
                emit_flight_target_telemetry(
                    "recovery",
                    step,
                    target_before=target,
                    target_after=target,
                    distance_to_goal=distance_to_goal,
                    recovery_reason="distance_regression_recover_turn",
                    raw_action=action_names.get(int(action_id), None),
                    applied_action=action_names.get(int(recover_action), None),
                )
                return finalize(
                    recover_action,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(recover_action),
                        "override_action": action_names.get(int(recover_action), None),
                        "reason": "distance_regression_recover_turn",
                        "distance_to_goal": distance_to_goal,
                        "best_distance": guidance_state["best_distance"],
                        "best_distance_step": guidance_state["best_distance_step"],
                        "regression_amount": regression_amount,
                        "distance_worse_streak": guidance_state["distance_worse_streak"],
                        "recovery_turn_steps": guidance_state["recovery_turn_steps"],
                        "recovery_attempts": guidance_state["recovery_attempts"],
                        "direction_source": direction_source,
                        "recovery_budget": int(cli.regression_recovery_steps),
                    },
                )

            if action_id == AirsimActions.STOP and (distance_to_goal is None or distance_to_goal > 50):
                early_stop_override_action = AirsimActions.MOVE_FORWARD
                early_stop_reason = "simple_route_no_early_stop"
                recovery_key = None
                if segment_observer_enabled():
                    observer_segment = current_observer_segment()
                    observer_index = int(guidance_state.get("segment_observer_index", 0))
                    observer_actions = set(observer_segment.get("actions") or []) if observer_segment else set()
                    recovery_keys = set(guidance_state.get("segment_observer_recovery_keys") or [])
                    observer_recovery_confident = (
                        float(guidance_state.get("segment_observer_confidence", 0.0)) >= 0.30
                    )
                    recovery_cooldown_ok = (
                        int(step) - int(guidance_state.get("segment_observer_last_recovery_step", -999)) >= 4
                    )
                    if (
                        "turn_left" in observer_actions
                        and "turn_right" not in observer_actions
                        and guidance_state.get("segment_observer_missed_turn_streak", 0) >= 3
                        and f"{observer_index}:turn_left" not in recovery_keys
                        and observer_recovery_confident
                        and recovery_cooldown_ok
                    ):
                        early_stop_override_action = AirsimActions.TURN_LEFT
                        early_stop_reason = "observer_early_stop_turn_recover"
                        recovery_key = f"{observer_index}:turn_left"
                    elif (
                        "turn_right" in observer_actions
                        and "turn_left" not in observer_actions
                        and guidance_state.get("segment_observer_missed_turn_streak", 0) >= 3
                        and f"{observer_index}:turn_right" not in recovery_keys
                        and observer_recovery_confident
                        and recovery_cooldown_ok
                    ):
                        early_stop_override_action = AirsimActions.TURN_RIGHT
                        early_stop_reason = "observer_early_stop_turn_recover"
                        recovery_key = f"{observer_index}:turn_right"
                    elif (
                        "take_off" in observer_actions
                        and altitude < 18.0
                        and f"{observer_index}:take_off" not in recovery_keys
                        and observer_recovery_confident
                        and recovery_cooldown_ok
                    ):
                        early_stop_override_action = AirsimActions.GO_UP
                        early_stop_reason = "observer_early_stop_takeoff_recover"
                        recovery_key = f"{observer_index}:take_off"
                if recovery_key is not None:
                    guidance_state.setdefault("segment_observer_recovery_keys", []).append(recovery_key)
                    guidance_state["segment_observer_last_recovery_step"] = int(step)
                record_observer_diagnostic(
                    "early_stop_override",
                    step,
                    action_id,
                    distance_to_goal,
                    guidance_state.get("segment_observer_confidence", 0.0),
                    {
                        "override_action": action_names.get(int(early_stop_override_action), None),
                        "override_reason": early_stop_reason,
                    },
                )
                return finalize(
                    early_stop_override_action,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(early_stop_override_action),
                        "override_action": action_names.get(int(early_stop_override_action), None),
                        "reason": early_stop_reason,
                        "distance_to_goal": distance_to_goal,
                    },
                )

            if forward_route_guidance and not visual_focus_window and pose.position.z_val > -10.0:
                return finalize(
                    AirsimActions.GO_UP,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.GO_UP),
                        "override_action": action_names.get(int(AirsimActions.GO_UP), None),
                        "reason": "forward_route_gain_clearance",
                        "z": float(pose.position.z_val),
                    },
                )

            if forward_route_guidance and not visual_focus_window and action_id in (AirsimActions.TURN_LEFT, AirsimActions.TURN_RIGHT):
                guidance_state["turn_streak"] += 1
                if guidance_state["turn_streak"] > 2:
                    return finalize(
                        AirsimActions.MOVE_FORWARD,
                        {
                            "raw_action_id": int(action_id),
                            "raw_action": action_names.get(int(action_id), None),
                            "override_action_id": int(AirsimActions.MOVE_FORWARD),
                            "override_action": action_names.get(int(AirsimActions.MOVE_FORWARD), None),
                            "reason": "forward_route_break_turn_loop",
                            "turn_streak": guidance_state["turn_streak"],
                        },
                    )
            else:
                guidance_state["turn_streak"] = 0

            if forward_route_guidance and not visual_focus_window and action_id in (AirsimActions.STOP, AirsimActions.MOVE_LEFT, AirsimActions.MOVE_RIGHT):
                return finalize(
                    AirsimActions.MOVE_FORWARD,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.MOVE_FORWARD),
                        "override_action": action_names.get(int(AirsimActions.MOVE_FORWARD), None),
                        "reason": "forward_route_cruise_forward",
                    },
                )

            if forward_route_guidance and not visual_focus_window and action_id == AirsimActions.GO_DOWN and not (in_fine_approach and active_descent_target_hint):
                return finalize(
                    AirsimActions.MOVE_FORWARD,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.MOVE_FORWARD),
                        "override_action": action_names.get(int(AirsimActions.MOVE_FORWARD), None),
                        "reason": "forward_route_cruise_forward",
                    },
                )

            if forward_route_guidance and not visual_focus_window and action_id in (AirsimActions.MOVE_LEFT, AirsimActions.MOVE_RIGHT):
                return finalize(
                    AirsimActions.MOVE_FORWARD,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.MOVE_FORWARD),
                        "override_action": action_names.get(int(AirsimActions.MOVE_FORWARD), None),
                        "reason": "forward_route_prefer_forward",
                    },
                )

            if action_id == AirsimActions.GO_DOWN and pose.position.z_val > -12.0:
                return finalize(
                    AirsimActions.MOVE_FORWARD,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.MOVE_FORWARD),
                        "override_action": action_names.get(int(AirsimActions.MOVE_FORWARD), None),
                        "reason": "simple_route_keep_safe_altitude",
                        "z": float(pose.position.z_val),
                    },
                )

            if action_id == AirsimActions.GO_UP and pose.position.z_val < -45.0:
                return finalize(
                    AirsimActions.MOVE_FORWARD,
                    {
                        "raw_action_id": int(action_id),
                        "raw_action": action_names.get(int(action_id), None),
                        "override_action_id": int(AirsimActions.MOVE_FORWARD),
                        "override_action": action_names.get(int(AirsimActions.MOVE_FORWARD), None),
                        "reason": "simple_route_altitude_ceiling",
                        "z": float(pose.position.z_val),
                    },
                )

            return finalize(action_id, None)

        live_jpeg_quality = max(45, min(85, int(os.environ.get("AIRVLN_LIVE_JPEG_QUALITY", "82"))))
        live_model_frame_stride = max(0, int(os.environ.get("AIRVLN_LIVE_MODEL_FRAME_STRIDE", "1")))

        def save_frame(step, obs, info, action_id=None, safety_override=None):
            rgb = obs.get("rgb")
            frame_depth = depth_diagnostics(obs)
            should_write_frame = bool(
                rgb is not None
                and live_model_frame_stride > 0
                and (step == 0 or int(step) % live_model_frame_stride == 0)
            )
            if should_write_frame:
                frame_path = frames_dir / f"step_{step:04d}.jpg"
                Image.fromarray(np.asarray(rgb).astype(np.uint8)).save(frame_path, quality=live_jpeg_quality, optimize=False)
            else:
                frame_path = None
            state = env.sim_states[0]
            pose = state.pose
            distance_to_goal = float(info.get("distance_to_goal", -1)) if info else None
            vertical_error_to_goal = None
            distance_3d_to_goal = None
            if goal_position is not None:
                dx_goal = float(goal_position[0]) - float(pose.position.x_val)
                dy_goal = float(goal_position[1]) - float(pose.position.y_val)
                dz_goal = float(goal_position[2]) - float(pose.position.z_val)
                vertical_error_to_goal = dz_goal
                distance_3d_to_goal = math.sqrt(dx_goal * dx_goal + dy_goal * dy_goal + dz_goal * dz_goal)
            best_distance = guidance_state.get("best_distance")
            if distance_to_goal is not None and distance_to_goal >= 0:
                if guidance_state.get("initial_distance") is None:
                    guidance_state["initial_distance"] = float(distance_to_goal)
                if best_distance is None or distance_to_goal < best_distance:
                    best_distance = float(distance_to_goal)
                    guidance_state["best_distance"] = best_distance
                    guidance_state["best_distance_step"] = int(step)
            initial_distance = guidance_state.get("initial_distance")
            if initial_distance and initial_distance > 0 and distance_to_goal is not None and distance_to_goal >= 0:
                progress_to_goal = max(0.0, min(1.0, (initial_distance - distance_to_goal) / initial_distance))
            else:
                progress_to_goal = None
            success_20m = bool(distance_to_goal is not None and 0 <= distance_to_goal <= cli.success_distance)
            oracle_success_20m = bool(best_distance is not None and 0 <= best_distance <= cli.success_distance)
            altitude_aligned = bool(
                vertical_error_to_goal is not None
                and abs(vertical_error_to_goal) <= float(cli.oracle_goal_homing_z_tolerance)
            )
            surface_clearance = guidance_state.get("surface_clearance")
            surface_aligned = bool(
                cli.disable_surface_landing
                or (
                    surface_clearance is not None
                    and surface_clearance <= float(cli.surface_landing_clearance)
                )
            )
            success_20m_with_altitude = bool(success_20m and altitude_aligned)
            success_20m_with_surface = bool(success_20m and surface_aligned)
            if success_20m_with_surface:
                navigation_phase = "success_zone"
            elif success_20m:
                navigation_phase = "landing_to_surface"
            elif distance_to_goal is not None and 0 <= distance_to_goal <= cli.arrival_fine_approach_distance:
                navigation_phase = "final_approach"
            elif distance_to_goal is not None and initial_distance and distance_to_goal < initial_distance:
                navigation_phase = "closing"
            else:
                navigation_phase = "exploring"
            item = {
                "step": step,
                "action_id": int(action_id) if action_id is not None else None,
                "action": action_names.get(int(action_id), None) if action_id is not None else None,
                "done": bool(dones[0]) if dones else False,
                "position": [
                    float(pose.position.x_val),
                    float(pose.position.y_val),
                    float(pose.position.z_val),
                ],
                "distance_to_goal": distance_to_goal,
                "distance_3d_to_goal": distance_3d_to_goal,
                "vertical_error_to_goal": vertical_error_to_goal,
                "altitude_aligned": altitude_aligned,
                "surface_clearance": surface_clearance,
                "surface_aligned": surface_aligned,
                "surface_probe_ok": guidance_state.get("surface_probe_ok"),
                "surface_probe_error": guidance_state.get("surface_probe_error"),
                "best_distance": best_distance,
                "best_distance_step": guidance_state.get("best_distance_step"),
                "initial_distance": initial_distance,
                "progress_to_goal": progress_to_goal,
                "success_distance": cli.success_distance,
                "success_20m": success_20m,
                "success_20m_with_altitude": success_20m_with_altitude,
                "success_20m_with_surface": success_20m_with_surface,
                "oracle_success_20m": oracle_success_20m,
                "navigation_phase": navigation_phase,
                "goal_position": goal_position,
                "collision_rollback_count": int(guidance_state.get("collision_rollback_count", 0)),
                "visual_occupancy_rollback_count": int(
                    guidance_state.get("visual_occupancy_rollback_count", 0)
                ),
                "vertical_reversal_suppressed": int(
                    guidance_state.get("vertical_reversal_suppressed", 0)
                ),
                "descent_surface_blocks": int(guidance_state.get("descent_surface_blocks", 0)),
                "front_depth_p10": (
                    float(frame_depth["front_p10"]) if frame_depth is not None else None
                ),
                "front_contact_ratio": (
                    float(frame_depth["front_contact_ratio"])
                    if frame_depth is not None
                    else None
                ),
                "last_collision": (
                    guidance_state.get("collision_rollbacks", [])[-1]
                    if guidance_state.get("collision_rollbacks")
                    else None
                ),
                "frame": str(frame_path) if frame_path else None,
                "segment_state": current_segment_snapshot(step),
            }
            if safety_override:
                item["safety_override"] = safety_override
                terminal_override = next(
                    (override for override in safety_override if override.get("reason") == "distance_regression_safety_stop"),
                    None,
                )
                if terminal_override:
                    item.update(
                        {
                            "post_recovery_outcome": terminal_override["post_recovery_outcome"],
                            "terminal_stop_reason": terminal_override["terminal_stop_reason"],
                            "strict_safety_counters": terminal_override["strict_safety_counters"],
                        }
                    )
            trace.append(item)
            with trace_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(item, ensure_ascii=False) + "\n")

        save_frame(0, observations[0], infos[0] if infos else {}, None)

        for step in range(1, cli.max_steps + 1):
            actions, rnn_states = trainer.policy.act(
                batch,
                rnn_states,
                prev_actions,
                not_done_masks,
                deterministic=True,
                step=step - 1,
            )
            action_ids = [int(temp[0]) for temp in actions.cpu().numpy()]
            overrides = []
            action_ids[0], route_override = route_guidance_override(infos[0], action_ids[0], step)
            if route_override:
                overrides.append(route_override)
            action_ids[0], vertical_override = vertical_stability_override(
                observations[0],
                infos[0],
                action_ids[0],
                step,
            )
            if vertical_override:
                overrides.append(vertical_override)
            action_ids[0], surface_finish_override = near_success_surface_finish_override(
                infos[0],
                action_ids[0],
                step,
            )
            if surface_finish_override:
                overrides.append(surface_finish_override)
            action_ids[0], final_structure_descent_override = (
                final_structure_descent_clearance_override(
                    observations[0],
                    infos[0],
                    action_ids[0],
                    step,
                    route_override,
                )
            )
            if final_structure_descent_override:
                overrides.append(final_structure_descent_override)
            escape_diagnostics = depth_escape_diagnostics(observations[0])
            route_distance_to_goal = None
            try:
                route_distance_to_goal = float((infos[0] or {}).get("distance_to_goal", -1))
            except Exception:
                route_distance_to_goal = None
            route_around_allowed = should_enable_depth_route_around(
                current_segment(),
                route_distance_to_goal,
                guidance_state.get("initial_distance"),
                step,
                depth_route_around_min_step,
            )
            if (
                not cli.disable_safety_depth_override
                and route_override
                and route_override.get("homing_target_kind") == "learned_segment"
                and route_around_allowed
            ):
                route_action_id = int(action_ids[0])
                action_ids[0], route_around_override = depth_route_around_guard.select(
                    route_action_id,
                    escape_diagnostics,
                    {
                        "left": int(AirsimActions.TURN_LEFT),
                        "right": int(AirsimActions.TURN_RIGHT),
                    },
                )
                if route_around_override:
                    route_around_override.update(
                        {
                            "raw_action_id": route_action_id,
                            "raw_action": action_names.get(route_action_id, None),
                            "override_action": action_names.get(int(action_ids[0]), None),
                        }
                    )
                    overrides.append(route_around_override)
            elif (
                not route_override
                or route_override.get("homing_target_kind") != "learned_segment"
                or not route_around_allowed
            ):
                depth_route_around_guard.reset()
            raw_action_id = int(action_ids[0])
            state = env.sim_states[0]
            position_xy = (
                float(state.pose.position.x_val),
                float(state.pose.position.y_val),
            )

            def final_depth_guard(action_id):
                return depth_safety_override(observations[0], action_id, infos[0])

            action_ids[0], escape_override, depth_override = apply_depth_escape_override(
                action_id=action_ids[0],
                step=step,
                avoidance_streak=guidance_state.get("avoidance_streak") or 0,
                planner=depth_escape_planner,
                diagnostics=escape_diagnostics
                if not cli.disable_safety_depth_override
                else None,
                position_xy=position_xy,
                action_ids={
                    "forward": int(AirsimActions.MOVE_FORWARD),
                    "left": int(AirsimActions.TURN_LEFT),
                    "right": int(AirsimActions.TURN_RIGHT),
                    "stop": int(AirsimActions.STOP),
                },
                final_depth_guard=final_depth_guard,
                enabled=not cli.disable_safety_depth_override,
            )
            if escape_override:
                escape_override = enrich_depth_escape_override(
                    escape_override,
                    raw_action_id=raw_action_id,
                    action_names=action_names,
                    diagnostics=escape_diagnostics,
                )
                overrides.append(escape_override)
                if escape_override.get("terminal"):
                    guidance_state["turn_streak"] = 0
            if depth_override:
                overrides.append(depth_override)
                record_learned_segment_depth_guard_turn(
                    guidance_state,
                    depth_override.get("reason"),
                    route_override.get("homing_target_kind") if route_override else None,
                )
                if (
                    depth_override.get("reason") == "depth_building_guard_turn"
                    and route_override
                    and route_override.get("homing_target_kind") == "learned_segment"
                    and route_around_allowed
                ):
                    depth_route_around_guard.record_guard_turn(
                        action_ids[0],
                        entry_distance=guidance_state.get(
                            "learned_segment_homing_entry_distance"
                        ),
                        best_distance=guidance_state.get(
                            "learned_segment_homing_best_distance"
                        ),
                    )
                depth_distance = None
                try:
                    depth_distance = float((infos[0] or {}).get("distance_to_goal", -1))
                except Exception:
                    depth_distance = None
                depth_recovery_ready = (
                    depth_override.get("reason") == "depth_building_guard_turn"
                    and int(depth_override.get("avoidance_streak") or 0) >= 30
                    and depth_distance is not None
                    and depth_distance > float(cli.success_distance)
                    and int(step) - int(guidance_state.get("depth_block_recovery_last_step", -999)) >= 40
                )
                if depth_recovery_ready:
                    recovery_override = recover_from_remote_done(
                        step,
                        depth_distance,
                        reason="depth_block_recovery_pose",
                    )
                    if recovery_override:
                        overrides.append(recovery_override)
                        guidance_state["depth_block_recovery_last_step"] = int(step)
                        guidance_state["avoidance_streak"] = 0
                        action_ids[0] = AirsimActions.MOVE_FORWARD
                        target = guidance_state.get("learned_segment_homing_target")
                        emit_flight_target_telemetry(
                            "recovery",
                            step,
                            target_before=target,
                            target_after=target,
                            distance_to_goal=depth_distance,
                            recovery_reason=recovery_override.get("reason"),
                            raw_action=action_names.get(raw_action_id, None),
                        )
            safety_override = overrides or None
            if action_ids[0] == AirsimActions.STOP and safety_override:
                stop_reason = safety_override[-1].get("reason") or stop_reason
            state_before_action = env.sim_states[0]
            safe_pose_before_action = copy.deepcopy(state_before_action.pose)
            safe_trajectory_length = len(state_before_action.trajectory)
            safe_pre_action = state_before_action.pre_action
            prev_actions.fill_(int(action_ids[0]))
            env.makeActions(action_ids)
            consume_collision = getattr(env.simulator_tool, "consumeCollisionEvent", None)
            collision_event = consume_collision() if callable(consume_collision) else None
            if collision_event:
                guidance_state["collision_rollback_count"] = int(
                    guidance_state.get("collision_rollback_count", 0)
                ) + 1
                collision_record = {
                    **collision_event,
                    "step": int(step),
                    "action_id": int(action_ids[0]),
                    "action": action_names.get(int(action_ids[0]), None),
                }
                guidance_state.setdefault("collision_rollbacks", []).append(collision_record)
                overrides.append(
                    {
                        "raw_action_id": int(action_ids[0]),
                        "raw_action": action_names.get(int(action_ids[0]), None),
                        "override_action_id": None,
                        "override_action": "HOLD_POSITION",
                        "reason": "hard_collision_rollback",
                        "collision": collision_record,
                    }
                )
                guidance_state["avoidance_streak"] = max(
                    1,
                    int(guidance_state.get("avoidance_streak", 0)),
                )
                safety_override = overrides or None
            outputs = env.get_obs()
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            post_depth = depth_diagnostics(observations[0])
            post_state = env.sim_states[0]
            visual_contact = bool(
                post_depth is not None
                and (
                    float(post_depth["front_contact_ratio"]) >= 0.03
                    or (
                        float(post_depth["front_p10"]) < 0.01
                        and float(post_depth["front_very_close_ratio"]) >= 0.08
                    )
                )
            )
            visual_occupancy_violation = bool(
                int(action_ids[0])
                in (
                    int(AirsimActions.MOVE_FORWARD),
                    int(AirsimActions.MOVE_LEFT),
                    int(AirsimActions.MOVE_RIGHT),
                    int(AirsimActions.GO_UP),
                    int(AirsimActions.GO_DOWN),
                )
                and (bool(post_state.is_collisioned) or visual_contact)
            )
            if visual_occupancy_violation:
                collision_sensor_triggered = bool(post_state.is_collisioned)
                restore_poses = getattr(env.simulator_tool, "restorePoses", None)
                restored = bool(
                    callable(restore_poses)
                    and restore_poses([[copy.deepcopy(safe_pose_before_action)]])
                )
                if restored:
                    post_state.pose = copy.deepcopy(safe_pose_before_action)
                    post_state.is_collisioned = False
                    post_state.is_end = False
                    post_state.pre_action = safe_pre_action
                    del post_state.trajectory[safe_trajectory_length:]
                    env.update_measurements()
                    rollback_record = {
                        "step": int(step),
                        "action_id": int(action_ids[0]),
                        "action": action_names.get(int(action_ids[0]), None),
                        "reason": "visual_occupancy_rollback",
                        "env_collision_sensor": collision_sensor_triggered,
                        "front_contact_ratio": (
                            float(post_depth["front_contact_ratio"])
                            if post_depth is not None
                            else None
                        ),
                        "front_p10": (
                            float(post_depth["front_p10"])
                            if post_depth is not None
                            else None
                        ),
                        "safe_position": [
                            float(safe_pose_before_action.position.x_val),
                            float(safe_pose_before_action.position.y_val),
                            float(safe_pose_before_action.position.z_val),
                        ],
                    }
                    guidance_state["visual_occupancy_rollback_count"] = int(
                        guidance_state.get("visual_occupancy_rollback_count", 0)
                    ) + 1
                    guidance_state["collision_rollback_count"] = int(
                        guidance_state.get("collision_rollback_count", 0)
                    ) + 1
                    guidance_state.setdefault("visual_occupancy_rollbacks", []).append(rollback_record)
                    guidance_state.setdefault("collision_rollbacks", []).append(rollback_record)
                    guidance_state["avoidance_streak"] = max(
                        1,
                        int(guidance_state.get("avoidance_streak", 0)),
                    )
                    overrides.append(
                        {
                            "raw_action_id": int(action_ids[0]),
                            "raw_action": action_names.get(int(action_ids[0]), None),
                            "override_action_id": None,
                            "override_action": "HOLD_POSITION",
                            "reason": "visual_occupancy_rollback",
                            "visual_occupancy": rollback_record,
                        }
                    )
                    safety_override = overrides or None
                    outputs = env.get_obs()
                    observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            distance_after_step = None
            try:
                distance_after_step = float((infos[0] or {}).get("distance_to_goal", -1))
            except Exception:
                distance_after_step = None
            surface_after_step = guidance_state.get("surface_clearance")
            surface_finish_pending = bool(
                distance_after_step is not None
                and 0 <= float(distance_after_step) <= float(cli.success_distance)
                and (
                    surface_after_step is None
                    or float(surface_after_step) > float(cli.surface_landing_clearance)
                )
            )
            remote_done_ignored = bool(
                dones
                and dones[0]
                and action_ids[0] != AirsimActions.STOP
                and (
                    distance_after_step is None
                    or distance_after_step < 0
                    or distance_after_step > float(cli.success_distance)
                    or surface_finish_pending
                )
            )
            if remote_done_ignored:
                recovery_override = recover_from_remote_done(step, distance_after_step)
                dones[0] = False
                overrides.append(
                    {
                        "raw_action_id": int(action_ids[0]),
                        "raw_action": action_names.get(int(action_ids[0]), None),
                        "override_action_id": int(action_ids[0]),
                        "override_action": action_names.get(int(action_ids[0]), None),
                        "reason": "remote_env_done_ignored",
                        "distance_to_goal": distance_after_step,
                        "success_distance": float(cli.success_distance),
                    }
                )
                if recovery_override:
                    overrides.append(recovery_override)
                    target = guidance_state.get("learned_segment_homing_target")
                    emit_flight_target_telemetry(
                        "recovery",
                        step,
                        target_before=target,
                        target_after=target,
                        distance_to_goal=distance_after_step,
                        recovery_reason=recovery_override.get("reason"),
                        raw_action=action_names.get(int(action_ids[0]), None),
                    )
                    outputs = env.get_obs()
                    observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                    if dones:
                        dones[0] = False
                safety_override = overrides or None
            batch = batch_obs(observations, trainer.device)
            not_done_masks = torch.tensor(
                [[0] if done else [1] for done in dones],
                dtype=torch.uint8,
                device=trainer.device,
            )
            save_frame(step, observations[0], infos[0], action_ids[0], safety_override=safety_override)
            if np.array(dones).all():
                if stop_reason is None:
                    if action_ids[0] == AirsimActions.STOP:
                        stop_reason = "model_stop"
                    elif step >= cli.max_steps:
                        stop_reason = "max_steps"
                    else:
                        stop_reason = "env_done"
                break
        else:
            stop_reason = stop_reason or "max_steps"

    episode_metadata = build_episode_metadata(episode, goal_position)
    meta = {
        "episode_id": cli.episode_id,
        "split": cli.split,
        **episode_metadata,
        "original_instruction": original_instruction,
        "model_instruction": model_instruction,
        "model_instruction_source": model_instruction_source,
        "keep_original_motion_chain_instruction": keep_original_motion_chain_instruction,
        "reference_dense_intervention_enabled": reference_dense_intervention_enabled,
        "original_instruction_word_count": original_instruction_word_count,
        "grounding_guidance": not cli.disable_grounding_guidance,
        "segmented_grounding": segmented_grounding,
        "instruction_profile": instruction_profile,
        "grounding": compact_grounding_view(grounding),
        "grounding_full": grounding,
        "segment_plan": segment_plan,
        "segment_switches": guidance_state.get("segment_switches", []),
        "segment_observer_switches": guidance_state.get("segment_observer_switches", []),
        "segment_observer_diagnostics": guidance_state.get("segment_observer_diagnostics", []),
        "simple_route_guidance": simple_route_guidance,
        "forward_route_guidance": forward_route_guidance,
        "ckpt": cli.ckpt,
        "max_steps": cli.max_steps,
        "arrival_stop_distance": cli.arrival_stop_distance,
        "arrival_fine_approach_distance": cli.arrival_fine_approach_distance,
        "arrival_overshoot_distance": cli.arrival_overshoot_distance,
        "success_distance": cli.success_distance,
        "oracle_goal_homing_enabled": cli.enable_oracle_goal_homing,
        "oracle_path_homing_enabled": cli.enable_oracle_path_homing,
        "learned_segment_homing_enabled": learned_segment_homing_enabled,
        "segment_grounding_ridge": cli.segment_grounding_ridge,
        "learned_segment_homing_min_step": cli.learned_segment_homing_min_step,
        "learned_segment_target_radius": cli.learned_segment_target_radius,
        "learned_segment_max_age": cli.learned_segment_max_age,
        "learned_segment_z_clamp": cli.learned_segment_z_clamp,
        "learned_segment_min_xy_step": cli.learned_segment_min_xy_step,
        "learned_segment_max_xy_step": cli.learned_segment_max_xy_step,
        "learned_segment_conservative_max_xy_step": cli.learned_segment_conservative_max_xy_step,
        "learned_segment_resample_worse_streak": cli.learned_segment_resample_worse_streak,
        "learned_segment_resample_regression": cli.learned_segment_resample_regression,
        "learned_segment_goal_regression_worse_streak": cli.learned_segment_goal_regression_worse_streak,
        "learned_segment_goal_regression": cli.learned_segment_goal_regression,
        "learned_segment_goal_regression_min_age": cli.learned_segment_goal_regression_min_age,
        "enable_final_goal_progress_gate": cli.enable_final_goal_progress_gate,
        "final_goal_progress_tolerance": cli.final_goal_progress_tolerance,
        "final_goal_progress_max_retries": cli.final_goal_progress_max_retries,
        "learned_segment_cruise_altitude": cli.learned_segment_cruise_altitude,
        "learned_segment_descent_altitude": cli.learned_segment_descent_altitude,
        "oracle_goal_homing_min_step": cli.oracle_goal_homing_min_step,
        "oracle_goal_homing_distance": cli.oracle_goal_homing_distance,
        "oracle_path_lookahead": cli.oracle_path_lookahead,
        "oracle_path_waypoint_radius": cli.oracle_path_waypoint_radius,
        "oracle_path_waypoint_max_age": cli.oracle_path_waypoint_max_age,
        "surface_landing_enabled": not cli.disable_surface_landing,
        "surface_landing_clearance": cli.surface_landing_clearance,
        "surface_landing_probe_camera": cli.surface_landing_probe_camera,
        "regression_recover_distance": cli.regression_recover_distance,
        "regression_stop_distance": cli.regression_stop_distance,
        "regression_worse_streak": cli.regression_worse_streak,
        "regression_recovery_steps": cli.regression_recovery_steps,
        "grounding_boundary_window_steps": cli.grounding_boundary_window_steps,
        "grounding_recovery_window_steps": cli.grounding_recovery_window_steps,
        "segment_switch_confidence": cli.segment_switch_confidence,
        "segment_min_steps": cli.segment_min_steps,
        "segment_progress_distance": cli.segment_progress_distance,
        "safety_depth_override": not cli.disable_safety_depth_override,
        "collision_rollback_count": int(guidance_state.get("collision_rollback_count", 0)),
        "collision_rollbacks": guidance_state.get("collision_rollbacks", []),
        "visual_occupancy_rollback_count": int(
            guidance_state.get("visual_occupancy_rollback_count", 0)
        ),
        "visual_occupancy_rollbacks": guidance_state.get("visual_occupancy_rollbacks", []),
        "vertical_reversal_suppressed": int(
            guidance_state.get("vertical_reversal_suppressed", 0)
        ),
        "descent_surface_blocks": int(guidance_state.get("descent_surface_blocks", 0)),
        "descent_depth_blocks": int(guidance_state.get("descent_depth_blocks", 0)),
        "stop_reason": stop_reason,
        "initial_distance": guidance_state.get("initial_distance"),
        "best_distance": guidance_state.get("best_distance"),
        "best_distance_step": guidance_state.get("best_distance_step"),
        "final_distance": trace[-1].get("distance_to_goal") if trace else None,
        "final_distance_3d": trace[-1].get("distance_3d_to_goal") if trace else None,
        "final_vertical_error": trace[-1].get("vertical_error_to_goal") if trace else None,
        "final_surface_clearance": trace[-1].get("surface_clearance") if trace else None,
        "surface_probe_ok": bool(trace and trace[-1].get("surface_probe_ok")),
        "surface_probe_error": trace[-1].get("surface_probe_error") if trace else None,
        "success_20m": bool(trace and trace[-1].get("success_20m")),
        "success_20m_with_altitude": bool(trace and trace[-1].get("success_20m_with_altitude")),
        "success_20m_with_surface": bool(trace and trace[-1].get("success_20m_with_surface")),
        "oracle_success_20m": bool(trace and trace[-1].get("oracle_success_20m")),
        "navigation_phase": trace[-1].get("navigation_phase") if trace else None,
        "out_dir": str(out_dir),
        "flight_target_telemetry": (
            build_flight_target_telemetry_manifest(telemetry_path, flight_target_telemetry_event_count)
            if telemetry_path.exists()
            else None
        ),
        "num_frames": len(list(frames_dir.glob("*.jpg"))),
        "trace": trace,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out_dir": str(out_dir), "num_frames": meta["num_frames"], "last": trace[-1]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
