import math
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_live_once import (
    choose_learned_segment_xy_step,
    learned_segment_homing_diagnostics,
    record_learned_segment_depth_guard_turn,
    learned_segment_target_xy_adjustment,
    should_break_learned_homing_turn_loop,
    should_defer_final_descent_far_from_goal,
    should_defer_close_final_structure_descent,
    should_enable_depth_route_around,
    should_block_final_structure_descent_for_visual_clearance,
    should_release_close_final_structure_depth_guard,
    should_release_stale_learned_clearance_climb,
    should_release_final_corridor_depth_guard,
    should_resample_stalled_learned_homing_target,
    should_resample_learned_target_for_goal_regression,
    should_reject_final_learned_target_for_goal_progress,
    should_continue_success_zone_surface_descent,
    recent_valid_surface_clearance,
    estimate_surface_clearance_from_recent_probe,
    parse_args,
    create_output_dir,
    normalize_flight_target_number,
    normalize_flight_target_position,
    choose_route_prior_segment_delta,
    route_prior_adjusted_learned_delta,
    select_route_prior_candidate_for_goal_bearing,
    build_flight_target_ranker_context,
    build_flight_target_telemetry_event,
    append_flight_target_telemetry,
    build_flight_target_telemetry_manifest,
    build_episode_metadata,
)


class LearnedHomingDiagnosticsTests(unittest.TestCase):
    def test_route_prior_selects_progress_matched_segment_delta(self):
        candidate = {
            "segment_delta_templates": [
                {
                    "segment_index": 0,
                    "segment_count": 4,
                    "segment_ratio": 0.125,
                    "delta_xyz": [24.0, 0.0, 0.0],
                    "actions": ["forward"],
                    "target_landmark": "road",
                },
                {
                    "segment_index": 3,
                    "segment_count": 4,
                    "segment_ratio": 0.875,
                    "delta_xyz": [0.0, 24.0, 0.0],
                    "actions": ["descend"],
                    "target_landmark": "road",
                },
            ]
        }

        delta = choose_route_prior_segment_delta(
            candidate,
            segment_index=3,
            segment_count=4,
            segment={"actions": ["descend"], "target_landmark": "road"},
        )

        self.assertEqual(delta, [0.0, 24.0, 0.0])

    def test_route_prior_adjusts_direction_without_changing_xy_scale_or_z(self):
        adjusted = route_prior_adjusted_learned_delta(
            [24.0, 0.0, -4.0],
            {
                "segment_delta_templates": [
                    {
                        "segment_ratio": 0.5,
                        "delta_xyz": [0.0, 10.0, 12.0],
                        "actions": ["forward"],
                    }
                ]
            },
            segment_index=0,
            segment_count=1,
            segment={"actions": ["forward"]},
            blend=0.5,
        )

        self.assertAlmostEqual(adjusted[0], 12.0)
        self.assertAlmostEqual(adjusted[1], 12.0)
        self.assertAlmostEqual(adjusted[2], -4.0)

    def test_route_prior_candidate_selection_prefers_matching_goal_bearing(self):
        selected = select_route_prior_candidate_for_goal_bearing(
            [
                {"episode_id": "east", "route_goal_unit_xy": [1.0, 0.0]},
                {"episode_id": "north", "route_goal_unit_xy": [0.0, 1.0]},
            ],
            current_position=[0.0, 0.0, 0.0],
            goal_position=[0.0, 20.0, 0.0],
        )

        self.assertEqual(selected["episode_id"], "north")

    def test_final_goal_progress_gate_is_opt_in_with_bounded_defaults(self):
        with patch.object(sys, "argv", ["model_live_once.py"]):
            defaults = parse_args()
        self.assertFalse(defaults.enable_final_goal_progress_gate)
        self.assertEqual(defaults.final_goal_progress_tolerance, 8.0)
        self.assertEqual(defaults.final_goal_progress_max_retries, 2)

    def test_final_goal_progress_gate_rejects_only_enabled_regressing_targets(self):
        self.assertFalse(should_reject_final_learned_target_for_goal_progress(
            enabled=False,
            final_segment=True,
            current_distance=100.0,
            candidate_distance=140.0,
            tolerance=5.0,
        ))
        self.assertFalse(should_reject_final_learned_target_for_goal_progress(
            enabled=True,
            final_segment=False,
            current_distance=100.0,
            candidate_distance=140.0,
            tolerance=5.0,
        ))
        self.assertTrue(should_reject_final_learned_target_for_goal_progress(
            enabled=True,
            final_segment=True,
            current_distance=100.0,
            candidate_distance=106.0,
            tolerance=5.0,
        ))

    def test_final_goal_progress_gate_accepts_improving_and_invalid_distances(self):
        for current_distance, candidate_distance in ((100.0, 90.0), (100.0, 105.0), (None, 140.0), (100.0, None)):
            self.assertFalse(should_reject_final_learned_target_for_goal_progress(
                enabled=True,
                final_segment=True,
                current_distance=current_distance,
                candidate_distance=candidate_distance,
                tolerance=5.0,
            ))

    def test_goal_regression_gate_is_learned_only(self):
        kwargs = dict(
            target_active=True,
            target_age=10,
            worse_streak=4,
            regression_from_best=55.0,
            min_age=8,
            worse_streak_threshold=4,
            regression_threshold=55.0,
        )
        self.assertTrue(should_resample_learned_target_for_goal_regression(
            learned_enabled=True, oracle_path_enabled=False, **kwargs
        ))
        self.assertFalse(should_resample_learned_target_for_goal_regression(
            learned_enabled=False, oracle_path_enabled=True, **kwargs
        ))

    def test_goal_regression_gate_requires_all_thresholds(self):
        kwargs = dict(
            learned_enabled=True,
            oracle_path_enabled=False,
            target_active=True,
            min_age=8,
            worse_streak_threshold=4,
            regression_threshold=55.0,
        )
        self.assertFalse(should_resample_learned_target_for_goal_regression(
            **kwargs, target_age=7, worse_streak=4, regression_from_best=70.0
        ))
        self.assertFalse(should_resample_learned_target_for_goal_regression(
            **kwargs, target_age=10, worse_streak=3, regression_from_best=70.0
        ))
        self.assertFalse(should_resample_learned_target_for_goal_regression(
            **kwargs, target_age=10, worse_streak=4, regression_from_best=54.9
        ))

    def test_reports_active_target_lifecycle_without_target_coordinates(self):
        diagnostics = learned_segment_homing_diagnostics(
            {
                "learned_segment_homing_target": [1.0, 2.0, 3.0],
                "learned_segment_homing_index": 2,
                "learned_segment_homing_created_step": 40,
                "learned_segment_homing_distance": 26.5,
                "learned_segment_homing_entry_distance": 150.0,
                "learned_segment_homing_best_distance": 124.0,
                "learned_segment_homing_resamples": 3,
                "learned_segment_homing_safety_turn_count": 4,
                "learned_segment_homing_segment_safety_turn_count": 7,
                "learned_segment_homing_last_safety_stall_resample_step": 31,
                "learned_segment_homing_goal_regression_resamples": 2,
                "learned_segment_homing_last_goal_regression_resample_step": 42,
                "learned_segment_homing_base_xy_step": 54.0,
                "learned_segment_homing_xy_step": 36.0,
                "learned_segment_homing_risk_tier": 1,
                "learned_segment_homing_xy_step_reason": "safety_turn_reduced",
            },
            step=45,
        )

        self.assertEqual(diagnostics["target_index"], 2)
        self.assertEqual(diagnostics["target_age"], 6)
        self.assertEqual(diagnostics["resamples"], 3)
        self.assertEqual(diagnostics["safety_turn_count"], 4)
        self.assertEqual(diagnostics["goal_regression_resamples"], 2)
        self.assertEqual(diagnostics["last_goal_regression_resample_step"], 42)
        self.assertNotIn("target", diagnostics)
        self.assertNotIn("delta", diagnostics)

    def test_reports_no_target_age_without_an_active_target(self):
        diagnostics = learned_segment_homing_diagnostics({}, step=45)

        self.assertFalse(diagnostics["active"])
        self.assertIsNone(diagnostics["target_age"])


class EpisodeMetadataTests(unittest.TestCase):
    def test_metadata_uses_episode_record_after_target_recovery_state_changes(self):
        episode = {
            "scene_id": 16,
            "trajectory_id": "trajectory-1",
            "instruction": {"instruction_text": "head to the park"},
        }

        metadata = build_episode_metadata(episode, goal_position=[1.0, 2.0, 3.0])

        self.assertEqual(metadata["scene_id"], 16)
        self.assertEqual(metadata["trajectory_id"], "trajectory-1")
        self.assertEqual(metadata["instruction"], "head to the park")
        self.assertEqual(metadata["goal_position"], [1.0, 2.0, 3.0])


class Scene16SupportRankerArgumentTests(unittest.TestCase):
    def test_support_ranker_is_opt_in(self):
        with patch.object(sys, "argv", ["model_live_once.py"]):
            defaults = parse_args()
        with patch.object(
            sys,
            "argv",
            [
                "model_live_once.py",
                "--enable-scene16-support-ranker",
                "--scene16-support-ranker",
                "/tmp/ranker",
            ],
        ):
            enabled = parse_args()

        self.assertFalse(defaults.enable_scene16_support_ranker)
        self.assertEqual(defaults.scene16_support_ranker, "")
        self.assertTrue(enabled.enable_scene16_support_ranker)
        self.assertEqual(enabled.scene16_support_ranker, "/tmp/ranker")

    def test_ranker_context_does_not_copy_reference_path(self):
        context = build_flight_target_ranker_context(
            {"target_landmark": "road", "landmarks": []},
            {
                "reference_path": [[0, 0, 0], [1, 1, 1]],
                "support_examples": [{"episode_id": "stable", "landmarks": ["road"]}],
                "scene16_support_ranker_decision": {
                    "applied": True,
                    "final_episode_ids": ["stable"],
                    "scored_candidates": [{"episode_id": "stable", "score": 0.9}],
                },
            },
        )
        self.assertNotIn("reference_path", json.dumps(context).lower())


class ModelRunNamespaceTests(unittest.TestCase):
    def test_model_run_id_uses_exact_new_directory_and_rejects_collisions(self):
        with tempfile.TemporaryDirectory() as root:
            out_root = Path(root)
            run_dir = create_output_dir(out_root, "v014-" + "a" * 32)

            self.assertEqual(run_dir, out_root / ("v014-" + "a" * 32))
            self.assertTrue(run_dir.is_dir())
            with self.assertRaises(FileExistsError):
                create_output_dir(out_root, "v014-" + "a" * 32)

    def test_model_run_id_must_be_a_safe_v014_component(self):
        with tempfile.TemporaryDirectory() as root:
            for value in ("", "../run", "run/name", "run\\name", "v014-" + "A" * 32):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    create_output_dir(Path(root), value)

    def test_pre_reserved_model_run_requires_the_exact_existing_non_reparse_directory(self):
        with tempfile.TemporaryDirectory() as root:
            out_root = Path(root) / "model_runs"
            run_id = "v014-" + "b" * 32
            reserved = out_root / run_id
            reserved.mkdir(parents=True)

            self.assertEqual(
                create_output_dir(out_root, run_id, pre_reserved=True),
                reserved,
            )
            with self.assertRaisesRegex(ValueError, "pre-reserved"):
                create_output_dir(out_root, "v014-" + "c" * 32, pre_reserved=True)

    def test_legacy_output_directory_remains_timestamped_when_model_run_id_is_absent(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("model_live_once.time.strftime", return_value="20260817-120000"):
                run_dir = create_output_dir(Path(root), None)

            self.assertEqual(run_dir, Path(root) / "20260817-120000")


class Scene16V014TelemetryContractTests(unittest.TestCase):
    def test_recovery_override_emits_authoritative_direction_and_explicit_budget(self):
        source = (Path(__file__).resolve().parent / "model_live_once.py").read_text(encoding="utf-8")
        recovery_start = source.index('if should_stop_regression:')
        recovery_end = source.index('if action_id == AirsimActions.STOP', recovery_start)
        recovery = source[recovery_start:recovery_end]

        self.assertIn('direction_source = "default_left"', recovery)
        self.assertIn('direction_source = "segment_right_constraint"', recovery)
        self.assertIn('"direction_source": direction_source', recovery)
        self.assertIn('"recovery_budget": int(cli.regression_recovery_steps)', recovery)

    def test_terminal_regression_stop_is_lifted_to_trace_row_with_actual_counters(self):
        source = (Path(__file__).resolve().parent / "model_live_once.py").read_text(encoding="utf-8")
        terminal_start = source.index('if should_stop_regression:')
        terminal_end = source.index('if (\n                guidance_state["recovery_attempts"]', terminal_start)
        terminal = source[terminal_start:terminal_end]
        save_frame_start = source.index('def save_frame(step, obs, info, action_id=None, safety_override=None):')
        save_frame_end = source.index('save_frame(0, observations[0]', save_frame_start)
        save_frame = source[save_frame_start:save_frame_end]

        self.assertIn('"post_recovery_outcome": "route_miss"', terminal)
        self.assertIn('"terminal_stop_reason": "distance_regression_safety_stop"', terminal)
        self.assertIn('"strict_safety_counters"', terminal)
        self.assertIn('"post_recovery_outcome"', save_frame)
        self.assertIn('"terminal_stop_reason"', save_frame)
        self.assertIn('"strict_safety_counters"', save_frame)

    def test_subthreshold_regression_has_one_bounded_target_resampling_retry(self):
        source = (Path(__file__).resolve().parent / "model_live_once.py").read_text(encoding="utf-8")
        recovery_start = source.index('if should_stop_regression:')
        recovery_end = source.index('if action_id == AirsimActions.STOP', recovery_start)
        recovery = source[recovery_start:recovery_end]

        self.assertIn('"regression_retry_attempts": 0', source)
        self.assertIn('regression_retry_attempts', recovery)
        self.assertIn('target_resampled_for_regression_retry', recovery)
        self.assertIn('"recovery_retry"', recovery)
        self.assertIn('regression_amount >= cli.regression_stop_distance', source)
        self.assertIn('and not excessive_regression', recovery)
        self.assertIn('guidance_state["recovery_turn_steps"] <= 2', recovery)
        self.assertIn('homing_must_yield_to_regression_retry', source)
        self.assertIn('bounded_retry_alternate', recovery)
        self.assertNotIn('guidance_state["recovery_attempts"] >= 1\n                )', recovery)

    def test_retry_opens_bounded_visual_exploration_window_before_homing_resumes(self):
        source = (Path(__file__).resolve().parent / "model_live_once.py").read_text(encoding="utf-8")
        self.assertIn('"regression_retry_exploration_steps": 0', source)
        self.assertIn('guidance_state["regression_retry_exploration_steps"] = int(', source)
        self.assertIn('cli.regression_recovery_steps', source)
        self.assertIn('guidance_state["regression_retry_exploration_steps"] > 0', source)

    def test_retry_has_turn_then_forward_probe_phase(self):
        source = (Path(__file__).resolve().parent / "model_live_once.py").read_text(encoding="utf-8")
        recovery_start = source.index('if should_stop_regression:')
        recovery_end = source.index('if action_id == AirsimActions.STOP', recovery_start)
        recovery = source[recovery_start:recovery_end]

        self.assertIn('"regression_retry_phase": None', source)
        self.assertIn('guidance_state["regression_retry_phase"] = "forward_probe"', recovery)
        self.assertIn('AirsimActions.MOVE_FORWARD', recovery)
        self.assertIn('distance_regression_recovery_forward_probe', recovery)

    def test_retry_forward_probe_precedes_homing_resume(self):
        source = (Path(__file__).resolve().parent / "model_live_once.py").read_text(encoding="utf-8")
        homing_start = source.index('homing_must_yield_to_regression_retry = (')
        homing_end = source.index('if oracle_action is not None', homing_start)
        homing = source[homing_start:homing_end]

        self.assertIn('regression_retry_phase', homing)
        self.assertIn('forward_probe', homing)
        self.assertIn('guidance_state["regression_retry_exploration_steps"] > 0', homing)

    def test_retry_blocks_learned_target_lifecycle_before_forward_probe(self):
        source = (Path(__file__).resolve().parent / "model_live_once.py").read_text(encoding="utf-8")
        homing_start = source.index('def oracle_goal_homing_action():')
        target_lifecycle_start = source.index('homing_target = goal_position', homing_start)
        homing_prefix = source[homing_start:target_lifecycle_start]

        self.assertIn('regression_retry_phase', homing_prefix)
        self.assertIn('"turn", "forward_probe"', homing_prefix)
        self.assertIn('guidance_state["regression_retry_exploration_steps"] > 0', homing_prefix)
        self.assertIn('return None', homing_prefix)

    def test_retry_preserves_target_provenance_across_release_and_forward_probe(self):
        source = (Path(__file__).resolve().parent / "model_live_once.py").read_text(encoding="utf-8")
        recovery_start = source.index('if should_stop_regression:')
        recovery_end = source.index('if action_id == AirsimActions.STOP', recovery_start)
        recovery = source[recovery_start:recovery_end]

        self.assertIn('"regression_retry_target": None', source)
        self.assertIn('guidance_state["regression_retry_target"] = list(target)', recovery)
        self.assertIn('target = guidance_state.get("regression_retry_target")', recovery)
        self.assertIn('guidance_state["regression_retry_target"] = None', recovery)


class LearnedHomingTurnLoopTests(unittest.TestCase):
    def test_breaks_only_for_a_mature_stationary_learned_target(self):
        self.assertFalse(
            should_break_learned_homing_turn_loop(
                "learned_segment", target_age=11, stationary_turn_streak=5, collision_rollback_count=0
            )
        )
        self.assertFalse(
            should_break_learned_homing_turn_loop(
                "learned_segment", target_age=12, stationary_turn_streak=4, collision_rollback_count=0
            )
        )
        self.assertTrue(
            should_break_learned_homing_turn_loop(
                "learned_segment", target_age=12, stationary_turn_streak=5, collision_rollback_count=0
            )
        )

    def test_does_not_probe_forward_after_a_rollback(self):
        self.assertFalse(
            should_break_learned_homing_turn_loop(
                "learned_segment", target_age=20, stationary_turn_streak=8, collision_rollback_count=1
            )
        )

    def test_does_not_apply_to_non_learned_homing(self):
        self.assertFalse(
            should_break_learned_homing_turn_loop(
                "reference_path", target_age=20, stationary_turn_streak=8, collision_rollback_count=0
            )
        )


class LearnedHomingStallTests(unittest.TestCase):
    def test_resamples_a_mature_learned_target_after_safety_turns_without_progress(self):
        self.assertTrue(
            should_resample_stalled_learned_homing_target(
                "learned_segment",
                target_age=17,
                entry_distance=129.9,
                best_distance=129.9,
                safety_turn_count=8,
            )
        )

    def test_keeps_targets_that_are_not_stalled_learned_homing(self):
        self.assertFalse(
            should_resample_stalled_learned_homing_target(
                "reference_path",
                target_age=17,
                entry_distance=129.9,
                best_distance=129.9,
                safety_turn_count=8,
            )
        )
        self.assertFalse(
            should_resample_stalled_learned_homing_target(
                "learned_segment",
                target_age=17,
                entry_distance=129.9,
                best_distance=120.0,
                safety_turn_count=8,
            )
        )
        self.assertFalse(
            should_resample_stalled_learned_homing_target(
                "learned_segment",
                target_age=17,
                entry_distance=129.9,
                best_distance=129.9,
                safety_turn_count=3,
            )
        )

    def test_applies_a_24_step_cooldown_between_safety_stall_resamples(self):
        self.assertFalse(
            should_resample_stalled_learned_homing_target(
                "learned_segment",
                target_age=17,
                entry_distance=129.9,
                best_distance=129.9,
                safety_turn_count=8,
                step=209,
                last_safety_stall_resample_step=186,
            )
        )
        self.assertTrue(
            should_resample_stalled_learned_homing_target(
                "learned_segment",
                target_age=17,
                entry_distance=129.9,
                best_distance=129.9,
                safety_turn_count=8,
                step=210,
                last_safety_stall_resample_step=186,
            )
        )

    def test_invalid_or_future_cooldown_markers_do_not_block_resampling(self):
        for marker in ("209", "invalid", None, True, 209.5, float("nan"), float("inf"), 211):
            with self.subTest(marker=marker):
                self.assertTrue(
                    should_resample_stalled_learned_homing_target(
                        "learned_segment",
                        target_age=17,
                        entry_distance=129.9,
                        best_distance=129.9,
                        safety_turn_count=8,
                        step=210,
                        last_safety_stall_resample_step=marker,
                    )
                )
        self.assertTrue(
            should_resample_stalled_learned_homing_target(
                "learned_segment",
                target_age=17,
                entry_distance=129.9,
                best_distance=129.9,
                safety_turn_count=8,
                step=186,
                last_safety_stall_resample_step=None,
            )
        )


class LearnedHomingClearanceClimbReleaseTests(unittest.TestCase):
    def test_releases_mature_stale_target_climb_near_cruise_altitude(self):
        self.assertTrue(
            should_release_stale_learned_clearance_climb(
                "learned_segment",
                18,
                24.0,
                30.0,
                39.5,
                45.0,
                6.0,
            )
        )

    def test_keeps_clearance_climb_for_young_far_or_low_altitude_targets(self):
        cases = (
            {"target_age": 17, "xy_distance": 24.0, "z_error": 30.0, "altitude": 39.5},
            {"target_age": 18, "xy_distance": 42.0, "z_error": 30.0, "altitude": 39.5},
            {"target_age": 18, "xy_distance": 24.0, "z_error": 30.0, "altitude": 28.0},
            {"target_age": 18, "xy_distance": 24.0, "z_error": 4.0, "altitude": 39.5},
        )
        for case in cases:
            with self.subTest(case=case):
                self.assertFalse(
                    should_release_stale_learned_clearance_climb(
                        "learned_segment",
                        case["target_age"],
                        case["xy_distance"],
                        case["z_error"],
                        case["altitude"],
                        45.0,
                        6.0,
                    )
                )

    def test_does_not_apply_to_reference_path_homing(self):
        self.assertFalse(
            should_release_stale_learned_clearance_climb(
                "reference_path",
                18,
                24.0,
                30.0,
                39.5,
                45.0,
                6.0,
            )
        )


class FinalDescentDistanceGateTests(unittest.TestCase):
    def test_defers_final_descent_when_actual_goal_is_still_far(self):
        self.assertTrue(should_defer_final_descent_far_from_goal(329.5, 50.0))

    def test_allows_final_descent_inside_fine_approach(self):
        self.assertFalse(should_defer_final_descent_far_from_goal(45.0, 50.0))
        self.assertFalse(should_defer_final_descent_far_from_goal(20.0, 50.0))

    def test_missing_distance_keeps_existing_behavior(self):
        self.assertFalse(should_defer_final_descent_far_from_goal(None, 50.0))
        self.assertFalse(should_defer_final_descent_far_from_goal(-1.0, 50.0))


class SuccessZoneSurfaceDescentTests(unittest.TestCase):
    def test_continues_descent_when_surface_probe_is_missing_above_landing_clearance(self):
        self.assertTrue(
            should_continue_success_zone_surface_descent(
                distance_to_goal=18.6,
                success_distance=20.0,
                surface_clearance=None,
                altitude=4.0,
                surface_landing_clearance=1.0,
            )
        )

    def test_stops_descent_when_surface_probe_is_missing_but_altitude_is_already_low(self):
        self.assertFalse(
            should_continue_success_zone_surface_descent(
                distance_to_goal=18.6,
                success_distance=20.0,
                surface_clearance=None,
                altitude=1.2,
                surface_landing_clearance=1.0,
            )
        )

    def test_uses_known_surface_clearance_when_probe_succeeds(self):
        self.assertTrue(
            should_continue_success_zone_surface_descent(18.6, 20.0, 2.2, 1.2, 1.0)
        )
        self.assertFalse(
            should_continue_success_zone_surface_descent(18.6, 20.0, 1.0, 8.0, 1.0)
        )

    def test_does_not_apply_outside_success_zone_or_on_invalid_values(self):
        self.assertFalse(
            should_continue_success_zone_surface_descent(22.0, 20.0, None, 8.0, 1.0)
        )
        self.assertFalse(
            should_continue_success_zone_surface_descent(None, 20.0, None, 8.0, 1.0)
        )


class RecentValidSurfaceClearanceTests(unittest.TestCase):
    def test_reuses_recent_valid_clearance(self):
        self.assertEqual(recent_valid_surface_clearance(38, 37, 12.625), 12.625)
        self.assertEqual(recent_valid_surface_clearance(38, 36, "4.5"), 4.5)

    def test_rejects_stale_future_or_invalid_clearance(self):
        self.assertIsNone(recent_valid_surface_clearance(38, 35, 12.625))
        self.assertIsNone(recent_valid_surface_clearance(38, 39, 12.625))
        self.assertIsNone(recent_valid_surface_clearance(38, 37, -1.0))
        self.assertIsNone(recent_valid_surface_clearance(38, 37, "invalid"))
        self.assertIsNone(recent_valid_surface_clearance("bad-step", 37, 12.625))


class SurfaceClearanceEstimateTests(unittest.TestCase):
    def test_estimates_clearance_from_recent_downward_motion(self):
        self.assertEqual(
            estimate_surface_clearance_from_recent_probe(
                current_z=5.4,
                last_valid_z=3.4,
                last_valid_clearance=12.6,
                current_step=38,
                last_valid_step=37,
            ),
            10.6,
        )
        self.assertEqual(
            estimate_surface_clearance_from_recent_probe(20.0, 3.0, 12.0, 45, 37),
            0.0,
        )

    def test_rejects_stale_upward_or_invalid_estimates(self):
        self.assertIsNone(
            estimate_surface_clearance_from_recent_probe(5.0, 3.0, 12.0, 50, 37)
        )
        self.assertIsNone(
            estimate_surface_clearance_from_recent_probe(2.0, 3.0, 12.0, 38, 37)
        )
        self.assertIsNone(
            estimate_surface_clearance_from_recent_probe(5.0, 3.0, -1.0, 38, 37)
        )
        self.assertIsNone(
            estimate_surface_clearance_from_recent_probe("bad-z", 3.0, 12.0, 38, 37)
        )


class DepthRouteAroundContextTests(unittest.TestCase):
    def test_blocks_intersection_road_route_even_when_landing_is_mentioned(self):
        segment = {
            "text": "now move forward towards the buildings and reach the intersection road and get down and land there",
            "target_landmark": "road",
            "target_zone": "corridor",
            "actions": ["forward", "descend"],
            "relations": [],
        }
        self.assertFalse(
            should_enable_depth_route_around(
                segment,
                distance_to_goal=127.0,
                initial_distance=547.0,
                step=395,
                min_step=60,
            )
        )

    def test_blocks_intersection_landmark_route_around_for_corridor_target(self):
        segment = {
            "text": "now move forward over the buildings and towards the road and reach the intersection road and land there",
            "target_landmark": "intersection",
            "target_zone": "corridor",
            "actions": ["forward", "fly_over", "descend"],
            "relations": [],
        }
        self.assertFalse(
            should_enable_depth_route_around(
                segment,
                distance_to_goal=31.4,
                initial_distance=550.0,
                step=886,
                min_step=60,
            )
        )

    def test_allows_near_terminal_building_or_anchor_segments(self):
        building_landing = {
            "text": "now go beside to the road and move forward and get down to the building and land near to the bench near to that building",
            "target_landmark": "road",
            "target_zone": "corridor",
            "actions": ["forward", "descend"],
            "relations": [{"type": "near", "text": "near to the bench near to that building"}],
        }
        self.assertTrue(
            should_enable_depth_route_around(
                building_landing,
                distance_to_goal=129.0,
                initial_distance=295.0,
                step=153,
                min_step=60,
            )
        )
        facing_building = {
            "text": "turn left fly above and land by facing the tall building",
            "target_landmark": "building",
            "target_zone": "structure",
            "actions": ["turn_left", "turn", "fly_over", "descend"],
            "relations": [{"type": "facing", "text": "facing the tall building"}],
        }
        self.assertTrue(
            should_enable_depth_route_around(
                facing_building,
                distance_to_goal=45.0,
                initial_distance=332.0,
                step=298,
                min_step=60,
            )
        )

    def test_blocks_far_route_around_before_target_region(self):
        segment = {
            "text": "now go beside to the road and move forward and get down to the building and land near to the bench",
            "target_landmark": "building",
            "target_zone": "structure",
            "actions": ["forward", "descend"],
            "relations": [],
        }
        self.assertFalse(
            should_enable_depth_route_around(
                segment,
                distance_to_goal=260.0,
                initial_distance=295.0,
                step=153,
                min_step=60,
            )
        )


class FinalCorridorDepthReleaseTests(unittest.TestCase):
    def test_releases_low_altitude_final_intersection_stall_when_depth_is_not_contacting(self):
        segment = {
            "text": "now move forward over the buildings and towards the road and reach the intersection road and land there",
            "target_landmark": "intersection",
            "target_zone": "corridor",
            "actions": ["forward", "fly_over", "descend"],
        }
        self.assertTrue(
            should_release_final_corridor_depth_guard(
                segment,
                segment_index=3,
                segment_count=4,
                distance_to_goal=31.4,
                surface_clearance=5.34,
                front_contact_ratio=0.0,
                front_very_close_ratio=0.01,
                front_p10=0.119,
                safety_turn_count=40,
                avoidance_streak=18,
            )
        )

    def test_releases_close_final_intersection_when_slightly_above_surface(self):
        segment = {
            "text": "now move forward over the buildings and towards the road and reach the intersection road and land there",
            "target_landmark": "intersection",
            "target_zone": "corridor",
            "actions": ["forward", "fly_over", "descend"],
        }
        self.assertTrue(
            should_release_final_corridor_depth_guard(
                segment,
                segment_index=3,
                segment_count=4,
                distance_to_goal=26.6,
                surface_clearance=15.13,
                front_contact_ratio=0.0,
                front_very_close_ratio=0.0,
                front_p10=0.0407,
                safety_turn_count=272,
                avoidance_streak=14,
            )
        )

    def test_keeps_guard_for_contact_or_non_final_segments(self):
        segment = {
            "text": "now move forward over the buildings and towards the road and reach the intersection road and land there",
            "target_landmark": "intersection",
            "target_zone": "corridor",
            "actions": ["forward", "descend"],
        }
        self.assertFalse(
            should_release_final_corridor_depth_guard(
                segment,
                segment_index=3,
                segment_count=4,
                distance_to_goal=31.4,
                surface_clearance=5.34,
                front_contact_ratio=0.01,
                front_very_close_ratio=0.01,
                front_p10=0.119,
                safety_turn_count=40,
                avoidance_streak=18,
            )
        )
        self.assertFalse(
            should_release_final_corridor_depth_guard(
                segment,
                segment_index=1,
                segment_count=4,
                distance_to_goal=31.4,
                surface_clearance=5.34,
                front_contact_ratio=0.0,
                front_very_close_ratio=0.01,
                front_p10=0.119,
                safety_turn_count=40,
                avoidance_streak=18,
            )
        )

    def test_keeps_guard_for_structure_targets_or_high_clearance(self):
        building_segment = {
            "text": "reach the red building and land there",
            "target_landmark": "building",
            "target_zone": "structure",
            "actions": ["forward", "descend"],
        }
        self.assertFalse(
            should_release_final_corridor_depth_guard(
                building_segment,
                segment_index=3,
                segment_count=4,
                distance_to_goal=31.4,
                surface_clearance=5.34,
                front_contact_ratio=0.0,
                front_very_close_ratio=0.01,
                front_p10=0.119,
                safety_turn_count=40,
                avoidance_streak=18,
            )
        )
        corridor_segment = {
            "text": "reach the intersection road and land there",
            "target_landmark": "intersection",
            "target_zone": "corridor",
            "actions": ["forward", "descend"],
        }
        self.assertFalse(
            should_release_final_corridor_depth_guard(
                corridor_segment,
                segment_index=3,
                segment_count=4,
                distance_to_goal=31.4,
                surface_clearance=19.0,
                front_contact_ratio=0.0,
                front_very_close_ratio=0.01,
                front_p10=0.119,
                safety_turn_count=40,
                avoidance_streak=18,
            )
        )


class CloseFinalStructureDescentDeferralTests(unittest.TestCase):
    def test_defers_structure_descent_outside_close_success_tail(self):
        segment = {
            "text": "landing by the charlie a chocolate building",
            "target_landmark": "building",
            "target_zone": "structure",
            "actions": ["descend", "land"],
        }

        self.assertTrue(
            should_defer_close_final_structure_descent(
                segment,
                distance_to_goal=32.65,
                success_distance=20.0,
                surface_clearance=19.18,
            )
        )

    def test_does_not_defer_success_zone_or_corridor_descent(self):
        structure_segment = {
            "text": "landing by the charlie a chocolate building",
            "target_landmark": "building",
            "target_zone": "structure",
            "actions": ["descend", "land"],
        }
        road_segment = {
            "text": "move straight to the zebra crossing and land",
            "target_landmark": "road",
            "target_zone": "corridor",
            "actions": ["descend", "land"],
        }

        self.assertFalse(
            should_defer_close_final_structure_descent(
                structure_segment,
                distance_to_goal=24.0,
                success_distance=20.0,
                surface_clearance=19.18,
            )
        )
        self.assertFalse(
            should_defer_close_final_structure_descent(
                road_segment,
                distance_to_goal=32.65,
                success_distance=20.0,
                surface_clearance=19.18,
            )
        )


class CloseFinalStructureDepthReleaseTests(unittest.TestCase):
    def test_releases_close_final_structure_when_depth_is_not_contact_like(self):
        segment = {
            "text": "landing by the charlie a chocolate building",
            "target_landmark": "building",
            "target_zone": "structure",
            "actions": ["descend", "land"],
        }

        self.assertTrue(
            should_release_close_final_structure_depth_guard(
                segment,
                segment_index=3,
                segment_count=4,
                distance_to_goal=32.87,
                surface_clearance=19.17,
                front_contact_ratio=None,
                front_very_close_ratio=0.0,
                front_p10=0.078,
                avoidance_streak=2,
            )
        )

    def test_keeps_guard_for_far_contact_or_non_final_cases(self):
        segment = {
            "text": "landing by the charlie a chocolate building",
            "target_landmark": "building",
            "target_zone": "structure",
            "actions": ["descend", "land"],
        }

        self.assertFalse(
            should_release_close_final_structure_depth_guard(
                segment,
                segment_index=3,
                segment_count=4,
                distance_to_goal=55.0,
                surface_clearance=19.17,
                front_contact_ratio=0.0,
                front_very_close_ratio=0.0,
                front_p10=0.078,
                avoidance_streak=2,
            )
        )
        self.assertFalse(
            should_release_close_final_structure_depth_guard(
                segment,
                segment_index=3,
                segment_count=4,
                distance_to_goal=32.87,
                surface_clearance=19.17,
                front_contact_ratio=0.02,
                front_very_close_ratio=0.0,
                front_p10=0.078,
                avoidance_streak=2,
            )
        )
        self.assertFalse(
            should_release_close_final_structure_depth_guard(
                segment,
                segment_index=0,
                segment_count=4,
                distance_to_goal=32.87,
                surface_clearance=19.17,
                front_contact_ratio=0.0,
                front_very_close_ratio=0.0,
                front_p10=0.078,
                avoidance_streak=2,
            )
        )


class FinalStructureDescentVisualClearanceTests(unittest.TestCase):
    def test_blocks_near_final_structure_descent_when_visual_clearance_is_low(self):
        segment = {
            "text": "go over the buildings and reach the white building and land there",
            "target_landmark": "building",
            "target_zone": "structure",
            "actions": ["forward", "fly_over", "descend"],
        }
        self.assertTrue(
            should_block_final_structure_descent_for_visual_clearance(
                segment,
                segment_index=3,
                segment_count=4,
                distance_to_goal=28.9,
                success_distance=20.0,
                surface_clearance=7.1,
                front_p10=0.043,
                front_contact_ratio=0.0,
                front_very_close_ratio=0.0,
            )
        )

    def test_allows_success_zone_surface_descent_to_be_handled_by_finish_logic(self):
        segment = {
            "text": "reach the white building and land there",
            "target_landmark": "building",
            "target_zone": "structure",
            "actions": ["descend"],
        }
        self.assertFalse(
            should_block_final_structure_descent_for_visual_clearance(
                segment,
                segment_index=3,
                segment_count=4,
                distance_to_goal=19.5,
                success_distance=20.0,
                surface_clearance=7.1,
                front_p10=0.043,
                front_contact_ratio=0.0,
                front_very_close_ratio=0.0,
            )
        )

    def test_does_not_block_final_road_or_clear_structure_descent(self):
        road_segment = {
            "text": "get down towards the road and land there",
            "target_landmark": "road",
            "target_zone": "corridor",
            "actions": ["descend"],
        }
        self.assertFalse(
            should_block_final_structure_descent_for_visual_clearance(
                road_segment,
                segment_index=3,
                segment_count=4,
                distance_to_goal=28.9,
                success_distance=20.0,
                surface_clearance=7.1,
                front_p10=0.043,
                front_contact_ratio=0.0,
                front_very_close_ratio=0.0,
            )
        )
        building_segment = {
            "text": "reach the white building and land there",
            "target_landmark": "building",
            "target_zone": "structure",
            "actions": ["descend"],
        }
        self.assertFalse(
            should_block_final_structure_descent_for_visual_clearance(
                building_segment,
                segment_index=3,
                segment_count=4,
                distance_to_goal=28.9,
                success_distance=20.0,
                surface_clearance=7.1,
                front_p10=0.08,
                front_contact_ratio=0.0,
                front_very_close_ratio=0.0,
            )
        )


class LearnedSegmentXyStepTests(unittest.TestCase):
    class HostileFloat:
        def __float__(self):
            raise RuntimeError("float conversion failed")

    class HostileCooldown:
        def __bool__(self):
            raise RuntimeError("bool conversion failed")

    def test_uses_clamped_base_before_the_safety_turn_threshold(self):
        self.assertEqual(
            choose_learned_segment_xy_step(45.0, 22.0, 38.0, 2, False),
            (38.0, 0, "base"),
        )

    def test_uses_caution_multiplier_for_three_through_five_safety_turns(self):
        self.assertEqual(
            choose_learned_segment_xy_step(30.0, 22.0, 38.0, 3, False),
            (25.5, 1, "safety_turn_caution"),
        )
        xy_step, multiplier, reason = choose_learned_segment_xy_step(
            38.0, 22.0, 38.0, 5, False
        )
        self.assertEqual((xy_step, multiplier, reason), (32.3, 1, "safety_turn_caution"))
        self.assertLessEqual(xy_step, 38.0)

    def test_uses_stall_multiplier_without_a_stronger_cooldown_penalty(self):
        self.assertEqual(
            choose_learned_segment_xy_step(30.0, 22.0, 38.0, 6, False),
            (22.0, 2, "safety_stall"),
        )
        self.assertEqual(
            choose_learned_segment_xy_step(38.0, 22.0, 38.0, 6, "active"),
            (26.599999999999998, 2, "safety_stall_cooldown"),
        )

    def test_invalid_base_still_applies_safety_turn_caution(self):
        self.assertEqual(
            choose_learned_segment_xy_step(float("nan"), 22.0, 38.0, 3, False),
            (22.0, 1, "safety_turn_caution"),
        )

    def test_invalid_base_still_applies_safety_stall_cooldown(self):
        self.assertEqual(
            choose_learned_segment_xy_step("invalid", 22.0, 38.0, 6, True),
            (22.0, 2, "safety_stall_cooldown"),
        )

    def test_sanitizes_bounds_and_safety_turn_count_without_raising(self):
        self.assertEqual(
            choose_learned_segment_xy_step(8.0, 10.0, 5.0, "not-a-count", False),
            (10.0, 0, "base"),
        )
        self.assertEqual(
            choose_learned_segment_xy_step("8", None, float("inf"), -1, False),
            (0.0, 0, "base"),
        )

    def test_normalizes_nonfinite_safety_turn_counts_to_base(self):
        for safety_turn_count in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(safety_turn_count=safety_turn_count):
                self.assertEqual(
                    choose_learned_segment_xy_step(45.0, 22.0, 38.0, safety_turn_count, False),
                    (38.0, 0, "base"),
                )

    def test_hostile_numeric_inputs_are_normalized_without_raising(self):
        hostile = self.HostileFloat()
        cases = (
            (hostile, 22.0, 38.0, 0, False),
            (30.0, hostile, 38.0, 0, False),
            (30.0, 22.0, hostile, 0, False),
            (30.0, 22.0, 38.0, hostile, False),
        )
        for args in cases:
            with self.subTest(args=args):
                result = choose_learned_segment_xy_step(*args)
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 3)
                self.assertIsInstance(result[0], float)
                self.assertIsInstance(result[1], int)
                self.assertIsInstance(result[2], str)
        self.assertEqual(
            choose_learned_segment_xy_step(hostile, 22.0, 38.0, 0, False),
            (22.0, 0, "base"),
        )

    def test_sanitizes_invalid_minimum_and_maximum_bounds_independently(self):
        self.assertEqual(
            choose_learned_segment_xy_step(8.0, "invalid", 10.0, 0, False),
            (8.0, 0, "base"),
        )
        self.assertEqual(
            choose_learned_segment_xy_step(8.0, 10.0, "invalid", 0, False),
            (10.0, 0, "base"),
        )
        self.assertEqual(
            choose_learned_segment_xy_step(8.0, 10.0, 5.0, 0, False),
            (10.0, 0, "base"),
        )

    def test_each_policy_tier_clamps_to_the_sanitized_lower_bound(self):
        for safety_turn_count, multiplier, reason in (
            (0, 0, "base"),
            (3, 1, "safety_turn_caution"),
            (6, 2, "safety_stall"),
        ):
            with self.subTest(safety_turn_count=safety_turn_count):
                self.assertEqual(
                    choose_learned_segment_xy_step(
                        1.0, 10.0, 40.0, safety_turn_count, False
                    ),
                    (10.0, multiplier, reason),
                )

    def test_base_policy_clamps_to_the_sanitized_upper_bound(self):
        self.assertEqual(
            choose_learned_segment_xy_step(100.0, 10.0, 40.0, 0, False),
            (40.0, 0, "base"),
        )

    def test_falsy_non_boolean_cooldown_uses_stall_reason(self):
        xy_step, multiplier, reason = choose_learned_segment_xy_step(38.0, 22.0, 38.0, 6, "")
        self.assertEqual(xy_step, 38.0 * 0.70)
        self.assertLessEqual(xy_step, 38.0)
        self.assertEqual((multiplier, reason), (2, "safety_stall"))

    def test_hostile_cooldown_bool_uses_safe_stall_fallback(self):
        self.assertEqual(
            choose_learned_segment_xy_step(30.0, 10.0, 60.0, 6, self.HostileCooldown()),
            (21.0, 2, "safety_stall"),
        )


class LearnedSegmentTargetAdjustmentTests(unittest.TestCase):
    def test_route_like_adjustment_preserves_xy_direction_and_z_with_audit_values(self):
        adjusted_delta, base_xy_step, xy_step, risk_tier, reason = (
            learned_segment_target_xy_adjustment(
                [30.0, 40.0, -7.0], True, 22.0, 38.0, 3, 210, None
            )
        )

        self.assertEqual(math.hypot(adjusted_delta[0], adjusted_delta[1]), xy_step)
        self.assertNotEqual(math.hypot(30.0, 40.0), xy_step)
        self.assertAlmostEqual(adjusted_delta[1] / adjusted_delta[0], 40.0 / 30.0)
        self.assertEqual(adjusted_delta[2], -7.0)
        self.assertEqual(base_xy_step, 38.0)
        self.assertAlmostEqual(xy_step, 32.3)
        self.assertEqual((risk_tier, reason), (1, "safety_turn_caution"))

    def test_non_route_like_adjustment_keeps_short_delta_without_policy_scaling(self):
        with patch("model_live_once.choose_learned_segment_xy_step") as choose_step:
            result = learned_segment_target_xy_adjustment(
                [3.0, 4.0, 7.0], False, 22.0, 38.0, 8, 210, 209
            )

        self.assertEqual(result, ([3.0, 4.0, 7.0], 5.0, 5.0, 0, "not_route_like"))
        choose_step.assert_not_called()

    def test_non_route_like_adjustment_clamps_long_delta_to_maximum(self):
        with patch("model_live_once.choose_learned_segment_xy_step") as choose_step:
            adjusted_delta, base_xy_step, xy_step, risk_tier, reason = (
                learned_segment_target_xy_adjustment(
                    [180.0, 60.0, -5.0], False, 22.0, 54.0, 0, 1, None
                )
            )

        self.assertAlmostEqual(math.hypot(adjusted_delta[0], adjusted_delta[1]), 54.0)
        self.assertAlmostEqual(adjusted_delta[1] / adjusted_delta[0], 60.0 / 180.0)
        self.assertEqual(adjusted_delta[2], -5.0)
        self.assertAlmostEqual(base_xy_step, math.hypot(180.0, 60.0))
        self.assertEqual((xy_step, risk_tier, reason), (54.0, 0, "not_route_like_clamped"))
        choose_step.assert_not_called()

    def test_valid_cooldown_is_active_but_future_and_invalid_markers_are_not(self):
        active = learned_segment_target_xy_adjustment(
            [30.0, 40.0, 1.0], True, 22.0, 38.0, 6, 210, 209
        )
        future = learned_segment_target_xy_adjustment(
            [30.0, 40.0, 1.0], True, 22.0, 38.0, 6, 210, 211
        )
        invalid = learned_segment_target_xy_adjustment(
            [30.0, 40.0, 1.0], True, 22.0, 38.0, 6, 210, "invalid"
        )

        self.assertEqual(active[-2:], (2, "safety_stall_cooldown"))
        self.assertEqual(future[-2:], (2, "safety_stall"))
        self.assertEqual(invalid[-2:], (2, "safety_stall"))

    def test_non_route_like_output_is_unscaled_with_audit_values(self):
        original_delta = [3.0, 4.0, 7.0]
        adjusted_delta, base_xy_step, xy_step, risk_tier, reason = (
            learned_segment_target_xy_adjustment(
                original_delta, False, 22.0, 38.0, 8, 210, 209
            )
        )

        self.assertEqual(adjusted_delta, original_delta)
        self.assertEqual((base_xy_step, xy_step), (5.0, 5.0))
        self.assertEqual((risk_tier, reason), (0, "not_route_like"))


class LearnedSegmentSafetyTurnStateTests(unittest.TestCase):
    def test_learned_depth_guard_tracks_target_and_segment_counts(self):
        guidance_state = {
            "learned_segment_homing_safety_turn_count": 5,
            "learned_segment_homing_segment_safety_turn_count": 8,
        }

        changed = record_learned_segment_depth_guard_turn(
            guidance_state,
            "depth_building_guard_turn",
            "learned_segment",
        )

        self.assertTrue(changed)
        self.assertEqual(guidance_state["learned_segment_homing_safety_turn_count"], 6)
        self.assertEqual(guidance_state["learned_segment_homing_segment_safety_turn_count"], 9)

    def test_only_learned_building_guard_tracks_safety_counts(self):
        for depth_reason, homing_target_kind in (
            ("depth_escape_scan_turn", "learned_segment"),
            ("depth_building_guard_turn", "reference_path"),
        ):
            with self.subTest(depth_reason=depth_reason, homing_target_kind=homing_target_kind):
                guidance_state = {
                    "learned_segment_homing_safety_turn_count": 2,
                    "learned_segment_homing_segment_safety_turn_count": 7,
                }
                self.assertFalse(
                    record_learned_segment_depth_guard_turn(
                        guidance_state,
                        depth_reason,
                        homing_target_kind,
                    )
                )
                self.assertEqual(guidance_state["learned_segment_homing_safety_turn_count"], 2)
                self.assertEqual(guidance_state["learned_segment_homing_segment_safety_turn_count"], 7)

    def test_segment_count_keeps_risk_after_target_local_count_resets(self):
        target_local_count = 0
        segment_count = 8

        _, _, xy_step, risk_tier, reason = learned_segment_target_xy_adjustment(
            [30.0, 40.0, 1.0],
            True,
            22.0,
            54.0,
            segment_count,
            210,
            186,
        )

        self.assertEqual(target_local_count, 0)
        self.assertEqual(xy_step, 35.0)
        self.assertEqual((risk_tier, reason), (2, "safety_stall"))


class FlightTargetTelemetryTests(unittest.TestCase):
    def test_normalizes_only_finite_three_dimension_positions(self):
        self.assertEqual(normalize_flight_target_number(2), 2.0)
        self.assertIsNone(normalize_flight_target_number(True))
        self.assertIsNone(normalize_flight_target_number(float("nan")))
        self.assertEqual(normalize_flight_target_position([1, 2.5, -3]), [1.0, 2.5, -3.0])
        for value in (None, [1, 2], [1, 2, float("inf")], [1, 2, True], "1,2,3"):
            with self.subTest(value=value):
                self.assertIsNone(normalize_flight_target_position(value))

    def test_event_copies_targets_and_normalizes_optional_values(self):
        before = [1.0, 2.0, 3.0]
        event = build_flight_target_telemetry_event(
            event="target_selected",
            step=9,
            target_before=before,
            target_after=[4, 5, 6],
            target_changed_reason="created",
            selected_landmark=None,
            target_source="fallback",
            ranker_top_support=None,
            ranker_influenced_target=False,
            current_position=[4, 5, 2],
            distance_to_dataset_goal=float("nan"),
            regression_start_step=None,
            regression_stop_step=None,
            recovery_reason=None,
            raw_action=None,
            applied_action=None,
        )
        before[0] = 99.0

        self.assertEqual(event["schema_version"], "scene16_flight_target_telemetry_v1")
        self.assertEqual(event["target_before"], [1.0, 2.0, 3.0])
        self.assertEqual(event["selected_target_position"], [4.0, 5.0, 6.0])
        self.assertEqual(event["distance_to_selected_target"], 4.0)
        self.assertIsNone(event["distance_to_dataset_goal"])
        self.assertEqual(event["trace_step"], 9)

    def test_action_only_recovery_keeps_target_and_has_no_change_reason(self):
        event = build_flight_target_telemetry_event(
            event="recovery",
            step=10,
            target_before=[1, 2, 3],
            target_after=[1, 2, 3],
            target_changed_reason=None,
            selected_landmark="road",
            target_source="explicit_landmark",
            ranker_top_support=None,
            ranker_influenced_target=False,
            current_position=[1, 2, 1],
            distance_to_dataset_goal=42.0,
            regression_start_step=9,
            regression_stop_step=None,
            recovery_reason="remote_env_done_recovery_pose",
            raw_action="MOVE_FORWARD",
            applied_action=None,
        )

        self.assertEqual(event["target_before"], event["target_after"])
        self.assertIsNone(event["target_changed_reason"])
        self.assertEqual(event["selected_target_position"], [1.0, 2.0, 3.0])

    def test_ranker_context_requires_top_support_landmark_for_causality(self):
        explicit = build_flight_target_ranker_context(
            {"target_landmark": "road", "landmarks": ["road"]},
            {"scene16_support_ranker_decision": {"applied": True}},
        )
        ranker = build_flight_target_ranker_context(
            {"target_landmark": "tower", "landmarks": [], "support_landmarks": ["tower"]},
            {
                "scene16_support_ranker_decision": {
                    "applied": True,
                    "final_episode_ids": ["episode-7"],
                    "scored_candidates": [{"episode_id": "episode-7", "score": 0.75}],
                },
                "support_examples": [{"episode_id": "episode-7", "landmarks": ["tower"]}],
            },
        )

        self.assertEqual(explicit["target_source"], "explicit_landmark")
        self.assertFalse(explicit["ranker_influenced_target"])
        self.assertEqual(ranker["target_source"], "ranker_artifact")
        self.assertTrue(ranker["ranker_influenced_target"])

    def test_appends_restricted_jsonl_and_builds_coordinate_free_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            telemetry_path = Path(root) / "flight_target_telemetry.jsonl"
            append_flight_target_telemetry(telemetry_path, {"event": "target_selected"})
            append_flight_target_telemetry(telemetry_path, {"event": "recovery"})
            rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
            manifest = build_flight_target_telemetry_manifest(telemetry_path, len(rows))

        self.assertEqual([row["event"] for row in rows], ["target_selected", "recovery"])
        self.assertEqual(set(manifest), {"path", "sha256", "event_count"})
        self.assertEqual(manifest["event_count"], 2)


if __name__ == "__main__":
    unittest.main()
