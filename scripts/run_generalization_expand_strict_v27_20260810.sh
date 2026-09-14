#!/usr/bin/env bash
set -euo pipefail

episode="33LKR6A5KNILLZ1T7XC7T891SBY1TO"
report="live_server/runtime/generalization_expand_strict_v27_20260810.json"
log="live_server/runtime/generalization_expand_strict_v27_20260810.log"

if [[ -e "$report" || -e "$log" ]]; then
  echo "Refusing to overwrite existing strict evidence: $report or $log" >&2
  exit 1
fi

/opt/conda/envs/AirVLN/bin/python -u scripts/run_scene16_gen_eval.py \
  --base-url http://127.0.0.1:18080 \
  --split train --scene-id 16 --only-episode-id "$episode" --limit 1 --resume \
  --out "$report" \
  --segment-grounding-ridge live_server/runtime/segment_grounding/ridge_all16k_finalgoal_v14_scene_interactions/segment_grounding_ridge.npz \
  --segment-grounding-final-ridge live_server/runtime/segment_grounding/final_tfidf_episode_anchor_v45/final_goal_tfidf_knn.npz \
  --disable-oracle-goal-homing --disable-oracle-path-homing \
  --max-black-drops 0 --max-collision-rollbacks 0 --surface-clearance 1.5 \
  --sample-sec 8 --stream-query "fps=24&turbo=1&latest=0" --stream-wait-first-sec 140 \
  --case-timeout-sec 620 --max-steps 900 \
  --learned-segment-target-radius 12 --learned-segment-max-age 34 \
  --learned-segment-min-xy-step 24 --learned-segment-max-xy-step 54 \
  --learned-segment-resample-worse-streak 4 --learned-segment-resample-regression 55 \
  --learned-segment-z-clamp 6 --learned-segment-cruise-altitude 42 \
  --learned-segment-descent-altitude 18 --segment-switch-confidence 0.48 \
  --segment-progress-distance 22 --segment-min-steps 3 \
  2>&1 | tee "$log"
