import json
import math
import os
import pickle
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path


DATASET_DATA_DIR = Path(os.environ.get("AIRVLN_DATASET_DATA_DIR", str(Path(__file__).resolve().parents[1] / "DATA" / "data")))
SCENE16_SUPPORT_RANKER_MODEL = "tfidf_cosine_instruction_ranker"
SCENE16_SUPPORT_RANKER_SPLIT = "candidate_episode_connected_component"
SCENE16_ROUTE_PRIOR_FORMAT = "scene16_target_route_prior_v1"
SCENE16_ROUTE_PRIOR_FORBIDDEN = (
    "reference_path", "oracle_path", "gt_path", "oracle_diagnostic_only",
    "offline_diagnostic_only", "target_off_route", "route_progress_regression",
    "reference_path_distance",
)

LANDMARK_PATTERNS = {
    "runway": ["runway", "airstrip", "跑道"],
    "taxiway": ["taxiway", "滑行道"],
    "hangar": ["hangar", "机库"],
    "airport": ["airport", "机场"],
    "apron": ["apron", "停机坪"],
    "bridge": ["bridge", "桥", "索桥", "高架桥"],
    "tower": ["tower", "塔", "塔台"],
    "road": ["road", "street", "avenue", "道路", "公路", "路面", "街道"],
    "intersection": ["intersection", "junction", "crossroad", "路口", "交叉口", "十字路口"],
    "building": ["building", "buildings", "tower block", "大楼", "建筑", "楼房"],
    "park": ["park", "garden", "公园", "花园"],
    "shop": ["shop", "store", "商店", "店铺"],
    "billboard": ["billboard", "advertisement", "广告牌", "横幅广告"],
    "rooftop": ["rooftop", "roof", "屋顶", "楼顶"],
    "tree": ["tree", "trees", "树林", "树木"],
    "path": ["path", "trail", "walkway", "小路", "路径"],
    "water": ["water", "river", "lake", "ocean", "sea", "beach", "水面", "海边"],
    "statue": ["statue", "雕像"],
    "skyscraper": ["skyscraper", "high-rise", "high rise", "摩天楼"],
    "city": ["city", "城区", "城市"],
    "ground": ["ground", "地面"],
}

ACTION_PATTERNS = {
    "take_off": ["take off", "ascend", "go up", "rise", "起飞", "上升"],
    "forward": [
        "forward",
        "ahead",
        "straight",
        "fly through",
        "fly to",
        "fly toward",
        "fly towards",
        "fly past",
        "flys to",
        "flys toward",
        "flys towards",
        "flys past",
        "move to",
        "move toward",
        "move towards",
        "continue along",
        "continue on",
        "follow along",
        "fly close to",
        "go to",
        "go across",
        "toward",
        "towards",
        "move forward",
        "fly along",
        "head back",
        "head back onto",
        "head to",
        "head toward",
        "head towards",
        "proceed forward",
        "proceed further",
        "pass ",
        "前进",
        "向前",
        "穿过",
        "飞向",
    ],
    "turn_left": ["turn left", "left turn", "veer left", "bear left", "slight left", "slightly left", "左转", "向左"],
    "turn_right": ["turn right", "right turn", "veer right", "bear right", "slight right", "slightly right", "右转", "向右"],
    "fly_over": ["fly over", "pass over", "越过", "飞过"],
    "descend": ["go down", "get down", "descend", "lower", "land", "decrease the height", "fly down", "下降", "降落", "降低高度"],
    "look_up": ["look up", "tilt up", "tilt upwards", "pan up", "camera up", "fly up", "slightly up", "抬头", "向上看"],
    "look_down": ["look down", "tilt down", "tilt downwards", "pan down", "camera down", "slightly down", "向下看"],
    "look_left": ["look left", "pan left", "tilt left", "camera left", "向左看"],
    "look_right": ["look right", "pan right", "tilt right", "camera right", "向右看"],
    "stop": ["stop", "halt", "停止"],
}

ACTION_PATTERN_ALIASES = {
    "take_off": [
        "taking off",
        "ascending",
        "going up",
        "fly up",
        "rising",
        "raise up",
        "raising up",
        "increase elevation",
        "increased elevation",
        "increasing elevation",
        "raise elevation",
        "raised elevation",
        "raising elevation",
        "raised",
        "left up",
        "right up",
        "up left",
        "up right",
        "up rights",
    ],
    "forward": [
        "flying through",
        "flying to",
        "flying toward",
        "flying towards",
        "flying past",
        "continued straight",
        "continued straight for",
        "flys down",
        "flys over",
        "flys to",
        "move to",
        "move toward",
        "move towards",
        "continue along",
        "continue on",
        "follow along",
        "fly close to",
        "fly along",
        "head back",
        "head back onto",
        "onto the road",
        "along the road",
        "going to",
        "moving forward",
        "heading to",
        "heading toward",
        "heading towards",
        "zoom in",
        "zooming in",
        "approach",
        "approaching",
        "passing",
    ],
    "turn_left": [
        "turning left",
        "turned left",
        "take left",
        "take a left",
        "turn to the left",
        "make a left turn",
        "turn the left",
        "rotating left",
        "rotate left",
        "turn further left",
        "zig bag left",
        "zig zag left",
        "left up",
        "up left",
        "turn around",
    ],
    "turn_right": [
        "turning right",
        "turned right",
        "take right",
        "take a right",
        "turn to the right",
        "make a right turn",
        "turn the right",
        "rotating right",
        "rotate right",
        "turn further right",
        "right up",
        "up right",
        "up rights",
        "rights right",
        "turn around",
    ],
    "fly_over": [
        "flying over",
        "fly above",
        "flying above",
        "fly plane over",
        "go over",
        "going over",
        "passing over",
        "cross over",
        "cross top",
        "go across the building",
        "go across building",
        "across the building",
        "over the",
    ],
    "descend": [
        "going down",
        "getting down",
        "drop down",
        "dropping down",
        "turn down",
        "turning down",
        "descending",
        "lowered",
        "lowered elevation",
        "lowering",
        "landing",
        "flying down",
        "down toward",
        "down towards",
        "onto the building",
        "onto the roof",
        "onto the rooftop",
    ],
    "look_down": ["face down", "facing down"],
    "look_left": ["look to your left", "look to the left", "looking to your left", "looking to the left"],
    "look_right": ["look to your right", "look to the right", "looking to your right", "looking to the right"],
    "turn": ["turn", "turns", "face", "faces", "facing", "look at", "looks at", "look toward", "looks toward"],
}
for _action_label, _aliases in ACTION_PATTERN_ALIASES.items():
    ACTION_PATTERNS.setdefault(_action_label, []).extend(_aliases)

GENERIC_TARGETS = {"city", "ground"}
LOW_ALTITUDE_TARGETS = {"runway", "taxiway", "road", "intersection", "apron", "park", "path", "ground", "water"}
STRUCTURE_TARGETS = {"bridge", "tower", "building", "hangar", "rooftop", "billboard", "shop", "statue", "skyscraper"}
AREA_TARGETS = {"airport", "park", "city", "apron", "water"}
SEGMENT_SPLIT_RE = re.compile(r"\bthen\b|[.;。！？；]\s*|(?:之后|然后|接着|再)\s*")
ASCII_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
LONG_ENGLISH_ACTION_BOUNDARY_RE = re.compile(
    r"\b(?:and|then|after|before|until|when|once)\s+"
    r"(?=(?:turn\s+(?:left|right|back|around)|fly|move|proceed|continue|cross|"
    r"descend|ascend|rise|lower|land|stop|head)\b)|"
    r"\b(?=(?:turn\s+(?:left|right|back|around)|fly|move|proceed|continue|cross|"
    r"descend|ascend|rise|lower|land|stop|head)\b)",
    re.IGNORECASE,
)
RELATION_PATTERNS = [
    (
        "right_of",
        re.compile(
            r"\b(?:to|on)\s+the\s+right\s+side\s+of\s+(?:the\s+)?([a-z0-9 ,'-]+?)(?=$|[.;]|"
            r"\b(?:and|then|while|before|after)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "left_of",
        re.compile(
            r"\b(?:to|on)\s+the\s+left\s+side\s+of\s+(?:the\s+)?([a-z0-9 ,'-]+?)(?=$|[.;]|"
            r"\b(?:and|then|while|before|after)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "near",
        re.compile(
            r"\b(?:when\s+)?near\s+(?:the\s+)?([a-z0-9 ,'-]+?)(?=$|[.;]|"
            r"\b(?:and|then|while|before|after)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "between",
        re.compile(
            r"\bbetween\s+(?:the\s+)?([a-z0-9 ,'-]+?)\s+and\s+(?:the\s+)?([a-z0-9 ,'-]+?)(?=$|[.;])",
            re.IGNORECASE,
        ),
    ),
    (
        "facing",
        re.compile(
            r"\bfacing\s+(?:the\s+)?([a-z0-9 ,'-]+?)(?=$|[.;]|"
            r"\b(?:and|then|while|before|after)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "target_right",
        re.compile(r"\b(?:to|on)\s+your\s+right\b", re.IGNORECASE),
    ),
    (
        "target_left",
        re.compile(r"\b(?:to|on)\s+your\s+left\b", re.IGNORECASE),
    ),
]
MOTION_ACTIONS = {"take_off", "forward", "turn_left", "turn_right", "turn", "fly_over", "descend"}
VISUAL_ACTIONS = {"look_up", "look_down", "look_left", "look_right", "stop"}


def contains_cjk(text):
    return any("\u4e00" <= char <= "\u9fff" for char in str(text or ""))


def _match_positions(text, patterns_map):
    raw_text = str(text or "")
    lower = raw_text.lower()
    hits = []
    for label, patterns in patterns_map.items():
        best = None
        for pattern in patterns:
            pos = raw_text.find(pattern) if contains_cjk(pattern) else lower.find(pattern.lower())
            if pos >= 0 and (best is None or pos < best):
                best = pos
        if best is not None:
            hits.append((best, label))
    hits.sort(key=lambda item: item[0])
    return [label for _, label in hits]


def ordered_unique(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def has_action_cue(text):
    return bool(extract_actions(text))


def split_long_english_clause(part):
    if contains_cjk(part) or len(ASCII_TOKEN_RE.findall(part)) <= 28:
        return [part]
    return [
        clause.strip(" .;,\n\t")
        for clause in LONG_ENGLISH_ACTION_BOUNDARY_RE.split(part)
        if clause.strip(" .;,\n\t")
    ]


def split_segments(text):
    initial_parts = [
        part.strip(" .;,\n\t")
        for part in SEGMENT_SPLIT_RE.split(str(text or ""))
        if part.strip(" .;,\n\t")
    ]
    raw_parts = [clause for part in initial_parts for clause in split_long_english_clause(part)]
    merged = []
    for part in raw_parts:
        words = part.split()
        lower = part.lower()
        if not merged:
            merged.append(part)
            continue
        prev = merged[-1]
        prev_lower = prev.lower()
        prev_has_action = has_action_cue(prev)
        part_has_action = has_action_cue(part)
        if prev_lower.endswith((" and", " then", " after that")):
            merged[-1] = f"{prev.rstrip(' ,.;')} {part}".strip()
            continue
        if lower.startswith(("and ", "then ", "after that ")):
            merged[-1] = f"{prev.rstrip(' ,.;')} {part}".strip()
            continue
        if len(words) <= 2 and not part_has_action:
            merged[-1] = f"{prev.rstrip(' ,.;')} {part}".strip()
            continue
        if len(words) <= 3 and not part_has_action and prev_has_action:
            merged[-1] = f"{prev.rstrip(' ,.;')} {part}".strip()
            continue
        merged.append(part)
    return merged


def tokenize_query(text):
    return set(ASCII_TOKEN_RE.findall(str(text or "").lower()))


def normalize_phrase(text):
    return re.sub(r"\s+", " ", str(text or "")).strip(" .,:;\n\t")


def with_article(noun_phrase):
    phrase = normalize_phrase(noun_phrase)
    if not phrase:
        return phrase
    if phrase.lower().startswith(("the ", "a ", "an ")):
        return phrase
    return f"the {phrase}"


def extract_landmarks(text):
    return ordered_unique(_match_positions(text, LANDMARK_PATTERNS))


def extract_actions(text):
    return ordered_unique(_match_positions(text, ACTION_PATTERNS))


def extract_relations(text):
    raw_text = str(text or "")
    relations = []
    seen = set()
    for relation_type, pattern in RELATION_PATTERNS:
        for match in pattern.finditer(raw_text):
            anchors = [normalize_phrase(group) for group in match.groups() if normalize_phrase(group)]
            key = (relation_type, tuple(anchor.lower() for anchor in anchors))
            if key in seen:
                continue
            seen.add(key)
            relations.append(
                {
                    "type": relation_type,
                    "text": normalize_phrase(match.group(0)),
                    "anchors": anchors,
                    "start": match.start(),
                }
            )
    relations.sort(key=lambda item: item["start"])
    for relation in relations:
        relation.pop("start", None)
    return relations


def instruction_text(episode):
    instruction = episode.get("instruction", "")
    if isinstance(instruction, dict):
        return str(instruction.get("instruction_text", "")).strip()
    return str(instruction).strip()


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


@lru_cache(maxsize=32)
def load_dataset_records(split=None, dataset="aerialvln"):
    records = []
    for dataset_name, split_name, path in dataset_files(split=split, dataset=dataset):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        episodes = raw.get("episodes", raw if isinstance(raw, list) else [])
        for episode in episodes:
            text = instruction_text(episode)
            if not text:
                continue
            relations = extract_relations(text)
            records.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "episode_id": episode.get("episode_id"),
                    "trajectory_id": episode.get("trajectory_id"),
                    "scene_id": int(episode.get("scene_id", -1)),
                    "instruction": text,
                    "tokens": tokenize_query(text),
                    "landmarks": extract_landmarks(text),
                    "actions": extract_actions(text),
                    "relation_types": [item["type"] for item in relations],
                }
            )
    return records


def search_support_examples(text, split=None, scene_id=None, dataset="aerialvln", limit=3, allowed_episode_ids=None):
    query_tokens = tokenize_query(text)
    query_landmarks = extract_landmarks(text)
    query_actions = extract_actions(text)
    query_relations = {item["type"] for item in extract_relations(text)}
    allowed_ids = None if allowed_episode_ids is None else set(allowed_episode_ids)
    results = []
    for record in load_dataset_records(split=split, dataset=dataset):
        if scene_id is not None and int(record["scene_id"]) != int(scene_id):
            continue
        if allowed_ids is not None and record.get("episode_id") not in allowed_ids:
            continue
        if record["instruction"].strip() == str(text or "").strip():
            continue
        overlap_tokens = len(query_tokens & record["tokens"])
        overlap_landmarks = len(set(query_landmarks) & set(record["landmarks"]))
        overlap_actions = len(set(query_actions) & set(record["actions"]))
        overlap_relations = len(query_relations & set(record.get("relation_types", [])))
        score = overlap_tokens + overlap_landmarks * 4 + overlap_actions * 2 + overlap_relations * 3
        if overlap_tokens == 0 and overlap_landmarks == 0 and overlap_relations == 0:
            continue
        results.append(
            {
                "score": score,
                "episode_id": record["episode_id"],
                "trajectory_id": record["trajectory_id"],
                "scene_id": record["scene_id"],
                "instruction": record["instruction"],
                "landmarks": record["landmarks"],
                "actions": record["actions"],
                "relation_types": record.get("relation_types", []),
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[: max(1, min(int(limit), 8))]


def directed_structure_target(segment):
    lower = str(segment.get("text") or "").lower()
    segment_landmarks = segment.get("landmarks", [])
    structure_landmarks = [item for item in segment_landmarks if item in STRUCTURE_TARGETS]
    if not structure_landmarks:
        return None
    if any(
        cue in lower
        for cue in (
            "skyscraper",
            "rooftop",
            "roof",
            "top of",
            "head to",
            "heading to",
            "land at",
            "land on",
            "land onto",
            "onto the building",
            "onto the roof",
            "to the other very tall",
            "to the large",
        )
    ):
        return structure_landmarks[-1]
    return None


def choose_target_landmark(segment_infos, landmarks, support_landmarks, descent_target=None, relations=None):
    relation_types = {item.get("type") for item in (relations or [])}
    for segment in reversed(segment_infos):
        directed_target = directed_structure_target(segment)
        if directed_target:
            return directed_target
    if descent_target:
        return descent_target
    if "near" in relation_types and len(landmarks) >= 2:
        for landmark in landmarks:
            if landmark not in GENERIC_TARGETS:
                return landmark
    for segment in reversed(segment_infos):
        low_targets = [item for item in segment["landmarks"] if item in LOW_ALTITUDE_TARGETS]
        if low_targets:
            return low_targets[-1]
    for segment in reversed(segment_infos):
        for landmark in reversed(segment["landmarks"]):
            if landmark not in GENERIC_TARGETS:
                return landmark
    for landmark in reversed(landmarks):
        if landmark not in GENERIC_TARGETS:
            return landmark
    for landmark in support_landmarks:
        if landmark not in GENERIC_TARGETS:
            return landmark
    if landmarks:
        return landmarks[-1]
    if support_landmarks:
        return support_landmarks[0]
    return None


def infer_target_zone(target_landmark):
    if target_landmark in LOW_ALTITUDE_TARGETS:
        return "corridor"
    if target_landmark in STRUCTURE_TARGETS:
        return "structure"
    if target_landmark in AREA_TARGETS:
        return "area"
    if target_landmark:
        return "target"
    return None


def infer_descent_target(segment_infos, landmarks):
    for segment in reversed(segment_infos):
        if "descend" in segment["actions"]:
            for landmark in reversed(segment["landmarks"]):
                if landmark in LOW_ALTITUDE_TARGETS:
                    return landmark
    for landmark in reversed(landmarks):
        if landmark in LOW_ALTITUDE_TARGETS:
            return landmark
    return None


def infer_segment_kind(segment_text, segment_actions, segment_relations):
    lower = str(segment_text or "").lower()
    motion_count = sum(1 for action in segment_actions if action in MOTION_ACTIONS)
    visual_count = sum(1 for action in segment_actions if action in VISUAL_ACTIONS)
    if any(item.get("type") == "facing" for item in segment_relations):
        visual_count += 1
    if any(token in lower for token in ("look ", "tilt ", "pan ", "view ", "camera ", "facing ")):
        visual_count += 1
    if visual_count > 0 and motion_count == 0:
        return "visual"
    if visual_count > 0 and motion_count > 0:
        return "mixed"
    return "motion"


def build_segment_plan(segment_infos, support_landmarks):
    plan = []
    for index, segment in enumerate(segment_infos):
        segment_landmarks = segment.get("landmarks", [])
        segment_relations = segment.get("relations", [])
        segment_actions = segment.get("actions", [])
        descent_target = infer_descent_target([segment], segment_landmarks)
        target_landmark = choose_target_landmark(
            [segment],
            segment_landmarks,
            support_landmarks,
            descent_target=descent_target,
            relations=segment_relations,
        )
        target_zone = infer_target_zone(target_landmark)
        segment_kind = infer_segment_kind(
            segment.get("text", ""),
            segment_actions,
            segment_relations,
        )
        grounded_instruction = compose_grounded_instruction(
            segment.get("text", ""),
            {
                "landmarks": segment_landmarks,
                "relations": segment_relations,
                "support_landmarks": support_landmarks,
                "target_landmark": target_landmark,
                "actions": segment_actions,
                "descent_target": descent_target,
            },
        )
        plan.append(
            {
                "index": index,
                "text": segment.get("text", ""),
                "actions": segment_actions,
                "landmarks": segment_landmarks,
                "relations": segment_relations,
                "target_landmark": target_landmark,
                "target_zone": target_zone,
                "segment_kind": segment_kind,
                "descent_target": descent_target,
                "forward_route": bool(segment_landmarks or segment_relations)
                and ("forward" in segment_actions or "fly_over" in segment_actions or "take_off" in segment_actions),
                "grounded_instruction": grounded_instruction,
            }
        )
    return plan


def summarize_instruction_profile(segment_plan, relations, landmarks):
    motion_segments = sum(1 for item in segment_plan if item.get("segment_kind") == "motion")
    mixed_segments = sum(1 for item in segment_plan if item.get("segment_kind") == "mixed")
    visual_segments = sum(1 for item in segment_plan if item.get("segment_kind") == "visual")
    relation_count = len(relations or [])
    facing_relations = sum(1 for item in (relations or []) if item.get("type") == "facing")
    landmark_count = len(landmarks or [])
    relation_dense_reference = relation_count >= 4 or (relation_count >= 2 and landmark_count >= 5)
    direction_token_segments = 0
    for item in segment_plan:
        text = normalize_phrase(item.get("text", "")).lower()
        words = text.split()
        actions = set(item.get("actions") or [])
        if (
            1 <= len(words) <= 5
            and actions
            and actions <= {"take_off", "turn_left", "turn_right", "look_up"}
            and any(token in words for token in ("left", "right", "up", "rights"))
        ):
            direction_token_segments += 1

    profile = "balanced"
    if (
        segment_plan
        and len(segment_plan) >= 12
        and direction_token_segments / max(1, len(segment_plan)) >= 0.75
        and relation_count == 0
    ):
        profile = "direction_token_chain"
    elif (
        visual_segments == 0
        and mixed_segments >= 4
        and motion_segments >= 3
        and facing_relations >= 4
        and relation_dense_reference
    ):
        profile = "reference_dense"
    elif (
        visual_segments >= 1
        and mixed_segments >= 3
        and visual_segments + mixed_segments >= max(3, motion_segments)
        and (
            facing_relations >= 3
            or relation_count >= 4
            or (mixed_segments >= 4 and visual_segments >= 2 and landmark_count >= 6)
        )
    ):
        profile = "visual_mixed"
    elif relation_dense_reference:
        profile = "reference_dense"
    elif visual_segments >= 2 or (facing_relations >= 2 and visual_segments + mixed_segments >= motion_segments):
        profile = "visual_boundary"
    elif motion_segments >= max(4, visual_segments + mixed_segments) and relation_count <= 2:
        profile = "motion_chain"

    alignment_strength = "balanced"
    if profile == "motion_chain":
        alignment_strength = "strong"
    elif profile in {"reference_dense", "visual_boundary", "direction_token_chain"}:
        alignment_strength = "light"

    return {
        "profile": profile,
        "alignment_strength": alignment_strength,
        "prefer_original_instruction": profile in {"reference_dense", "visual_boundary"},
        "semantic_quality": "terse_direction_tokens" if profile == "direction_token_chain" else "normal",
        "motion_segments": motion_segments,
        "mixed_segments": mixed_segments,
        "visual_segments": visual_segments,
        "relation_count": relation_count,
        "facing_relations": facing_relations,
        "landmark_count": landmark_count,
        "direction_token_segments": direction_token_segments,
        "segment_count": len(segment_plan),
    }


def compose_grounded_instruction(text, grounding):
    target = grounding.get("target_landmark") or "target area"
    relations = grounding.get("relations", [])
    references = [item for item in grounding.get("landmarks", []) if item != target][:2]
    support_refs = [
        item for item in grounding.get("support_landmarks", []) if item not in references and item != target
    ][:2]
    if support_refs and len(references) < 2 and not relations:
        references.extend(support_refs[: 2 - len(references)])

    actions = grounding.get("actions", [])
    descent_target = grounding.get("descent_target")
    parts = []
    if "take_off" in actions:
        parts.append("take off and fly forward")
    else:
        parts.append("fly forward")
    parts.append(f"toward {with_article(target)}")
    if references:
        if len(references) == 1:
            parts.append(f"using the {references[0]} as the main reference landmark")
        else:
            parts.append(f"using the {references[0]} and {references[1]} as reference landmarks")
    for relation in relations[:2]:
        anchors = relation.get("anchors", [])
        if relation["type"] == "near" and anchors:
            parts.append(f"use {with_article(anchors[0])} as the near-range cue before the final approach")
        elif relation["type"] == "right_of" and anchors:
            parts.append(f"approach on the right side of {with_article(anchors[0])}")
        elif relation["type"] == "left_of" and anchors:
            parts.append(f"approach on the left side of {with_article(anchors[0])}")
        elif relation["type"] == "between" and len(anchors) >= 2:
            parts.append(f"stay between {with_article(anchors[0])} and {with_article(anchors[1])} on approach")
        elif relation["type"] == "facing" and anchors:
            parts.append(f"keep the camera and heading aligned with {with_article(anchors[0])}")
        elif relation["type"] == "target_right":
            parts.append("keep the target slightly to your right during approach")
        elif relation["type"] == "target_left":
            parts.append("keep the target slightly to your left during approach")
    if "turn_left" in actions and "turn_right" not in actions:
        parts.append("favor the left turn cues along the route")
    elif "turn_right" in actions and "turn_left" not in actions:
        parts.append("favor the right turn cues along the route")
    if "fly_over" in actions and target in STRUCTURE_TARGETS:
        parts.append(f"clear {with_article(target)} before descending")
    if descent_target:
        parts.append(f"descend only when close to {with_article(descent_target)}")
    else:
        parts.append("keep safe altitude until the target area is close")
    parts.append("avoid obstacles and do not stop early")
    return ". ".join(parts) + "."


def compact_grounding_view(grounding):
    return {
        "segments": [
            {
                "index": item.get("index", index),
                "text": item.get("text", ""),
                "actions": item.get("actions", []),
                "landmarks": item.get("landmarks", []),
                "relations": item.get("relations", []),
            }
            for index, item in enumerate(grounding.get("segments", [])[:8])
        ],
        "segment_plan": grounding.get("segment_plan", [])[:8],
        "instruction_profile": grounding.get("instruction_profile"),
        "actions": grounding.get("actions", []),
        "landmarks": grounding.get("landmarks", []),
        "relations": grounding.get("relations", []),
        "support_landmarks": grounding.get("support_landmarks", []),
        "target_landmark": grounding.get("target_landmark"),
        "target_zone": grounding.get("target_zone"),
        "descent_target": grounding.get("descent_target"),
        "forward_route": grounding.get("forward_route", False),
        "grounded_instruction": grounding.get("grounded_instruction"),
        "scene16_support_ranker_decision": grounding.get("scene16_support_ranker_decision"),
    }


def _load_trusted_scene16_support_ranker(artifact_path):
    if not artifact_path:
        return None, "not_configured"
    try:
        artifact = Path(artifact_path)
        metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
        parameters = metadata.get("parameters") or {}
        if (
            parameters.get("model") != SCENE16_SUPPORT_RANKER_MODEL
            or parameters.get("validation_split") != SCENE16_SUPPORT_RANKER_SPLIT
        ):
            return None, "untrusted_artifact"
        model = pickle.loads((artifact / "model.pkl").read_bytes())
        if not isinstance(model, dict):
            return None, "artifact_error"
        targets = model.get("targets")
        vocabulary = model.get("vocabulary")
        if not isinstance(targets, dict) or not isinstance(vocabulary, dict):
            return None, "artifact_error"
        return model, None
    except Exception:
        return None, "artifact_error"


def _contains_forbidden_route_prior_token(value):
    if isinstance(value, dict):
        return any(
            _contains_forbidden_route_prior_token(key) or _contains_forbidden_route_prior_token(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_route_prior_token(item) for item in value)
    return isinstance(value, str) and any(token in value.lower() for token in SCENE16_ROUTE_PRIOR_FORBIDDEN)


def _is_finite_vector3(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in value
        )
    )


def _is_finite_vector2(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in value
        )
    )


def _compact_route_prior_candidate(candidate):
    if not isinstance(candidate, dict):
        return None
    compact = {
        "episode_id": candidate.get("episode_id"),
        "route_prior": candidate.get("route_prior"),
        "off_route_penalty": candidate.get("off_route_penalty", 0.0),
    }
    if _is_finite_vector2(candidate.get("route_goal_unit_xy")):
        compact["route_goal_unit_xy"] = [
            float(item) for item in candidate["route_goal_unit_xy"]
        ]
    templates = []
    for template in candidate.get("segment_delta_templates") or []:
        if not isinstance(template, dict) or not _is_finite_vector3(template.get("delta_xyz")):
            continue
        templates.append({
            "segment_index": template.get("segment_index"),
            "segment_count": template.get("segment_count"),
            "segment_ratio": template.get("segment_ratio"),
            "delta_xyz": [float(item) for item in template["delta_xyz"]],
            "actions": template.get("actions") or [],
            "target_landmark": template.get("target_landmark"),
            "target_zone": template.get("target_zone"),
        })
    if templates:
        compact["segment_delta_templates"] = templates
    return compact


def load_scene16_target_route_prior(artifact_path):
    if not artifact_path:
        return None, "not_configured"
    try:
        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        if _contains_forbidden_route_prior_token(artifact):
            return None, "contaminated_artifact"
        if artifact.get("format") != SCENE16_ROUTE_PRIOR_FORMAT:
            return None, "incompatible_artifact"
        candidates = artifact.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return None, "artifact_error"
        for candidate in candidates:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("episode_id"), str):
                return None, "artifact_error"
            for field in ("route_prior", "off_route_penalty"):
                if field in candidate:
                    score = candidate[field]
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                        return None, "artifact_error"
            for template in candidate.get("segment_delta_templates") or []:
                if not isinstance(template, dict) or not _is_finite_vector3(template.get("delta_xyz")):
                    return None, "artifact_error"
            if "route_goal_unit_xy" in candidate and not _is_finite_vector2(candidate.get("route_goal_unit_xy")):
                return None, "artifact_error"
        return artifact, None
    except Exception:
        return None, "artifact_error"


def is_scene16_route_prior_enabled(config):
    return bool(
        isinstance(config, dict)
        and config.get("enable") is True
        and isinstance(config.get("path"), str)
        and bool(config["path"])
    )


def _rerank_scene16_support_examples_with_decision(
    text, examples, artifact_path, minimum_margin=0.01, model=None, load_reason=None,
    route_prior_model=None, route_prior_enabled=False, route_prior_minimum_margin=0.01,
    route_prior_load_reason=None,
):
    original = list(examples or [])
    original_episode_ids = [item["episode_id"] for item in original if isinstance(item.get("episode_id"), str)]

    def decision(
        applied=False,
        fallback_reason=None,
        scored_candidates=None,
        top_two_margin=None,
        final=None,
        candidate_source=None,
        route_prior_top_candidate=None,
        route_prior_candidates=None,
    ):
        final_examples = original if final is None else final
        record = {
            "applied": applied,
            "fallback_reason": fallback_reason,
            "original_episode_ids": original_episode_ids,
            "final_episode_ids": [
                item["episode_id"] for item in final_examples if isinstance(item.get("episode_id"), str)
            ],
            "scored_candidates": scored_candidates or [],
            "top_two_margin": top_two_margin,
            "route_prior_enabled": bool(route_prior_enabled),
        }
        if candidate_source is not None:
            record["candidate_source"] = candidate_source
        if route_prior_top_candidate is not None:
            record["route_prior_top_candidate"] = route_prior_top_candidate
        if route_prior_candidates is not None:
            record["route_prior_candidates"] = route_prior_candidates
        return record

    if model is None and load_reason is None and not route_prior_enabled:
        model, load_reason = _load_trusted_scene16_support_ranker(artifact_path)
    if route_prior_enabled and route_prior_model is None:
        return original, decision(fallback_reason=route_prior_load_reason or "route_prior_artifact_error")
    if model is None and not route_prior_enabled:
        return original, decision(fallback_reason=load_reason or "artifact_error")
    if len(original) < 2:
        return original, decision(fallback_reason="fewer_than_two_candidates")
    try:
        vocabulary = model["vocabulary"] if model is not None else {}
        targets = model["targets"] if model is not None else {}
        query = Counter(re.findall(r"[a-z0-9]+", str(text or "").lower()))

        def score(candidate):
            target = Counter(re.findall(r"[a-z0-9]+", str(targets[candidate["episode_id"]]).lower()))
            query_vector = {token: count * vocabulary[token] for token, count in query.items() if token in vocabulary}
            target_vector = {token: count * vocabulary[token] for token, count in target.items() if token in vocabulary}
            numerator = sum(value * target_vector.get(token, 0.0) for token, value in query_vector.items())
            denominator = math.sqrt(sum(value * value for value in query_vector.values())) * math.sqrt(
                sum(value * value for value in target_vector.values())
            )
            return numerator / denominator if denominator else 0.0

        route_candidates = {
            item["episode_id"]: item for item in (route_prior_model or {}).get("candidates", [])
        }
        if route_prior_enabled:
            query_tokens = set(re.findall(r"[a-z0-9]+", str(text or "").lower()))
            scored = []
            for index, item in enumerate(original):
                candidate = route_candidates.get(item.get("episode_id"))
                if candidate is None:
                    continue
                base_score = score(item) if model is not None and item.get("episode_id") in targets else 0.0
                candidate_tokens = {
                    token.lower()
                    for token in candidate.get("tokens", [])
                    if isinstance(token, str)
                }
                token_overlap = len(query_tokens.intersection(candidate_tokens))
                combined = (
                    base_score
                    + float(candidate.get("route_prior", 0.0))
                    + float(token_overlap)
                    - float(candidate.get("off_route_penalty", 0.0))
                )
                scored.append((index, item, combined))
        else:
            scored = [(index, item, score(item)) for index, item in enumerate(original) if item.get("episode_id") in targets]
        if len(scored) < 2:
            return original, decision(fallback_reason="fewer_than_two_scored_candidates")
        ranked = sorted(scored, key=lambda item: (-item[2], item[0]))
        top_two_margin = ranked[0][2] - ranked[1][2]
        scored_candidates = [
            {"episode_id": item["episode_id"], "score": score}
            for _, item, score in scored
            if isinstance(item.get("episode_id"), str) and math.isfinite(score)
        ]
        if len(scored_candidates) != len(scored):
            return original, decision(fallback_reason="artifact_error")
        required_margin = route_prior_minimum_margin if route_prior_enabled else minimum_margin
        if top_two_margin < float(required_margin):
            return original, decision(
                fallback_reason="route_prior_insufficient_margin" if route_prior_enabled else "insufficient_margin",
                scored_candidates=scored_candidates,
                top_two_margin=top_two_margin,
            )
        ranked_ids = {id(item) for _, item, _ in ranked}
        final = [item for _, item, _ in ranked] + [item for item in original if id(item) not in ranked_ids]
        return final, decision(
            applied=True,
            scored_candidates=scored_candidates,
            top_two_margin=top_two_margin,
            final=final,
            route_prior_top_candidate=(
                _compact_route_prior_candidate(route_candidates.get(ranked[0][1].get("episode_id")))
                if route_prior_enabled else None
            ),
            route_prior_candidates=(
                [
                    compact
                    for _index, item, _score in ranked
                    for compact in [_compact_route_prior_candidate(route_candidates.get(item.get("episode_id")))]
                    if compact is not None
                ]
                if route_prior_enabled else None
            ),
        )
    except Exception:
        return original, decision(fallback_reason="artifact_error")


def rerank_scene16_support_examples(text, examples, artifact_path, minimum_margin=0.01):
    ranked, _ = _rerank_scene16_support_examples_with_decision(
        text, examples, artifact_path, minimum_margin=minimum_margin
    )
    return ranked


def build_grounding(
    text,
    split=None,
    scene_id=None,
    dataset="aerialvln",
    support_limit=3,
    enable_scene16_support_ranker=False,
    scene16_support_ranker="",
    enable_scene16_route_prior=False,
    scene16_route_prior="",
):
    text = str(text or "").strip()
    segments = split_segments(text)
    segment_infos = []
    for index, segment in enumerate(segments):
        segment_infos.append(
            {
                "index": index,
                "text": segment,
                "landmarks": extract_landmarks(segment),
                "actions": extract_actions(segment),
                "relations": extract_relations(segment),
            }
        )

    landmarks = ordered_unique([item for segment in segment_infos for item in segment["landmarks"]])
    actions = ordered_unique([item for segment in segment_infos for item in segment["actions"]])
    relations = [item for segment in segment_infos for item in segment["relations"]]
    scene16_support_ranker_decision = None
    ranker_model = None
    ranker_load_reason = None
    route_prior_model = None
    route_prior_load_reason = None
    scene16_ranker_enabled = enable_scene16_support_ranker and int(scene_id or -1) == 16
    if scene16_ranker_enabled:
        ranker_model, ranker_load_reason = _load_trusted_scene16_support_ranker(scene16_support_ranker)
    route_prior_enabled = bool(enable_scene16_route_prior and scene16_route_prior and int(scene_id or -1) == 16)
    if route_prior_enabled:
        route_prior_model, route_prior_load_reason = load_scene16_target_route_prior(scene16_route_prior)
    retrieval_allowed_episode_ids = None
    if ranker_model is not None:
        retrieval_allowed_episode_ids = set(ranker_model["targets"])
    elif route_prior_model is not None:
        retrieval_allowed_episode_ids = {
            candidate["episode_id"]
            for candidate in route_prior_model.get("candidates", [])
            if isinstance(candidate, dict) and isinstance(candidate.get("episode_id"), str)
        }
    support_examples = search_support_examples(
        text,
        split=split,
        scene_id=scene_id,
        dataset=dataset,
        limit=support_limit,
        allowed_episode_ids=retrieval_allowed_episode_ids,
    )
    if scene16_ranker_enabled or route_prior_enabled:
        support_examples, scene16_support_ranker_decision = _rerank_scene16_support_examples_with_decision(
            text,
            support_examples,
            scene16_support_ranker,
            model=ranker_model,
            load_reason=ranker_load_reason,
            route_prior_model=route_prior_model,
            route_prior_enabled=route_prior_enabled,
            route_prior_load_reason=route_prior_load_reason,
        )
        scene16_support_ranker_decision["candidate_source"] = (
            "artifact_targets"
            if ranker_model is not None or route_prior_model is not None
            else "default_retrieval_fallback"
        )
        scene16_support_ranker_decision["route_prior_artifact"] = bool(route_prior_model is not None)
    support_landmarks = ordered_unique([item for example in support_examples for item in example.get("landmarks", [])])
    if not landmarks and support_landmarks:
        landmarks = support_landmarks[:]
    segment_plan = build_segment_plan(segment_infos, support_landmarks)
    instruction_profile = summarize_instruction_profile(segment_plan, relations, landmarks)

    descent_target = infer_descent_target(segment_infos, landmarks)
    target_landmark = choose_target_landmark(
        segment_infos,
        landmarks,
        support_landmarks,
        descent_target=descent_target,
        relations=relations,
    )
    grounded_instruction = compose_grounded_instruction(
        text,
        {
            "landmarks": landmarks,
            "relations": relations,
            "support_landmarks": support_landmarks,
            "target_landmark": target_landmark,
            "actions": actions,
            "descent_target": descent_target,
        },
    )

    return {
        "text": text,
        "contains_cjk": contains_cjk(text),
        "segments": segment_infos,
        "segment_plan": segment_plan,
        "instruction_profile": instruction_profile,
        "actions": actions,
        "landmarks": landmarks,
        "relations": relations,
        "support_examples": support_examples,
        "support_landmarks": support_landmarks,
        "target_landmark": target_landmark,
        "target_zone": infer_target_zone(target_landmark),
        "descent_target": descent_target,
        "forward_route": bool(landmarks or relations) and ("forward" in actions or "fly_over" in actions or "take_off" in actions),
        "grounded_instruction": grounded_instruction,
        "scene16_support_ranker_decision": scene16_support_ranker_decision,
    }
