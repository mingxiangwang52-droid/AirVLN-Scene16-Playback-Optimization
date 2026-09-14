"""Display-only stream pacing helpers."""


def resolve_hold_last_frame(args, turbo):
    """Keep a heartbeat in turbo mode unless the client explicitly opts out."""
    value = args.get("hold")
    if value is None:
        return True
    return str(value).lower() not in ("0", "false", "no")


def resolve_hold_max_age(args, cinematic=False, default_sec=None):
    """How long a valid frame may be repeated to bridge source-frame stalls."""
    if default_sec is None:
        default_sec = 3.5 if cinematic else 1.2
    try:
        value = float(args.get("hold_age", default_sec))
    except (TypeError, ValueError):
        value = default_sec
    return min(6.0, max(0.25, value))


def should_hold_frame(previous_frame, frame_at, now, max_age_sec):
    """Only repeat a recently accepted frame; invalid frames never become held frames."""
    return bool(previous_frame) and frame_at is not None and 0.0 <= now - frame_at <= max_age_sec
