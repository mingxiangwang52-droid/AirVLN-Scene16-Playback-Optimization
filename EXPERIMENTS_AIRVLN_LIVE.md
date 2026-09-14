# Reproducibility Notes

The original experiments were run on a remote GPU host. This public snapshot intentionally omits host-specific SSH commands, absolute paths, runtime logs, model weights, simulator binaries, datasets, and generated videos.

Configure local paths with environment variables such as `AIRVLN_SCENE16_DEMO_STABLE_POOL`, `AIRVLN_SCENE16_PREVIEW_MANIFEST`, `AIRVLN_SEGMENT_GROUNDING_RIDGE`, and `AIRVLN_BLACKWELL_RUNTIME_MARKER`.

For a local demo server:

```bash
python live_server/app.py --host 127.0.0.1 --port 18080
```

For high-resolution replay, see `scripts/render_scene16_highres_trace.py` and the main README.
