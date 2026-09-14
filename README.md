# AirVLN Scene16 Playback Optimization

An engineering extension of [AerialVLN](https://github.com/AirVLN/AirVLN) for reliable Scene16 demonstrations: strict successful-flight selection, deterministic high-resolution AirSim replay, and browser playback that avoids ghosting and frame distortion.

This repository is based on **Liu et al., 2023, “AerialVLN: Vision-and-Language Navigation for UAVs.”** The upstream paper and code define the UAV vision-language navigation task, simulator, data, and CMA-style baseline. This repository documents the follow-up optimization work; it does not redistribute the simulator binaries, datasets, model checkpoints, or generated videos.

## What changed

- Keeps only flights passing the strict success gate: final distance <= 20 m, surface aligned, and `success_20m_with_surface=true`.
- Replays the recorded successful trace offline, so presentation rendering cannot change the verified endpoint.
- Captures AirSim `preview_0` frames at `1280x720` and encodes H.264 at 8 fps.
- Uses real-frame hold playback instead of optical-flow interpolation, avoiding duplicated edges, warped buildings, and invented detail.
- Adds a Scene16 demo builder and a browser player with cached-video fallback behavior.

## Verified result

The Scene16 stable pool contains 82 records. The completed optimization run produced 76 high-resolution replays; one eligible trace failed because AirSim returned an empty image response and was excluded. The remaining five records were already outside the strict completed set.

The verified sample is `1280x720`, H.264, 8 fps, 228 frames. Its rendered endpoint matches the recorded endpoint coordinate-for-coordinate, with recorded distance to goal 15.87 m.

## Repository layout

```text
Model/                 CMA policy and encoders
airsim_plugin/         AirSim client/server integration
src/                   Training and environment code
live_server/           Demo API, session handling, and browser frontend
scripts/               Evaluation and high-resolution replay tools
docs/                  Optimization notes and reproducibility details
```

Generated runtime data is intentionally excluded. Download the original AerialVLN assets following the upstream project instructions before running training or simulation.

## High-resolution replay

```bash
python scripts/render_scene16_highres_trace.py \\
  --session-id <successful-session-id> \\
  --tool-port 30014 \\
  --camera preview_0 \\
  --fps 8
```

For a manifest containing eligible Scene16 sessions, use `--all-eligible --skip-existing`. The tool opens a scene once, reuses the AirSim connection, writes H.264 output under the session `replay/` directory, and updates the manifest only after a video is complete.

## Run the demo server

```bash
python live_server/app.py --host 127.0.0.1 --port 18080
```

Open `http://127.0.0.1:18080/`. The simulator and dataset paths are environment-specific; configure them locally rather than committing machine paths.

## Reproduce the optimization workflow

1. Clone this repository and the upstream AerialVLN assets. The simulator package, AerialVLN/AerialVLN-S annotations, and checkpoints are distributed separately by the upstream project.

2. Create the Python 3.8 environment and install dependencies:

```bash
conda create -n AirVLN-opt python=3.8
conda activate AirVLN-opt
pip install -r requirements.txt
pip install airsim==1.7.0
```

3. Copy `config.example.env` to `.env` and set the local dataset, checkpoint, stable-pool, and manifest paths. The example values are intentionally placeholders.

4. Start the AirSim scene server with a `preview_0` capture configured to 1280x720. `airsim_plugin/AirVLNSimulatorServerTool.py` can generate the preview capture settings; use `AIRVLN_PREVIEW_CAMERA_WIDTH=1280`, `AIRVLN_PREVIEW_CAMERA_HEIGHT=720`, and `AIRVLN_PREVIEW_CAMERA_MAX_RAW_BYTES=3000000`.

5. Start the demo API with `scripts/run_demo_server.sh`. Verify it with `GET /api/health` and inspect the catalog with `GET /api/scene16/demo-builder?max_seeds_per_family=120`.

6. For an existing successful session, render one deterministic high-resolution replay:

```bash
python scripts/render_scene16_highres_trace.py \\
  --session-id <successful-session-id> \\
  --manifest "$AIRVLN_SCENE16_PREVIEW_MANIFEST" \\
  --tool-port 30014 \\
  --skip-existing
python scripts/validate_highres_replay.py \\
  live_server/runtime/agent_sessions/<successful-session-id>/replay/replay_h264_highres_8fps.json
```

7. After validating one trace, render the eligible set with `--all-eligible --skip-existing`. The renderer reuses one Scene16 connection, writes each video atomically before updating its manifest entry, and continues past an individual failure.

The exact published result cannot be regenerated without the original successful traces and licensed simulator assets. To make a result auditable, preserve the manifest, metadata JSON, source trace hash, simulator settings hash, and the output video SHA256 alongside your experiment record.

## Citation

```bibtex
@inproceedings{liu2023aerialvln,
  title={AerialVLN: Vision-and-Language Navigation for UAVs},
  author={Liu, Shubo and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2023}
}
```

Please cite the upstream AerialVLN paper and repository for the original task, simulator, and dataset.

## Status and limitations

This is a research/engineering snapshot, not a packaged end-user application. Reproducing the exact numbers requires the same AirSim scene assets, GPU environment, model checkpoint, and successful trace set. High-resolution rendering improves source detail and playback stability; it cannot recover geometry absent from the simulator render.
