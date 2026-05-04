# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the main navigation system
python main.py

# Run speech engine unit tests
python test_speech_engine.py

# Test phone camera stream
python phone_camera_test.py

# Benchmarking / analysis
python bench_speech.py
python batch_analyze.py
python latency_report.py
python stats_analyzer.py
python stats_plots.py
python protocol_evaluator.py
```

No package manager or build step — pure Python. Install deps with `pip install -r requirements.txt`.

## Architecture

**ManeNavSystem** is a real-time navigation assistant for visually impaired users. Camera frames flow through a perception pipeline, produce prioritized verbal guidance, and are spoken via TTS.

### Data Flow

```
Camera Frame
  → detector.py       (YOLOv8 object detection, with YOLO-World fallback)
  → tracker.py        (multi-object tracking, stable IDs across frames)
  → depth_estimator.py (Depth-Anything V2 Metric depth map)
  → distance_fusion.py (fuses geometric + depth estimates per object)
  → navigator.py      (bearing, urgency ranking, verbal distance encoding)
  → speech_policy.py  (risk-band throttling: critical / warn / info)
  → speech_engine.py  (utterance formatting → pyttsx3 or OpenAI Realtime TTS)
```

Parallel: `floor_segmenter.py` + `free_space.py` analyze open path sectors; `landmarks.py` maintains spatial memory; `imu_sensor.py` feeds optical-flow stabilization.

### Key Files

| File | Role |
|---|---|
| `pipeline.py` | Core orchestrator — threaded capture/detect/depth loops, `SharedState`, voice command dispatch |
| `config.py` | Single `Config` dataclass with 100+ fields (models, thresholds, speech windows, sector geometry, protocol modes) |
| `navigator.py` | Prioritization logic, directional reasoning, distance-to-word encoding |
| `speech_policy.py` | Risk-band stratification; suppresses redundant utterances |
| `speech_engine.py` | Utterance formatting ("Stop. Person ahead."), TTS backend |
| `conversation.py` | Free-form voice dialogue via LLM |
| `phone_camera.py` | HTTPS + WebRTC remote phone camera streaming |
| `web_output_server.py` | Annotated frame web viewer |
| `renderer.py` | Real-time OpenCV frame annotation |

### Outputs / Logs

- `events.jsonl` — timestamped detection + speech event stream
- `navigation.jsonl` — high-level navigation records
- `protocol_events.jsonl` — protocol evaluation events
- `session_summary_*.json` — end-of-session stats
- `navvision.log` — application log

### Models & Calibration

- `yolov8s-oiv7.pt`, `yolov8s-worldv2.pt` — pre-trained detectors (OIV7 primary, World fallback)
- `camera_calibration.npz` — saved camera matrix + distortion coefficients
- `vocabulary.txt` — custom class list for YOLO-World detection
