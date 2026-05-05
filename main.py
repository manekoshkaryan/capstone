
import argparse
import cv2
import logging
import os
import sys
import time
import numpy as np

from config import AppConfig
from perception.calibration import CalibrationData, run_calibration_workflow
from pipeline import Pipeline
from interfaces.renderer import Renderer
from speech.speech_engine import SpeechEngine
from speech.speech_policy import ConversationMode
from utils.event_logger import EventLogger
from stats.stats_analyzer import analyze_events, print_summary_table, save_summary_json
from stats.stats_plots import generate_plots
from protocols.communication_protocols import ProtocolMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/logs/navvision.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

def _window_name(config: AppConfig) -> str:
    return config.system_name

def _make_splash(width: int, height: int, title: str, body: str = "", sub: str = "",
                 brand: str = "ManeNavSystem") -> np.ndarray:
    frame = np.full((height, width, 3), 28, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def _center_text(text, scale, thickness, y, color):
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = max(0, (width - tw) // 2)
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    _center_text(brand, 1.6, 2, height // 2 - 80, (60, 220, 140))
    if title:
        _center_text(title, 0.7, 1, height // 2 - 20, (210, 210, 210))
    if body:
        _center_text(body, 0.5, 1, height // 2 + 20, (160, 160, 160))
    if sub:
        _center_text(sub, 0.42, 1, height // 2 + 55, (110, 110, 110))
    return frame

def _build_calib_label(calib: CalibrationData, is_calibrated_after: bool) -> str:
    if is_calibrated_after:
        return f"Calibrated | fx={calib.fx:.0f} fy={calib.fy:.0f} err={calib.reprojection_error:.3f}px"
    return "UNCALIBRATED — distances are best-effort (online fx estimation active)"

def _run_calibration(config: AppConfig, calib: CalibrationData, speech, window_name: str) -> bool:
    logger.info("Entering calibration mode (in-window)")
    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        logger.error("Cannot open camera for calibration")
        return False
    result = run_calibration_workflow(cap, config, speech=speech, window_name=window_name)
    cap.release()
    if result is not None:
        result.save(config.calibration_file)
        calib.camera_matrix = result.camera_matrix
        calib.dist_coeffs = result.dist_coeffs
        calib.reprojection_error = result.reprojection_error
        calib.image_size = result.image_size
        logger.info("Calibration complete and saved")
        return True
    return False

def _parse_args():
    ap = argparse.ArgumentParser(description=f"ManeNavSystem — voice-to-voice navigation assistant")
    ap.add_argument("--no-voice-input", action="store_true",
                    help="Disable microphone listener (debugging without mic)")
    ap.add_argument("--no-speech", action="store_true",
                    help="Disable text-to-speech output")
    ap.add_argument("--no-event-log", action="store_true",
                    help="Disable JSONL event logging")
    ap.add_argument("--generate-plots", action="store_true",
                    help="On shutdown, generate session_plots PNG via matplotlib")
    ap.add_argument("--camera-index", type=int, default=None)
    ap.add_argument("--voice-backend", choices=("local", "openai_realtime"), default=None,
                    help="Voice conversation backend. Defaults to VOICE_BACKEND env var or 'local'.")
    return ap.parse_args()

def main():
    args = _parse_args()
    config = AppConfig()
    # Pull voice settings (VOICE_BACKEND, OPENAI_*) from environment first,
    # so CLI overrides win.
    config.apply_env_overrides()

    if args.no_speech:
        config.enable_speech = False
    if args.no_event_log:
        config.enable_event_log = False
    if args.no_voice_input:
        config.enable_voice_input = False
    if args.generate_plots:
        config.auto_generate_plots = True
    if args.camera_index is not None:
        config.camera_index = args.camera_index
    if args.voice_backend is not None:
        config.voice_backend = args.voice_backend

    if config.voice_backend == "openai_realtime" and not config.openai_api_key:
        logger.error(
            "voice_backend=openai_realtime but OPENAI_API_KEY is empty. "
            "Set the env var or rerun with --voice-backend local."
        )
        sys.exit(1)
    logger.info(f"Voice backend: {config.voice_backend}")

    win = _window_name(config)

    calib = CalibrationData()
    calib.load(config.calibration_file)

    speech = SpeechEngine(rate=185) if config.enable_speech else None
    event_logger = EventLogger(
        enabled=config.enable_event_log,
        output_path=config.event_log_file,
        nav_output_path=config.nav_log_file,
    )

    pipeline = Pipeline(
        config, calib, speech, event_logger,
        enable_voice_input=config.enable_voice_input,
    )
    renderer = Renderer(config)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, config.frame_width, config.frame_height)
    cv2.imshow(win, _make_splash(
        config.frame_width, config.frame_height,
        "Loading models, please wait...",
        "Detector + depth estimator initializing",
        "First launch downloads ~400 MB from HuggingFace",
        brand=config.system_name,
    ))
    cv2.waitKey(1)

    try:
        pipeline.start()
    except RuntimeError as e:
        logger.error(f"Pipeline start failed: {e}")
        sys.exit(1)

    if calib.is_calibrated:
        pipeline._say(f"{config.system_name} ready.")
    else:
        pipeline._say(f"{config.system_name} ready.")

    if config.phone_mode:
        cam_url = pipeline.phone_camera_url
        view_url = pipeline.phone_view_url
        logger.info("=" * 78)
        logger.info(" PHONE INSTRUCTIONS")
        logger.info(f"  1) On phone (Chrome/Safari) for camera + controls: {cam_url}")
        logger.info(f"  2) On any browser to view annotated output       : {view_url}")
        logger.info("     (the camera page already shows the annotated stream too)")
        logger.info("=" * 78)

    calib_label = _build_calib_label(calib, calib.is_calibrated)
    last_screenshot_path: str = ""

    logger.info(
        "Controls: q=quit  p=pause  s=screenshot  r=reload-vocab  c=calibrate  d=depth-view  t=tracking  v=verbose  +/-=tune-distance  0=reset-scale  g=guidance  f=floor-overlay  n=clear-target | "
        "Modes: 1=passive  2=continuous-guidance  3=minimal-alert  4=conversation | "
        "Protocols: 5=minimal  6=descriptive  7=proactive  8=adaptive  e=run-protocol-eval | "
        f"voice_input={'ON' if config.enable_voice_input else 'OFF'}"
    )

    try:
        while True:
            if pipeline.state.consume_shutdown_request():
                logger.info("Shutdown requested via voice")
                break

            if pipeline.state.consume_calibration_request():
                logger.info("Calibration requested via voice")
                pipeline.stop()
                _run_calibration(config, calib, speech, window_name=win)
                calib_label = _build_calib_label(calib, calib.is_calibrated)
                pipeline = Pipeline(
                    config, calib, speech, event_logger,
                    enable_voice_input=config.enable_voice_input,
                )
                renderer = Renderer(config)
                cv2.imshow(win, _make_splash(
                    config.frame_width, config.frame_height, "Restarting pipeline...",
                    brand=config.system_name,
                ))
                cv2.waitKey(1)
                try:
                    pipeline.start()
                except RuntimeError as e:
                    logger.error(f"Pipeline restart failed: {e}")
                    break
                continue

            snap = pipeline.get_display_snapshot()
            raw_frame = snap.get("raw_frame")

            if raw_frame is None:
                cv2.imshow(win, _make_splash(
                    config.frame_width, config.frame_height,
                    "Waiting for camera...",
                    f"Camera index: {config.camera_index}",
                    "Check that the camera is connected and not used by another app",
                    brand=config.system_name,
                ))
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                continue

            metrics = {
                "fps": snap.get("fps", 0.0),
                "cap_ms": snap.get("cap_ms", 0.0),
                "det_ms": snap.get("det_ms", 0.0),
                "dep_ms": snap.get("dep_ms", 0.0),
                "pipe_ms": snap.get("pipe_ms", 0.0),
                "device": snap.get("device", "CPU"),
                "det_model": snap.get("det_model", "?"),
                "dep_model": snap.get("dep_model", "?"),
                "vocab_n": snap.get("vocab_n", 0),
                "n_objects": snap.get("n_objects", 0),
                "fusion_global_scale": snap.get("fusion_global_scale", 1.0),
                "fusion_estimated_fx": snap.get("fusion_estimated_fx"),
                "fusion_samples": snap.get("fusion_samples", 0),
                "fusion_method_counts": snap.get("fusion_method_counts", {}),
                "effective_fx": snap.get("effective_fx", 0.0),
                "effective_fy": snap.get("effective_fy", 0.0),
                "calib_reproj_px": snap.get("calib_reproj_px"),
                "calib_quality_warn_px": snap.get("calib_quality_warn_px", 1.0),
                "user_scale_factor": config.user_scale_factor,
                "floor_mask": snap.get("floor_mask"),
                "floor_coverage": snap.get("floor_coverage", 0.0),
                "floor_horizon_y": snap.get("floor_horizon_y", 0),
                "free_space": snap.get("free_space"),
                "guidance_action": snap.get("guidance_action", ""),
                "guidance_text": snap.get("guidance_text", ""),
                "target_landmark": snap.get("target_landmark", ""),
                "imu_pitch_deg": snap.get("imu_pitch_deg", 0.0),
                "imu_yaw_deg": snap.get("imu_yaw_deg", 0.0),
                "imu_valid": snap.get("imu_valid", False),
                "landmark_count": snap.get("landmark_count", 0),
                "protocol_mode": snap.get("protocol_mode", ""),
                "protocol_last_message": snap.get("protocol_last_message", ""),
                "protocol_last_word_count": snap.get("protocol_last_word_count", 0),
                "protocol_last_action": snap.get("protocol_last_action", ""),
                "protocol_last_response_ms": snap.get("protocol_last_response_ms"),
                "protocol_last_speech_ms": snap.get("protocol_last_speech_ms"),
                "protocol_overlap_count": snap.get("protocol_overlap_count", 0),
                "protocol_duplicate_count": snap.get("protocol_duplicate_count", 0),
            }

            tracked_objects = snap.get("tracked_objects", [])
            depth_map = snap.get("depth_map")
            depth_colorized = snap.get("depth_colorized")

            rendered = renderer.draw_frame(
                raw_frame,
                tracked_objects,
                depth_map,
                depth_colorized,
                metrics,
                calib_label,
            )

            cv2.imshow(win, rendered)
            try:
                pipeline._lidar_server.push_annotated_frame(rendered)
            except Exception:
                pass
            try:
                pipeline.push_phone_render(rendered)
                voice_status = "off"
                if speech is not None:
                    voice_status = f"{getattr(speech, 'backend_name', '?')} vol={int(speech.volume*100)}%"
                pipeline.push_phone_status(
                    fps=f"{snap.get('fps', 0.0):.1f}",
                    device=snap.get("device", "?"),
                    det_model=snap.get("det_model", "?"),
                    dep_model=snap.get("dep_model", "?"),
                    voice=voice_status,
                    phone="active" if config.phone_mode else "off",
                    n_objects=snap.get("n_objects", 0),
                    guidance_action=snap.get("guidance_action", ""),
                )
            except Exception:
                pass

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.info("Quit requested")
                break

            elif key == ord("p"):
                config.paused = not config.paused
                state_str = "PAUSED" if config.paused else "RESUMED"
                logger.info(f"Pipeline {state_str}")

            elif key == ord("s"):
                last_screenshot_path = pipeline.save_screenshot(rendered)
                logger.info(f"Screenshot: {last_screenshot_path}")

            elif key == ord("r"):
                pipeline.reload_vocabulary()
                logger.info("Vocabulary reloaded")

            elif key == ord("c"):
                pipeline.stop()
                _run_calibration(config, calib, speech, window_name=win)
                calib_label = _build_calib_label(calib, calib.is_calibrated)
                pipeline = Pipeline(
                    config, calib, speech, event_logger,
                    enable_voice_input=config.enable_voice_input,
                )
                renderer = Renderer(config)
                cv2.imshow(win, _make_splash(
                    config.frame_width, config.frame_height, "Restarting pipeline...",
                    brand=config.system_name,
                ))
                cv2.waitKey(1)
                try:
                    pipeline.start()
                except RuntimeError as e:
                    logger.error(f"Pipeline restart failed: {e}")
                    break

            elif key == ord("d"):
                config.show_depth_view = not config.show_depth_view
                logger.info(f"Depth view: {'ON' if config.show_depth_view else 'OFF'}")

            elif key == ord("t"):
                config.show_tracking = not config.show_tracking
                logger.info(f"Track IDs: {'ON' if config.show_tracking else 'OFF'}")

            elif key == ord("v"):
                config.verbose_diagnostics = not config.verbose_diagnostics
                logger.info(f"Verbose diagnostics: {'ON' if config.verbose_diagnostics else 'OFF'}")

            elif key in (ord("+"), ord("=")):
                pipeline.adjust_user_scale(+1)

            elif key in (ord("-"), ord("_")):
                pipeline.adjust_user_scale(-1)

            elif key == ord("0"):
                config.user_scale_factor = 1.0
                logger.info("user_scale_factor reset to 1.00")

            elif key == ord("g"):
                config.enable_guidance = not config.enable_guidance
                logger.info(f"Guidance: {'ON' if config.enable_guidance else 'OFF'}")
                pipeline.stop()
                pipeline = Pipeline(
                    config, calib, speech, event_logger,
                    enable_voice_input=config.enable_voice_input,
                )
                renderer = Renderer(config)
                try:
                    pipeline.start()
                except RuntimeError as e:
                    logger.error(f"Pipeline restart failed: {e}")
                    break

            elif key == ord("f"):
                config.show_floor_overlay = not config.show_floor_overlay
                logger.info(f"Floor overlay: {'ON' if config.show_floor_overlay else 'OFF'}")

            elif key == ord("n"):
                pipeline.state.set_target_landmark("")
                logger.info("Cleared navigation target")

            elif key == ord("1"):
                pipeline.set_mode(ConversationMode.PASSIVE)
            elif key == ord("2"):
                pipeline.set_mode(ConversationMode.CONTINUOUS_GUIDANCE)
            elif key == ord("3"):
                pipeline.set_mode(ConversationMode.MINIMAL_ALERT)
            elif key == ord("4"):
                pipeline.set_mode(ConversationMode.CONVERSATION)
            elif key == ord("5"):
                pipeline.set_protocol_mode(ProtocolMode.MINIMAL)
            elif key == ord("6"):
                pipeline.set_protocol_mode(ProtocolMode.DESCRIPTIVE)
            elif key == ord("7"):
                pipeline.set_protocol_mode(ProtocolMode.PROACTIVE)
            elif key == ord("8"):
                pipeline.set_protocol_mode(ProtocolMode.ADAPTIVE)
            elif key == ord("e"):
                logger.info("Running protocol evaluation...")
                info = pipeline.run_protocol_evaluation(include_scenarios=True)
                logger.info(
                    f"Protocol evaluation done: metrics_rows={info.get('metrics_rows', 0)} "
                    f"plots={len(info.get('plot_paths', []))}"
                )
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down")

    pipeline.stop()
    if speech is not None:
        speech.stop()
    event_logger.close()
    cv2.destroyAllWindows()

    try:
        stats = analyze_events(config.event_log_file, config.nav_log_file)
        print_summary_table(stats)
        save_summary_json(stats, config.session_summary_dir)
        if config.auto_generate_plots:
            generate_plots(config.event_log_file, config.session_plots_dir)
    except Exception as e:
        logger.warning(f"Session post-processing failed: {e}")

    logger.info(f"{config.system_name} exited cleanly")

if __name__ == "__main__":
    main()
