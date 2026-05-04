import cv2
import logging
import os
import queue
import threading
import time
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from dotenv import load_dotenv
load_dotenv()

from config import AppConfig
from perception.detector import ObjectDetector
from perception.depth_estimator import DepthEstimator, LiDARDepthSource
from interfaces.lidar_depth_server import LiDARDepthServer
from perception.tracker import MultiObjectTracker, TrackedObject, RawDetection
from perception.calibration import CalibrationData
from perception.distance_fusion import HybridDistanceEstimator, colorize_depth
from perception.floor_segmenter import FloorSegmenter, FloorResult
from navigation.free_space import FreeSpaceAnalyzer, FreeSpaceFrame
from navigation.landmarks import LandmarkMemory
from perception.imu_sensor import IMUSensor, SerialIMU, FlowEstimatedIMU
from utils.utils import FPSCounter, Timer, load_vocabulary
from navigation.navigator import (
    prioritize_objects, build_navigation_record, tracked_to_detection,
    get_urgency, get_direction, meters_to_verbal,
)
from speech.speech_engine import SpeechEngine
from speech.speech_policy import (
    ConversationMode, ConversationModeManager,
)
from utils.event_logger import EventLogger
from speech.speech_input import SpeechListener, VoiceCommand, parse_command
from speech.conversation import ConversationHandler
from speech.voice_context import VoiceContextProvider
from protocols.communication_protocols import (
    PerceptionInput, ProtocolMode, ProtocolMessage,
    render as render_protocol, coerce_mode as coerce_protocol_mode,
    normalize_direction, ACTION_STOP, ACTION_DIRECTIONAL, ACTION_CLEAR, ACTION_INFO,
    CascadedBaselineConfig,
)
from protocols.protocol_logger import ProtocolLogger, ProtocolEvent
from protocols.protocol_evaluator import (
    run_evaluation as run_protocol_evaluation_offline,
    builtin_scenarios,
    run_scenarios as run_protocol_scenarios,
)
from navigation.llm_guide import LLMGuide
from speech.piper_tts import PiperTTS

logger = logging.getLogger(__name__)

_URGENCY_RANK_MAP = {"critical": 0, "warn": 1, "info": 2, "beyond": 3}


def _urgency_for_action(action: str, perception: "PerceptionInput") -> str:
    if action == ACTION_STOP:
        return "critical"
    if action == ACTION_DIRECTIONAL:
        return "warn"
    if action == ACTION_CLEAR:
        return "info"
    if perception.distance_m is not None and perception.distance_m < 0.8:
        return "critical"
    if perception.distance_m is not None and perception.distance_m < 2.0:
        return "warn"
    return "info"

class SharedState:
    def __init__(self):
        self._lock = threading.RLock()
        self.raw_frame: Optional[np.ndarray] = None
        self.raw_frame_id: int = -1
        self.raw_frame_ts: float = 0.0
        self.raw_frame_wall_ts: float = 0.0

        self.detections: List[RawDetection] = []
        self.det_frame_id: int = -1
        self.det_latency_ms: float = 0.0
        self.det_done_wall_ts: float = 0.0

        self.depth_map: Optional[np.ndarray] = None
        self.depth_frame_id: int = -1
        self.depth_latency_ms: float = 0.0

        self.tracked_objects: List[TrackedObject] = []
        self.depth_colorized: Optional[np.ndarray] = None
        self.fusion_latency_ms: float = 0.0

        self.cap_latency_ms: float = 0.0
        self.frame_counter: int = 0

        self.command_queue: "queue.Queue[VoiceCommand]" = queue.Queue(maxsize=32)
        self.calibration_requested: bool = False
        self.shutdown_requested: bool = False
        self.session_start_ts: float = time.time()

        self.floor_mask: Optional[np.ndarray] = None
        self.floor_coverage: float = 0.0
        self.floor_horizon_y: int = 0
        self.free_space: Optional[FreeSpaceFrame] = None
        self.last_guidance_action: str = ""
        self.last_guidance_text: str = ""
        self.target_landmark: str = ""
        self.imu_pitch_deg: float = 0.0
        self.imu_yaw_deg: float = 0.0
        self.imu_valid: bool = False

    def set_frame(self, frame: np.ndarray, frame_id: int, latency_ms: float):
        with self._lock:
            self.raw_frame = frame
            self.raw_frame_id = frame_id
            self.raw_frame_ts = time.perf_counter()
            self.raw_frame_wall_ts = time.time()
            self.cap_latency_ms = latency_ms
            self.frame_counter = frame_id

    def get_frame(self) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self.raw_frame is None:
                return None, -1
            return self.raw_frame.copy(), self.raw_frame_id

    def set_detections(self, dets: List[RawDetection], frame_id: int, latency_ms: float):
        with self._lock:
            self.detections = dets
            self.det_frame_id = frame_id
            self.det_latency_ms = latency_ms
            self.det_done_wall_ts = time.time()

    def set_depth(self, depth_map: Optional[np.ndarray], frame_id: int, latency_ms: float):
        with self._lock:
            self.depth_map = depth_map
            self.depth_frame_id = frame_id
            self.depth_latency_ms = latency_ms

    def set_fusion(self, tracked: List[TrackedObject], depth_colored: Optional[np.ndarray], latency_ms: float):
        with self._lock:
            self.tracked_objects = tracked
            self.depth_colorized = depth_colored
            self.fusion_latency_ms = latency_ms

    def set_navigation(
        self,
        floor: Optional[FloorResult],
        free_space: Optional[FreeSpaceFrame],
        guidance_action: str,
        guidance_text: str,
        imu_pitch: float,
        imu_yaw: float,
        imu_valid: bool,
    ):
        with self._lock:
            if floor is not None:
                self.floor_mask = floor.mask
                self.floor_coverage = float(floor.coverage)
                self.floor_horizon_y = int(floor.horizon_y)
            self.free_space = free_space
            self.last_guidance_action = guidance_action
            self.last_guidance_text = guidance_text
            self.imu_pitch_deg = float(imu_pitch)
            self.imu_yaw_deg = float(imu_yaw)
            self.imu_valid = bool(imu_valid)

    def set_target_landmark(self, label: str):
        with self._lock:
            self.target_landmark = label or ""

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "raw_frame": self.raw_frame.copy() if self.raw_frame is not None else None,
                "frame_id": self.raw_frame_id,
                "raw_frame_wall_ts": self.raw_frame_wall_ts,
                "det_done_wall_ts": self.det_done_wall_ts,
                "tracked_objects": list(self.tracked_objects),
                "depth_map": self.depth_map.copy() if self.depth_map is not None else None,
                "depth_colorized": self.depth_colorized.copy() if self.depth_colorized is not None else None,
                "det_frame_id": self.det_frame_id,
                "depth_frame_id": self.depth_frame_id,
                "cap_ms": self.cap_latency_ms,
                "det_ms": self.det_latency_ms,
                "dep_ms": self.depth_latency_ms,
                "pipe_ms": self.fusion_latency_ms,
                "floor_mask": self.floor_mask,
                "floor_coverage": self.floor_coverage,
                "floor_horizon_y": self.floor_horizon_y,
                "free_space": self.free_space,
                "guidance_action": self.last_guidance_action,
                "guidance_text": self.last_guidance_text,
                "target_landmark": self.target_landmark,
                "imu_pitch_deg": self.imu_pitch_deg,
                "imu_yaw_deg": self.imu_yaw_deg,
                "imu_valid": self.imu_valid,
            }

    def push_command(self, cmd: VoiceCommand) -> bool:
        try:
            self.command_queue.put_nowait(cmd)
            return True
        except queue.Full:
            return False

    def request_calibration(self):
        with self._lock:
            self.calibration_requested = True

    def consume_calibration_request(self) -> bool:
        with self._lock:
            v = self.calibration_requested
            self.calibration_requested = False
            return v

    def request_shutdown(self):
        with self._lock:
            self.shutdown_requested = True

    def consume_shutdown_request(self) -> bool:
        with self._lock:
            return self.shutdown_requested

class Pipeline:
    def __init__(
        self,
        config: AppConfig,
        calib: CalibrationData,
        speech: "Optional[SpeechEngine]",
        event_logger: EventLogger,
        enable_voice_input: bool = True,
    ):
        self._config = config
        self._calib = calib
        self._speech = speech
        self._event_logger = event_logger
        # In OpenAI Realtime mode, the user's mic is captured client-side
        # by WebRTC and sent straight to OpenAI — running the local STT
        # listener at the same time would double-capture the mic and fight
        # for the audio device. Safety TTS still uses the local engine.
        self._voice_backend = (getattr(config, "voice_backend", "local") or "local").strip().lower()
        if self._voice_backend not in ("local", "openai_realtime"):
            self._voice_backend = "local"
        self._enable_voice_input = (
            enable_voice_input
            and config.enable_voice_input
            and self._voice_backend == "local"
        )

        self._detector = ObjectDetector(config)
        self._depth_estimator = DepthEstimator(config)
        self._lidar_server = LiDARDepthServer(port=8444)
        self._lidar_source = LiDARDepthSource(self._lidar_server)
        self._tracker = MultiObjectTracker(
            max_age=config.tracker_max_age,
            min_hits=config.tracker_min_hits,
            iou_threshold=config.tracker_iou_threshold,
            ema_alpha_pos=config.ema_alpha_position,
            obstacle_dist=config.obstacle_warning_distance_m,
        )
        self._fusion = HybridDistanceEstimator(config, calib)

        if config.enable_imu and config.imu_serial_port:
            self._imu: IMUSensor = SerialIMU(config.imu_serial_port, config.imu_baud)
        elif config.imu_use_optical_flow_fallback:
            self._imu = FlowEstimatedIMU()
        else:
            self._imu = IMUSensor()
        self._floor = FloorSegmenter(config) if config.enable_floor_segmentation else None
        self._free_space = FreeSpaceAnalyzer(config) if config.enable_guidance else None
        self._landmarks = LandmarkMemory(config, imu=self._imu) if config.enable_landmarks else None

        self._state = SharedState()
        self._fps_counter = FPSCounter(window=30)
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._detection_thread: Optional[threading.Thread] = None
        self._depth_thread: Optional[threading.Thread] = None
        self._fusion_thread: Optional[threading.Thread] = None

        self._cap: Optional[cv2.VideoCapture] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._phone_server = None
        self._phone_active_label: str = ""

        self._listener: Optional[SpeechListener] = None
        if self._enable_voice_input:
            self._listener = SpeechListener(
                on_command=self._on_voice_command,
                on_text=self._on_voice_freeform,
                is_speaking=(speech.is_speaking if speech is not None else None),
                on_barge_in=self._on_barge_in,
                reverb_tail_s=getattr(config, "speech_tts_reverb_tail_s", 0.3),
                energy_threshold=config.voice_input_energy_threshold,
                pause_threshold=config.voice_input_pause_threshold,
                phrase_time_limit=config.voice_input_phrase_time_limit,
                ambient_calibration_s=config.voice_input_ambient_calibration_s,
                dynamic_energy=config.voice_input_dynamic_energy,
            )

        self._distance_method_counts: Dict[str, int] = {
            "geometric": 0,
            "calibrated_depth": 0,
            "fallback_depth": 0,
            "none": 0,
        }

        self._mode_manager = ConversationModeManager(ConversationMode.PASSIVE)
        self._conversation = ConversationHandler(self)

        # LLM-based guidance engine + Piper TTS.
        self._piper = PiperTTS()
        self._llm_guide = LLMGuide(
            model=getattr(config, "llm_guide_model", "gpt-4o-mini"),
            stop_distance_m=getattr(config, "guidance_stop_distance_m", 0.8),
            periodic_interval_s=getattr(config, "llm_guide_interval_s", 5.0),
            on_guidance=self._on_llm_guidance,
        )
        logger.info("LLM guidance engine active (GPT + Piper TTS)")

        # Hazard timestamp = first time critical guidance appeared in the
        # current critical episode. Cleared when no critical command in flight.
        # Used to measure end-to-end safety alert latency (hazard detected →
        # audio actually playing).
        self._critical_hazard_ts: Optional[float] = None

        self._protocol_mode = coerce_protocol_mode(getattr(config, "protocol_mode", "adaptive"))
        self._protocol_logger = ProtocolLogger(
            events_path=getattr(config, "protocol_events_file", "protocol_events.jsonl"),
            metrics_path=getattr(config, "protocol_metrics_file", "protocol_metrics.csv"),
            comparison_path=getattr(config, "latency_comparison_file", "latency_comparison.csv"),
            baseline=CascadedBaselineConfig(
                asr_ms=getattr(config, "cascaded_baseline_asr_ms", 600.0),
                llm_ms=getattr(config, "cascaded_baseline_llm_ms", 900.0),
                tts_ms=getattr(config, "cascaded_baseline_tts_ms", 700.0),
            ),
            enabled=bool(getattr(config, "enable_protocol_logging", True)),
        )
        self._pending_speech_meta: Dict[str, Any] = {}
        self._last_protocol_message: Optional[ProtocolMessage] = None
        self._last_protocol_perception: Optional[PerceptionInput] = None
        self._last_protocol_event_record: Optional[Dict[str, Any]] = None
        self._device_seen: Dict[str, float] = {}
        self._device_last_speech: Dict[str, float] = {}

        if self._speech is not None:
            self._speech.set_announce_listener(self._on_announce_complete)
            self._speech.set_overlap_listener(self._on_speech_overlap)
        if self._listener is not None:
            self._listener.set_stt_listener(self._on_stt_complete)

        # Shared voice context — read by both the local ConversationHandler
        # and the OpenAI Realtime client in the phone UI. Read-only for the
        # LLM; safety decisions never originate here.
        self._voice_context = VoiceContextProvider(
            self,
            min_interval_s=getattr(config, "voice_context_min_interval_s", 1.0),
        )

    @property
    def state(self) -> SharedState:
        return self._state

    @property
    def mode_manager(self) -> ConversationModeManager:
        return self._mode_manager

    def set_mode(self, mode: ConversationMode, announce: bool = True) -> None:
        """Switch conversation mode and optionally voice the change."""
        if self._mode_manager.set(mode):

            if announce:
                phrases = {
                    ConversationMode.PASSIVE: "Passive mode.",
                    ConversationMode.CONTINUOUS_GUIDANCE: "Continuous guidance.",
                    ConversationMode.MINIMAL_ALERT: "Minimal alert. I will only warn of danger.",
                    ConversationMode.CONVERSATION: "Conversation mode.",
                }
                self._say(phrases.get(mode, mode.value))

    def start(self):
        logger.info("Pipeline starting — loading models...")
        self._lidar_server.start()
        logger.info("LiDAR depth server started on ws://0.0.0.0:8444/depth")
        self._detector.load()
        self._depth_estimator.load()

        src = (self._config.camera_source or "webcam").strip().lower()
        if src in ("phone", "webrtc", "xiaomi", "iphone"):
            logger.info(f"Opening phone camera (source={src})")
        else:
            logger.info(f"Opening camera index {self._config.camera_index}")
        self._cap = self._open_camera()
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera (source={src})")

        if self._config.enable_recording:
            self._start_recording()

        self._running = True
        self._state.session_start_ts = time.time()

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="CaptureThread")
        self._detection_thread = threading.Thread(target=self._detection_loop, daemon=True, name="DetectionThread")
        self._depth_thread = threading.Thread(target=self._depth_loop, daemon=True, name="DepthThread")
        self._fusion_thread = threading.Thread(target=self._fusion_loop, daemon=True, name="FusionThread")

        self._capture_thread.start()
        self._detection_thread.start()
        self._depth_thread.start()
        self._fusion_thread.start()

        if self._listener is not None:
            ok = self._listener.start()
            if not ok:
                logger.warning("Voice input disabled (init failed)")

        try:
            self._imu.start()
        except Exception as e:
            logger.debug(f"IMU start failed: {e}")

        logger.info("Pipeline running")

    def _open_camera(self):
        src = (self._config.camera_source or "webcam").strip().lower()
        if src in ("phone", "webrtc", "xiaomi", "iphone"):
            return self._open_phone_camera(src)
        import platform
        _sys = platform.system()
        if _sys == "Darwin":
            _backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        elif _sys == "Windows":
            _backends = [getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY), cv2.CAP_ANY]
        else:
            _backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        for backend in _backends:
            try:
                cap = cv2.VideoCapture(self._config.camera_index, backend)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)
                cap.set(cv2.CAP_PROP_FPS, self._config.capture_fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if cap.isOpened():
                    ret, f = cap.read()
                    if ret and f is not None:
                        logger.info(f"Camera opened: {f.shape[1]}x{f.shape[0]} @ backend={backend}")
                        return cap
                cap.release()
            except Exception as e:
                logger.debug(f"Camera backend {backend} failed: {e}")
        return None

    def _open_phone_camera(self, src: str):
        from interfaces.phone_camera import PhoneCameraServer, PhoneCameraSource, _local_ip
        from interfaces.web_output_server import WebOutputServer
        cfg = self._config
        cfg.phone_mode = True
        if src in ("xiaomi", "iphone"):
            label = src
        else:
            label = (cfg.phone_camera_label or "phone").strip().lower() or "phone"
        if self._phone_server is None:
            web_out = WebOutputServer(jpeg_quality=cfg.web_output_jpeg_quality) if cfg.enable_web_output else WebOutputServer()
            self._phone_server = PhoneCameraServer(
                port=cfg.phone_camera_port,
                cert_dir=cfg.phone_camera_cert_dir,
                web_output=web_out,
            )
            self._phone_server.set_command_callback(self._on_phone_command)
            self._phone_server.set_button_callback(self._on_phone_button)
            try:
                self._phone_server.configure_voice(
                    backend=self._voice_backend,
                    context_provider=self._voice_context.get,
                    distance_callback=self.adjust_user_scale,
                    openai_api_key=getattr(cfg, "openai_api_key", "") or "",
                    openai_model=getattr(cfg, "openai_realtime_model", "gpt-realtime-mini"),
                    openai_voice=getattr(cfg, "openai_realtime_voice", "alloy"),
                    openai_session_url=getattr(cfg, "openai_realtime_session_url",
                                               "https://api.openai.com/v1/realtime/sessions"),
                    openai_instructions=getattr(cfg, "openai_realtime_instructions", ""),
                )
            except Exception as e:
                logger.warning(f"phone server voice config failed: {e}")
            self._phone_server.start()
            if self._speech is not None:
                self._speech.set_text_listener(self._on_speech_text)
                if cfg.mute_local_speech_when_phone_active:
                    self._speech.set_mute_local(True)
                    logger.info("Local TTS muted (phone will speak instead)")
                else:
                    self._speech.set_mute_local(False)
                    logger.info("Local TTS active (PC speakers will play voice)")
        self._phone_active_label = label
        url = self._phone_server.url_for(label)
        view_url = self._phone_server.view_url()
        logger.info("=" * 78)
        logger.info(f" PhoneCam source selected: label='{label}'")
        logger.info(f" Open ON PHONE  (camera + controls): {url}")
        logger.info(f" Open ON ANY BROWSER (view-only)  : {view_url}")
        logger.info("  Same Wi-Fi. Accept the self-signed cert. Tap 'Start'.")
        logger.info("=" * 78)
        if cfg.phone_passthrough:
            target_size = None
            rotate_arg = 0
            logger.info("PhoneCam passthrough ON: frames delivered 1:1 (no rotate, no resize)")
        else:
            target_size = (int(cfg.phone_output_width), int(cfg.phone_output_height))
            rotate_arg = cfg.rotate_phone_frame
        source = PhoneCameraSource(
            self._phone_server,
            label=label,
            target_size=target_size,
            wait_first_frame_s=cfg.phone_camera_wait_first_frame_s,
            rotate=rotate_arg,
            preserve_aspect_ratio=cfg.preserve_aspect_ratio,
        )
        if not source.isOpened():
            logger.error(f"PhoneCam '{label}' did not connect within {cfg.phone_camera_wait_first_frame_s}s")
            return None
        ok, f = source.read()
        if not ok or f is None:
            logger.error(f"PhoneCam '{label}' connected but no frames received")
            return None
        logger.info(f"PhoneCam '{label}' connected: {f.shape[1]}x{f.shape[0]}")
        return source

    def _start_recording(self):
        if self._cap is None:
            return
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        try:
            self._writer = cv2.VideoWriter(self._config.recording_file, fourcc, 20.0, (w, h))
            logger.info(f"Recording to {self._config.recording_file}")
        except Exception as e:
            logger.warning(f"Recording init failed: {e}")

    def _capture_loop(self):
        frame_id = 0
        while self._running:
            if self._config.paused:
                time.sleep(0.03)
                continue
            t0 = time.perf_counter()
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("Camera read failed — retrying")
                time.sleep(0.1)
                continue
            if (
                self._config.apply_undistortion
                and self._calib.is_calibrated
                and self._calib.is_safe_for_undistort(self._config.apply_undistortion_max_reproj_px)
            ):
                try:
                    frame = self._calib.undistort_frame(frame)
                except Exception as e:
                    logger.debug(f"Undistort error: {e}")
            latency_ms = (time.perf_counter() - t0) * 1000.0
            frame_id += 1
            self._fps_counter.tick()
            self._state.set_frame(frame, frame_id, latency_ms)
            if self._writer is not None:
                try:
                    self._writer.write(frame)
                except Exception:
                    pass

    def _detection_loop(self):
        last_det_frame_id = -1
        interval = self._config.detection_interval_frames

        while self._running:
            if self._config.paused or not self._detector.is_loaded:
                time.sleep(0.03)
                continue

            frame, frame_id = self._state.get_frame()
            if frame is None or frame_id == last_det_frame_id:
                time.sleep(0.005)
                continue

            if (frame_id - last_det_frame_id) < interval:
                time.sleep(0.005)
                continue

            last_det_frame_id = frame_id
            t0 = time.perf_counter()
            try:
                dets = self._detector.detect(frame)
            except Exception as e:
                logger.warning(f"Detection exception: {e}")
                dets = []
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._state.set_detections(dets, frame_id, latency_ms)

    def _depth_loop(self):
        last_depth_frame_id = -1
        interval = self._config.depth_interval_frames
        _lidar_active_logged = False

        while self._running:
            if self._config.paused:
                time.sleep(0.05)
                continue

            frame, frame_id = self._state.get_frame()
            if frame is None or frame_id == last_depth_frame_id:
                time.sleep(0.01)
                continue

            if (frame_id - last_depth_frame_id) < interval:
                time.sleep(0.01)
                continue

            last_depth_frame_id = frame_id
            t0 = time.perf_counter()

            # Try LiDAR first (iOS app) — zero Mac compute, ±1-2cm accuracy
            depth_map = self._lidar_source.get(frame)
            if depth_map is not None:
                if not _lidar_active_logged:
                    logger.info("LiDAR depth active — Depth-Anything bypassed")
                    _lidar_active_logged = True
            else:
                _lidar_active_logged = False
                if self._depth_estimator.is_loaded:
                    try:
                        depth_map = self._depth_estimator.estimate(frame)
                    except Exception as e:
                        logger.warning(f"Depth exception: {e}")
                        depth_map = None

            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._state.set_depth(depth_map, frame_id, latency_ms)

    def _fusion_loop(self):
        last_fused_frame_id = -1
        while self._running:
            self._process_commands()

            if self._config.paused:
                time.sleep(0.03)
                continue

            snap = self._state.snapshot()
            frame_id = snap["frame_id"]

            if frame_id == last_fused_frame_id or frame_id < 0:
                time.sleep(0.003)
                continue
            last_fused_frame_id = frame_id

            t0 = time.perf_counter()

            det_frame_id_snap = snap["det_frame_id"]
            depth_frame_id_snap = snap["depth_frame_id"]

            with self._state._lock:
                raw_dets = list(self._state.detections) if det_frame_id_snap >= 0 else []

            depth_map = snap["depth_map"]
            dep_staleness = abs(frame_id - depth_frame_id_snap)
            depth_available = depth_map is not None and dep_staleness <= self._config.depth_stale_frames_limit

            tracked = self._tracker.update(list(raw_dets))

            depth_colored = None
            if depth_available:
                depth_colored = colorize_depth(depth_map, self._config.depth_max_range_m)

            frame_shape = snap["raw_frame"].shape if snap.get("raw_frame") is not None else (
                self._config.frame_height, self._config.frame_width, 3,
            )

            any_distance = False
            for obj in tracked:
                result = self._fusion.fuse(
                    obj.box,
                    obj.label,
                    depth_map if depth_available else None,
                    frame_shape,
                    obj.det_conf,
                    track_age=obj.frames_seen,
                )
                if result is None:
                    self._distance_method_counts["none"] += 1
                    continue
                self._tracker.update_depth_for_track(
                    obj.track_id,
                    result.distance_m,
                    result.nearest_m,
                    result.variance,
                    result.pixel_count,
                    result.confidence,
                    method=result.method,
                )
                self._distance_method_counts[result.method] = (
                    self._distance_method_counts.get(result.method, 0) + 1
                )
                any_distance = True

            if any_distance:
                confirmed = self._tracker.get_confirmed_tracks()
                tracked = [t.to_tracked_object(self._config.obstacle_warning_distance_m) for t in confirmed]

            fusion_ms = (time.perf_counter() - t0) * 1000.0
            self._state.set_fusion(tracked, depth_colored, fusion_ms)

            self._navigation_step(
                depth_map=depth_map if depth_available else None,
                frame_shape=frame_shape,
                tracked=tracked,
                frame_id=frame_id,
                raw_frame=snap.get("raw_frame"),
            )

            if self._config.enable_event_log:
                nav_record = build_navigation_record(
                    tracked, frame_id, self._config.frame_width,
                    max_dist_m=self._config.nav_max_announce_dist_m,
                )
                self._event_logger.log_detections(tracked, frame_id)
                self._event_logger.log_navigation(nav_record)


    def _on_llm_guidance(self, guidance) -> None:
        """Fires when LLMGuide has a GPT response. Synthesizes once with Piper,
        plays on laptop and streams the same WAV to the phone."""
        import base64
        mode = self._mode_manager.mode
        if mode not in (ConversationMode.CONTINUOUS_GUIDANCE, ConversationMode.PASSIVE):
            return
        text = guidance.speech
        urgency = guidance.urgency
        # Synthesize once — reuse bytes for both laptop and phone.
        try:
            wav = self._piper.synthesize_bytes(text)
        except Exception as e:
            logger.warning(f"Piper synthesis failed: {e}")
            wav = b""
        # Play on laptop.
        if wav:
            import threading, tempfile, subprocess, os
            def _play():
                import tempfile, subprocess, os
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(wav); tmp = f.name
                try:
                    subprocess.run(["afplay", tmp], check=True, capture_output=True)
                except Exception:
                    pass
                finally:
                    try: os.unlink(tmp)
                    except OSError: pass
            threading.Thread(target=_play, daemon=True).start()
        # Push same audio to phone.
        if wav and self._phone_server is not None and self._phone_active_label:
            try:
                wav_b64 = base64.b64encode(wav).decode()
                self._phone_server.push_audio(self._phone_active_label, wav_b64, text, urgency)
            except Exception as e:
                logger.debug(f"Phone audio push failed: {e}")

    def _on_voice_command(self, cmd: VoiceCommand):
        if not self._state.push_command(cmd):
            logger.warning(f"Command queue full — dropping {cmd.type}")

    def _on_announce_complete(
        self, text: str, urgency: str, ts_enqueue: float,
        ts_speak_start: float, ts_speak_end: float, meta: dict,
    ) -> None:
        if self._config.enable_event_log:
            try:
                self._event_logger.log_announcement(
                    text=text,
                    urgency=urgency,
                    ts_enqueue=ts_enqueue,
                    ts_speak_start=ts_speak_start,
                    ts_speak_end=ts_speak_end,
                    hazard_ts=meta.get("hazard_ts"),
                    source=meta.get("source"),
                    action=meta.get("action"),
                    is_repeat=bool(meta.get("is_repeat", False)),
                    reason=meta.get("reason"),
                )
            except Exception as e:
                logger.debug(f"announce log error: {e}")
        try:
            self._log_protocol_announcement(text, urgency, ts_enqueue, ts_speak_start, ts_speak_end, meta)
        except Exception as e:
            logger.debug(f"protocol announce log error: {e}")

    def _log_protocol_announcement(
        self, text: str, urgency: str, ts_enqueue: float,
        ts_speak_start: float, ts_speak_end: float, meta: dict,
    ) -> None:
        if self._protocol_logger is None:
            return
        protocol_mode = meta.get("protocol_mode") if meta else None
        if not protocol_mode:
            return
        meta_key = f"{text}|{urgency}"
        stored = self._pending_speech_meta.pop(meta_key, None)
        meta_full = stored if stored is not None else (meta if isinstance(meta, dict) else {})
        perception_data = meta_full.get("perception") or {}
        action = meta_full.get("protocol_action") or ""
        word_count = int(meta_full.get("protocol_word_count", 0) or 0)
        msg = ProtocolMessage(
            text=text,
            protocol=protocol_mode,
            action_category=action,
            word_count=word_count,
            timestamp=ts_speak_start,
        )
        perception = PerceptionInput(
            label=str(perception_data.get("label") or ""),
            distance_m=(
                float(perception_data.get("distance_m"))
                if perception_data.get("distance_m") is not None
                else None
            ),
            direction=str(perception_data.get("direction") or "ahead"),
            urgency=urgency,
            front_clear=bool(perception_data.get("front_clear", True)),
            left_clear_m=perception_data.get("left_clear_m"),
            right_clear_m=perception_data.get("right_clear_m"),
            device_label=str(meta_full.get("device") or ""),
        )
        selected = coerce_protocol_mode(meta_full.get("selected_protocol") or protocol_mode)
        self._record_protocol_event(
            msg=msg,
            perception=perception,
            source=str(meta_full.get("source") or "auto"),
            device=str(meta_full.get("device") or self._active_device_label()),
            ts_frame=meta_full.get("t_frame_received"),
            ts_det=meta_full.get("t_detection_done"),
            ts_decision=meta_full.get("t_decision_started"),
            ts_message=meta_full.get("t_message_generated"),
            ts_tts=meta_full.get("ts_tts_requested") or ts_enqueue,
            ts_audio_start=ts_speak_start,
            ts_audio_end=ts_speak_end,
            is_duplicate=False,
            overlap_skipped=False,
            selected=selected,
        )

    def _on_stt_complete(
        self, text: str, ts_audio_end: float, ts_transcribe_end: float,
        engine: str, ok: bool,
    ) -> None:
        if not self._config.enable_event_log:
            return
        try:
            self._event_logger.log_stt(
                text=text, ts_audio_end=ts_audio_end,
                ts_transcribe_end=ts_transcribe_end, engine=engine, ok=ok,
            )
        except Exception as e:
            logger.debug(f"stt log error: {e}")

    def _on_barge_in(self) -> None:
        """User audio captured — kill in-flight non-critical TTS plus pending
        queue so the user is heard immediately. Critical safety alerts are
        preserved (they may be the reason the user spoke)."""
        if self._speech is not None:
            self._speech.cancel_pending(keep_critical=True, stop_current=True)

    def _on_voice_freeform(self, text: str):
        """Free-form text from STT that didn't match the rigid command parser.
        Send it through the conversational handler and speak the reply."""
        text = (text or "").strip()
        if not text:
            return
        try:
            reply = self._conversation.respond(text, None)
        except Exception as e:
            logger.error(f"conversation handler error: {e}")
            reply = None
        if reply:
            logger.info(f"VOICE freeform: {text!r} -> {reply!r}")
            self._say(reply)

    def _on_phone_command(self, label: str, text: str):
        text = (text or "").strip()
        if not text:
            return
        cmd = parse_command(text)
        if cmd is None:
            # Phone free-form input → conversational handler instead of
            # treating every typed sentence as a "find X" search.
            logger.info(f"PHONE[{label}] freeform: {text!r}")
            self._on_voice_freeform(text)
            return
        logger.info(f"PHONE[{label}] cmd: {cmd.type} target={cmd.target!r} raw={text!r}")
        self._on_voice_command(cmd)

    def _on_phone_button(self, label: str, action: str, target: str):
        action = (action or "").strip().lower()
        target = (target or "").strip()
        if not action:
            return
        logger.info(f"PHONE[{label}] button: {action} target={target!r}")
        cmd = VoiceCommand(
            type=action, target=(target or None), raw=f"button:{action}",
            ts=time.time(), source="phone_button",
        )
        self._on_voice_command(cmd)

    def _on_speech_text(self, text: str, urgency: str):
        if self._phone_server is None or not self._phone_active_label:
            return
        try:
            self._phone_server.push_speech(self._phone_active_label, text, urgency)
        except Exception as e:
            logger.debug(f"phone speech forward failed: {e}")

    def push_phone_render(self, frame) -> None:
        if self._phone_server is None or frame is None:
            return
        try:
            self._phone_server.push_render_frame(self._phone_active_label, frame)
        except Exception as e:
            logger.debug(f"phone render push failed: {e}")

    def push_phone_status(self, **status) -> None:
        if self._phone_server is None:
            return
        # Always include the live user-scale value so the phone UI's
        # Distance display can render a current number.
        status.setdefault("user_scale", f"{self._config.user_scale_factor:.2f}")
        try:
            self._phone_server.update_status(**status)
        except Exception as e:
            logger.debug(f"phone status push failed: {e}")

    @property
    def phone_view_url(self) -> str:
        if self._phone_server is None:
            return ""
        try:
            return self._phone_server.view_url()
        except Exception:
            return ""

    @property
    def phone_camera_url(self) -> str:
        if self._phone_server is None or not self._phone_active_label:
            return ""
        try:
            return self._phone_server.url_for(self._phone_active_label)
        except Exception:
            return ""

    def _process_commands(self):
        while True:
            try:
                cmd = self._state.command_queue.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle_command(cmd)
            except Exception as e:
                logger.error(f"Voice command handler error ({cmd.type}): {e}")

    def _handle_command(self, cmd: VoiceCommand):
        logger.info(
            f"VOICE command: raw={cmd.raw!r} -> intent={cmd.type} target={cmd.target!r}"
        )
        ctype = cmd.type
        source = getattr(cmd, "source", None) or "voice"

        # Drop near-duplicate commands within a short window. Real session
        # log shows STT misrecognizing background TTS as "pause"/"resume"/
        # "louder", producing a flood of "Already paused.", "Already
        # running.", "Volume 100 percent." Squash them.
        now = time.time()
        key = (ctype, (cmd.target or "").lower())
        last = getattr(self, "_last_cmd_handled", {})
        if last.get(key, 0.0) and (now - last[key]) < 2.5:
            logger.debug(f"command dedupe: dropping repeat {ctype}")
            return
        last[key] = now
        self._last_cmd_handled = last

        if ctype == "scene_description":
            self._speak_scene_description()
        elif ctype == "find_object":
            self._speak_object_location(cmd.target or "")
        elif ctype == "pause":
            if not self._config.paused:
                self._config.paused = True
                self._say("Paused.")
            else:
                self._say("Already paused.")
        elif ctype == "resume":
            if self._config.paused:
                self._config.paused = False
                self._say("Resumed.")
            else:
                self._say("Already running.")
        elif ctype == "calibrate":
            self._say("Starting calibration. Please look at the screen.")
            self._state.request_calibration()
        elif ctype == "louder":
            if self._speech is not None:
                v = self._speech.louder()
                self._say(f"Volume {int(round(v * 100))} percent.")
        elif ctype == "quieter":
            if self._speech is not None:
                v = self._speech.quieter()
                self._say(f"Volume {int(round(v * 100))} percent.")
        elif ctype == "status":
            self._speak_status()
        elif ctype == "shutdown":
            self._say("Shutting down.")
            self._state.request_shutdown()
        elif ctype in ("navigate_to", "guide_to", "go_to"):
            self._set_navigation_target(cmd.target or "")
        elif ctype in ("stop_navigation", "cancel_navigation", "stop_guidance"):
            self._set_navigation_target("")
            # "stop guiding" also drops continuous-guidance mode back to passive.
            if self._mode_manager.mode == ConversationMode.CONTINUOUS_GUIDANCE:
                self._mode_manager.set(ConversationMode.PASSIVE)
    
            self._say("Guidance stopped.")
        elif ctype == "set_minimal_alert":
            self._mode_manager.set(ConversationMode.MINIMAL_ALERT)

            self._say("Minimal alert mode. I will only warn you of danger.")
        elif ctype == "set_continuous_guidance":
            self._mode_manager.set(ConversationMode.CONTINUOUS_GUIDANCE)

            self._say("Continuous guidance on.")
        elif ctype == "set_verbose_describe":
            # One-shot rich description; mode itself returns to passive after.
            self._speak_scene_description()
        elif ctype == "query_direction":
            self._speak_direction_query(cmd.target or "")
        elif ctype == "query_path_clear":
            self._speak_path_clear_query()
        elif ctype in ("distance_plus", "distance_minus"):
            v = self.adjust_user_scale(+1 if ctype == "distance_plus" else -1)
            self._say(f"Distance {int(round(v * 100))} percent.")
        elif ctype in ("where_is", "locate"):
            self._speak_landmark_location(cmd.target or "")
        elif ctype in ("forget", "forget_landmark"):
            self._forget_landmark(cmd.target or "")
        elif ctype in ("zero_heading", "set_north"):
            try:
                self._imu.zero_heading()
                self._say("Heading reset.")
            except Exception:
                pass
        elif ctype == "what_to_do":
            self._speak_what_to_do(source=source)
        elif ctype == "what_is_closest":
            self._speak_what_is_closest(source=source)
        elif ctype == "what_is_around_me":
            self._speak_scene_description()
        elif ctype == "repeat":
            self._speak_repeat()
        elif ctype == "help":
            self._say(
                "Say what is closest, what to do, what is around me, "
                "is the path clear, mute, unmute, repeat, or run protocol evaluation."
            )
        elif ctype == "mute":
            if self._speech is not None:
                self._speech.set_mute_local(True)
                self._say("Muted.")
        elif ctype == "unmute":
            if self._speech is not None:
                self._speech.set_mute_local(False)
                self._say("Unmuted.")
        elif ctype == "stop_speaking":
            if self._speech is not None:
                self._speech.cancel_pending(keep_critical=False, stop_current=True)
        elif ctype == "set_protocol_minimal":
            self.set_protocol_mode(ProtocolMode.MINIMAL)
        elif ctype == "set_protocol_descriptive":
            self.set_protocol_mode(ProtocolMode.DESCRIPTIVE)
        elif ctype == "set_protocol_proactive":
            self.set_protocol_mode(ProtocolMode.PROACTIVE)
        elif ctype == "set_protocol_adaptive":
            self.set_protocol_mode(ProtocolMode.ADAPTIVE)
        elif ctype == "run_protocol_evaluation":
            self.run_protocol_evaluation(include_scenarios=True)

    def _set_navigation_target(self, target: str):
        target = (target or "").strip().lower()
        if not target:
            self._state.set_target_landmark("")
            return
        if self._landmarks is None:
            self._say("Landmark memory disabled.")
            return
        lm = self._landmarks.find(target)
        if lm is None:
            self._say(f"I have not seen {target} yet. Please look toward it once.")
            self._state.set_target_landmark(target)
            return
        self._state.set_target_landmark(target)
        self._say(f"Guiding you to {target}.")

    def _speak_landmark_location(self, target: str):
        target = (target or "").strip()
        if not target or self._landmarks is None:
            self._say("I do not have any landmarks remembered.")
            return
        lm = self._landmarks.find(target)
        if lm is None:
            self._say(f"I do not remember a {target}.")
            return
        rel, dist = self._landmarks.relative_bearing(lm)
        if rel < 0.35:
            direction = "to your left"
        elif rel > 0.65:
            direction = "to your right"
        else:
            direction = "ahead"
        self._say(f"{lm.label} {direction}, about {dist:.1f} meters.")

    def _forget_landmark(self, target: str):
        if self._landmarks is None:
            return
        if target.strip().lower() in ("all", "everything"):
            self._landmarks.clear()
            self._say("All landmarks forgotten.")
            return
        if self._landmarks.forget(target):
            self._say(f"Forgot {target}.")
        else:
            self._say(f"I do not remember a {target}.")

    def _navigation_step(self, depth_map, frame_shape, tracked, frame_id, raw_frame):
        cfg = self.config_or_default()
        H, W = frame_shape[:2]

        if isinstance(self._imu, FlowEstimatedIMU) and raw_frame is not None and cfg.imu_use_optical_flow_fallback:
            try:
                self._imu.update_from_frame(raw_frame)
            except Exception:
                pass
        imu_reading = self._imu.read()

        floor_result: Optional[FloorResult] = None
        floor_mask = None
        if self._floor is not None and depth_map is not None:
            fx = self._fusion.effective_fx(W)
            fy = self._fusion.effective_fy(H)
            cx = float(self._calib.cx) if self._calib.is_calibrated and self._calib.cx else W / 2.0
            cy = float(self._calib.cy) if self._calib.is_calibrated and self._calib.cy else H / 2.0
            try:
                floor_result = self._floor.segment(
                    depth_map, fx, fy, cx, cy,
                    cfg.camera_height_m,
                    pitch_deg=imu_reading.pitch_deg if imu_reading.valid else 0.0,
                )
                floor_mask = floor_result.mask
            except Exception as e:
                logger.debug(f"Floor segmentation failed: {e}")

        free_space_frame = None
        if self._free_space is not None:
            try:
                free_space_frame = self._free_space.analyze(depth_map, tracked, floor_mask)
            except Exception as e:
                logger.debug(f"Free-space analysis failed: {e}")

        # LLM guidance — feed scene to GPT engine for natural language direction.
        if free_space_frame is not None:
            llm_tracked = []
            for obj in tracked:
                if not obj.is_stable or obj.distance_m is None:
                    continue
                llm_tracked.append(tracked_to_detection(obj, W))
            try:
                self._llm_guide.on_scene_update(free_space_frame, llm_tracked)
            except Exception as e:
                logger.debug(f"LLMGuide update failed: {e}")

        if self._landmarks is not None:
            try:
                self._landmarks.update(tracked, W, frame_id)
            except Exception as e:
                logger.debug(f"Landmark update failed: {e}")

        self._state.set_navigation(
            floor=floor_result,
            free_space=free_space_frame,
            guidance_action="",
            guidance_text="",
            imu_pitch=imu_reading.pitch_deg,
            imu_yaw=imu_reading.yaw_deg,
            imu_valid=imu_reading.valid,
        )

    def config_or_default(self):
        return self._config

    def _say(self, text: str, urgency: str = "info"):
        import base64, threading, tempfile, subprocess, os
        try:
            wav = self._piper.synthesize_bytes(text)
        except Exception:
            wav = b""
        if wav:
            def _play():
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(wav); tmp = f.name
                try: subprocess.run(["afplay", tmp], check=True, capture_output=True)
                except Exception: pass
                finally:
                    try: os.unlink(tmp)
                    except OSError: pass
            threading.Thread(target=_play, daemon=True).start()
            if self._phone_server is not None and self._phone_active_label:
                try:
                    self._phone_server.push_audio(
                        self._phone_active_label, base64.b64encode(wav).decode(), text, urgency
                    )
                except Exception: pass

    @property
    def protocol_mode(self) -> ProtocolMode:
        return self._protocol_mode

    @property
    def protocol_logger(self) -> ProtocolLogger:
        return self._protocol_logger

    @property
    def last_protocol_message(self) -> Optional[ProtocolMessage]:
        return self._last_protocol_message

    def set_protocol_mode(self, mode: Any, announce: bool = True) -> ProtocolMode:
        new_mode = coerce_protocol_mode(mode)
        if new_mode == self._protocol_mode:
            return self._protocol_mode
        self._protocol_mode = new_mode
        logger.info(f"protocol-mode -> {new_mode.value}")
        if announce:
            phrases = {
                ProtocolMode.MINIMAL: "Minimal protocol.",
                ProtocolMode.DESCRIPTIVE: "Descriptive protocol.",
                ProtocolMode.PROACTIVE: "Proactive protocol.",
                ProtocolMode.ADAPTIVE: "Adaptive protocol.",
            }
            self._say(phrases.get(new_mode, new_mode.value))
        return self._protocol_mode

    def _active_device_label(self) -> str:
        if self._phone_active_label:
            return self._phone_active_label
        src = (self._config.camera_source or "webcam").strip().lower()
        return src or "webcam"

    def _aggregate_closest(self) -> Optional[TrackedObject]:
        best: Optional[TrackedObject] = None
        best_d = float("inf")
        for obj in self._stable_objects():
            d = obj.nearest_dist_m if obj.nearest_dist_m is not None else obj.distance_m
            if d is None:
                continue
            if d < best_d:
                best_d = d
                best = obj
        return best

    def _build_perception(
        self,
        target: Optional[Any] = None,
        urgency: str = "info",
        include_free_space: bool = True,
    ) -> PerceptionInput:
        from navigation.navigator import get_bearing
        fw = self._config.frame_width
        label = ""
        distance = None
        direction = "ahead"
        device = self._active_device_label()
        front_clear = True
        left_clear = None
        right_clear = None
        if include_free_space:
            with self._state._lock:
                fs = self._state.free_space
            if fs is not None and getattr(fs, "sectors", None):
                sectors = fs.sectors
                n = len(sectors)
                center_idx = n // 2
                front = float(sectors[center_idx].free_distance_m)
                left_clear = max(
                    (float(s.free_distance_m) for s in sectors[:center_idx]),
                    default=0.0,
                )
                right_clear = max(
                    (float(s.free_distance_m) for s in sectors[center_idx + 1:]),
                    default=0.0,
                )
                front_clear = front >= self._config.guidance_min_go_distance_m
        if target is not None:
            if isinstance(target, dict):
                label = str(target.get("class") or target.get("label") or "")
                distance = target.get("distance_m")
                if distance is not None:
                    distance = float(distance)
                direction = str(target.get("bearing") or target.get("direction") or "ahead")
            else:
                label = str(getattr(target, "label", ""))
                d = getattr(target, "nearest_dist_m", None)
                if d is None:
                    d = getattr(target, "distance_m", None)
                distance = float(d) if d is not None else None
                box = getattr(target, "box", None)
                if box is not None:
                    direction = get_bearing(box, fw)
        return PerceptionInput(
            label=label,
            distance_m=distance,
            direction=direction,
            urgency=urgency or "info",
            front_clear=front_clear,
            left_clear_m=left_clear,
            right_clear_m=right_clear,
            device_label=device,
        )


    def _emit_protocol_speech(
        self,
        perception: PerceptionInput,
        source: str,
        mode_override: Optional[ProtocolMode] = None,
        urgency_override: Optional[str] = None,
        action_override: str = "",
        t_decision_started: Optional[float] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[ProtocolMessage]:
        if self._speech is None:
            return None
        selected = self._protocol_mode
        effective = mode_override or selected
        t_decision = t_decision_started if t_decision_started is not None else time.time()
        msg = render_protocol(effective, perception)
        t_message = time.time()
        urgency = urgency_override or _urgency_for_action(msg.action_category, perception)
        device = perception.device_label or self._active_device_label()
        device_dedup_window = 0.6
        now = time.time()
        last_device_ts = self._device_last_speech.get(device, 0.0)
        if last_device_ts and (now - last_device_ts) < device_dedup_window:
            self._protocol_logger.mark_duplicate(msg.protocol)
        snap = self._state.snapshot()
        meta_payload = {
            "protocol_text": msg.text,
            "protocol_mode": msg.protocol,
            "selected_protocol": selected.value,
            "protocol_action": msg.action_category,
            "protocol_word_count": msg.word_count,
            "perception": {
                "label": perception.label,
                "distance_m": perception.distance_m,
                "direction": perception.direction,
                "front_clear": perception.front_clear,
                "left_clear_m": perception.left_clear_m,
                "right_clear_m": perception.right_clear_m,
            },
            "source": source,
            "device": device,
            "action_override": action_override,
            "t_frame_received": snap.get("raw_frame_wall_ts") or None,
            "t_detection_done": snap.get("det_done_wall_ts") or None,
            "t_decision_started": t_decision,
            "t_message_generated": t_message,
        }
        if extra_meta:
            meta_payload.update(extra_meta)
        meta_key = f"{msg.text}|{urgency}"
        self._pending_speech_meta[meta_key] = meta_payload
        accepted = True
        self._say(msg.text, urgency=urgency)
        if not accepted:
            self._pending_speech_meta.pop(meta_key, None)
            self._record_protocol_event(
                msg=msg,
                perception=perception,
                source=source,
                device=device,
                ts_frame=meta_payload["t_frame_received"],
                ts_det=meta_payload["t_detection_done"],
                ts_decision=t_decision,
                ts_message=t_message,
                ts_tts=now,
                ts_audio_start=None,
                ts_audio_end=None,
                is_duplicate=True,
                overlap_skipped=False,
                selected=selected,
            )
            return msg
        self._device_last_speech[device] = now
        self._last_protocol_message = msg
        self._last_protocol_perception = perception
        return msg

    def _record_protocol_event(
        self,
        msg: ProtocolMessage,
        perception: PerceptionInput,
        source: str,
        device: str,
        ts_frame: Optional[float],
        ts_det: Optional[float],
        ts_decision: Optional[float],
        ts_message: Optional[float],
        ts_tts: Optional[float],
        ts_audio_start: Optional[float],
        ts_audio_end: Optional[float],
        is_duplicate: bool,
        overlap_skipped: bool,
        selected: ProtocolMode,
    ) -> Optional[Dict[str, Any]]:
        if self._protocol_logger is None:
            return None
        event = ProtocolEvent(
            protocol=msg.protocol,
            selected_protocol=selected.value,
            source=source,
            device=device,
            message=msg.text,
            word_count=msg.word_count,
            closest_label=perception.label,
            distance_m=perception.distance_m,
            direction=normalize_direction(perception.direction),
            action_category=msg.action_category,
            t_frame_received=ts_frame,
            t_detection_done=ts_det,
            t_decision_started=ts_decision,
            t_message_generated=ts_message,
            t_tts_requested=ts_tts,
            t_audio_started=ts_audio_start,
            t_audio_completed=ts_audio_end,
            is_duplicate=is_duplicate,
            overlap_skipped=overlap_skipped,
        )
        record = self._protocol_logger.log(event)
        self._last_protocol_event_record = record
        return record

    def _on_speech_overlap(self, text: str, urgency: str, reason: str, meta: dict) -> None:
        if self._protocol_logger is None:
            return
        self._protocol_logger.mark_overlap(meta.get("protocol_mode", "") if meta else "")

    def run_protocol_evaluation(self, include_scenarios: bool = True) -> Dict[str, Any]:
        cfg = self._config
        try:
            result = run_protocol_evaluation_offline(
                metrics_path=cfg.protocol_metrics_file,
                plot_dir=cfg.protocol_plot_dir,
                summary_csv=cfg.protocol_summary_csv,
                summary_json=cfg.protocol_summary_json,
                scenario_output=cfg.scenario_output_file,
                ground_truth_path=cfg.ground_truth_protocol_file,
                accuracy_path=cfg.protocol_accuracy_csv,
                include_scenarios=include_scenarios,
            )
        except Exception as e:
            logger.error(f"Protocol evaluation failed: {e}")
            return {"ok": False, "error": str(e)}
        info = {
            "ok": True,
            "metrics_rows": result.metrics_rows,
            "scenarios": len(result.scenario_rows),
            "summary": result.summary,
            "accuracy": result.accuracy,
            "plot_paths": result.plot_paths,
        }
        try:
            self._say("Protocol evaluation saved.")
        except Exception:
            pass
        return info

    def _broadcast_safety(self, action: str, text: str, urgency: str) -> None:
        """Record a SafetyCommand in voice context + push it to SSE clients.

        The LLM never sees this as an instruction — it sees it as a fact
        in the read-only navigation context (`last_safety_command`). The
        client uses the SSE event to duck/cancel its Realtime audio so
        the local safety TTS is heard cleanly.
        """
        action = (action or "").strip().upper()
        try:
            self._voice_context.record_safety_command(action, text)
        except Exception as e:
            logger.debug(f"voice_context.record_safety_command failed: {e}")
        if self._phone_server is not None:
            try:
                self._phone_server.push_safety_event(action, text, urgency)
            except Exception as e:
                logger.debug(f"phone_server.push_safety_event failed: {e}")

    def adjust_user_scale(self, direction: int) -> float:
        """Bump the user_scale_factor by one step. Used by both keyboard
        +/- in main.py and the phone UI Distance +/- buttons so calibration
        logic stays in one place.
        """
        cfg = self._config
        step = cfg.user_scale_step * (1 if direction > 0 else -1)
        cfg.user_scale_factor = max(
            cfg.user_scale_min,
            min(cfg.user_scale_max, round(cfg.user_scale_factor + step, 3)),
        )
        logger.info(f"user_scale_factor -> {cfg.user_scale_factor:.2f}")
        return cfg.user_scale_factor

    def _stable_objects(self) -> List[TrackedObject]:
        with self._state._lock:
            return [o for o in self._state.tracked_objects if o.is_stable]

    def _speak_scene_description(self):
        objs = self._stable_objects()
        if not objs:
            self._say("Nothing in view.")
            return

        fw = self._config.frame_width
        items = []
        for obj in objs:
            d = obj.nearest_dist_m if obj.nearest_dist_m is not None else obj.distance_m
            if d is None:
                continue
            items.append((obj, d))
        items.sort(key=lambda t: t[1])
        items = items[:6]

        if not items:
            # Have objects but no distance fix yet — stay silent rather than
            # speak "I can see N objects but distances are not available
            # yet." which the session log showed firing repeatedly with no
            # value to the user.
            return

        parts = []
        for obj, d in items:
            direction = get_direction(obj.box, fw)
            verbal = meters_to_verbal(d)
            parts.append(f"{obj.label} {direction}, {verbal}")
        msg = "I see " + "; ".join(parts) + "."
        self._say(msg)

    def _speak_what_is_closest(self, source: str = "voice") -> None:
        target = self._aggregate_closest()
        urgency = "info"
        if target is not None:
            d = target.nearest_dist_m if target.nearest_dist_m is not None else target.distance_m
            if d is not None:
                if d < 0.8:
                    urgency = "critical"
                elif d < 2.0:
                    urgency = "warn"
        perception = self._build_perception(target=target, urgency=urgency)
        self._emit_protocol_speech(perception, source=source)

    def _speak_what_to_do(self, source: str = "voice") -> None:
        target = self._aggregate_closest()
        urgency = "info"
        if target is not None:
            d = target.nearest_dist_m if target.nearest_dist_m is not None else target.distance_m
            if d is not None:
                if d < 0.8:
                    urgency = "critical"
                elif d < 2.0:
                    urgency = "warn"
        perception = self._build_perception(target=target, urgency=urgency)
        if self._protocol_mode == ProtocolMode.DESCRIPTIVE:
            mode_override = ProtocolMode.PROACTIVE
        elif self._protocol_mode == ProtocolMode.MINIMAL:
            mode_override = ProtocolMode.PROACTIVE
        else:
            mode_override = ProtocolMode.ADAPTIVE
        self._emit_protocol_speech(
            perception, source=source, mode_override=mode_override,
        )

    def _speak_repeat(self) -> None:
        if self._last_protocol_message is None:
            self._say("Nothing to repeat yet.")
            return
        msg = self._last_protocol_message
        urgency = _urgency_for_action(msg.action_category, self._last_protocol_perception or PerceptionInput())
        meta = {
            "protocol_text": msg.text,
            "protocol_mode": msg.protocol,
            "selected_protocol": self._protocol_mode.value,
            "protocol_action": msg.action_category,
            "protocol_word_count": msg.word_count,
            "perception": {
                "label": (self._last_protocol_perception.label
                          if self._last_protocol_perception else ""),
                "distance_m": (self._last_protocol_perception.distance_m
                               if self._last_protocol_perception else None),
                "direction": (self._last_protocol_perception.direction
                              if self._last_protocol_perception else "ahead"),
                "front_clear": True,
                "left_clear_m": None,
                "right_clear_m": None,
            },
            "source": "repeat",
            "device": self._active_device_label(),
            "t_decision_started": time.time(),
            "t_message_generated": time.time(),
        }
        meta_key = f"{msg.text}|{urgency}"
        self._pending_speech_meta[meta_key] = meta
        self._say(msg.text, urgency=urgency)

    def _speak_direction_query(self, direction: str):
        """Answer 'what is on my left/right/front?' from currently tracked objs.
        User question — overrides passive landmark silence (announces landmarks
        explicitly). Heavy logging for diagnosing voice flow.
        """
        direction = (direction or "").strip().lower()
        logger.info(f"voice-query direction={direction!r}")
        if direction not in ("left", "right", "front"):
            self._say("Please ask about the left, right, or front.")
            return

        # Direction → eligible bearing buckets.
        bearings = {
            "left": ("far_left", "left", "slight_left"),
            "right": ("far_right", "right", "slight_right"),
            "front": ("slight_left", "center", "slight_right"),
        }[direction]

        from navigation.navigator import get_bearing
        fw = self._config.frame_width

        # Front-clear short-circuit: if user asked about ahead and the path
        # really is clear, say so directly instead of listing nothing.
        if direction == "front":
            with self._state._lock:
                fs = self._state.free_space
            if fs is not None and fs.sectors:
                center_clear = fs.center_clear_m >= self._config.guidance_min_go_distance_m
                if center_clear and not any(
                    get_bearing(o.box, fw) in bearings for o in self._stable_objects()
                ):
                    logger.info("voice-query front: path clear, no objects -> 'path ahead is clear'")
                    self._say("The path ahead is clear.")
                    return

        items = []
        for obj in self._stable_objects():
            b = get_bearing(obj.box, fw)
            if b not in bearings:
                continue
            d = obj.nearest_dist_m if obj.nearest_dist_m is not None else obj.distance_m
            items.append((obj, b, d))
        items.sort(key=lambda t: (t[2] is None, t[2] if t[2] is not None else 1e9))
        considered = [(o.label, b, d) for o, b, d in items]
        logger.info(f"voice-query considered={considered}")

        if not items:
            phrase = "ahead" if direction == "front" else f"on your {direction}"
            answer = f"I don't see anything important {phrase}."
            logger.info(f"voice-query answer={answer!r}")
            self._say(answer)
            return

        items = items[:3]
        parts = []
        for obj, _, d in items:
            article = "an" if obj.label.lower()[:1] in "aeiou" else "a"
            if d is None:
                parts.append(f"{article} {obj.label}")
            else:
                parts.append(f"{article} {obj.label} about {d:.1f} meters away")

        phrase = "ahead" if direction == "front" else f"on your {direction}"
        if direction == "front":
            answer = f"Ahead, I see " + ", and ".join(parts) + "."
        else:
            answer = f"On your {direction}, I see " + ", and ".join(parts) + "."
        logger.info(f"voice-query answer={answer!r}")
        self._say(answer)

    def _speak_path_clear_query(self):
        """Answer 'can I go forward?' / 'is the path clear?'."""
        with self._state._lock:
            fs = self._state.free_space
        if fs is None or not fs.sectors:
            self._say("I cannot tell yet.")
            return
        cfg = self._config
        center_idx = len(fs.sectors) // 2
        center_clear = fs.center_clear_m
        if center_clear >= cfg.guidance_min_go_distance_m:
            self._say("Yes, the path ahead is clear.")
        else:
            left_clear = max((s.free_distance_m for s in fs.sectors[:center_idx]), default=0.0)
            right_clear = max((s.free_distance_m for s in fs.sectors[center_idx+1:]), default=0.0)
            if max(left_clear, right_clear) >= cfg.guidance_min_go_distance_m:
                side = "left" if left_clear > right_clear else "right"
                self._say(f"Front is blocked. The {side} side is open.")
            else:
                self._say("No clear path ahead. Please wait.")

    def _speak_object_location(self, target: str):
        target = (target or "").strip().lower()
        if not target:
            self._say("Please specify what to find.")
            return

        objs = self._stable_objects()
        matches = []
        for obj in objs:
            label = obj.label.lower()
            if target in label or label in target:
                d = obj.nearest_dist_m if obj.nearest_dist_m is not None else obj.distance_m
                matches.append((obj, d))

        if not matches:
            self._say(f"I do not see a {target}.")
            return

        matches.sort(key=lambda t: (t[1] is None, t[1] if t[1] is not None else 1e9))
        best, best_d = matches[0]
        urgency = "info"
        if best_d is not None:
            if best_d < 0.8:
                urgency = "critical"
            elif best_d < 2.0:
                urgency = "warn"
        perception = self._build_perception(target=best, urgency=urgency)
        self._emit_protocol_speech(perception, source="find_object")

    def _speak_status(self):
        fps = self._fps_counter.fps()
        with self._state._lock:
            n_obj = len([o for o in self._state.tracked_objects if o.is_stable])
        uptime_s = max(0.0, time.time() - self._state.session_start_ts)
        mins = int(uptime_s // 60)
        secs = int(uptime_s % 60)
        device = self._config.device.upper()
        calib = "calibrated" if self._calib.is_calibrated else "uncalibrated"
        msg = (
            f"Status: {fps:.1f} frames per second, {n_obj} stable objects, "
            f"running {mins} minutes {secs} seconds, device {device}, {calib}."
        )
        self._say(msg)

    def get_distance_method_counts(self) -> Dict[str, int]:
        return dict(self._distance_method_counts)

    def get_uptime_s(self) -> float:
        return max(0.0, time.time() - self._state.session_start_ts)

    def get_display_snapshot(self) -> Dict[str, Any]:
        snap = self._state.snapshot()
        snap["fps"] = self._fps_counter.fps()
        snap["device"] = self._config.device.upper()
        snap["det_model"] = "OIV7-600" if "oiv7" in self._config.detector_model else "YOLO-World"
        active = getattr(self._depth_estimator, "_active_model_id", self._config.depth_model_id)
        snap["dep_model"] = "DAV2-Indoor" if "Indoor" in active else "DAV2-Outdoor"
        snap["vocab_n"] = len(self._detector.vocabulary)
        snap["n_objects"] = len([o for o in snap["tracked_objects"] if o.is_stable])
        snap["is_calibrated"] = self._calib.is_calibrated
        snap["uptime_s"] = self.get_uptime_s()
        snap["dist_method_counts"] = self.get_distance_method_counts()
        diag = self._fusion.diagnostics
        snap["fusion_global_scale"] = diag.global_scale
        snap["fusion_estimated_fx"] = diag.estimated_fx
        snap["fusion_samples"] = diag.samples_seen
        snap["fusion_method_counts"] = dict(diag.method_counts)
        snap["effective_fx"] = self._fusion.effective_fx(self._config.frame_width)
        snap["effective_fy"] = self._fusion.effective_fy(self._config.frame_height)
        snap["calib_reproj_px"] = (
            float(self._calib.reprojection_error) if self._calib.is_calibrated else None
        )
        snap["calib_quality_warn_px"] = self._config.calibration_quality_warn_px
        snap["landmark_count"] = len(self._landmarks._landmarks) if self._landmarks is not None else 0
        snap["protocol_mode"] = self._protocol_mode.value
        if self._last_protocol_message is not None:
            snap["protocol_last_message"] = self._last_protocol_message.text
            snap["protocol_last_word_count"] = self._last_protocol_message.word_count
            snap["protocol_last_action"] = self._last_protocol_message.action_category
        else:
            snap["protocol_last_message"] = ""
            snap["protocol_last_word_count"] = 0
            snap["protocol_last_action"] = ""
        snap["protocol_last_response_ms"] = self._protocol_logger.last_response_ms
        snap["protocol_last_speech_ms"] = self._protocol_logger.last_speech_ms
        snap["protocol_overlap_count"] = self._protocol_logger.overlap_count
        snap["protocol_duplicate_count"] = self._protocol_logger.duplicate_count
        return snap

    def reload_vocabulary(self):
        vocab = load_vocabulary(self._config.vocabulary_file)
        if vocab:
            self._detector.set_vocabulary(vocab)
            logger.info(f"Vocabulary reloaded: {len(vocab)} classes")
        else:
            logger.warning("Vocabulary reload found empty file")

    def save_screenshot(self, frame: np.ndarray) -> str:
        os.makedirs(self._config.screenshots_dir, exist_ok=True)
        ts = int(time.time())
        path = os.path.join(self._config.screenshots_dir, f"screenshot_{ts}.png")
        cv2.imwrite(path, frame)
        logger.info(f"Screenshot saved: {path}")
        return path

    def stop(self):
        logger.info("Pipeline stopping...")
        self._running = False
        if self._listener is not None:
            self._listener.stop()
        try:
            self._imu.stop()
        except Exception:
            pass
        for t in [self._capture_thread, self._detection_thread, self._depth_thread, self._fusion_thread]:
            if t and t.is_alive():
                t.join(timeout=3.0)
        if self._cap:
            self._cap.release()
        if self._writer:
            self._writer.release()
        if self._phone_server is not None:
            try:
                self._phone_server.stop()
            except Exception:
                pass
            self._phone_server = None
            self._phone_active_label = ""
            self._config.phone_mode = False
            if self._speech is not None:
                try:
                    self._speech.set_text_listener(None)
                    self._speech.set_mute_local(False)
                except Exception:
                    pass
        self._event_logger.close()
        try:
            self._protocol_logger.close()
        except Exception:
            pass
        logger.info("Pipeline stopped")
