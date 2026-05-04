from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import os
import torch

def _resolve_device(preference: str) -> str:
    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preference

def _default_known_heights() -> Dict[str, float]:
    return {
        "person": 1.70,
        "pedestrian": 1.70,
        "man": 1.75,
        "woman": 1.65,
        "child": 1.20,
        "car": 1.50,
        "vehicle": 1.50,
        "bus": 3.00,
        "truck": 2.50,
        "motorcycle": 1.20,
        "bicycle": 1.10,
        "wheelchair": 1.20,
        "stroller": 1.05,
        "chair": 0.90,
        "bench": 0.85,
        "table": 0.75,
        "sofa": 0.85,
        "bed": 0.55,
        "door": 2.00,
        "doorway": 2.00,
        "traffic light": 0.60,
        "stop sign": 0.75,
        "traffic cone": 0.45,
        "fire hydrant": 0.65,
        "parking meter": 1.50,
        "bottle": 0.25,
        "cup": 0.10,
        "laptop": 0.25,
        "phone": 0.15,
        "monitor": 0.45,
        "tv": 0.55,
        "television": 0.55,
        "keyboard": 0.04,
        "mouse": 0.04,
        "book": 0.22,
        "dog": 0.50,
        "cat": 0.30,
        "tree": 4.00,
        "trash can": 0.95,
        "pole": 3.00,
        "pillar": 2.50,
        "bollard": 0.95,
        "fire extinguisher": 0.55,
        "shopping cart": 1.00,
        "suitcase": 0.65,
        "backpack": 0.50,
        "refrigerator": 1.70,
        "microwave": 0.30,
        "sink": 0.30,
        "toilet": 0.75,
        "face": 0.23,
        "head": 0.23,
    }

def _default_known_widths() -> Dict[str, float]:
    return {
        "person": 0.45,
        "pedestrian": 0.45,
        "man": 0.48,
        "woman": 0.42,
        "child": 0.32,
        "face": 0.16,
        "head": 0.16,
        "car": 1.80,
        "vehicle": 1.80,
        "bus": 2.55,
        "truck": 2.45,
        "motorcycle": 0.80,
        "bicycle": 0.55,
        "wheelchair": 0.65,
        "stroller": 0.55,
        "chair": 0.50,
        "bench": 1.50,
        "table": 1.20,
        "sofa": 2.00,
        "bed": 1.40,
        "door": 0.90,
        "doorway": 0.90,
        "traffic light": 0.30,
        "stop sign": 0.75,
        "traffic cone": 0.25,
        "fire hydrant": 0.30,
        "bottle": 0.07,
        "cup": 0.08,
        "laptop": 0.34,
        "phone": 0.07,
        "cell phone": 0.07,
        "mobile phone": 0.07,
        "monitor": 0.55,
        "tv": 1.10,
        "television": 1.10,
        "keyboard": 0.45,
        "mouse": 0.06,
        "book": 0.15,
        "dog": 0.40,
        "cat": 0.20,
        "trash can": 0.40,
        "pole": 0.10,
        "pillar": 0.40,
        "bollard": 0.20,
        "fire extinguisher": 0.18,
        "shopping cart": 0.55,
        "suitcase": 0.45,
        "backpack": 0.35,
        "refrigerator": 0.75,
        "microwave": 0.50,
        "sink": 0.55,
        "toilet": 0.45,
    }

@dataclass
class AppConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    capture_fps: int = 30

    detector_model: str = "yolov8s-oiv7.pt"
    detector_backend: str = "yolov8-oiv7"
    detector_conf_threshold: float = 0.28
    detector_iou_threshold: float = 0.45
    detector_imgsz: int = 640
    detection_interval_frames: int = 1

    depth_model_id: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    depth_model_id_fallback: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
    depth_input_size: int = 518
    depth_interval_frames: int = 2
    depth_max_range_m: float = 20.0

    device_preference: str = "auto"
    use_half_precision: bool = True

    vocabulary_file: str = "vocabulary.txt"
    calibration_file: str = "camera_calibration.npz"
    calibration_file_path: str = "camera_calibration.npz"
    screenshots_dir: str = "screenshots"

    tracker_max_age: int = 30
    tracker_min_hits: int = 2
    tracker_iou_threshold: float = 0.25

    ema_alpha_distance: float = 0.25
    ema_alpha_position: float = 0.4
    depth_roi_trim_low: float = 10.0
    depth_roi_trim_high: float = 90.0
    depth_min_valid_pixels: int = 20
    depth_variance_high_threshold: float = 0.8
    depth_variance_medium_threshold: float = 0.25
    roi_size_high_threshold: int = 2000
    roi_size_medium_threshold: int = 400
    det_conf_high_threshold: float = 0.65
    det_conf_medium_threshold: float = 0.40

    obstacle_warning_distance_m: float = 2.0
    depth_stale_frames_limit: int = 6
    detection_stale_frames_limit: int = 5

    enable_speech: bool = True
    enable_event_log: bool = True
    event_log_file: str = "data/logs/events.jsonl"
    nav_log_file: str = "data/logs/navigation.jsonl"

    # Wise speech policy thresholds — see speech_policy.SpeechPolicy.
    # Repetition / risk-band constants (assistive-nav spec).
    speech_dist_change_threshold_m: float = 0.5      # MIN_DISTANCE_CHANGE_M
    speech_repeat_critical_s: float = 1.5            # HIGH_RISK_REPEAT_SEC
    speech_repeat_warn_s: float = 4.0                # MEDIUM_RISK_REPEAT_SEC
    speech_repeat_info_s: float = 8.0                # LOW_RISK_REPEAT_SEC
    speech_critical_dist_m: float = 1.0              # DANGEROUS_DISTANCE_M
    speech_warn_dist_m: float = 2.0                  # CAUTION_DISTANCE_M
    speech_info_dist_m: float = 5.0
    speech_entry_ttl_s: float = 12.0
    speech_global_min_gap_s: float = 0.7
    speech_clear_path_repeat_s: float = 10.0         # CLEAR_PATH_REPEAT_SEC
    speech_tts_reverb_tail_s: float = 0.3            # TTS_REVERB_TAIL_SEC

    nav_max_announce_dist_m: float = 12.0
    nav_track_cooldown_s: float = 4.0
    nav_global_cooldown_s: float = 1.0
    nav_no_depth_cooldown_s: float = 5.0
    enable_recording: bool = False
    recording_file: str = "recording.avi"

    show_depth_view: bool = False
    show_tracking: bool = False
    verbose_diagnostics: bool = False
    paused: bool = False

    calibration_board_cols: int = 9
    calibration_board_rows: int = 6
    calibration_square_size_m: float = 0.025
    square_size_mm: float = 25.0
    calibration_frames_needed: int = 20
    min_calibration_frames: int = 15
    checkerboard_size: Tuple[int, int] = (9, 6)
    calibration_quality_warn_px: float = 1.0
    calibration_quality_reject_px: float = 1.5
    apply_undistortion_max_reproj_px: float = 1.0

    use_geometric_distance: bool = True
    apply_undistortion: bool = True
    known_object_heights: Dict[str, float] = field(default_factory=_default_known_heights)
    known_object_widths: Dict[str, float] = field(default_factory=_default_known_widths)

    system_name: str = "ManeNavSystem"

    distance_scale_alpha: float = 0.06
    online_fx_alpha: float = 0.04
    enable_online_calibration: bool = True
    default_h_fov_deg: float = 68.0
    default_v_fov_deg: float = 50.0
    near_field_threshold_m: float = 2.6
    near_field_min_factor: float = 0.38
    geometric_partial_clip_margin_px: int = 4
    geometric_min_box_dim_px: int = 10
    max_calibration_reprojection_px: float = 0.8
    calibration_min_distinct_angles: int = 4
    calibration_corner_position_min_var: float = 0.018
    initial_depth_scale_factor: float = 0.55
    geometric_height_weight: float = 1.0
    geometric_width_weight: float = 0.9
    fusion_log_interval_s: float = 30.0
    closeup_bbox_area_ratio: float = 0.35
    closeup_clip_margin_px: int = 2
    user_scale_step: float = 0.05
    user_scale_min: float = 0.10
    user_scale_max: float = 1.80
    user_scale_factor: float = 0.24

    enable_guidance: bool = True
    guidance_sector_names: Tuple[str, ...] = ("far_left", "left", "center", "right", "far_right")
    guidance_open_horizon_m: float = 5.0
    guidance_min_go_distance_m: float = 1.6
    guidance_stop_distance_m: float = 0.8
    guidance_distance_percentile: float = 8.0
    guidance_min_pixels_for_obstacle: int = 80
    guidance_sample_top_ratio: float = 0.45
    guidance_repeat_cooldown_s: float = 4.0
    guidance_stop_cooldown_s: float = 12.0
    guidance_min_speak_interval_s: float = 1.5
    guidance_h_fov_total_deg: float = 68.0
    # Per-sector free-distance smoothing — kills per-frame depth noise that
    # was flipping best_idx between adjacent sectors every cycle.
    guidance_sector_ema_alpha: float = 0.4
    # Center sector wins ties unless a side is at least this much clearer.
    guidance_center_bias_m: float = 0.5
    # Direction switch must improve over center by this many meters AND hold
    # for `switch_min_frames` frames before a new BEAR_*/GO_* fires.
    guidance_switch_margin_m: float = 0.4
    guidance_switch_min_frames: int = 3

    # CONTINUOUS_GUIDANCE-mode repeat intervals. Tighter than the default
    # passive-mode throttling: directional corrections must keep up with
    # walking pace, "Continue straight" should not spam every second, and
    # critical stops still have priority. Validator + SpeechPolicy still
    # gate every utterance — these only shorten the silence window.
    continuous_clear_interval_s: float = 6.0
    continuous_directional_interval_s: float = 3.0
    continuous_danger_interval_s: float = 1.5
    # After this many back-to-back STOP announcements with no risk-up,
    # back off to the long cooldown so we stop nagging a user who has
    # already stopped. Reset when the action changes (path opens, bearing
    # shifts, etc.). Real session log showed 60+ "Stop." in 90s — this is
    # the spam cap.
    continuous_stop_repeat_cap: int = 3
    continuous_stop_backoff_s: float = 12.0

    enable_floor_segmentation: bool = True
    camera_height_m: float = 1.10
    floor_depth_ratio_low: float = 0.70
    floor_depth_ratio_high: float = 1.35
    floor_min_below_horizon_px: int = 10

    enable_imu: bool = True
    imu_serial_port: str = ""
    imu_baud: int = 115200
    imu_use_optical_flow_fallback: bool = True

    enable_landmarks: bool = True
    landmark_min_det_conf: float = 0.45
    landmark_position_alpha: float = 0.25
    landmark_max_age_s: float = 90.0

    show_floor_overlay: bool = False
    show_sector_bars: bool = False

    enable_voice_input: bool = True
    voice_input_energy_threshold: int = 300
    voice_input_pause_threshold: float = 0.85
    voice_input_phrase_time_limit: float = 5.0
    voice_input_ambient_calibration_s: float = 0.6
    voice_input_dynamic_energy: bool = True

    session_summary_dir: str = "data/sessions"
    session_plots_dir: str = "."
    auto_generate_plots: bool = False

    camera_source: str = "iphone"
    phone_camera_port: int = 8443
    phone_camera_label: str = "iphone"
    phone_camera_cert_dir: str = "."
    phone_camera_wait_first_frame_s: float = 60.0
    mute_local_speech_when_phone_active: bool = False

    phone_mode: bool = False
    phone_passthrough: bool = True
    rotate_phone_frame: str = "0"
    preserve_aspect_ratio: bool = True
    phone_output_width: int = 1280
    phone_output_height: int = 720

    enable_web_output: bool = True
    web_output_jpeg_quality: int = 80

    # ------------------------------------------------------------------
    # Voice backend selection.
    #
    # "local"           — existing STT/local-TTS conversation stack. No
    #                     external services. Safety TTS uses local engine.
    # "openai_realtime" — OpenAI Realtime voice-to-voice for free-form
    #                     conversation. Safety TTS still local & deterministic.
    #
    # Both modes share the same NavigationContext / SpeechPolicy /
    # CommandValidator. The LLM never decides STOP / MOVE_*.
    # ------------------------------------------------------------------
    voice_backend: str = "local"
    openai_api_key: str = ""
    openai_realtime_model: str = "gpt-realtime-mini"
    openai_realtime_voice: str = "alloy"
    openai_realtime_session_url: str = "https://api.openai.com/v1/realtime/sessions"
    openai_realtime_webrtc_url: str = "https://api.openai.com/v1/realtime"
    openai_realtime_instructions: str = (
        "You are a calm voice assistant for a navigation system for visually "
        "impaired users. Be brief by default. Do not speak unless the user "
        "asks or backend context requires a non-safety answer. Use only the "
        "latest backend navigation context provided to you. Never independently "
        "tell the user to stop, turn, move left, move right, or continue. "
        "Safety commands come only from the local navigation system. "
        "If unsure, say briefly that you are not sure."
    )
    voice_context_min_interval_s: float = 1.0

    protocol_mode: str = "adaptive"
    enable_protocol_logging: bool = True
    protocol_events_file: str = "data/logs/protocol_events.jsonl"
    protocol_metrics_file: str = "protocol_metrics.csv"
    latency_comparison_file: str = "latency_comparison.csv"
    protocol_summary_csv: str = "protocol_summary.csv"
    protocol_summary_json: str = "protocol_summary.json"
    protocol_accuracy_csv: str = "protocol_accuracy.csv"
    scenario_output_file: str = "scenario_protocol_outputs.csv"
    ground_truth_protocol_file: str = "ground_truth_protocol.csv"
    protocol_plot_dir: str = "outputs/protocol_plots"
    cascaded_baseline_asr_ms: float = 600.0
    cascaded_baseline_llm_ms: float = 900.0
    cascaded_baseline_tts_ms: float = 700.0

    def apply_env_overrides(self) -> None:
        """Pull voice-related settings from environment variables.

        Only reads, never writes. Leaves field defaults intact when a var
        is unset, so the local mode keeps working with no env at all.
        """
        backend = os.environ.get("VOICE_BACKEND", "").strip().lower()
        if backend in ("local", "openai_realtime"):
            self.voice_backend = backend
        for env_key, attr in (
            ("OPENAI_API_KEY", "openai_api_key"),
            ("OPENAI_REALTIME_MODEL", "openai_realtime_model"),
            ("OPENAI_REALTIME_VOICE", "openai_realtime_voice"),
        ):
            val = os.environ.get(env_key, "").strip()
            if val:
                setattr(self, attr, val)

    @property
    def device(self) -> str:
        return _resolve_device(self.device_preference)

    @property
    def use_cuda(self) -> bool:
        return self.device.startswith("mps")

    def effective_half(self) -> bool:
        return self.use_half_precision and self.device == "mps"

    def known_height_for(self, label: str) -> Optional[float]:
        return _lookup_size(label, self.known_object_heights)

    def known_width_for(self, label: str) -> Optional[float]:
        return _lookup_size(label, self.known_object_widths)

def _lookup_size(label: Optional[str], table: Dict[str, float]) -> Optional[float]:
    if not label:
        return None
    key = label.strip().lower()
    if key in table:
        return table[key]
    for k, v in table.items():
        if k in key or key in k:
            return v
    return None
