import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from queue import Queue, Empty
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PHONE_KEYWORDS = [
    "iphone", "ipad", "apple", "camo", "ivcam", "epoccam", "reincubate",
    "xiaomi", "redmi", "poco", "miui", "droidcam", "iriun",
]


def _get_pnp_cameras_windows() -> List[Tuple[str, str]]:
    if platform.system() != "Windows":
        return []
    try:
        ps_cmd = (
            "Get-PnpDevice -Class Camera | "
            "Select-Object FriendlyName,Status | "
            "ConvertTo-Json -Compress"
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=8,
        )
        if not r.stdout.strip():
            return []
        data = json.loads(r.stdout)
        if isinstance(data, dict):
            data = [data]
        return [(d.get("FriendlyName") or "", d.get("Status") or "") for d in data]
    except Exception:
        return []


def _get_adb_devices() -> List[str]:
    try:
        r = subprocess.run(
            ["adb", "devices"], capture_output=True, text=True, timeout=8,
        )
        serials = []
        for line in r.stdout.strip().splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) == 2 and parts[1].strip() == "device":
                serials.append(parts[0].strip())
        return serials
    except Exception:
        return []


def _get_apple_usb_device_names() -> List[str]:
    if platform.system() != "Windows":
        return []
    try:
        ps_cmd = (
            "Get-PnpDevice | Where-Object { "
            "$_.FriendlyName -match '(?i)(apple|iphone|ipad)' -and "
            "$_.Status -eq 'OK' } | "
            "Select-Object FriendlyName | ConvertTo-Json -Compress"
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=8,
        )
        if not r.stdout.strip():
            return []
        data = json.loads(r.stdout)
        if isinstance(data, dict):
            data = [data]
        return [d.get("FriendlyName") or "" for d in data if d.get("FriendlyName")]
    except Exception:
        return []


def _is_phone_name(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in PHONE_KEYWORDS)


class ScrcpyCameraCapture:
    def __init__(self, serial: str, width: int, height: int, fps: int):
        self._width = width
        self._height = height
        self._frame_bytes = width * height * 3
        self._scrcpy: Optional[subprocess.Popen] = None
        self._ffmpeg: Optional[subprocess.Popen] = None
        self._opened = False

        try:
            subprocess.run(["scrcpy", "--version"], capture_output=True, timeout=3, check=True)
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=3, check=True)
        except Exception:
            return

        scrcpy_cmd = [
            "scrcpy",
            f"--serial={serial}",
            "--video-source=camera",
            "--camera-facing=back",
            f"--camera-size={width}x{height}",
            f"--camera-fps={fps}",
            "--no-audio",
            "--no-window",
            "--record=-",
            "--record-format=mkv",
        ]
        ffmpeg_cmd = [
            "ffmpeg", "-loglevel", "quiet",
            "-i", "pipe:0",
            "-vf", f"fps={fps}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]
        try:
            self._scrcpy = subprocess.Popen(
                scrcpy_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self._ffmpeg = subprocess.Popen(
                ffmpeg_cmd,
                stdin=self._scrcpy.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            time.sleep(1.5)
            if self._scrcpy.poll() is None and self._ffmpeg.poll() is None:
                self._opened = True
        except Exception as e:
            logger.debug(f"ScrcpyCameraCapture init failed for {serial}: {e}")
            self._teardown()

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened or self._ffmpeg is None:
            return False, None
        try:
            raw = self._ffmpeg.stdout.read(self._frame_bytes)
            if len(raw) != self._frame_bytes:
                self._opened = False
                return False, None
            return True, np.frombuffer(raw, dtype=np.uint8).reshape((self._height, self._width, 3)).copy()
        except Exception:
            return False, None

    def set(self, prop_id: int, value: float) -> bool:
        return False

    def release(self) -> None:
        self._opened = False
        self._teardown()

    def _teardown(self) -> None:
        for proc in (self._ffmpeg, self._scrcpy):
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass


def _open_camera(index: int, width: int, height: int, fps: int) -> Optional[cv2.VideoCapture]:
    sysname = platform.system()
    if sysname == "Darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    elif sysname == "Windows":
        backends = [getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY), cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY, cv2.CAP_AVFOUNDATION]
    for be in backends:
        try:
            cap = cv2.VideoCapture(index, be)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                ok, fr = cap.read()
                if ok and fr is not None:
                    logger.info(f"Camera {index} opened backend={be} resolution={fr.shape[1]}x{fr.shape[0]}")
                    return cap
            cap.release()
        except Exception as e:
            logger.debug(f"Camera {index} backend {be} failed: {e}")
    return None


def _open_camera_url(url: str, width: int, height: int, fps: int) -> Optional[cv2.VideoCapture]:
    try:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            ok, fr = cap.read()
            if ok and fr is not None:
                logger.info(f"Camera URL {url!r} opened resolution={fr.shape[1]}x{fr.shape[0]}")
                return cap
        cap.release()
    except Exception as e:
        logger.debug(f"Camera URL {url!r} failed: {e}")
    return None


def _open_adb_camera(serial: str, width: int, height: int, fps: int) -> Optional[Any]:
    scrcpy_cap = ScrcpyCameraCapture(serial, width, height, fps)
    if scrcpy_cap.isOpened():
        ok, fr = scrcpy_cap.read()
        if ok and fr is not None:
            logger.info(f"ADB camera {serial} opened via scrcpy {fr.shape[1]}x{fr.shape[0]}")
            return scrcpy_cap
        scrcpy_cap.release()

    try:
        subprocess.run(
            ["adb", "-s", serial, "forward", "tcp:8080", "tcp:8080"],
            capture_output=True, timeout=5,
        )
        cap = _open_camera_url("http://localhost:8080/video", width, height, fps)
        if cap is not None:
            logger.info(f"ADB camera {serial} opened via port-forwarded MJPEG")
            return cap
        subprocess.run(
            ["adb", "-s", serial, "forward", "--remove", "tcp:8080"],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass

    return None


def _resolve_camera(spec: str, width: int, height: int, fps: int) -> Optional[Any]:
    if spec.startswith("adb:"):
        return _open_adb_camera(spec[4:], width, height, fps)
    if "://" in spec:
        return _open_camera_url(spec, width, height, fps)
    if spec.lstrip("-").isdigit():
        return _open_camera(int(spec), width, height, fps)
    return None


def _auto_detect_phones() -> List[Tuple[str, str]]:
    detected = []
    for serial in _get_adb_devices():
        detected.append((f"adb:{serial}", "Android/Xiaomi (ADB)"))
    for idx, (name, status) in enumerate(_get_pnp_cameras_windows()):
        if status == "OK" and _is_phone_name(name):
            detected.append((str(idx), f"Phone camera: {name}"))
    return detected


def _scan_and_print_devices(width: int, height: int, fps: int) -> None:
    print("\n=== USB Phone Detection ===")

    adb_serials = _get_adb_devices()
    if adb_serials:
        print("\nAndroid/Xiaomi devices via ADB:")
        for s in adb_serials:
            print(f"  --left-cam adb:{s}   or   --right-cam adb:{s}")
    else:
        print("\nNo ADB Android devices found  (install platform-tools and enable USB debugging on device)")

    apple_names = _get_apple_usb_device_names()
    if apple_names:
        print("\nApple devices connected via USB:")
        for name in apple_names:
            print(f"  {name}")
        print("  iPhone camera appears as a numbered index below when Apple Devices app is installed")

    print("\n=== Available Camera Indices (0-9) ===")
    cam_names = _get_pnp_cameras_windows()
    found_any = False
    for idx in range(10):
        cap_obj = _open_camera(idx, width, height, fps)
        if cap_obj is not None:
            found_any = True
            name = cam_names[idx][0] if idx < len(cam_names) else ""
            tag = " [PHONE]" if _is_phone_name(name) else ""
            print(f"  index {idx}: {name or '(unnamed)'}{tag}")
            cap_obj.release()
    if not found_any:
        print("  No cameras found")
    print()


class CameraThread(threading.Thread):
    def __init__(self, cap: Any, name: str, queue: Queue):
        super().__init__(daemon=True, name=name)
        self._cap = cap
        self._queue = queue
        self._running = False

    def start_capture(self):
        self._running = True
        self.start()

    def run(self):
        while self._running:
            ok, frame = self._cap.read()
            ts = time.time()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            try:
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        pass
                self._queue.put_nowait((ts, frame))
            except Exception:
                pass

    def stop(self):
        self._running = False


def _frame_brightness(frame: np.ndarray) -> float:
    if frame is None or frame.size == 0:
        return 0.0
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return float(gray.mean())


def _classify_lighting(brightness: float) -> str:
    if brightness < 50:
        return "dark"
    if brightness < 100:
        return "dim"
    if brightness < 170:
        return "normal"
    return "bright"


def main():
    ap = argparse.ArgumentParser(description="Synchronized two-camera recorder for empirical research")
    ap.add_argument("--left-cam", type=str, default="0",
                    help="Index (0), 'adb:<serial>' for Android/Xiaomi, or stream URL")
    ap.add_argument("--right-cam", type=str, default="1",
                    help="Index (1), 'adb:<serial>' for Android/Xiaomi, or stream URL")
    ap.add_argument("--environment", type=str, default="home",
                    choices=["home", "street", "university", "mixed"])
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out-root", type=str, default="recordings")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = unlimited until 'q'")
    ap.add_argument("--show", action="store_true", help="Display side-by-side preview")
    ap.add_argument("--scan", action="store_true",
                    help="Detect and list all connected cameras and USB phones, then exit")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.scan:
        _scan_and_print_devices(args.width, args.height, args.fps)
        return

    phones = _auto_detect_phones()
    if phones and args.left_cam == "0" and args.right_cam == "1":
        logger.info("USB phone cameras detected — pass these as --left-cam / --right-cam:")
        for spec, label in phones:
            logger.info(f"  {spec}  ({label})")

    left = _resolve_camera(args.left_cam, args.width, args.height, args.fps)
    right = _resolve_camera(args.right_cam, args.width, args.height, args.fps)

    if left is None:
        logger.error(f"Cannot open left camera ({args.left_cam})")
        sys.exit(1)
    if right is None:
        logger.error(f"Cannot open right camera ({args.right_cam})")
        left.release()
        sys.exit(1)

    ts_label = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(args.out_root, f"{args.environment}_{ts_label}")
    left_dir = os.path.join(session_dir, "left")
    right_dir = os.path.join(session_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    left_q: Queue = Queue(maxsize=2)
    right_q: Queue = Queue(maxsize=2)
    left_th = CameraThread(left, "LeftCam", left_q)
    right_th = CameraThread(right, "RightCam", right_q)
    left_th.start_capture()
    right_th.start_capture()

    sync_deltas = []
    brightness_samples = []
    saved = 0
    started = time.time()
    sync_window_s = 0.05
    last_sync_warn = 0.0

    logger.info(f"Recording to {session_dir}  press q to stop")

    try:
        while True:
            try:
                lt, lframe = left_q.get(timeout=1.0)
            except Empty:
                continue
            try:
                rt, rframe = right_q.get(timeout=1.0)
            except Empty:
                continue

            delta = abs(lt - rt)
            sync_deltas.append(delta)

            if delta > sync_window_s:
                now = time.time()
                if now - last_sync_warn > 5.0:
                    logger.warning(f"Sync delta {delta*1000:.1f} ms exceeds {sync_window_s*1000:.0f} ms window")
                    last_sync_warn = now

            idx = saved
            cv2.imwrite(os.path.join(left_dir, f"frame_{idx:06d}.jpg"), lframe)
            cv2.imwrite(os.path.join(right_dir, f"frame_{idx:06d}.jpg"), rframe)

            brightness_samples.append(_frame_brightness(lframe))
            saved += 1

            if args.show:
                lh, lw = lframe.shape[:2]
                rh, rw = rframe.shape[:2]
                target_h = min(lh, rh, 540)
                lp = cv2.resize(lframe, (int(lw * target_h / lh), target_h))
                rp = cv2.resize(rframe, (int(rw * target_h / rh), target_h))
                preview = np.hstack([lp, rp])
                cv2.putText(
                    preview,
                    f"frames={saved}  sync={delta*1000:.1f}ms  env={args.environment}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 80), 2,
                )
                cv2.imshow("StereoRecorder (q to stop)", preview)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if args.max_frames > 0 and saved >= args.max_frames:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted")

    ended = time.time()

    left_th.stop()
    right_th.stop()
    left.release()
    right.release()
    if args.show:
        cv2.destroyAllWindows()

    avg_delta = float(np.mean(sync_deltas)) if sync_deltas else 0.0
    max_delta = float(np.max(sync_deltas)) if sync_deltas else 0.0
    avg_bright = float(np.mean(brightness_samples)) if brightness_samples else 0.0
    duration = ended - started
    fps_actual = saved / duration if duration > 0 else 0.0

    metadata = {
        "environment": args.environment,
        "left_cam": args.left_cam,
        "right_cam": args.right_cam,
        "resolution": {"width": args.width, "height": args.height},
        "requested_fps": args.fps,
        "actual_fps": round(fps_actual, 2),
        "total_frames": saved,
        "duration_s": round(duration, 3),
        "avg_sync_delta_ms": round(avg_delta * 1000.0, 3),
        "max_sync_delta_ms": round(max_delta * 1000.0, 3),
        "avg_frame_brightness": round(avg_bright, 2),
        "lighting_estimate": _classify_lighting(avg_bright),
        "started_at": started,
        "ended_at": ended,
        "session_dir": session_dir,
    }

    meta_path = os.path.join(session_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 60)
    print(" STEREO RECORDING SUMMARY")
    print("=" * 60)
    for k, v in metadata.items():
        print(f"  {k:<22s} {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
