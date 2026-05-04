import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class IMUReading:
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    yaw_deg: float = 0.0
    ts: float = 0.0
    valid: bool = False

class IMUSensor:
    def __init__(self):
        self._lock = threading.RLock()
        self._reading = IMUReading(ts=time.time(), valid=False)
        self._yaw_offset_deg: float = 0.0
        self._running = False

    def start(self) -> bool:
        self._running = True
        return True

    def stop(self) -> None:
        self._running = False

    def read(self) -> IMUReading:
        with self._lock:
            r = self._reading
            return IMUReading(r.pitch_deg, r.roll_deg, r.yaw_deg - self._yaw_offset_deg, r.ts, r.valid)

    def push_external(self, pitch_deg: float, roll_deg: float, yaw_deg: float) -> None:
        with self._lock:
            self._reading = IMUReading(
                pitch_deg=float(pitch_deg),
                roll_deg=float(roll_deg),
                yaw_deg=float(yaw_deg),
                ts=time.time(),
                valid=True,
            )

    def zero_heading(self) -> None:
        with self._lock:
            self._yaw_offset_deg = self._reading.yaw_deg

class SerialIMU(IMUSensor):
    def __init__(self, port: str, baud: int = 115200):
        super().__init__()
        self._port = port
        self._baud = baud
        self._thread: Optional[threading.Thread] = None
        self._serial = None

    def start(self) -> bool:
        try:
            import serial
            self._serial = serial.Serial(self._port, self._baud, timeout=0.5)
        except Exception as e:
            logger.warning(f"SerialIMU open failed on {self._port}: {e}")
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="IMUThread")
        self._thread.start()
        return True

    def _loop(self) -> None:
        while self._running and self._serial is not None:
            try:
                raw = self._serial.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                self.push_external(
                    obj.get("pitch", 0.0),
                    obj.get("roll", 0.0),
                    obj.get("yaw", 0.0),
                )
            except Exception:
                time.sleep(0.05)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            if self._serial is not None:
                self._serial.close()
        except Exception:
            pass

class FlowEstimatedIMU(IMUSensor):
    def __init__(self):
        super().__init__()
        self._prev_gray = None
        self._yaw_acc = 0.0
        self._pitch_acc = 0.0

    def update_from_frame(self, frame_bgr) -> None:
        try:
            import cv2
            import numpy as np
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))
            if self._prev_gray is None:
                self._prev_gray = gray
                return
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                0.5, 2, 20, 2, 5, 1.2, 0,
            )
            self._prev_gray = gray
            mean_dx = float(np.mean(flow[..., 0]))
            mean_dy = float(np.mean(flow[..., 1]))
            self._yaw_acc += mean_dx * 0.05
            self._pitch_acc += mean_dy * 0.04
            self.push_external(self._pitch_acc, 0.0, self._yaw_acc)
        except Exception:
            pass
