"""Unified WebSocket server that receives video + LiDAR depth from the iOS app.

Runs on port 8444 (plain HTTP/WebSocket).
Binary frame format (1-byte type prefix):
  0x01 + 8-byte f64 timestamp + JPEG bytes         (video frame)
  0x02 + 8-byte f64 ts + 2-byte u16 w + 2-byte u16 h + uint16[] depth (mm)
"""
from __future__ import annotations

import asyncio
import logging
import struct
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PORT = 8444
_FRAME_STALE_S = 0.5


class DepthFrame:
    __slots__ = ("depth_m", "width", "height", "timestamp", "received_at")

    def __init__(self, depth_m: np.ndarray, width: int, height: int, timestamp: float):
        self.depth_m = depth_m
        self.width = width
        self.height = height
        self.timestamp = timestamp
        self.received_at = time.time()


class VideoFrame:
    __slots__ = ("bgr", "timestamp", "received_at")

    def __init__(self, bgr: np.ndarray, timestamp: float):
        self.bgr = bgr
        self.timestamp = timestamp
        self.received_at = time.time()


class LiDARDepthServer:
    """Receives video + depth from iOS app. Thread-safe store + asyncio WS server."""

    def __init__(self, port: int = _PORT):
        self._port = port
        self._depth: Optional[DepthFrame] = None
        self._video: Optional[VideoFrame] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._client_ws = None  # active WebSocket to iOS app
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._annotated_skip = 0  # throttle: send every Nth frame

    # ------------------------------------------------------------------ #
    # Public API (pipeline thread)
    # ------------------------------------------------------------------ #

    def latest_depth(self, label: str = "iphone") -> Optional[DepthFrame]:
        with self._lock:
            frame = self._depth
        if frame is None:
            return None
        if time.time() - frame.received_at > _FRAME_STALE_S:
            return None
        return frame

    def latest_video(self) -> Optional[VideoFrame]:
        with self._lock:
            frame = self._video
        if frame is None:
            return None
        if time.time() - frame.received_at > _FRAME_STALE_S:
            return None
        return frame

    def is_video_connected(self) -> bool:
        v = self.latest_video()
        return v is not None

    def start(self):
        self._thread = threading.Thread(
            target=self._run_server, daemon=True, name="LiDARStreamServer"
        )
        self._thread.start()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _run_server(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        try:
            import websockets
        except ImportError:
            logger.error("websockets not installed — run: pip install websockets")
            return

        logger.info(f"LiDAR stream server listening on ws://0.0.0.0:{self._port}/stream")
        async with websockets.serve(self._handle, "0.0.0.0", self._port):
            await asyncio.Future()

    def push_annotated_frame(self, bgr_frame: np.ndarray):
        """Send annotated frame (type 0x03) to connected iOS app. Thread-safe."""
        with self._lock:
            ws = self._client_ws
        if ws is None:
            return
        ret, jpeg = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not ret:
            return
        payload = bytes([0x03]) + jpeg.tobytes()
        loop = self._loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(self._send_to_client(ws, payload), loop)

    async def _handle(self, websocket):
        logger.info("iOS app connected")
        with self._lock:
            self._client_ws = websocket
        try:
            async for message in websocket:
                if not isinstance(message, bytes) or len(message) < 9:
                    continue
                msg_type = message[0]
                if msg_type == 0x01:
                    self._parse_video(message)
                elif msg_type == 0x02:
                    self._parse_depth(message)
        except Exception as e:
            logger.debug(f"iOS WS closed: {e}")
        with self._lock:
            self._client_ws = None
        logger.info("iOS app disconnected")

    @staticmethod
    async def _send_to_client(ws, payload: bytes):
        try:
            await ws.send(payload)
        except Exception:
            pass

    def _parse_video(self, data: bytes):
        # 0x01 + 8-byte f64 ts + JPEG
        if len(data) < 10:
            return
        (ts,) = struct.unpack_from("<d", data, 1)
        jpeg = data[9:]
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        with self._lock:
            self._video = VideoFrame(bgr=bgr, timestamp=ts)

    def _parse_depth(self, data: bytes):
        # 0x02 + 8-byte f64 ts + 2-byte u16 w + 2-byte u16 h + uint16[]
        if len(data) < 13:
            return
        ts, w, h = struct.unpack_from("<dHH", data, 1)
        expected = 13 + w * h * 2
        if len(data) < expected:
            return
        uint16_flat = np.frombuffer(data, dtype=np.uint16, count=w * h, offset=13)
        depth_m = uint16_flat.reshape(h, w).astype(np.float32) / 1000.0
        with self._lock:
            self._depth = DepthFrame(depth_m=depth_m, width=w, height=h, timestamp=ts)
