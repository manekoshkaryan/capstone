"""Plain HTTP/WebSocket server that receives LiDAR depth frames from the iOS app.

Runs on port 8444 (plain HTTP, not HTTPS — iOS URLSession connects directly).
Stores the latest depth frame per device label for the pipeline to consume.
"""
from __future__ import annotations

import asyncio
import logging
import struct
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_PORT = 8444


class DepthFrame:
    __slots__ = ("depth_m", "width", "height", "timestamp", "received_at")

    def __init__(self, depth_m: np.ndarray, width: int, height: int, timestamp: float):
        self.depth_m = depth_m          # float32 array, shape (H, W), meters
        self.width = width
        self.height = height
        self.timestamp = timestamp
        self.received_at = time.time()


class LiDARDepthServer:
    """Thread-safe store for latest LiDAR depth frames + asyncio WebSocket server."""

    def __init__(self, port: int = _PORT):
        self._port = port
        self._frames: dict[str, DepthFrame] = {}
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Public API (called from pipeline thread)
    # ------------------------------------------------------------------ #

    def latest(self, label: str = "iphone") -> Optional[DepthFrame]:
        with self._lock:
            frame = self._frames.get(label)
        if frame is None:
            return None
        # Discard stale frames (>500ms)
        if time.time() - frame.received_at > 0.5:
            return None
        return frame

    def start(self):
        self._thread = threading.Thread(
            target=self._run_server, daemon=True, name="LiDARDepthServer"
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

        logger.info(f"LiDARDepthServer listening on ws://0.0.0.0:{self._port}/depth")
        async with websockets.serve(self._handle, "0.0.0.0", self._port):
            await asyncio.Future()  # run forever

    async def _handle(self, websocket):
        label = "iphone"
        # Parse label from query string if provided (?label=iphone2)
        path = getattr(websocket, "path", "") or ""
        if "label=" in path:
            label = path.split("label=")[-1].split("&")[0]

        logger.info(f"LiDAR depth client connected: label={label}")
        try:
            async for message in websocket:
                if not isinstance(message, bytes):
                    continue
                frame = self._parse(message)
                if frame:
                    with self._lock:
                        self._frames[label] = frame
        except Exception as e:
            logger.debug(f"LiDAR WS closed ({label}): {e}")
        logger.info(f"LiDAR depth client disconnected: label={label}")

    @staticmethod
    def _parse(data: bytes) -> Optional[DepthFrame]:
        # Header: 8 bytes f64 timestamp + 2 bytes u16 width + 2 bytes u16 height = 12 bytes
        if len(data) < 12:
            return None
        ts, w, h = struct.unpack_from("<dHH", data, 0)
        expected = 12 + w * h * 2  # uint16 payload
        if len(data) < expected:
            return None
        uint16_flat = np.frombuffer(data, dtype=np.uint16, count=w * h, offset=12)
        depth_m = uint16_flat.reshape(h, w).astype(np.float32) / 1000.0
        return DepthFrame(depth_m=depth_m, width=w, height=h, timestamp=ts)
