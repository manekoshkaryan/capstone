import asyncio
import logging
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from aiohttp import web
    _AIOHTTP_OK = True
except Exception as _e:
    _AIOHTTP_OK = False
    web = None


class WebOutputServer:
    def __init__(self, jpeg_quality: int = 80, max_fps: float = 20.0):
        self._jpeg_quality = int(jpeg_quality)
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_ts: float = 0.0
        self._last_encode_ts: float = 0.0
        self._min_encode_interval = 1.0 / max(1.0, float(max_fps))
        self._frame_count: int = 0
        self._viewer_count: int = 0
        self._status: Dict[str, Any] = {}

    def push_frame(self, bgr: np.ndarray) -> None:
        if bgr is None:
            return
        # Skip the JPEG encode when no MJPEG viewer is connected, or when the
        # previous encode is fresher than the min interval — burning ~10ms
        # per frame on the main loop for buffers nobody reads is wasted work.
        now = time.time()
        with self._lock:
            viewers = self._viewer_count
            stale = (now - self._last_encode_ts) < self._min_encode_interval
        if viewers == 0 or stale:
            return
        try:
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        except Exception as e:
            logger.debug(f"WebOutput encode error: {e}")
            return
        if not ok:
            return
        data = buf.tobytes()
        with self._lock:
            self._latest_jpeg = data
            self._latest_ts = now
            self._last_encode_ts = now
            self._frame_count += 1

    def update_status(self, **kw) -> None:
        with self._lock:
            self._status.update(kw)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            data = dict(self._status)
            data["frames"] = self._frame_count
            data["age_s"] = (time.time() - self._latest_ts) if self._latest_ts else None
        return data

    async def mjpeg_handler(self, request):
        boundary = "frame"
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": f"multipart/x-mixed-replace; boundary={boundary}",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Connection": "close",
            },
        )
        await resp.prepare(request)
        last_ts = 0.0
        with self._lock:
            self._viewer_count += 1
        try:
            while True:
                with self._lock:
                    jpeg = self._latest_jpeg
                    ts = self._latest_ts
                if jpeg is not None and ts != last_ts:
                    last_ts = ts
                    header = (
                        f"--{boundary}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n"
                    ).encode("ascii")
                    try:
                        await resp.write(header)
                        await resp.write(jpeg)
                        await resp.write(b"\r\n")
                    except (asyncio.CancelledError, ConnectionResetError, RuntimeError):
                        break
                    except Exception as e:
                        logger.debug(f"mjpeg client write error: {e}")
                        break
                else:
                    await asyncio.sleep(0.02)
        finally:
            with self._lock:
                self._viewer_count = max(0, self._viewer_count - 1)
            try:
                await resp.write_eof()
            except Exception:
                pass
        return resp

    async def view_handler(self, request):
        return web.Response(content_type="text/html", text=_VIEW_HTML)

    async def status_handler(self, request):
        return web.json_response(self.snapshot())

    def register_routes(self, app) -> None:
        if not _AIOHTTP_OK:
            return
        app.router.add_get("/stream.mjpg", self.mjpeg_handler)
        app.router.add_get("/view", self.view_handler)
        app.router.add_get("/output_status", self.status_handler)


_VIEW_HTML = r"""<!doctype html>
<html><head>
<meta charset="utf-8"><title>ManeNavSystem</title>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<style>
body{font-family:system-ui,sans-serif;background:#0b0e13;color:#e6edf3;margin:0;padding:6px}
h2{margin:2px 0 6px;font-size:16px;display:flex;justify-content:space-between;align-items:center}
.tag{font-size:11px;color:#8be08b;padding:2px 6px;background:#143;border-radius:5px}
#stream{width:100%;background:#000;border-radius:6px;display:block}
#status{font-size:12px;color:#8aa;margin-top:6px;line-height:1.4;background:#0a0d12;padding:8px;border-radius:6px}
</style></head><body>
<h2><span>ManeNavSystem</span><span class="tag" id="tag">view-only</span></h2>
<img id="stream" src="/stream.mjpg" alt="stream">
<div id="status">loading…</div>
<script>
async function refresh(){
  try{
    const r = await fetch('/output_status', {cache:'no-store'});
    const d = await r.json();
    const parts = [];
    for(const k of ['fps','device','det_model','dep_model','voice','phone','frames','n_objects','guidance_action']){
      if(d[k] !== undefined && d[k] !== null) parts.push(`<b>${k}</b>: ${d[k]}`);
    }
    document.getElementById('status').innerHTML = parts.join('  ·  ') || 'no status';
  }catch(e){ document.getElementById('status').textContent = 'status: '+e; }
}
refresh(); setInterval(refresh, 1000);
</script></body></html>
"""
