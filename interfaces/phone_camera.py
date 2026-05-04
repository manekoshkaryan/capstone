import asyncio
import ipaddress
import json
import logging
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import aiohttp
    from aiohttp import web
    from aiortc import RTCPeerConnection, RTCSessionDescription
    _WEBRTC_AVAILABLE = True
    _WEBRTC_IMPORT_ERROR = None
except Exception as _e:
    _WEBRTC_AVAILABLE = False
    _WEBRTC_IMPORT_ERROR = _e
    RTCPeerConnection = None
    RTCSessionDescription = None
    web = None
    aiohttp = None

from interfaces.web_output_server import WebOutputServer


_HTML_PAGE = r"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>ManeNavSystem</title>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<style>
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:#0b0e13;color:#e6edf3;margin:0;padding:6px}
body.cameraonly{padding:0;background:#000;overflow:hidden}
h2{margin:2px 0 6px;font-size:16px;display:flex;justify-content:space-between;align-items:center}
.tag{font-size:11px;color:#8be08b;padding:2px 6px;background:#143;border-radius:5px}
.tag.openai{background:#3a2c5a;color:#cdb6ff}
#stream{width:100%;background:#000;border-radius:6px;display:block}
body.cameraonly #stream{position:fixed;top:0;left:0;width:100vw;height:100vh;border-radius:0;object-fit:contain}
#localwrap{position:fixed;top:14px;right:14px;width:96px;height:64px;border:1px solid #555;border-radius:6px;overflow:hidden;background:#000;z-index:10}
#local{width:100%;height:100%;object-fit:cover}
body.cameraonly #localwrap{display:none}
body.cameraonly h2,body.cameraonly .banner,body.cameraonly .bar,body.cameraonly .row,body.cameraonly #micBtn,body.cameraonly #statusline,body.cameraonly #log{display:none}
#exitCamOnly{display:none;position:fixed;top:14px;right:14px;z-index:30;background:#722;color:#fff;border:0;border-radius:8px;padding:10px 14px;font-size:14px}
body.cameraonly #exitCamOnly{display:block}
.banner{margin:5px 0;padding:6px 8px;border-radius:6px;background:#141a23;font-size:13px;min-height:22px}
.banner.critical{background:#5a1a1a;color:#fff}
.banner.warn{background:#5a3a1a;color:#fff}
.banner.guide{background:#1a3a5a;color:#fff;font-weight:600}
.row{display:flex;flex-wrap:wrap;gap:5px;margin-top:5px}
button.act{flex:1 1 30%;min-height:46px;font-size:13px;background:#26303d;color:#e6edf3;border:0;border-radius:8px;padding:6px}
button.act:active{background:#3a4858}
button.warn{background:#553}
button.danger{background:#722}
button.voice{background:#2a72d4;color:#fff}
button.voice.live{background:#1a8a4a}
button.big{background:#2a72d4;flex:1 1 100%;min-height:54px;font-size:16px}
button.big.rec{background:#c33;animation:pulse 0.8s infinite}
@keyframes pulse{0%{opacity:1}50%{opacity:.55}100%{opacity:1}}
input[type=text]{flex:1;font-size:15px;padding:9px;border-radius:8px;border:0;background:#1c2330;color:#fff}
.bar{display:flex;gap:5px;margin-top:6px}
.bar button{padding:9px 12px;background:#2a72d4;color:#fff;border:0;border-radius:8px;font-size:14px}
#log{margin-top:8px;font-size:11px;color:#7aa;white-space:pre-wrap;background:#0a0d12;border-radius:6px;padding:6px;max-height:100px;overflow:auto}
#startbar{text-align:center;padding:14px}
#startbar button{font-size:18px;padding:14px 24px;background:#2a72d4;color:#fff;border:0;border-radius:10px}
.hidden{display:none}
.row label{flex:1 1 100%;font-size:11px;color:#8aa;margin-top:2px}
#statusline{font-size:11px;color:#8aa;margin-top:5px;background:#0a0d12;padding:6px;border-radius:6px}
#aiAudio{display:none}
</style></head><body>

<h2><span>ManeNavSystem</span>
  <span>
    <span class="tag" id="modetag">backend: __VOICE_BACKEND__</span>
    <span class="tag" id="status">disconnected — <b id="label">__LABEL__</b></span>
  </span>
</h2>

<button id="exitCamOnly">Exit camera-only</button>

<div id="startbar">
  <button id="startBtn">Start</button>
  <div style="font-size:12px;color:#8aa;margin-top:6px">Allow camera permission</div>
</div>

<div id="app" class="hidden">
  <div id="localwrap"><video id="local" autoplay muted playsinline></video></div>
  <img id="stream" src="" alt="">
  <audio id="aiAudio" autoplay></audio>

  <div class="banner guide" id="guideBan">guidance: idle</div>
  <div class="banner" id="speakBan">—</div>

  <div class="row">
    <label>Voice</label>
    <button class="act voice" id="startVoiceBtn">Start Voice</button>
    <button class="act voice" id="stopVoiceBtn">Stop Voice</button>
    <button class="act" id="distMinusBtn">Distance −</button>
    <button class="act" id="distPlusBtn">Distance +</button>
    <button class="act" id="cameraOnlyBtn">Camera Only</button>
  </div>

  <div class="bar">
    <input id="cmd" type="text" placeholder='e.g. "find chair", "guide me to door"'>
    <button id="sendBtn">Send</button>
  </div>

  <button id="micBtn" class="big">Hold to speak (local STT)</button>

  <div class="row">
    <label>Quick actions</label>
    <button class="act" data-cmd="scene_description">What do you see</button>
    <button class="act" data-cmd="status">Status</button>
    <button class="act" data-cmd="pause">Pause</button>
    <button class="act" data-cmd="resume">Resume</button>
    <button class="act" data-cmd="louder">Louder</button>
    <button class="act" data-cmd="quieter">Quieter</button>
    <button class="act" data-cmd="zero_heading">Zero heading</button>
    <button class="act" data-cmd="forget" data-target="all">Forget all</button>
    <button class="act warn" data-cmd="stop_navigation">Stop guidance</button>
    <button class="act warn" data-cmd="calibrate">Calibrate</button>
    <button class="act danger" data-cmd="shutdown">Shutdown</button>
  </div>

  <div id="statusline">status: loading…</div>
  <div id="log"></div>
</div>

<script>
const VOICE_BACKEND = "__VOICE_BACKEND__";
const lbl = document.getElementById('label').innerText;
const startBar = document.getElementById('startbar');
const startBtn = document.getElementById('startBtn');
const app = document.getElementById('app');
const status = document.getElementById('status');
const modeTag = document.getElementById('modetag');
const local = document.getElementById('local');
const stream = document.getElementById('stream');
const guideBan = document.getElementById('guideBan');
const speakBan = document.getElementById('speakBan');
const cmdInp = document.getElementById('cmd');
const sendBtn = document.getElementById('sendBtn');
const micBtn = document.getElementById('micBtn');
const logEl = document.getElementById('log');
const statusLine = document.getElementById('statusline');
const startVoiceBtn = document.getElementById('startVoiceBtn');
const stopVoiceBtn = document.getElementById('stopVoiceBtn');
const distMinusBtn = document.getElementById('distMinusBtn');
const distPlusBtn = document.getElementById('distPlusBtn');
const cameraOnlyBtn = document.getElementById('cameraOnlyBtn');
const exitCamOnlyBtn = document.getElementById('exitCamOnly');
const aiAudio = document.getElementById('aiAudio');

if (VOICE_BACKEND === 'openai_realtime') {
  modeTag.classList.add('openai');
}

let pc=null, dc=null, mediaStream=null, recog=null, recordingVoice=false, ttsUnlocked=false;
// OpenAI Realtime state.
let rtPc=null, rtDc=null, rtMicStream=null, rtSse=null, rtActive=false;
let rtDuckTimer=null;

function log(s){ logEl.textContent = (s + "\n" + logEl.textContent).slice(0, 4000); }
function setStatus(s, color){ status.innerHTML = s + ' — <b>' + lbl + '</b>'; status.style.background = color || '#143'; }

function unlockTTS(){
  if(ttsUnlocked) return;
  try{
    const u = new SpeechSynthesisUtterance(' ');
    u.volume = 0.0;
    speechSynthesis.speak(u);
    ttsUnlocked = true;
  }catch(e){}
}

let _audioCtx = null;
let _audioSource = null;
function getAudioCtx(){
  if(!_audioCtx) _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return _audioCtx;
}
async function playPiperAudio(b64){
  try{
    const ctx = getAudioCtx();
    if(ctx.state === 'suspended') await ctx.resume();
    const raw = atob(b64);
    const buf = new Uint8Array(raw.length);
    for(let i=0;i<raw.length;i++) buf[i]=raw.charCodeAt(i);
    const decoded = await ctx.decodeAudioData(buf.buffer);
    if(_audioSource){ try{ _audioSource.stop(); }catch(e){} }
    _audioSource = ctx.createBufferSource();
    _audioSource.buffer = decoded;
    _audioSource.connect(ctx.destination);
    _audioSource.start(0);
  }catch(e){ log('piper audio: '+e); }
}
function speak(text, urgency){
  if(!text) return;
  try{
    const u = new SpeechSynthesisUtterance(text);
    u.rate = urgency === 'critical' ? 1.15 : 1.0;
    u.volume = 1.0;
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  }catch(e){ log('tts: '+e); }
}

function sendDC(obj){
  if(!dc || dc.readyState !== 'open'){ log('dc not open'); return; }
  try{ dc.send(JSON.stringify(obj)); }catch(e){ log('dc send: '+e); }
}

async function start(){
  startBtn.disabled = true;
  unlockTTS();
  try{
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video:{facingMode:{ideal:'environment'}, width:{ideal:1280}, height:{ideal:720}, frameRate:{ideal:30}},
      audio:false
    });
  }catch(e){ log('camera err: '+e); startBtn.disabled = false; return; }
  local.srcObject = mediaStream;

  pc = new RTCPeerConnection({iceServers:[{urls:'stun:stun.l.google.com:19302'}]});
  dc = pc.createDataChannel('control', {ordered:true});
  dc.onopen = () => { setStatus('connected','#143'); log('data channel open'); };
  dc.onclose = () => setStatus('dc closed','#522');
  dc.onmessage = ev => {
    let m; try{ m = JSON.parse(ev.data); }catch(e){ return; }
    if(m.type==='audio'){
      speakBan.textContent = m.text || '';
      speakBan.className = 'banner ' + (m.urgency==='critical'?'critical':(m.urgency==='warn'?'warn':''));
      if(m.wav) playPiperAudio(m.wav);
    }else if(m.type==='speak'){
      speakBan.textContent = m.text || '';
      speakBan.className = 'banner ' + (m.urgency==='critical'?'critical':(m.urgency==='warn'?'warn':''));
      if(m.text) speak(m.text, m.urgency);
    }else if(m.type==='guide'){
      guideBan.textContent = 'guidance: ' + (m.action || 'idle') + (m.text ? ' — '+m.text : '');
    }else if(m.type==='ack'){
      log('ack: '+(m.text||''));
    }
  };

  pc.oniceconnectionstatechange = () => log('ice: '+pc.iceConnectionState);
  pc.onconnectionstatechange = () => log('pc: '+pc.connectionState);

  for(const t of mediaStream.getTracks()){ pc.addTrack(t, mediaStream); }

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await new Promise(r => {
    if(pc.iceGatheringState === 'complete') return r();
    const f = () => { if(pc.iceGatheringState === 'complete'){ pc.removeEventListener('icegatheringstatechange', f); r(); } };
    pc.addEventListener('icegatheringstatechange', f);
    setTimeout(r, 1500);
  });
  let resp;
  try{
    resp = await fetch('/offer?device='+encodeURIComponent(lbl), {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({sdp: pc.localDescription.sdp, type: pc.localDescription.type})
    });
  }catch(e){ log('post err: '+e); startBtn.disabled = false; return; }
  const ans = await resp.json();
  await pc.setRemoteDescription(ans);

  startBar.classList.add('hidden');
  app.classList.remove('hidden');
  setStatus('connecting','#235');
  stream.src = '/stream.mjpg?ts=' + Date.now();
}

startBtn.onclick = start;

document.querySelectorAll('button.act').forEach(b => {
  b.addEventListener('click', () => {
    unlockTTS();
    const c = b.getAttribute('data-cmd');
    const t = b.getAttribute('data-target') || '';
    sendDC({type:'button', action:c, target:t});
    log('button: '+c+(t?(' '+t):''));
  });
});

sendBtn.onclick = () => {
  unlockTTS();
  const txt = cmdInp.value.trim();
  if(!txt) return;
  sendDC({type:'command', text:txt});
  log('cmd: '+txt);
  cmdInp.value = '';
};
cmdInp.addEventListener('keydown', e => { if(e.key==='Enter') sendBtn.click(); });

const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
if(SR){
  recog = new SR();
  recog.continuous = false; recog.interimResults = false; recog.lang = 'en-US';
  recog.onresult = ev => {
    const txt = ev.results[0][0].transcript;
    log('heard: '+txt);
    sendDC({type:'command', text:txt});
  };
  recog.onerror = ev => log('stt err: '+ev.error);
  recog.onend = () => { recordingVoice = false; micBtn.classList.remove('rec'); micBtn.textContent='Hold to speak'; };
}else{
  micBtn.disabled = true;
  micBtn.textContent = 'Voice input unsupported (use buttons / typing)';
  micBtn.style.background = '#444';
}

function startVoice(){
  if(!recog || recordingVoice) return;
  unlockTTS();
  try{ recog.start(); recordingVoice=true; micBtn.classList.add('rec'); micBtn.textContent='Listening… release to send'; }catch(e){ log('stt start: '+e); }
}
function stopVoice(){
  if(!recog) return;
  try{ recog.stop(); }catch(e){}
}
micBtn.addEventListener('mousedown', startVoice);
micBtn.addEventListener('touchstart', e => { e.preventDefault(); startVoice(); });
micBtn.addEventListener('mouseup', stopVoice);
micBtn.addEventListener('mouseleave', stopVoice);
micBtn.addEventListener('touchend', e => { e.preventDefault(); stopVoice(); });
micBtn.addEventListener('touchcancel', stopVoice);

async function refreshStatus(){
  try{
    const r = await fetch('/output_status', {cache:'no-store'});
    const d = await r.json();
    const parts = [];
    for(const k of ['fps','device','det_model','dep_model','voice','phone','n_objects','guidance_action']){
      if(d[k] !== undefined && d[k] !== null && d[k] !== '') parts.push(`<b>${k}</b>: ${d[k]}`);
    }
    statusLine.innerHTML = parts.join('  ·  ') || 'no status';
  }catch(e){}
}
setInterval(refreshStatus, 1000);

// ---------------------------------------------------------------- Distance
async function pushDistance(direction){
  try{
    const r = await fetch('/api/distance', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({direction})
    });
    const d = await r.json();
    if (d && typeof d.scale === 'number') log('distance scale: '+d.scale.toFixed(2));
  }catch(e){ log('distance err: '+e); }
}
distPlusBtn && (distPlusBtn.onclick = () => pushDistance('+'));
distMinusBtn && (distMinusBtn.onclick = () => pushDistance('-'));

// ---------------------------------------------------------------- Camera-Only
function setCameraOnly(on){
  document.body.classList.toggle('cameraonly', !!on);
}
cameraOnlyBtn && (cameraOnlyBtn.onclick = () => setCameraOnly(true));
exitCamOnlyBtn && (exitCamOnlyBtn.onclick = () => setCameraOnly(false));

// ---------------------------------------------------------------- Safety SSE
function startSafetyChannel(){
  if (rtSse) return;
  try{
    rtSse = new EventSource('/api/voice/events');
    rtSse.onmessage = ev => {
      let m; try{ m = JSON.parse(ev.data); }catch(e){ return; }
      if (m.type === 'safety') {
        log('safety: '+m.action+' "'+(m.text||'')+'"');
        // Duck or cancel Realtime audio so the local safety TTS is heard.
        duckRealtime(m.urgency === 'critical' ? 2500 : 1500);
        if (rtDc && rtDc.readyState === 'open') {
          try{ rtDc.send(JSON.stringify({type:'response.cancel'})); }catch(e){}
          // Send fresh context after the safety alert lands.
          setTimeout(pushContextToRealtime, 600);
        }
      }
    };
    rtSse.onerror = () => { try{ rtSse.close(); }catch(e){} rtSse=null; setTimeout(startSafetyChannel, 1500); };
  }catch(e){ log('sse err: '+e); }
}

function duckRealtime(ms){
  if (!aiAudio) return;
  aiAudio.volume = 0.0;
  if (rtDuckTimer) clearTimeout(rtDuckTimer);
  rtDuckTimer = setTimeout(() => { aiAudio.volume = 1.0; }, ms);
}

// ---------------------------------------------------------------- Voice mode
async function pushContextToRealtime(){
  if (!rtDc || rtDc.readyState !== 'open') return;
  try{
    const r = await fetch('/api/voice/context', {cache:'no-store'});
    const ctx = await r.json();
    const sys = 'NAV_CONTEXT (read-only, do not invent movement commands): '
              + JSON.stringify(ctx);
    rtDc.send(JSON.stringify({
      type:'conversation.item.create',
      item:{ type:'message', role:'system', content:[{type:'input_text', text:sys}] }
    }));
  }catch(e){ /* ignore */ }
}

async function startRealtime(){
  if (rtActive) return;
  let session;
  try{
    const sr = await fetch('/api/realtime/session', {method:'POST'});
    session = await sr.json();
  }catch(e){ log('rt session err: '+e); return; }
  if (!session || !session.client_secret) {
    log('rt session missing client_secret: '+JSON.stringify(session||{}).slice(0,160));
    return;
  }
  const ephemeral = session.client_secret.value || session.client_secret;
  const model = session.model || 'gpt-realtime-mini';

  try{
    rtMicStream = await navigator.mediaDevices.getUserMedia({
      audio:{echoCancellation:true, noiseSuppression:true, autoGainControl:true},
      video:false
    });
  }catch(e){ log('mic err: '+e); return; }

  rtPc = new RTCPeerConnection();
  rtPc.ontrack = ev => {
    if (ev.streams && ev.streams[0]) aiAudio.srcObject = ev.streams[0];
  };
  for (const t of rtMicStream.getAudioTracks()) rtPc.addTrack(t, rtMicStream);

  rtDc = rtPc.createDataChannel('oai-events');
  rtDc.onopen = () => { log('rt dc open'); pushContextToRealtime(); };
  rtDc.onmessage = ev => { /* could parse server events here */ };

  const offer = await rtPc.createOffer();
  await rtPc.setLocalDescription(offer);

  const url = 'https://api.openai.com/v1/realtime?model=' + encodeURIComponent(model);
  let resp;
  try{
    resp = await fetch(url, {
      method:'POST',
      headers:{
        'Authorization':'Bearer '+ephemeral,
        'Content-Type':'application/sdp'
      },
      body: rtPc.localDescription.sdp
    });
  }catch(e){ log('rt sdp post err: '+e); return; }
  if (!resp.ok) { log('rt sdp http '+resp.status); return; }
  const answerSdp = await resp.text();
  await rtPc.setRemoteDescription({type:'answer', sdp:answerSdp});

  rtActive = true;
  startVoiceBtn.classList.add('live');
  startVoiceBtn.textContent = 'Voice live';
  log('OpenAI Realtime connected ('+model+')');
  // Refresh nav context every 2s while live.
  rtPushTimer = setInterval(pushContextToRealtime, 2000);
}

let rtPushTimer = null;
function stopRealtime(){
  if (rtPushTimer) { clearInterval(rtPushTimer); rtPushTimer = null; }
  try{ if (rtDc) rtDc.close(); }catch(e){}
  try{ if (rtPc) rtPc.close(); }catch(e){}
  if (rtMicStream) { for (const t of rtMicStream.getTracks()) try{ t.stop(); }catch(e){} }
  rtDc = null; rtPc = null; rtMicStream = null;
  rtActive = false;
  startVoiceBtn.classList.remove('live');
  startVoiceBtn.textContent = 'Start Voice';
  log('OpenAI Realtime stopped');
}

function startLocalVoice(){
  // Local-mode "Start Voice" maps to the existing hold-to-speak STT.
  if (!recog) { log('local STT unsupported in this browser'); return; }
  startVoice();
}

function stopLocalVoice(){
  if (!recog) return;
  try{ recog.stop(); }catch(e){}
}

startVoiceBtn && (startVoiceBtn.onclick = () => {
  unlockTTS();
  if (VOICE_BACKEND === 'openai_realtime') startRealtime();
  else startLocalVoice();
});
stopVoiceBtn && (stopVoiceBtn.onclick = () => {
  if (VOICE_BACKEND === 'openai_realtime') stopRealtime();
  else stopLocalVoice();
});

// Safety SSE always on — it is shared between modes.
startSafetyChannel();

window.addEventListener('beforeunload', () => {
  try{ if(pc) pc.close(); }catch(e){}
  try{ stopRealtime(); }catch(e){}
});
</script>
</body></html>
"""


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _ensure_cert(cert_path: Path, key_path: Path) -> None:
    if cert_path.exists() and key_path.exists():
        return
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ManeNavSystem-PhoneCam")])
    san_list = [x509.DNSName("localhost")]
    try:
        san_list.append(x509.IPAddress(ipaddress.ip_address(_local_ip())))
    except Exception:
        pass
    try:
        san_list.append(x509.IPAddress(ipaddress.ip_address("127.0.0.1")))
    except Exception:
        pass
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow() - timedelta(days=1))
        .not_valid_after(datetime.utcnow() + timedelta(days=365 * 5))
        .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    logger.info(f"Generated self-signed cert at {cert_path.name}")


@dataclass
class _DeviceState:
    label: str
    last_frame: Optional[np.ndarray] = None
    last_ts: float = 0.0
    frame_count: int = 0
    pc: object = None
    data_channel: object = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class PhoneCameraServer:
    def __init__(self, port: int = 8443, cert_dir: str = ".", web_output: Optional[WebOutputServer] = None):
        if not _WEBRTC_AVAILABLE:
            raise RuntimeError(
                f"WebRTC stack unavailable: {_WEBRTC_IMPORT_ERROR}. "
                "Install with: pip install aiortc aiohttp cryptography av"
            )
        self._port = int(port)
        self._cert_dir = Path(cert_dir)
        self._devices: Dict[str, _DeviceState] = {}
        self._devices_lock = threading.RLock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._runner = None
        self._site = None
        self._pcs = set()
        self._started_evt = threading.Event()
        self._command_callback: Optional[Callable[[str, str], None]] = None
        self._button_callback: Optional[Callable[[str, str, str], None]] = None
        self._web_output = web_output if web_output is not None else WebOutputServer()
        # Voice integration hooks set by Pipeline at runtime.
        self._voice_backend: str = "local"
        self._voice_context_provider: Optional[Callable[[], dict]] = None
        self._distance_callback: Optional[Callable[[int], float]] = None
        self._openai_api_key: str = ""
        self._openai_model: str = "gpt-realtime-mini"
        self._openai_voice: str = "alloy"
        self._openai_session_url: str = "https://api.openai.com/v1/realtime/sessions"
        self._openai_instructions: str = ""
        # SSE subscribers for safety / voice events.
        self._sse_subs: list = []
        self._sse_lock = threading.Lock()

    @property
    def web_output(self) -> WebOutputServer:
        return self._web_output

    def url_for(self, device_label: str) -> str:
        return f"https://{_local_ip()}:{self._port}/?device={device_label}"

    def view_url(self) -> str:
        return f"https://{_local_ip()}:{self._port}/view"

    def get_devices(self):
        with self._devices_lock:
            return list(self._devices.keys())

    def get_frame(self, label: str) -> Optional[Tuple[np.ndarray, float]]:
        label = (label or "").strip().lower()
        with self._devices_lock:
            d = self._devices.get(label)
        if d is None:
            return None
        with d.lock:
            if d.last_frame is None:
                return None
            return d.last_frame.copy(), d.last_ts

    def has_frames(self, label: str) -> bool:
        return self.get_frame(label) is not None

    def set_command_callback(self, cb: Callable[[str, str], None]):
        self._command_callback = cb

    def set_button_callback(self, cb: Callable[[str, str, str], None]):
        self._button_callback = cb

    def push_render_frame(self, label: str, bgr: np.ndarray) -> None:
        self._web_output.push_frame(bgr)

    def update_status(self, **kw) -> None:
        self._web_output.update_status(**kw)

    def push_message(self, label: str, payload: dict) -> None:
        label = (label or "").strip().lower()
        with self._devices_lock:
            d = self._devices.get(label)
        if d is None or d.data_channel is None or self._loop is None:
            return
        try:
            data = json.dumps(payload)
        except Exception:
            return
        ch = d.data_channel
        def _send():
            try:
                if getattr(ch, "readyState", "open") == "open":
                    ch.send(data)
            except Exception as e:
                logger.debug(f"dc send[{label}] failed: {e}")
        try:
            self._loop.call_soon_threadsafe(_send)
        except Exception:
            pass

    def push_speech(self, label: str, text: str, urgency: str = "info") -> None:
        if not text:
            return
        self.push_message(label, {"type": "speak", "text": text, "urgency": urgency})

    def push_audio(self, label: str, wav_b64: str, text: str = "", urgency: str = "info") -> None:
        """Send Piper-synthesized WAV (base64) to phone for playback."""
        if not wav_b64:
            return
        self.push_message(label, {"type": "audio", "wav": wav_b64, "text": text, "urgency": urgency})

    def broadcast_audio(self, wav_b64: str, text: str = "", urgency: str = "info") -> None:
        for lbl in self.get_devices():
            self.push_audio(lbl, wav_b64, text, urgency)

    def push_guidance(self, label: str, action: str, text: str = "") -> None:
        self.push_message(label, {"type": "guide", "action": action or "idle", "text": text or ""})

    def broadcast_speech(self, text: str, urgency: str = "info") -> None:
        for lbl in self.get_devices():
            self.push_speech(lbl, text, urgency)

    def broadcast_guidance(self, action: str, text: str = "") -> None:
        for lbl in self.get_devices():
            self.push_guidance(lbl, action, text)

    # ------------------------------------------------------------------
    # Voice integration: configuration + event broadcast
    # ------------------------------------------------------------------
    def configure_voice(
        self,
        backend: str,
        context_provider: Optional[Callable[[], dict]] = None,
        distance_callback: Optional[Callable[[int], float]] = None,
        openai_api_key: str = "",
        openai_model: str = "gpt-realtime-mini",
        openai_voice: str = "alloy",
        openai_session_url: str = "https://api.openai.com/v1/realtime/sessions",
        openai_instructions: str = "",
    ) -> None:
        self._voice_backend = (backend or "local").strip().lower() or "local"
        if context_provider is not None:
            self._voice_context_provider = context_provider
        if distance_callback is not None:
            self._distance_callback = distance_callback
        self._openai_api_key = openai_api_key or ""
        if openai_model:
            self._openai_model = openai_model
        if openai_voice:
            self._openai_voice = openai_voice
        if openai_session_url:
            self._openai_session_url = openai_session_url
        if openai_instructions:
            self._openai_instructions = openai_instructions

    def push_safety_event(self, action: str, text: str = "",
                          urgency: str = "info") -> None:
        """Fan out a safety command to every connected SSE client.

        Safe to call from any thread. The actual queue.put is hopped onto
        the asyncio loop, since asyncio.Queue is not thread-safe.
        """
        payload = {
            "type": "safety",
            "action": (action or "").upper(),
            "text": text or "",
            "urgency": urgency or "info",
            "ts": time.time(),
        }
        loop = self._loop
        if loop is None:
            return
        try:
            data = json.dumps(payload)
        except Exception:
            return
        with self._sse_lock:
            subs = list(self._sse_subs)
        for q in subs:
            try:
                loop.call_soon_threadsafe(q.put_nowait, data)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # HTTP handlers for the voice layer
    # ------------------------------------------------------------------
    async def _voice_context(self, request):
        if self._voice_context_provider is None:
            return web.json_response({})
        try:
            ctx = self._voice_context_provider()
        except Exception as e:
            logger.debug(f"voice_context_provider error: {e}")
            ctx = {}
        return web.json_response(ctx or {})

    async def _voice_events(self, request):
        # Server-Sent Events stream. Each subscriber gets its own queue.
        resp = web.StreamResponse(status=200, reason="OK", headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })
        await resp.prepare(request)
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        with self._sse_lock:
            self._sse_subs.append(q)
        try:
            await resp.write(b": connected\n\n")
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=15.0)
                    await resp.write(("data: " + data + "\n\n").encode("utf-8"))
                except asyncio.TimeoutError:
                    await resp.write(b": ping\n\n")
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            with self._sse_lock:
                if q in self._sse_subs:
                    self._sse_subs.remove(q)
        return resp

    async def _distance(self, request):
        try:
            params = await request.json()
        except Exception:
            params = {}
        direction = (params.get("direction") or "").strip()
        step = +1 if direction in ("+", "plus", "up") else (-1 if direction in ("-", "minus", "down") else 0)
        if step == 0 or self._distance_callback is None:
            return web.json_response({"ok": False, "scale": None})
        try:
            scale = self._distance_callback(step)
        except Exception as e:
            logger.warning(f"distance callback failed: {e}")
            return web.json_response({"ok": False, "error": str(e)})
        return web.json_response({"ok": True, "scale": float(scale)})

    async def _realtime_session(self, request):
        if self._voice_backend != "openai_realtime":
            return web.json_response(
                {"error": "voice_backend is not openai_realtime"},
                status=400,
            )
        api_key = self._openai_api_key
        if not api_key:
            return web.json_response(
                {"error": "OPENAI_API_KEY is not set on the server"},
                status=500,
            )
        body = {
            "model": self._openai_model,
            "voice": self._openai_voice,
            "modalities": ["audio", "text"],
            "instructions": self._openai_instructions,
            "turn_detection": {"type": "server_vad"},
            "input_audio_transcription": {"model": "whisper-1"},
        }
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    self._openai_session_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "OpenAI-Beta": "realtime=v1",
                    },
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as r:
                    text = await r.text()
                    if r.status >= 400:
                        logger.warning(f"OpenAI realtime session HTTP {r.status}: {text[:200]}")
                        return web.json_response(
                            {"error": "openai_session_failed", "status": r.status, "detail": text[:300]},
                            status=502,
                        )
                    try:
                        data = json.loads(text)
                    except Exception:
                        return web.json_response({"error": "openai_invalid_json", "detail": text[:300]}, status=502)
        except Exception as e:
            logger.warning(f"OpenAI realtime session error: {e}")
            return web.json_response({"error": "openai_unreachable", "detail": str(e)}, status=502)
        # Strip the long-lived API key from the response just in case.
        data.pop("api_key", None)
        return web.json_response(data)

    def _get_or_create_device(self, label: str) -> _DeviceState:
        label = (label or "phone").strip().lower() or "phone"
        with self._devices_lock:
            if label not in self._devices:
                self._devices[label] = _DeviceState(label=label)
            return self._devices[label]

    async def _index(self, request):
        label = (request.query.get("device") or "phone").strip().lower() or "phone"
        html = (
            _HTML_PAGE
            .replace("__LABEL__", label)
            .replace("__VOICE_BACKEND__", self._voice_backend)
        )
        return web.Response(content_type="text/html", text=html)

    async def _status(self, request):
        with self._devices_lock:
            data = {
                lbl: {
                    "frames": d.frame_count,
                    "age_s": (time.time() - d.last_ts) if d.last_ts else None,
                    "has_dc": d.data_channel is not None,
                }
                for lbl, d in self._devices.items()
            }
        return web.json_response(data)

    async def _offer(self, request):
        label = (request.query.get("device") or "phone").strip().lower() or "phone"
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()
        self._pcs.add(pc)
        device = self._get_or_create_device(label)
        if device.pc is not None:
            try:
                await device.pc.close()
            except Exception:
                pass
            self._pcs.discard(device.pc)
        device.pc = pc
        device.data_channel = None

        @pc.on("connectionstatechange")
        async def on_state():
            logger.info(f"phone[{label}] connection={pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                self._pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            logger.info(f"phone[{label}] track received: {track.kind}")
            if track.kind == "video":
                asyncio.ensure_future(self._consume(track, device))

        @pc.on("datachannel")
        def on_dc(channel):
            logger.info(f"phone[{label}] datachannel '{channel.label}' open")
            device.data_channel = channel

            @channel.on("message")
            def on_msg(msg):
                self._handle_dc_message(label, msg)

            @channel.on("close")
            def on_close():
                if device.data_channel is channel:
                    device.data_channel = None

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        })

    def _handle_dc_message(self, label: str, msg: object) -> None:
        try:
            if isinstance(msg, bytes):
                msg = msg.decode("utf-8", errors="ignore")
            data = json.loads(msg)
        except Exception:
            return
        mtype = data.get("type")
        if mtype == "command":
            text = (data.get("text") or "").strip()
            if text and self._command_callback:
                try:
                    self._command_callback(label, text)
                except Exception as e:
                    logger.warning(f"command callback failed: {e}")
        elif mtype == "button":
            action = (data.get("action") or "").strip()
            target = (data.get("target") or "").strip()
            if action and self._button_callback:
                try:
                    self._button_callback(label, action, target)
                except Exception as e:
                    logger.warning(f"button callback failed: {e}")
        elif mtype == "ping":
            self.push_message(label, {"type": "ack", "text": "pong"})

    async def _consume(self, track, device: _DeviceState):
        while True:
            try:
                frame = await track.recv()
            except Exception as e:
                logger.info(f"phone[{device.label}] track ended: {e}")
                return
            try:
                img = frame.to_ndarray(format="bgr24")
            except Exception as e:
                logger.debug(f"phone[{device.label}] decode error: {e}")
                continue
            with device.lock:
                device.last_frame = img
                device.last_ts = time.time()
                device.frame_count += 1

    async def _shutdown(self, app):
        coros = [pc.close() for pc in list(self._pcs)]
        await asyncio.gather(*coros, return_exceptions=True)
        self._pcs.clear()

    def start(self) -> None:
        if self._thread is not None:
            return
        cert_path = self._cert_dir / "phone_cam_cert.pem"
        key_path = self._cert_dir / "phone_cam_key.pem"
        _ensure_cert(cert_path, key_path)

        def runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            app = web.Application()
            app.router.add_get("/", self._index)
            app.router.add_get("/status", self._status)
            app.router.add_post("/offer", self._offer)
            app.router.add_get("/api/voice/context", self._voice_context)
            app.router.add_get("/api/voice/events", self._voice_events)
            app.router.add_post("/api/distance", self._distance)
            app.router.add_post("/api/realtime/session", self._realtime_session)
            self._web_output.register_routes(app)
            app.on_shutdown.append(self._shutdown)
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(str(cert_path), str(key_path))
            self._runner = web.AppRunner(app, access_log=None)
            self._loop.run_until_complete(self._runner.setup())
            self._site = web.TCPSite(self._runner, "0.0.0.0", self._port, ssl_context=ssl_ctx)
            try:
                self._loop.run_until_complete(self._site.start())
            except Exception as e:
                logger.error(f"PhoneCameraServer bind failed on :{self._port}: {e}")
                self._started_evt.set()
                return
            logger.info(f"PhoneCameraServer listening on https://0.0.0.0:{self._port}")
            self._started_evt.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=runner, daemon=True, name="PhoneCameraServer")
        self._thread.start()
        self._started_evt.wait(timeout=3.0)

    def stop(self) -> None:
        if self._loop is None:
            return
        async def _stop():
            try:
                if self._site is not None:
                    await self._site.stop()
            except Exception:
                pass
            try:
                if self._runner is not None:
                    await self._runner.cleanup()
            except Exception:
                pass
            self._loop.stop()
        try:
            asyncio.run_coroutine_threadsafe(_stop(), self._loop).result(timeout=3.0)
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._loop = None
        self._thread = None


def _normalize_rotation(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("auto", "a", ""):
            return -1
        try:
            value = int(v)
        except Exception:
            return 0
    try:
        v = int(value) % 360
    except Exception:
        return 0
    if v not in (0, 90, 180, 270):
        return 0
    return v


def _rotate_bgr(frame: np.ndarray, deg: int) -> np.ndarray:
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _letterbox(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == target_w and h == target_h:
        return frame
    scale = min(target_w / float(w), target_h / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    if (nw, nh) != (w, h):
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        resized = frame
    canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


class PhoneCameraSource:
    def __init__(
        self,
        server: PhoneCameraServer,
        label: str,
        target_size: Optional[Tuple[int, int]] = None,
        wait_first_frame_s: float = 60.0,
        rotate: Any = "auto",
        preserve_aspect_ratio: bool = True,
    ):
        self._server = server
        self._label = (label or "phone").strip().lower() or "phone"
        self._target_size = target_size
        self._last_ts = 0.0
        self._wait_first = float(wait_first_frame_s)
        self._opened = False
        self._rotate_cfg = _normalize_rotation(rotate)
        self._preserve_aspect = bool(preserve_aspect_ratio)
        self._auto_rot_logged = False

    @property
    def label(self) -> str:
        return self._label

    def isOpened(self) -> bool:
        if self._opened:
            return True
        deadline = time.time() + self._wait_first
        while time.time() < deadline:
            if self._server.has_frames(self._label):
                self._opened = True
                return True
            time.sleep(0.1)
        return False

    def _apply_orientation(self, frame: np.ndarray) -> np.ndarray:
        rot = self._rotate_cfg
        if rot == -1:
            h, w = frame.shape[:2]
            if h > w:
                if not self._auto_rot_logged:
                    logger.info(f"PhoneCam[{self._label}] auto-rotate: portrait {w}x{h} -> landscape (90° CW)")
                    self._auto_rot_logged = True
                frame = _rotate_bgr(frame, 90)
        elif rot in (90, 180, 270):
            frame = _rotate_bgr(frame, rot)
        return frame

    def _apply_target_size(self, frame: np.ndarray) -> np.ndarray:
        if self._target_size is None:
            return frame
        tw, th = int(self._target_size[0]), int(self._target_size[1])
        if frame.shape[1] == tw and frame.shape[0] == th:
            return frame
        if self._preserve_aspect:
            return _letterbox(frame, tw, th)
        return cv2.resize(frame, (tw, th))

    def read(self):
        deadline = time.time() + 5.0
        while time.time() < deadline:
            got = self._server.get_frame(self._label)
            if got is not None:
                frame, ts = got
                if ts != self._last_ts:
                    self._last_ts = ts
                    frame = self._apply_orientation(frame)
                    frame = self._apply_target_size(frame)
                    return True, frame
            time.sleep(0.005)
        return False, None

    def release(self):
        self._opened = False

    def get(self, prop):
        got = self._server.get_frame(self._label)
        if got is None:
            return 0.0
        h, w = got[0].shape[:2]
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(h)
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def set(self, prop, value):
        return False
