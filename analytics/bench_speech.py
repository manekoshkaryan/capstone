"""
Latency benchmark: local STT/TTS backends, optionally OpenAI Realtime.

Usage:
  # TTS — measure say() duration per backend over fixed phrases.
  python3 bench_speech.py tts [--repeats 3] [--phrases-file phrases.txt]

  # STT — feed a WAV file into Google + Sphinx, optional Whisper.
  python3 bench_speech.py stt --audio sample.wav [--repeats 3]

  # OpenAI Realtime first-audio latency (requires OPENAI_API_KEY).
  python3 bench_speech.py openai --audio sample.wav

Reports p50 / p95 / max in milliseconds. Writes JSONL to bench_<mode>.jsonl.
"""
import argparse
import json
import os
import statistics
import sys
import time
from typing import Callable, List, Optional


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _summarize(name: str, xs: List[float]) -> dict:
    if not xs:
        return {"backend": name, "n": 0}
    return {
        "backend": name,
        "n": len(xs),
        "p50_ms": round(_percentile(xs, 50), 2),
        "p95_ms": round(_percentile(xs, 95), 2),
        "max_ms": round(max(xs), 2),
        "mean_ms": round(statistics.fmean(xs), 2),
    }


def _print_table(rows: List[dict]) -> None:
    cols = ["backend", "n", "p50_ms", "p95_ms", "max_ms", "mean_ms"]
    print(" | ".join(c.ljust(12) for c in cols))
    print("-" * 80)
    for r in rows:
        print(" | ".join(str(r.get(c, "-")).ljust(12) for c in cols))


def _write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {path}")


def bench_tts(repeats: int, phrases: List[str]) -> None:
    from speech.speech_engine import _Pyttsx3Backend, _SAPIBackend, _MacSayBackend

    candidates = [
        ("pyttsx3", lambda: _Pyttsx3Backend(185, 0.95)),
        ("sapi-powershell", lambda: _SAPIBackend(185, 0.95)),
        ("mac-say", lambda: _MacSayBackend(185, 0.95)),
    ]
    rows = []
    raw = []
    for name, ctor in candidates:
        try:
            backend = ctor()
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue
        latencies: List[float] = []
        try:
            for _ in range(repeats):
                for phrase in phrases:
                    t0 = time.perf_counter()
                    try:
                        backend.say(phrase, 185)
                    except Exception as e:
                        print(f"[err] {name}: {e}")
                        continue
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    latencies.append(dt_ms)
                    raw.append({"backend": name, "phrase": phrase, "dur_ms": round(dt_ms, 2)})
        finally:
            try:
                backend.stop()
            except Exception:
                pass
        rows.append(_summarize(name, latencies))
    _print_table(rows)
    _write_jsonl("bench_tts.jsonl", raw)


def _stt_google(recognizer, audio) -> str:
    import speech_recognition as sr
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""


def _stt_sphinx(recognizer, audio) -> str:
    return recognizer.recognize_sphinx(audio)


def _stt_whisper_local(audio_path: str) -> str:
    import whisper
    model = whisper.load_model("base")
    return model.transcribe(audio_path)["text"]


def bench_stt(audio_path: str, repeats: int) -> None:
    if not os.path.isfile(audio_path):
        print(f"audio not found: {audio_path}")
        sys.exit(1)
    import speech_recognition as sr
    rec = sr.Recognizer()
    with sr.AudioFile(audio_path) as src:
        audio = rec.record(src)

    backends: List[tuple[str, Callable[[], str]]] = [
        ("google", lambda: _stt_google(rec, audio)),
        ("sphinx", lambda: _stt_sphinx(rec, audio)),
    ]
    try:
        import whisper  # noqa: F401
        backends.append(("whisper-base", lambda: _stt_whisper_local(audio_path)))
    except Exception:
        print("[info] whisper not installed — skipping")

    rows = []
    raw = []
    for name, fn in backends:
        latencies: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            try:
                text = fn()
            except Exception as e:
                print(f"[err] {name}: {e}")
                text = None
                continue
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)
            raw.append({"backend": name, "dur_ms": round(dt_ms, 2), "text": text})
        rows.append(_summarize(name, latencies))
    _print_table(rows)
    _write_jsonl("bench_stt.jsonl", raw)


def bench_openai_realtime(audio_path: str) -> None:
    """Stream audio_path into OpenAI Realtime and measure first-audio-out latency.
    Requires OPENAI_API_KEY. Uses websockets.

    Returns when the first audio delta arrives.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set")
        sys.exit(1)
    if not os.path.isfile(audio_path):
        print(f"audio not found: {audio_path}")
        sys.exit(1)
    try:
        import websocket  # websocket-client
        import wave
        import base64
    except ImportError as e:
        print(f"missing dep: {e}. pip install websocket-client")
        sys.exit(1)

    url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
    headers = [f"Authorization: Bearer {api_key}", "OpenAI-Beta: realtime=v1"]

    with wave.open(audio_path, "rb") as wf:
        if wf.getframerate() != 24000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            print("warning: audio expected 24kHz mono PCM16. Results may be off.")
        pcm = wf.readframes(wf.getnframes())
    audio_b64 = base64.b64encode(pcm).decode("ascii")

    t_first = {"v": None}

    def on_message(ws, msg):
        if t_first["v"] is None:
            try:
                data = json.loads(msg)
                if data.get("type") == "response.audio.delta":
                    t_first["v"] = time.perf_counter()
                    ws.close()
            except Exception:
                pass

    def on_open(ws):
        ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
        ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        ws.send(json.dumps({"type": "response.create"}))

    t0 = time.perf_counter()
    ws = websocket.WebSocketApp(url, header=headers, on_open=on_open, on_message=on_message)
    ws.run_forever()
    if t_first["v"] is None:
        print("no audio response received")
        return
    first_audio_ms = (t_first["v"] - t0) * 1000.0
    row = {"backend": "openai-realtime", "first_audio_ms": round(first_audio_ms, 2)}
    print(json.dumps(row, indent=2))
    _write_jsonl("bench_openai.jsonl", [row])


_DEFAULT_PHRASES = [
    "Stop. Path blocked.",
    "Continue straight.",
    "Move slightly left.",
    "Door, two meters ahead.",
    "Chair to the right at one and a half meters.",
]


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    p_tts = sub.add_parser("tts")
    p_tts.add_argument("--repeats", type=int, default=3)
    p_tts.add_argument("--phrases-file", default=None)

    p_stt = sub.add_parser("stt")
    p_stt.add_argument("--audio", required=True)
    p_stt.add_argument("--repeats", type=int, default=3)

    p_oai = sub.add_parser("openai")
    p_oai.add_argument("--audio", required=True)

    args = p.parse_args()

    if args.mode == "tts":
        if args.phrases_file and os.path.isfile(args.phrases_file):
            with open(args.phrases_file, encoding="utf-8") as f:
                phrases = [ln.strip() for ln in f if ln.strip()]
        else:
            phrases = _DEFAULT_PHRASES
        bench_tts(args.repeats, phrases)
    elif args.mode == "stt":
        bench_stt(args.audio, args.repeats)
    elif args.mode == "openai":
        bench_openai_realtime(args.audio)


if __name__ == "__main__":
    main()
