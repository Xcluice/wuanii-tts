import io, json, os, re, tempfile, urllib.request, wave
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices.npz")
MAX_CHARS = 700

VOICES = {
    "am_puck": "Puck (playful)", "am_fenrir": "Fenrir (deep)", "am_adam": "Adam (calm)",
    "am_michael": "Michael (clear)", "am_echo": "Echo", "am_liam": "Liam", "am_onyx": "Onyx (bass)",
    "am_eric": "Eric", "bm_george": "George (UK)", "bm_fable": "Fable (UK story)",
    "bm_lewis": "Lewis (UK)", "bm_daniel": "Daniel (UK)", "af_bella": "Bella (female)",
    "af_nicole": "Nicole (soft)", "af_sky": "Sky (female)", "af_nova": "Nova (female)",
    "af_jessica": "Jessica (female)", "bf_emma": "Emma (UK female)", "bf_isabella": "Isabella (UK female)",
}

STYLES = {
    # base speed, '!' boost, '?' boost, '...' factor, pauses (. ! ? ... ,)
    "normal":   dict(base=1.00, bang=1.00, ask=1.00, dots=0.95, p_dot=0.28, p_bang=0.25, p_ask=0.30, p_dots=0.40),
    "hype":     dict(base=1.10, bang=1.07, ask=1.04, dots=1.00, p_dot=0.16, p_bang=0.14, p_ask=0.20, p_dots=0.22),
    "dramatic": dict(base=0.90, bang=1.02, ask=0.97, dots=0.82, p_dot=0.42, p_bang=0.34, p_ask=0.45, p_dots=0.60),
}

_kokoro = None


def _ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 300_000_000:
        return
    tmp = MODEL_PATH + ".part"
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "wuanii-tts"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, MODEL_PATH)


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        _ensure_model()
        _kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
    return _kokoro


def _split(text):
    text = re.sub(r"\s+", " ", text.replace("…", "...")).strip()
    parts = re.findall(r".+?(?:\.\.\.|[.!?]+|$)", text)
    return [p.strip() for p in parts if re.search(r"\w", p)]


def _trim(a, sr, thr=0.008, keep=0.03):
    idx = np.where(np.abs(a) > thr)[0]
    if len(idx) == 0:
        return a
    s = max(0, idx[0] - int(sr * keep))
    e = min(len(a), idx[-1] + int(sr * keep))
    return a[s:e]


def synthesize(text, voice="am_puck", style="hype", speed=1.0):
    text = (text or "").strip()
    if not text:
        raise ValueError("Please type a script first.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"Script too long ({len(text)} chars). Keep it under {MAX_CHARS}.")
    if voice not in VOICES:
        voice = "am_puck"
    st = STYLES.get(style, STYLES["hype"])
    speed = min(1.5, max(0.6, float(speed)))
    k = _get_kokoro()
    out, sr = [], 24000
    for sent in _split(text):
        if sent.endswith("..."):
            sp, pause = st["dots"], st["p_dots"]
        elif sent.endswith("!"):
            sp, pause = st["bang"], st["p_bang"]
        elif sent.endswith("?"):
            sp, pause = st["ask"], st["p_ask"]
        else:
            sp, pause = 1.0, st["p_dot"]
        audio = None
        for attempt in range(3):  # retry with a tiny speed change if the model glitches
            s, sr = k.create(sent, voice=voice, speed=min(2.0, max(0.5, st["base"] * sp * speed + attempt * 0.013)),
                             lang="en-us", trim=False)
            s = np.asarray(s, dtype=np.float32)
            if len(s) > 0 and not np.isnan(s).any():
                audio = s
                break
        if audio is None:
            continue
        out.append(_trim(audio, sr))
        out.append(np.zeros(int(sr * pause), dtype=np.float32))
    if not out:
        raise RuntimeError("Voice engine returned no audio. Try again.")
    a = np.concatenate(out[:-1]) if len(out) > 1 else out[0]
    # gentle compression + loudness so it punches through phone speakers
    a = np.tanh(a * 1.6) / np.tanh(1.6)
    peak = float(np.max(np.abs(a))) or 1.0
    a = a / peak * 0.92
    pcm = (a * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        for k_, v_ in (extra or {}).items():
            self.send_header(k_, v_)
        self.end_headers()
        self.wfile.write(body)

    def _run(self, text, voice, style, speed):
        try:
            wav = synthesize(text, voice, style, speed)
            self._send(200, wav, "audio/wav", {"Content-Disposition": 'inline; filename="wuanii-voice.wav"'})
        except ValueError as e:
            self._send(400, json.dumps({"error": str(e)}).encode())
        except Exception as e:  # noqa
            self._send(500, json.dumps({"error": "Voice engine error: " + str(e)[:200]}).encode())

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain", {"Access-Control-Allow-Headers": "Content-Type", "Access-Control-Allow-Methods": "GET,POST,OPTIONS"})

    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        g = lambda n, d="": (q.get(n) or [d])[0]
        if g("info"):
            return self._send(200, json.dumps({"voices": VOICES, "styles": list(STYLES)}).encode())
        self._run(g("text"), g("voice", "am_puck"), g("style", "hype"), g("speed", "1"))

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            d = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, b'{"error":"Bad request"}')
        self._run(d.get("text"), d.get("voice", "am_puck"), d.get("style", "hype"), d.get("speed", 1))
