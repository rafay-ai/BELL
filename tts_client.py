import os
import time
import wave
import requests

TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:7860/v1/audio/speech")
SAMPLE_RATE = 24000
REF_AUDIO = os.path.join(os.path.dirname(__file__), "voice_ref.wav")
REF_TEXT = open(os.path.join(os.path.dirname(__file__), "voice_ref.txt")).read().strip()

# Delivery only. Never describe the speaker, because the reference fixes the identity.
EMOTIONS = {
    "stern":     "Speak in a serious, stern, firm tone, controlled and low, not shouting.",
    "neutral":   "Speak in a calm, natural tone.",
    "happy":     "Speak in a cheerful, bright tone, slightly faster.",
    "proud":     "Speak in a warm, proud, encouraging tone.",
    "concerned": "Speak in a gentle, caring tone, slightly worried and a little slower.",
    "sleepy":    "Speak in a tired, soft tone, slowly.",
}

def speak(text: str, emotion: str = "neutral", out_path: str = "") -> str:
    out_path = out_path or f"replies/{int(time.time())}_{emotion}.wav"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(REF_AUDIO, "rb") as ref:
        resp = requests.post(
            TTS_URL,
            data={
                "text": text,
                "instruction": EMOTIONS.get(emotion, EMOTIONS["neutral"]),
                "cfg_scale": 4,
                "ref_text": REF_TEXT,
            },
            files={"ref_audio": ("voice_ref.wav", ref, "audio/wav")},
            timeout=120,
        )
    resp.raise_for_status()
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(resp.content)
    return out_path


def synthesize(text: str, emotion: str = "neutral") -> bytes:
    """Same as speak(), but returns WAV bytes instead of saving a file."""
    import io
    with open(REF_AUDIO, "rb") as ref:
        resp = requests.post(
            TTS_URL,
            data={
                "text": text,
                "instruction": EMOTIONS.get(emotion, EMOTIONS["neutral"]),
                "cfg_scale": 4,
                "ref_text": REF_TEXT,
            },
            files={"ref_audio": ("voice_ref.wav", ref, "audio/wav")},
            timeout=120,
        )
    resp.raise_for_status()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(resp.content)
    return buf.getvalue()


def synthesize_stream(text: str, emotion: str = "neutral"):
    """Yield raw 16-bit mono PCM chunks (24 kHz) as Breeze produces them."""
    with open(REF_AUDIO, "rb") as ref:
        resp = requests.post(
            TTS_URL,
            data={
                "text": text,
                "instruction": EMOTIONS.get(emotion, EMOTIONS["neutral"]),
                "cfg_scale": 4,
                "ref_text": REF_TEXT,
            },
            files={"ref_audio": ("voice_ref.wav", ref, "audio/wav")},
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
        leftover = b""
        for chunk in resp.iter_content(chunk_size=None):
            if not chunk:
                continue
            data = leftover + chunk
            usable = len(data) - (len(data) % 2)  # keep whole 16-bit samples
            leftover = data[usable:]
            if usable:
                yield data[:usable]
