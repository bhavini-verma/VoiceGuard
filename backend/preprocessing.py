"""
VoiceGuard — Preprocessing Pipeline
─────────────────────────────────────────────────────────────────────────
Implements the preprocessing steps specified in the VoiceGuard project plan:

  1. Load audio (any supported format) via librosa
  2. Convert to mono
  3. Resample to 16kHz
  4. Simulate phone codec degradation (16kHz -> 8kHz -> 16kHz)
     so the model sees the same distribution at inference that it saw
     during training (bank calls arrive codec-compressed).
  5. Voice Activity Detection (VAD) — trim leading/trailing silence and
     flag/skip near-silent chunks.
  6. Chunk into 3-second windows with 50% overlap (1.5s hop).

This module is the single source of truth for "how raw audio becomes
model-ready chunks" — both extract_bio.py and extract_deep.py should be
called AFTER this module, per-chunk, not on the raw upload directly.
"""

import io
import os
import numpy as np
import librosa
import soundfile as sf

TARGET_SR = 16000
CHUNK_SECONDS = 3.0
HOP_SECONDS = 1.5
MIN_CHUNK_SECONDS = 1.0          # extract_bio_features() will reject anything shorter
SILENCE_RMS_THRESHOLD = 1e-4     # matches "completely silent" guard in extract_bio.py
VAD_TOP_DB = 30                  # librosa.effects.trim default-ish threshold


class AudioMetadata:
    def __init__(self, filename, original_sr, channels, duration_sec, file_size_bytes, file_format):
        self.filename = filename
        self.original_sr = original_sr
        self.channels = channels
        self.duration_sec = duration_sec
        self.file_size_bytes = file_size_bytes
        self.file_format = file_format

    def to_dict(self):
        return {
            "filename": self.filename,
            "format": self.file_format,
            "sample_rate": f"{self.original_sr}Hz",
            "channels": "1 (Mono)" if self.channels == 1 else f"{self.channels} (Stereo)" if self.channels == 2 else f"{self.channels} Ch",
            "duration": round(self.duration_sec, 2),
            "file_size_bytes": self.file_size_bytes,
        }


def simulate_phone_codec(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Simulate GSM/AMR phone codec degradation by downsampling to 8kHz and back.

    This MUST mirror the transform applied in extract_bio.py / extract_deep.py
    during training (simulate_phone_codec). Skipping this at inference would
    feed the model a cleaner distribution than it was trained on, since real
    bank calls arrive 8kHz-codec-compressed.
    """
    y_8k = librosa.resample(y, orig_sr=sr, target_sr=8000)
    y_degraded = librosa.resample(y_8k, orig_sr=8000, target_sr=sr)
    return y_degraded


def load_and_normalize(file_bytes: bytes, filename: str) -> tuple[np.ndarray, AudioMetadata]:
    """Load arbitrary audio bytes (wav/mp3/m4a/ogg/flac/opus), return
    16kHz mono float32 PCM plus metadata captured from the ORIGINAL file
    (before resampling) so the dashboard shows what the user actually uploaded.
    """
    file_size = len(file_bytes)
    ext = (filename.rsplit(".", 1)[-1].upper() if "." in filename else "AUDIO")

    buf = io.BytesIO(file_bytes)
    try:
        # soundfile gives us the true original sample rate / channel count
        info = sf.info(buf)
        original_sr = info.samplerate
        channels = info.channels
        original_duration = info.duration
    except Exception:
        # Fall back to librosa's prober for formats soundfile can't read
        # directly (e.g. some mp3/m4a encodings depend on audioread/ffmpeg)
        buf.seek(0)
        y_probe, sr_probe = librosa.load(buf, sr=None, mono=False)
        original_sr = sr_probe
        channels = 1 if y_probe.ndim == 1 else y_probe.shape[0]
        original_duration = librosa.get_duration(y=y_probe, sr=sr_probe)

    buf.seek(0)
    y, sr = librosa.load(buf, sr=TARGET_SR, mono=True)

    metadata = AudioMetadata(
        filename=filename,
        original_sr=original_sr,
        channels=channels,
        duration_sec=original_duration,
        file_size_bytes=file_size,
        file_format=ext,
    )
    return y, metadata


def trim_silence(y: np.ndarray, top_db: int = VAD_TOP_DB) -> np.ndarray:
    """Voice Activity Detection (lightweight): trim leading/trailing silence.
    Uses librosa's energy-based trim, matching the plan's recommended
    'librosa.effects.split() or energy threshold' approach.
    """
    if len(y) == 0:
        return y
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed if len(y_trimmed) > 0 else y


def is_silent(y: np.ndarray) -> bool:
    return len(y) == 0 or np.max(np.abs(y)) < SILENCE_RMS_THRESHOLD


def chunk_audio(y: np.ndarray, sr: int = TARGET_SR,
                 chunk_seconds: float = CHUNK_SECONDS,
                 hop_seconds: float = HOP_SECONDS) -> list[dict]:
    """Slice audio into 3-second windows with 50% (1.5s) overlap.

    Returns a list of dicts: {start_sec, end_sec, audio (np.ndarray)}.
    Chunks shorter than MIN_CHUNK_SECONDS are dropped (extract_bio_features
    rejects sub-1s clips), EXCEPT we always keep at least one chunk even if
    the whole clip is shorter than 3s, so very short uploads still get analyzed.
    """
    chunk_len = int(chunk_seconds * sr)
    hop_len = int(hop_seconds * sr)
    total_len = len(y)

    chunks = []

    if total_len <= 0:
        return chunks

    if total_len <= chunk_len:
        # Whole clip is shorter than one chunk — analyze it as a single chunk
        # (pad lightly only if needed downstream; extraction functions handle
        # variable-length audio natively via librosa, so no padding required).
        duration = total_len / sr
        if duration >= MIN_CHUNK_SECONDS:
            chunks.append({"start_sec": 0.0, "end_sec": round(duration, 2), "audio": y})
        return chunks

    start = 0
    while start < total_len:
        end = start + chunk_len
        segment = y[start:end]
        seg_duration = len(segment) / sr
        if seg_duration >= MIN_CHUNK_SECONDS:
            chunks.append({
                "start_sec": round(start / sr, 2),
                "end_sec": round(min(end, total_len) / sr, 2),
                "audio": segment,
            })
        if end >= total_len:
            break
        start += hop_len

    return chunks


def preprocess_for_inference(file_bytes: bytes, filename: str,
                              apply_codec_sim: bool = True) -> dict:
    """Full preprocessing entrypoint used by the backend's /predict and
    /stream handlers.

    Returns:
        {
            "metadata": AudioMetadata,
            "full_audio": np.ndarray (16kHz mono, codec-simulated, VAD-trimmed),
            "chunks": [{"start_sec", "end_sec", "audio"}, ...]
        }
    """
    y, metadata = load_and_normalize(file_bytes, filename)
    y = trim_silence(y)

    if is_silent(y):
        return {"metadata": metadata, "full_audio": y, "chunks": [], "silent": True}

    if apply_codec_sim:
        y = simulate_phone_codec(y, sr=TARGET_SR)

    chunks = chunk_audio(y, sr=TARGET_SR)

    return {"metadata": metadata, "full_audio": y, "chunks": chunks, "silent": False}
