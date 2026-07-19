"""
build_concat_audio.py
=====================
Takes a folder of AI voice clone clips and builds a single long WAV file
with an audible marker tone between each clip.

When you record this long audio over a phone call on another device,
the split_recorded_audio.py script can detect those marker tones and
automatically split the recording back into individual clips.

USAGE:
    python build_concat_audio.py --clips_dir ./ai_clips --output concat_master.wav

REQUIREMENTS:
    pip install numpy soundfile librosa
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
import librosa
import json


# ------------------------------------------------------------------
# MARKER TONE CONFIGURATION
# These settings must match split_recorded_audio.py exactly.
# ------------------------------------------------------------------
MARKER_FREQ_HZ   = 880      # Pure tone frequency (survives phone codecs well)
MARKER_DURATION_S = 1.0     # Length of each marker in seconds
MARKER_AMPLITUDE  = 0.6     # Loud enough to survive phone/speaker chain
SILENCE_BEFORE_S  = 0.5     # Silence before each marker
SILENCE_AFTER_S   = 0.5     # Silence after each marker
TARGET_SR         = 16000   # All audio standardized to 16kHz


def make_tone(freq, duration, sr, amplitude=0.6):
    """Generate a pure sine tone."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Add a brief fade in/out to avoid clicks
    tone = amplitude * np.sin(2 * np.pi * freq * t)
    fade = int(0.01 * sr)  # 10ms fade
    tone[:fade] *= np.linspace(0, 1, fade)
    tone[-fade:] *= np.linspace(1, 0, fade)
    return tone


def make_silence(duration, sr):
    return np.zeros(int(sr * duration))


def load_clip(path, target_sr):
    """Load any audio file and resample to target SR, mono."""
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    # Normalize to 80% peak to leave headroom for marker tone
    peak = np.max(np.abs(y))
    if peak > 1e-4:
        y = (y / peak) * 0.80
    return y


def build_concat(clips_dir, output_path, max_clip_duration=None):
    # Collect clips
    exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    clips = sorted([
        os.path.join(clips_dir, f)
        for f in os.listdir(clips_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])

    if not clips:
        print(f"[ERROR] No audio files found in: {clips_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(clips)} clips in {clips_dir}")

    sr = TARGET_SR
    marker = make_tone(MARKER_FREQ_HZ, MARKER_DURATION_S, sr, MARKER_AMPLITUDE)
    silence_pre  = make_silence(SILENCE_BEFORE_S, sr)
    silence_post = make_silence(SILENCE_AFTER_S, sr)

    # A "marker block" = silence + tone + silence
    marker_block = np.concatenate([silence_pre, marker, silence_post])
    marker_block_duration = SILENCE_BEFORE_S + MARKER_DURATION_S + SILENCE_AFTER_S

    segments = []       # list of (start_time_s, end_time_s, clip_filename)
    full_audio = []
    cursor = 0.0        # current time position in seconds

    # Always start with a marker block so the recording device has
    # a clear signal to detect even if it missed the very start
    full_audio.append(marker_block)
    cursor += marker_block_duration

    manifest = []

    for i, clip_path in enumerate(clips):
        clip_name = os.path.basename(clip_path)
        print(f"  [{i+1:03d}/{len(clips)}] Loading: {clip_name}")

        try:
            y = load_clip(clip_path, sr)
        except Exception as e:
            print(f"  [WARN] Skipping {clip_name}: {e}")
            continue

        if max_clip_duration:
            max_samples = int(max_clip_duration * sr)
            y = y[:max_samples]

        clip_start = cursor
        clip_end   = cursor + len(y) / sr

        manifest.append({
            "index":     i + 1,
            "filename":  clip_name,
            "start_s":   round(clip_start, 3),
            "end_s":     round(clip_end, 3),
            "duration_s": round(len(y) / sr, 3)
        })

        full_audio.append(y)
        cursor = clip_end

        # Append marker block after each clip
        full_audio.append(marker_block)
        cursor += marker_block_duration

    # Concatenate everything
    concat = np.concatenate(full_audio).astype(np.float32)
    total_duration = len(concat) / sr

    print(f"\n[INFO] Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"[INFO] Writing: {output_path}")
    sf.write(output_path, concat, sr)

    # Save manifest (tells split_recorded_audio.py what to expect)
    manifest_path = output_path.replace('.wav', '_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump({
            "marker_freq_hz":     MARKER_FREQ_HZ,
            "marker_duration_s":  MARKER_DURATION_S,
            "silence_before_s":   SILENCE_BEFORE_S,
            "silence_after_s":    SILENCE_AFTER_S,
            "target_sr":          TARGET_SR,
            "total_clips":        len(manifest),
            "total_duration_s":   round(total_duration, 2),
            "clips":              manifest
        }, f, indent=2)

    print(f"[INFO] Manifest saved: {manifest_path}")
    print(f"\n✅ Done! Play '{output_path}' through your phone.")
    print(f"   Record it on another phone, then run:")
    print(f"   python split_recorded_audio.py --recording <your_recording.wav> --output_dir ./splits/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build concatenated AI clip audio with marker tones')
    parser.add_argument('--clips_dir',  required=True, help='Directory containing AI voice clone clips')
    parser.add_argument('--output',     default='concat_master.wav', help='Output WAV file path')
    parser.add_argument('--max_dur',    type=float, default=None, help='Max seconds per clip (optional)')
    args = parser.parse_args()

    build_concat(args.clips_dir, args.output, args.max_dur)
