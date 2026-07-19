"""
split_recorded_audio.py
========================
After you record the concat_master.wav through a phone call on another device,
feed that recording into this script. It detects the 880Hz marker tones and
splits the audio into individual clips — one per AI voice clip.

USAGE:
    python split_recorded_audio.py --recording my_phone_recording.wav --output_dir ./fake_ota_splits/

    # With a manifest from build_concat_audio.py (better accuracy):
    python split_recorded_audio.py --recording my_phone_recording.wav \
                                   --manifest concat_master_manifest.json \
                                   --output_dir ./fake_ota_splits/

REQUIREMENTS:
    pip install numpy soundfile librosa scipy
"""

import os
import sys
import argparse
import json
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, lfilter


# ------------------------------------------------------------------
# MUST MATCH build_concat_audio.py settings
# ------------------------------------------------------------------
MARKER_FREQ_HZ    = 880
MARKER_DURATION_S = 1.0
SILENCE_BEFORE_S  = 0.5
SILENCE_AFTER_S   = 0.5
TARGET_SR         = 16000


def bandpass_filter(data, lowcut, highcut, sr, order=4):
    """Narrow bandpass around the marker frequency to isolate it."""
    nyq = 0.5 * sr
    low  = max(lowcut  / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)


def detect_marker_regions(y, sr, freq=MARKER_FREQ_HZ, min_duration=0.4, threshold_percentile=92):
    """
    Returns list of (start_sample, end_sample) for each marker tone region.
    Strategy:
    1. Bandpass filter tightly around marker frequency (±40 Hz)
    2. Compute short-time energy of filtered signal
    3. Threshold to find high-energy (= tone present) regions
    """
    # Step 1: Bandpass around marker frequency
    filtered = bandpass_filter(y, freq - 40, freq + 40, sr)

    # Step 2: Short-time energy (frame size ~50ms, hop ~10ms)
    frame_len = int(0.05 * sr)
    hop_len   = int(0.01 * sr)
    energy = np.array([
        np.sum(filtered[i:i+frame_len] ** 2)
        for i in range(0, len(filtered) - frame_len, hop_len)
    ])

    # Step 3: Threshold
    threshold = np.percentile(energy, threshold_percentile)
    is_tone = energy > threshold

    # Step 4: Find contiguous tone regions
    regions = []
    in_tone = False
    start_frame = 0
    min_frames = int(min_duration / (hop_len / sr))

    for i, v in enumerate(is_tone):
        if v and not in_tone:
            in_tone = True
            start_frame = i
        elif not v and in_tone:
            in_tone = False
            length = i - start_frame
            if length >= min_frames:
                start_s = start_frame * hop_len
                end_s   = i * hop_len + frame_len
                regions.append((start_s, min(end_s, len(y))))
    
    # Handle tone that runs to end of file
    if in_tone and (len(is_tone) - start_frame) >= min_frames:
        start_s = start_frame * hop_len
        regions.append((start_s, len(y)))

    return regions


def extract_clips_between_markers(y, sr, marker_regions, min_clip_duration=1.0, max_clip_duration=30.0):
    """
    Given marker regions, extract the audio BETWEEN consecutive markers.
    These are the actual AI voice clone clips.
    """
    if len(marker_regions) < 2:
        print("[WARN] Fewer than 2 marker regions found. Cannot extract clips.")
        return []

    clips = []
    # Clip i = audio between end of marker[i] and start of marker[i+1]
    for i in range(len(marker_regions) - 1):
        clip_start = marker_regions[i][1]      # end of current marker
        clip_end   = marker_regions[i + 1][0]  # start of next marker

        # Skip tiny or huge gaps
        duration = (clip_end - clip_start) / sr
        if duration < min_clip_duration:
            print(f"  [SKIP] Gap {i} is {duration:.2f}s — too short, skipping")
            continue
        if duration > max_clip_duration:
            print(f"  [WARN] Gap {i} is {duration:.2f}s — unexpectedly long, still saving")

        clips.append({
            "index":       len(clips) + 1,
            "start_sample": clip_start,
            "end_sample":   clip_end,
            "duration_s":  round(duration, 3)
        })

    return clips


def split_recording(recording_path, output_dir, manifest_path=None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading recording: {recording_path}")
    y, sr = librosa.load(recording_path, sr=TARGET_SR, mono=True)
    print(f"[INFO] Duration: {len(y)/sr:.1f}s at {sr}Hz")

    # Load manifest if provided (helps with labeling)
    manifest = None
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"[INFO] Manifest loaded: expects {manifest['total_clips']} clips")

    # Detect marker tones
    print(f"[INFO] Detecting {MARKER_FREQ_HZ}Hz marker tones...")
    marker_regions = detect_marker_regions(y, sr)
    print(f"[INFO] Found {len(marker_regions)} marker regions")

    if len(marker_regions) == 0:
        print("[ERROR] No markers detected. Check that:")
        print("  - The recording volume was adequate")
        print("  - The correct MARKER_FREQ_HZ is set (must match build_concat_audio.py)")
        print("  - The phone call didn't filter out 880Hz (unlikely but possible)")
        sys.exit(1)

    # Print detected marker positions
    for i, (s, e) in enumerate(marker_regions):
        print(f"  Marker {i+1}: {s/sr:.2f}s → {e/sr:.2f}s (duration: {(e-s)/sr:.2f}s)")

    # Extract clips between markers
    clips = extract_clips_between_markers(y, sr, marker_regions)
    print(f"\n[INFO] Extracting {len(clips)} clips...")

    results = []
    for clip in clips:
        i          = clip['index']
        audio_clip = y[clip['start_sample']:clip['end_sample']]

        # Normalize clip
        peak = np.max(np.abs(audio_clip))
        if peak > 1e-4:
            audio_clip = (audio_clip / peak) * 0.95

        # Build filename
        if manifest and i <= len(manifest['clips']):
            orig_name = os.path.splitext(manifest['clips'][i-1]['filename'])[0]
            out_name  = f"{i:04d}_{orig_name}_ota.wav"
        else:
            out_name  = f"{i:04d}_fake_ota.wav"

        out_path = os.path.join(output_dir, out_name)
        sf.write(out_path, audio_clip, sr)

        print(f"  [{i:04d}] {clip['duration_s']:.2f}s → {out_name}")
        results.append({"file": out_name, "duration_s": clip['duration_s']})

    # Save split report
    report_path = os.path.join(output_dir, "_split_report.json")
    with open(report_path, 'w') as f:
        json.dump({
            "recording":        recording_path,
            "markers_detected": len(marker_regions),
            "clips_extracted":  len(clips),
            "clips":            results
        }, f, indent=2)

    print(f"\n✅ Done! {len(clips)} clips saved to: {output_dir}")
    print(f"   Split report: {report_path}")

    if manifest:
        expected = manifest['total_clips']
        got      = len(clips)
        if got < expected:
            print(f"\n⚠️  Expected {expected} clips but only got {got}.")
            print("   Likely cause: some marker tones were too quiet to detect.")
            print("   Try re-recording with speaker volume turned up.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto-split phone recording using 880Hz marker tones')
    parser.add_argument('--recording',   required=True, help='Path to the phone-recorded audio file')
    parser.add_argument('--output_dir',  default='./ota_splits', help='Directory to save split clips')
    parser.add_argument('--manifest',    default=None, help='Optional manifest JSON from build_concat_audio.py')
    parser.add_argument('--min_dur',     type=float, default=1.0, help='Minimum clip duration in seconds')
    parser.add_argument('--max_dur',     type=float, default=30.0, help='Maximum clip duration in seconds')
    args = parser.parse_args()

    split_recording(args.recording, args.output_dir, args.manifest)
