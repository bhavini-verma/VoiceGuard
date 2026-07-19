"""
dataset_audit.py
================
Feed this your existing real call recordings directory.
It will report:
  - Total clips and duration
  - Duration distribution (histogram)
  - Recommended number of fake OTA clips to record
  - Recommended fake OTA recording time needed

USAGE:
    python dataset_audit.py --genuine_dir ./real_call_recordings/

REQUIREMENTS:
    pip install numpy soundfile librosa
"""

import os
import sys
import argparse
import json
import numpy as np
import soundfile as sf
import librosa
from collections import Counter


SUPPORTED_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.opus'}


def get_audio_duration(path):
    """Fast duration check using soundfile (no decode needed for WAV/FLAC)."""
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        try:
            # Fallback for MP3/other formats
            y, sr = librosa.load(path, sr=None, mono=True, duration=0.1)
            # Get actual duration via mutagen-style header read
            info = sf.info(path) if path.endswith('.wav') else None
            if info:
                return info.duration
            # Last resort: full load
            y, sr = librosa.load(path, sr=None, mono=True)
            return len(y) / sr
        except Exception as e:
            return None


def audit_directory(directory):
    """Scan a directory recursively for audio files and report stats."""
    clips = []
    skipped = []

    print(f"[INFO] Scanning: {directory}")
    for root, dirs, files in os.walk(directory):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            fpath = os.path.join(root, fname)
            dur = get_audio_duration(fpath)
            if dur is None:
                skipped.append(fpath)
                continue
            size_kb = os.path.getsize(fpath) / 1024
            clips.append({
                "path":     fpath,
                "filename": fname,
                "duration_s": round(dur, 2),
                "size_kb":  round(size_kb, 1)
            })

    return clips, skipped


def print_histogram(durations, bins=10):
    """Print a simple ASCII histogram of clip durations."""
    if not durations:
        return
    min_d, max_d = min(durations), max(durations)
    bin_size = (max_d - min_d) / bins if max_d > min_d else 1
    counts = [0] * bins
    for d in durations:
        idx = min(int((d - min_d) / bin_size), bins - 1)
        counts[idx] += 1
    max_count = max(counts) if counts else 1
    print("\n  Duration Histogram:")
    for i in range(bins):
        lo = min_d + i * bin_size
        hi = lo + bin_size
        bar = '█' * int(counts[i] / max_count * 30)
        print(f"  {lo:5.1f}s–{hi:5.1f}s | {bar:<30} {counts[i]}")


def recommend_fake_count(genuine_clips, clip_duration_s=10):
    """
    Recommend how many fake OTA clips to record.
    Targets a 1:1 class ratio which is ideal for training.
    Returns (n_clips_needed, recording_minutes_needed)
    """
    n_genuine = len(genuine_clips)
    total_genuine_s = sum(c['duration_s'] for c in genuine_clips)

    # Estimate how many 10s clips we'd get from genuine data
    # (assuming they'll be split too)
    n_genuine_splits = int(total_genuine_s / clip_duration_s)

    # Fake should match genuine splits for 1:1
    n_fake_needed = n_genuine_splits

    # Each fake OTA session needs raw recording time = n_clips * clip_duration
    # Plus ~2s marker overhead per clip
    marker_overhead_s = 2.0
    raw_recording_s = n_fake_needed * (clip_duration_s + marker_overhead_s)

    return n_genuine, n_genuine_splits, n_fake_needed, raw_recording_s


def run_audit(genuine_dir, output_report=None, split_duration_s=10):
    clips, skipped = audit_directory(genuine_dir)

    if not clips:
        print("[ERROR] No valid audio files found.")
        sys.exit(1)

    durations = [c['duration_s'] for c in clips]
    total_s   = sum(durations)
    total_min = total_s / 60

    print("\n" + "="*60)
    print("  GENUINE DATASET AUDIT REPORT")
    print("="*60)
    print(f"  Total clips found    : {len(clips)}")
    print(f"  Total duration       : {total_s:.0f}s  ({total_min:.1f} min)")
    print(f"  Average clip length  : {np.mean(durations):.1f}s")
    print(f"  Median clip length   : {np.median(durations):.1f}s")
    print(f"  Shortest clip        : {min(durations):.1f}s")
    print(f"  Longest clip         : {max(durations):.1f}s")
    if skipped:
        print(f"  Skipped (unreadable) : {len(skipped)}")

    print_histogram(durations)

    # Duration buckets
    short  = sum(1 for d in durations if d < 5)
    medium = sum(1 for d in durations if 5 <= d < 15)
    long_  = sum(1 for d in durations if d >= 15)
    print(f"\n  Duration Buckets:")
    print(f"    < 5s  : {short}  clips  (too short for 10s splits — will be discarded)")
    print(f"    5–15s : {medium} clips  (yields ~1 split each)")
    print(f"    > 15s : {long_}  clips  (yields multiple splits)")

    # Recommend fake OTA recording
    n_genuine, n_splits, n_fake_needed, raw_s = recommend_fake_count(clips, split_duration_s)
    raw_min = raw_s / 60

    print(f"\n" + "="*60)
    print(f"  FAKE OTA RECORDING RECOMMENDATION  (for 1:1 class balance)")
    print(f"="*60)
    print(f"  Genuine clips after {split_duration_s}s splits  : ~{n_splits}")
    print(f"  Fake OTA clips needed (1:1 ratio)  : ~{n_fake_needed}")
    print(f"  Raw recording time needed           : ~{raw_s:.0f}s  ({raw_min:.1f} min)")
    print(f"")
    print(f"  With 3 environments (recommended):")
    per_env = n_fake_needed // 3
    per_env_min = (per_env * (split_duration_s + 2)) / 60
    print(f"    → {per_env} clips per environment")
    print(f"    → {per_env_min:.1f} min of recording per environment")

    print(f"\n  NOTE: 1:1 ratio is ideal. Minimum viable is 1:2 (fake:genuine).")
    print(f"        At 1:2, you need ~{n_fake_needed//2} fake clips ({raw_min/2:.1f} min recording).")

    # Save report
    report = {
        "genuine_dir": genuine_dir,
        "total_clips": len(clips),
        "total_duration_s": round(total_s, 1),
        "avg_duration_s": round(np.mean(durations), 1),
        "median_duration_s": round(np.median(durations), 1),
        "estimated_splits_at_10s": n_splits,
        "recommended_fake_ota_clips": n_fake_needed,
        "recommended_recording_time_min": round(raw_min, 1),
        "per_environment_3env": per_env,
        "clips": clips
    }

    if output_report:
        with open(output_report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  Full report saved: {output_report}")

    print("="*60)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audit genuine call recordings and recommend fake OTA targets')
    parser.add_argument('--genuine_dir',  required=True, help='Directory of genuine call recordings')
    parser.add_argument('--output',       default='dataset_audit_report.json', help='Output JSON report path')
    parser.add_argument('--split_dur',    type=float, default=10.0, help='Target split duration in seconds')
    args = parser.parse_args()

    run_audit(args.genuine_dir, args.output, args.split_dur)
