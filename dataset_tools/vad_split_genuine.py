"""
vad_split_genuine.py
=====================
Processes real phone call recordings with Voice Activity Detection (VAD).
Strips silence, only keeps chunks with real speech, outputs clean clips
ready for model training.

WHAT IT DOES:
  1. Loads each .m4a / .wav recording
  2. Runs Google WebRTC VAD (same engine used in Chrome/WebRTC)
  3. Merges nearby speech segments
  4. Discards any chunk where speech < MIN_SPEECH_RATIO
  5. Saves clean 10s clips as WAV @ 16kHz

USAGE:
    python vad_split_genuine.py \
        --input_dir "C:/Users/rajas/Downloads/Call Recordings" \
        --output_dir ./genuine_splits/ \
        --chunk_dur 10 \
        --min_speech 0.5

    # min_speech=0.5 means: keep chunk only if >=50% of it is actual speech

REQUIREMENTS:
    pip install webrtcvad-wheels soundfile librosa numpy
"""

import os
import sys
import argparse
import struct
import wave
import io
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import json


def load_webrtcvad():
    try:
        import webrtcvad
        return webrtcvad
    except ImportError:
        print("[ERROR] webrtcvad not installed. Run: pip install webrtcvad-wheels")
        sys.exit(1)


# ----------------------------------------------------------------
# VAD CONFIGURATION
# ----------------------------------------------------------------
VAD_SAMPLE_RATE    = 16000    # WebRTC VAD supports 8k/16k/32k/48k
VAD_FRAME_MS       = 20       # Frame size in ms (10, 20, or 30 only)
VAD_AGGRESSIVENESS = 2        # 0=least aggressive, 3=most aggressive
                               # 2 is good for phone call recordings
MIN_SPEECH_RATIO   = 0.50     # Chunk needs >= 50% speech to be kept
MERGE_GAP_S        = 0.3      # Merge speech regions within 300ms of each other
SUPPORTED_EXTS     = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.amr'}


def float32_to_pcm16(audio):
    """Convert float32 numpy array to raw 16-bit PCM bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    return pcm.tobytes()


def run_vad(y, sr, aggressiveness=VAD_AGGRESSIVENESS):
    """
    Run WebRTC VAD on audio.
    Returns: list of (start_sample, end_sample) for speech regions.
    """
    webrtcvad = load_webrtcvad()
    vad = webrtcvad.Vad(aggressiveness)

    frame_samples = int(sr * VAD_FRAME_MS / 1000)  # samples per frame
    speech_flags  = []

    for start in range(0, len(y) - frame_samples, frame_samples):
        frame = y[start:start + frame_samples]
        pcm   = float32_to_pcm16(frame)
        try:
            is_speech = vad.is_speech(pcm, sr)
        except Exception:
            is_speech = False
        speech_flags.append((start, start + frame_samples, is_speech))

    # Merge nearby speech regions
    merge_samples = int(MERGE_GAP_S * sr)
    speech_regions = []
    in_speech  = False
    region_start = 0

    for start, end, is_speech in speech_flags:
        if is_speech and not in_speech:
            in_speech    = True
            region_start = start
        elif not is_speech and in_speech:
            # Check if next speech region is close (merge gap)
            in_speech = False
            speech_regions.append((region_start, end))

    if in_speech:
        speech_regions.append((region_start, len(y)))

    # Merge regions that are within merge_samples of each other
    if not speech_regions:
        return []

    merged = [speech_regions[0]]
    for start, end in speech_regions[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_samples:
            merged[-1] = (prev_start, end)  # merge
        else:
            merged.append((start, end))

    return merged


def split_into_chunks(y, sr, chunk_dur_s, min_speech_ratio, speech_regions):
    """
    Divide the full audio into fixed-length chunks.
    For each chunk, compute how much of it overlaps with speech regions.
    Keep only chunks above the min_speech_ratio.
    """
    chunk_samples = int(chunk_dur_s * sr)
    kept = []
    discarded_silence = 0
    discarded_short   = 0

    # Build a boolean speech mask (True = speech frame)
    speech_mask = np.zeros(len(y), dtype=bool)
    for s, e in speech_regions:
        speech_mask[s:e] = True

    for chunk_start in range(0, len(y) - chunk_samples + 1, chunk_samples):
        chunk_end   = chunk_start + chunk_samples
        chunk_audio = y[chunk_start:chunk_end]
        chunk_mask  = speech_mask[chunk_start:chunk_end]

        speech_ratio = np.mean(chunk_mask)

        if speech_ratio >= min_speech_ratio:
            kept.append({
                "audio":        chunk_audio,
                "speech_ratio": round(float(speech_ratio), 3),
                "start_s":      round(chunk_start / sr, 2),
                "end_s":        round(chunk_end / sr, 2)
            })
        else:
            discarded_silence += 1

    # Handle last partial chunk (if remaining > 50% of chunk duration)
    remainder_start = (len(y) // chunk_samples) * chunk_samples
    remainder       = y[remainder_start:]
    if len(remainder) >= chunk_samples * 0.5:
        # Pad to full chunk length
        padded = np.pad(remainder, (0, chunk_samples - len(remainder)))
        mask_r = speech_mask[remainder_start:]
        if len(mask_r) > 0 and np.mean(mask_r) >= min_speech_ratio:
            kept.append({
                "audio":        padded,
                "speech_ratio": round(float(np.mean(mask_r)), 3),
                "start_s":      round(remainder_start / sr, 2),
                "end_s":        round(len(y) / sr, 2)
            })
        else:
            discarded_silence += 1
    else:
        discarded_short += 1

    return kept, discarded_silence, discarded_short


def process_file(input_path, output_dir, file_idx, chunk_dur_s, min_speech_ratio):
    """Process a single recording file."""
    basename = os.path.splitext(os.path.basename(input_path))[0]
    # Sanitize filename for windows (remove special chars)
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in basename)[:40]

    # Load and resample using raw ffmpeg (bypasses numba DLL block and missing ffprobe)
    try:
        import subprocess
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        # We output raw 16-bit PCM (s16le) at VAD_SAMPLE_RATE Hz, mono
        cmd = [
            ffmpeg_exe,
            "-i", input_path,
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", str(VAD_SAMPLE_RATE),
            "-ac", "1",
            "-"
        ]
        
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            return 0, 0, f"ffmpeg error: {proc.stderr.decode('utf-8', 'ignore')}"
            
        raw_audio = proc.stdout
        # Convert bytes to numpy float32
        samples = np.frombuffer(raw_audio, dtype=np.int16)
        y = samples.astype(np.float32) / 32768.0
    except Exception as e:
        return 0, 0, str(e)

    # Peak normalize
    peak = np.max(np.abs(y))
    if peak > 1e-4:
        y = (y / peak) * 0.95

    # Run VAD
    speech_regions = run_vad(y, VAD_SAMPLE_RATE)

    total_speech_s = sum((e - s) / VAD_SAMPLE_RATE for s, e in speech_regions)
    total_dur_s    = len(y) / VAD_SAMPLE_RATE
    speech_pct     = 100 * total_speech_s / max(total_dur_s, 1)

    # Split into chunks
    chunks, disc_silence, disc_short = split_into_chunks(
        y, VAD_SAMPLE_RATE, chunk_dur_s, min_speech_ratio, speech_regions
    )

    # Save chunks
    saved = 0
    for c_idx, chunk in enumerate(chunks):
        out_name = f"{file_idx:04d}_{safe_name}_{c_idx+1:03d}.wav"
        out_path = os.path.join(output_dir, out_name)
        sf.write(out_path, chunk["audio"].astype(np.float32), VAD_SAMPLE_RATE)
        saved += 1

    return saved, disc_silence, None


def process_directory(input_dir, output_dir, chunk_dur_s, min_speech_ratio):
    os.makedirs(output_dir, exist_ok=True)

    # Collect files
    files = []
    for root, dirs, fnames in os.walk(input_dir):
        for fn in fnames:
            if os.path.splitext(fn)[1].lower() in SUPPORTED_EXTS:
                files.append(os.path.join(root, fn))
    files.sort()

    if not files:
        print(f"[ERROR] No audio files found in: {input_dir}")
        sys.exit(1)

    print(f"[INFO] Processing {len(files)} recordings...")
    print(f"[INFO] Chunk duration  : {chunk_dur_s}s")
    print(f"[INFO] Min speech ratio: {min_speech_ratio*100:.0f}%")
    print(f"[INFO] VAD aggressiveness: {VAD_AGGRESSIVENESS}/3")
    print(f"[INFO] Output dir      : {output_dir}")
    print()

    total_saved    = 0
    total_silenced = 0
    errors         = []

    for i, fpath in enumerate(files):
        fn_display = os.path.basename(fpath).encode('ascii', 'replace').decode()[:50]
        print(f"  [{i+1:03d}/{len(files)}] {fn_display} ...", end="", flush=True)
        saved, silenced, err = process_file(fpath, output_dir, i+1, chunk_dur_s, min_speech_ratio)
        if err:
            print(f" FAILED: {err}")
            errors.append({"file": fpath, "error": err})
        else:
            print(f" -> {saved} clips kept, {silenced} silent chunks discarded")
        total_saved    += saved
        total_silenced += silenced

    print()
    print("=" * 55)
    print(f"  DONE")
    print(f"  Total clips saved    : {total_saved}")
    print(f"  Silent chunks dropped: {total_silenced}")
    print(f"  Files with errors    : {len(errors)}")
    print(f"  Training data        : {total_saved * chunk_dur_s / 60:.1f} minutes of genuine speech")
    print("=" * 55)

    # Save summary
    summary_path = os.path.join(output_dir, "_vad_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "input_dir":         input_dir,
            "chunk_duration_s":  chunk_dur_s,
            "min_speech_ratio":  min_speech_ratio,
            "vad_aggressiveness": VAD_AGGRESSIVENESS,
            "total_clips_saved": total_saved,
            "silent_discarded":  total_silenced,
            "errors":            errors
        }, f, indent=2)
    print(f"  Summary: {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAD-based speech extraction from call recordings')
    parser.add_argument('--input_dir',    required=True,  help='Directory of call recordings')
    parser.add_argument('--output_dir',   required=True,  help='Where to save clean speech clips')
    parser.add_argument('--chunk_dur',    type=float, default=10.0, help='Clip duration in seconds (default: 10)')
    parser.add_argument('--min_speech',   type=float, default=0.50, help='Min fraction of chunk that must be speech (default: 0.5)')
    parser.add_argument('--aggressiveness', type=int, default=2,   help='VAD aggressiveness 0-3 (default: 2). Use 3 for noisy recordings.')
    args = parser.parse_args()

    VAD_AGGRESSIVENESS = args.aggressiveness
    process_directory(args.input_dir, args.output_dir, args.chunk_dur, args.min_speech)
