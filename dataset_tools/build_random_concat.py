"""
build_random_concat.py
======================
Takes a base directory of AI voice clone clips, excludes specific folders,
and builds multiple ~30-minute WAV files with random, non-repeating clips.
Each clip is separated by an 880Hz marker tone for auto-splitting later.

USAGE:
    python build_random_concat.py \
        --base_dir "C:/Users/rajas/.gemini/antigravity-ide/scratch/VoiceGaurd/data/2_synthetic" \
        --output_dir ./fake_ota_chunks/

REQUIREMENTS:
    pip install numpy soundfile librosa imageio-ffmpeg
"""

import os
import sys
import argparse
import random
import json
import numpy as np
import soundfile as sf
import librosa
import imageio_ffmpeg
import subprocess

# ------------------------------------------------------------------
# MARKER TONE CONFIGURATION
# ------------------------------------------------------------------
MARKER_FREQ_HZ    = 880
MARKER_DURATION_S = 1.0
SILENCE_BEFORE_S  = 0.5
SILENCE_AFTER_S   = 0.5
TARGET_SR         = 16000
CHUNK_LENGTH_MINS = 30
CHUNK_LENGTH_S    = CHUNK_LENGTH_MINS * 60

SUPPORTED_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.opus'}
EXCLUDED_FOLDERS = {'generic'}


def get_ffmpeg_exe():
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_clip(path, sr):
    """Load audio file via raw ffmpeg subprocess to avoid DLL issues."""
    ffmpeg_exe = get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-i", path, "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(sr), "-ac", "1", "-"
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise Exception(f"ffmpeg error: {proc.stderr.decode('utf-8', 'ignore')}")
        
    raw_audio = proc.stdout
    samples = np.frombuffer(raw_audio, dtype=np.int16)
    y = samples.astype(np.float32) / 32768.0
    
    peak = np.max(np.abs(y))
    if peak > 1e-4:
        y = (y / peak) * 0.80
    return y


def make_tone(freq, duration, sr, amplitude=0.6):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = amplitude * np.sin(2 * np.pi * freq * t)
    fade = int(0.01 * sr)
    tone[:fade] *= np.linspace(0, 1, fade)
    tone[-fade:] *= np.linspace(1, 0, fade)
    return tone


def make_silence(duration, sr):
    return np.zeros(int(sr * duration))


def gather_files(base_dir):
    files = []
    for root, dirs, fnames in os.walk(base_dir):
        # Exclude specified folders (like "generic")
        if any(exc in [d.lower() for d in root.split(os.sep)] for exc in EXCLUDED_FOLDERS):
            continue
        # Modify dirs in-place to prevent os.walk from visiting excluded folders
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDED_FOLDERS]
        
        for fn in fnames:
            if os.path.splitext(fn)[1].lower() in SUPPORTED_EXTS:
                files.append(os.path.join(root, fn))
    return files


def build_chunks(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print("[INFO] Gathering files...")
    all_files = gather_files(base_dir)
    print(f"[INFO] Found {len(all_files)} audio files. Shuffling...")
    
    # Randomize the list (without replacement)
    random.seed(42)  # For reproducibility, though you can change this
    random.shuffle(all_files)

    sr = TARGET_SR
    marker       = make_tone(MARKER_FREQ_HZ, MARKER_DURATION_S, sr)
    silence_pre  = make_silence(SILENCE_BEFORE_S, sr)
    silence_post = make_silence(SILENCE_AFTER_S, sr)
    marker_block = np.concatenate([silence_pre, marker, silence_post])
    marker_block_duration = SILENCE_BEFORE_S + MARKER_DURATION_S + SILENCE_AFTER_S

    chunk_index = 1
    file_index = 0
    total_files = len(all_files)
    
    while file_index < total_files:
        print(f"\n[INFO] Starting building Chunk #{chunk_index:02d} (Target: {CHUNK_LENGTH_MINS} mins)")
        
        full_audio = []
        manifest = []
        cursor = 0.0
        
        # Start each chunk with a marker
        full_audio.append(marker_block)
        cursor += marker_block_duration
        
        chunk_clips_used = 0
        
        # Add files until we hit ~30 mins
        while file_index < total_files and cursor < CHUNK_LENGTH_S:
            clip_path = all_files[file_index]
            clip_name = os.path.basename(clip_path)
            
            try:
                y = load_clip(clip_path, sr)
            except Exception as e:
                print(f"  [WARN] Skipping {clip_name}: {e}")
                file_index += 1
                continue
                
            clip_start = cursor
            clip_end   = cursor + len(y) / sr
            
            manifest.append({
                "chunk_index": chunk_index,
                "global_file_index": file_index + 1,
                "original_path": clip_path,
                "filename": clip_name,
                "start_s": round(clip_start, 3),
                "end_s": round(clip_end, 3),
                "duration_s": round(len(y) / sr, 3)
            })
            
            full_audio.append(y)
            cursor = clip_end
            
            # Add marker block
            full_audio.append(marker_block)
            cursor += marker_block_duration
            
            file_index += 1
            chunk_clips_used += 1
            
            if chunk_clips_used % 50 == 0:
                print(f"  ... added {chunk_clips_used} clips, duration is {cursor/60:.1f} / {CHUNK_LENGTH_MINS} mins")

        # Save this chunk
        concat = np.concatenate(full_audio).astype(np.float32)
        total_duration = len(concat) / sr
        
        out_wav = os.path.join(output_dir, f"fake_ota_chunk_{chunk_index:02d}.wav")
        out_json = os.path.join(output_dir, f"fake_ota_chunk_{chunk_index:02d}_manifest.json")
        
        print(f"[INFO] Saving {out_wav} ({total_duration/60:.1f} mins, {chunk_clips_used} clips)")
        sf.write(out_wav, concat, sr)
        
        with open(out_json, 'w') as f:
            json.dump({
                "chunk_id": chunk_index,
                "marker_freq_hz": MARKER_FREQ_HZ,
                "marker_duration_s": MARKER_DURATION_S,
                "total_clips": len(manifest),
                "total_duration_s": round(total_duration, 2),
                "clips": manifest
            }, f, indent=2)
            
        chunk_index += 1
        
    print(f"\n✅ All {total_files} clips successfully grouped into {chunk_index - 1} chunks!")
    print(f"   Outputs saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build 30-min randomized concat audio blocks for OTA recording')
    parser.add_argument('--base_dir', required=True, help='Directory containing AI voice clone clips')
    parser.add_argument('--output_dir', required=True, help='Output directory for chunks and manifests')
    args = parser.parse_args()

    build_chunks(args.base_dir, args.output_dir)
