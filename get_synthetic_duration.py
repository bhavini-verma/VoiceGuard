import os, sys, subprocess, json

folder = r"C:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGaurd\data\2_synthetic"
exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}

if not os.path.exists(folder):
    print(f"Folder not found: {folder}")
    sys.exit(1)

import imageio_ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffprobe_exe = os.path.join(os.path.dirname(ffmpeg_exe), "ffprobe.exe")

files = []
for root, dirs, fnames in os.walk(folder):
    # Exclude "generic" folder
    if "generic" in [d.lower() for d in root.split(os.sep)]:
        continue
    if "generic" in [d.lower() for d in dirs]:
        dirs.remove("generic")
        
    for fn in fnames:
        if os.path.splitext(fn)[1].lower() in exts:
            files.append(os.path.join(root, fn))

import soundfile as sf
total_s = 0
failed = 0

for fpath in files:
    try:
        # Try soundfile first (fast for wav/flac)
        info = sf.info(fpath)
        total_s += info.duration
    except Exception:
        # Fallback to ffmpeg decode if sf fails (e.g. for mp3)
        try:
            cmd = [ffmpeg_exe, "-i", fpath, "-f", "null", "-"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            # Find duration in stderr: Duration: 00:00:03.14
            import re
            match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d+)", proc.stderr)
            if match:
                h, m, s = match.groups()
                dur = int(h)*3600 + int(m)*60 + float(s)
                total_s += dur
            else:
                failed += 1
        except Exception:
            failed += 1

hours = int(total_s // 3600)
mins  = int((total_s % 3600) // 60)
secs  = int(total_s % 60)

print(f"Total AI voice files  : {len(files)}")
print(f"Failed to read        : {failed}")
print(f"Total duration        : {hours}h {mins}m {secs}s ({total_s:.1f} seconds)")
