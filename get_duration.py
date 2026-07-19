import os, sys
import mutagen
from mutagen.mp4 import MP4

folder = r"C:\Users\rajas\Downloads\Call Recordings"

files = []
for root, dirs, fnames in os.walk(folder):
    for fn in fnames:
        if fn.lower().endswith(".m4a"):
            files.append(os.path.join(root, fn))

total_s = 0
failed = 0
for fpath in files:
    try:
        audio = MP4(fpath)
        total_s += audio.info.length
    except Exception as e:
        failed += 1

hours = int(total_s // 3600)
mins  = int((total_s % 3600) // 60)
secs  = int(total_s % 60)

sys.stdout.buffer.write(f"Total files     : {len(files)}\n".encode())
sys.stdout.buffer.write(f"Failed to read  : {failed}\n".encode())
sys.stdout.buffer.write(f"Total duration  : {hours}h {mins}m {secs}s\n".encode())
sys.stdout.buffer.write(f"Avg per clip    : {total_s/max(1,len(files)-failed):.1f}s\n".encode())
sys.stdout.buffer.write(f"Est 10s splits  : ~{int(total_s/10)}\n".encode())
sys.stdout.buffer.write(f"Est 5s splits   : ~{int(total_s/5)}\n".encode())
