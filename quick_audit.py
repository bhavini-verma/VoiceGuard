import os, sys

folder = r"C:\Users\rajas\Downloads\Call Recordings"
exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".amr", ".3gp"}

files = []
for root, dirs, fnames in os.walk(folder):
    for fn in fnames:
        if os.path.splitext(fn)[1].lower() in exts:
            files.append(os.path.join(root, fn))

print(f"Total audio files found: {len(files)}")
if files:
    print("Formats:")
    from collections import Counter
    c = Counter(os.path.splitext(f)[1].lower() for f in files)
    for ext, cnt in c.most_common():
        print(f"  {ext}: {cnt}")
    print("\nFirst 5 files:")
    for f in files[:5]:
        print(" ", os.path.basename(f))
