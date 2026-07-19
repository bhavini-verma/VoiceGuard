import os
folder = r"C:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGaurd\data\2_synthetic"
exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}

stats = {}
total_files = 0

for root, dirs, fnames in os.walk(folder):
    audio_count = sum(1 for fn in fnames if os.path.splitext(fn)[1].lower() in exts)
    if audio_count > 0:
        rel_path = os.path.relpath(root, folder)
        stats[rel_path] = audio_count
        total_files += audio_count

print("Audio files by folder inside 2_synthetic:")
for k, v in sorted(stats.items()):
    if "generic" in k.lower():
        print(f"  [EXCLUDED] {k}: {v} files")
    else:
        print(f"  [INCLUDED] {k}: {v} files")
        
print(f"\nTotal files across all folders: {total_files}")
