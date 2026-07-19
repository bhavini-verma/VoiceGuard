import os, random, shutil

input_dir = r"C:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGaurd_TelephonyExp\data\1_genuine\vad_splits"
output_dir = r"C:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGaurd_TelephonyExp\data\1_genuine\sampled_for_training"

os.makedirs(output_dir, exist_ok=True)

files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
random.seed(42)
random.shuffle(files)

# We want 5.6 hours. At 10s per clip, that's 5.6 * 360 = 2016 clips.
target_clips = 2016
sampled = files[:target_clips]

for f in sampled:
    src = os.path.join(input_dir, f)
    dst = os.path.join(output_dir, f)
    shutil.copy2(src, dst)

print(f"Sampled {len(sampled)} clips (~5.6 hours) to {output_dir}")
