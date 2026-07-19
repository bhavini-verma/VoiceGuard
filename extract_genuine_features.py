import os
import subprocess
import time
import sys

input_dir = r"C:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGaurd_TelephonyExp\data\1_genuine\sampled_for_training"
output_dir_deep = r"C:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGaurd_TelephonyExp\features\deep_genuine"
output_dir_bio = r"C:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGaurd_TelephonyExp\features\bio_genuine"

extract_deep_script = r"C:\Users\rajas\.gemini\antigravity-ide\scratch\voiceguard_repo\src\extract_deep.py"
extract_bio_script = r"C:\Users\rajas\.gemini\antigravity-ide\scratch\voiceguard_repo\src\extract_bio.py"

os.makedirs(output_dir_deep, exist_ok=True)
os.makedirs(output_dir_bio, exist_ok=True)

print("Starting Deep Spectral Extraction...")
start = time.time()
subprocess.run([sys.executable, extract_deep_script, "--input_dir", input_dir, "--output_dir", output_dir_deep])
deep_time = time.time() - start

print("Starting Bio-Feature Extraction...")
start = time.time()
subprocess.run([sys.executable, extract_bio_script, "--input_dir", input_dir, "--output_dir", output_dir_bio])
bio_time = time.time() - start

print(f"\nExtraction complete!")
print(f"Deep extraction time: {deep_time/60:.1f} mins")
print(f"Bio extraction time: {bio_time/60:.1f} mins")
