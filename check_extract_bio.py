import codecs, sys

base = r"c:\Users\rajas\.gemini\antigravity-ide\scratch"
# Check VoiceGaurd_TelephonyExp path
import os

# Find extract_bio.py
for root, dirs, files in os.walk(r"c:\Users\rajas\.gemini\antigravity-ide\scratch"):
    for fn in files:
        if "extract_bio" in fn:
            print(os.path.join(root, fn))
