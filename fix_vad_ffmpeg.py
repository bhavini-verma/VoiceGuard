import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\dataset_tools\vad_split_genuine.py"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

import re

# Add imageio_ffmpeg and set pydub ffmpeg path
imports = """import librosa
import json"""

new_imports = """import librosa
import json
import imageio_ffmpeg
from pydub import AudioSegment
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()"""

text = text.replace(imports, new_imports)

with codecs.open(path, "w", "utf-8") as f:
    f.write(text)

print("Updated script to use imageio-ffmpeg for pydub")
