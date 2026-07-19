import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\dataset_tools\vad_split_genuine.py"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

import re

# Remove librosa import
text = re.sub(r"import librosa\n", "from pydub import AudioSegment\n", text)

# Replace load logic
old_load = """    # Load and resample to 16kHz mono
    try:
        y, sr = librosa.load(input_path, sr=VAD_SAMPLE_RATE, mono=True)
    except Exception as e:
        return 0, 0, str(e)"""

new_load = """    # Load and resample using pydub (bypasses numba DLL block)
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(VAD_SAMPLE_RATE).set_channels(1)
        samples = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            y = samples.astype(np.float32) / 32768.0
        elif audio.sample_width == 4:
            y = samples.astype(np.float32) / 2147483648.0
        else:
            return 0, 0, "Unsupported sample width"
    except Exception as e:
        return 0, 0, str(e)"""

text = text.replace(old_load, new_load)

with codecs.open(path, "w", "utf-8") as f:
    f.write(text)

print("Updated script to use pydub")
