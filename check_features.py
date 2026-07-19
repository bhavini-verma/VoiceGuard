import codecs, sys

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\fastapi_app.py"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Find librosa.load line and feature extraction
idx = text.find("y, sr = librosa.load")
print(text[max(0, idx-100):idx+2000])
