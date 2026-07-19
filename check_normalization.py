import codecs, sys

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\fastapi_app.py"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Find peak normalization
idx = text.find("peak = np.max")
print(text[max(0, idx-200):idx+600])
