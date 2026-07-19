import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\fastapi_app.py"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Find the analyze endpoint
idx = text.find("/analyze")
print(text[max(0, idx-100):idx+2000])
