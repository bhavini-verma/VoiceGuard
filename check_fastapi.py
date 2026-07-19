import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\fastapi_app.py"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()
print(text[:3000])
