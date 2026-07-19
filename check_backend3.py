import codecs, sys

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\fastapi_app.py"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

idx = text.find("@app.post(\"/analyze\")")
idx_end = text.find("\n@app.", idx+1)
sys.stdout.buffer.write(text[idx:idx_end].encode("utf-8"))
