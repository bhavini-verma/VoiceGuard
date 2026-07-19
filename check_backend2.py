import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\fastapi_app.py"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Read the whole analyze function
idx = text.find("@app.post(\"/analyze\")")
idx_end = text.find("\n@app.", idx+1)
print(text[idx:idx_end])
