import codecs, sys

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\fastapi_app.py"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Find wav2vec and feature extraction
idx = text.find("wav2vec") 
while idx != -1:
    print(text[max(0, idx-50):idx+200])
    print("---")
    idx = text.find("wav2vec", idx+1)
    if idx > 250000:
        break
