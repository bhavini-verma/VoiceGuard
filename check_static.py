import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\fastapi_app.py"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Find static files and HTML route
idx = text.find("StaticFiles")
while idx != -1:
    print(text[max(0, idx-50):idx+200])
    print("---")
    idx = text.find("StaticFiles", idx+1)

idx2 = text.find("voiceguard_uco_bank_platform.html")
if idx2 != -1:
    print(text[max(0, idx2-200):idx2+300])
