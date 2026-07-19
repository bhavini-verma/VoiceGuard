import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Check switchPage function - does it hide things?
idx = text.find("function switchPage")
if idx != -1:
    print(text[idx:idx+1000])
