import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

idx2 = text.find("function startLiveSweepLoop")
idx2_end = text.find("\nfunction ", idx2+1)
print(text[idx2:idx2_end])
