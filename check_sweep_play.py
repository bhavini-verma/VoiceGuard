import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

idx = text.find("function togglePlay")
idx_end = text.find("function ", idx+1)
print("=== togglePlay ===")
print(text[idx:idx_end])

idx2 = text.find("function startLiveSweepLoop")
idx2_end = text.find("function ", idx2+1)
print("\n=== startLiveSweepLoop ===")
print(text[idx2:idx2_end])
