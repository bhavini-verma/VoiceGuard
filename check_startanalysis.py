import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# 2. Check how startAnalysis sends the file
idx = text.find("function startAnalysis")
idx_end = text.find("\nfunction ", idx+1)
print("=== startAnalysis ===")
print(text[idx:idx_end])
