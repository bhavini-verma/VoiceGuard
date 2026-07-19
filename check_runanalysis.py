import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Find runAnalysis
idx = text.find("function runAnalysis")
idx_end = text.find("\nfunction ", idx+1)
print("=== runAnalysis ===")
print(text[idx:idx_end])
