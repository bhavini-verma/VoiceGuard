import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

idx = text.find("function apiPost")
idx_end = text.find("\nfunction ", idx+1)
print("=== apiPost ===")
print(text[idx:idx_end])
