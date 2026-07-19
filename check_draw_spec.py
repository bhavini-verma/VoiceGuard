import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

idx = text.find("drawSpectrogram(")
while idx != -1:
    start_idx = max(0, text.rfind("\n", 0, idx))
    end_idx = text.find("\n", idx)
    print("Found at", idx, ":")
    print(text[start_idx:end_idx].strip())
    idx = text.find("drawSpectrogram(", idx+1)
