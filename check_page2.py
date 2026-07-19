import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Find the detect page and whether the waveform/spectrogram section has a page-id
idx = text.find("page-dashboard")
if idx != -1:
    print("page-dashboard at:", idx)
    print(text[max(0, idx-50):idx+2000])
