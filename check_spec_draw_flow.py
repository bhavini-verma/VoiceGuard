import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Check the FULL drawSpectrogram function
idx = text.find("function drawSpectrogram")
idx_end = text.find("\nfunction ", idx+1)
print("=== drawSpectrogram ===")
print(text[idx:idx_end])

# Check what switchPage does for page-intel
idx2 = text.find("function switchPage")
idx2_end = text.find("\nfunction ", idx2+1)
print("\n=== switchPage ===")
print(text[idx2:idx2_end])
