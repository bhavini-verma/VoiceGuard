import codecs
import re

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Find and show the exact content around the freqBins declaration
idx = text.find("const freqBins=fftSize/2;")
print("Found freqBins line at:", idx)
print(repr(text[max(0, idx-200):idx+300]))
