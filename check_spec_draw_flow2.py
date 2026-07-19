import codecs
import sys

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Check the FULL drawSpectrogram function
idx = text.find("function drawSpectrogram")
idx_end = text.find("\nfunction ", idx+1)
sys.stdout.buffer.write(text[idx:idx_end].encode("utf-8"))
