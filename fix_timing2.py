import codecs
import re

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Find the onstop block to verify current state
idx = text.find("mr.onstop")
idx2 = text.find("function stopMic()", idx)
block = text[idx:idx2]
print("CURRENT mr.onstop block:")
print(block[:3000])
