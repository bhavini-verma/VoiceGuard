import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

idx_start = text.find("mr.onstop=()")
idx_end = text.find("function stopMic()", idx_start)
print("START:", idx_start)
print("END:", idx_end)
print("BLOCK:")
import sys
sys.stdout.buffer.write(text[idx_start:idx_end].encode("utf-8"))
