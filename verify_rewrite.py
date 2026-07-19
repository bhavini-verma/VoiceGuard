import codecs, sys

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

idx = text.find("mr.onstop=()")
idx2 = text.find("function stopMic()", idx)
print("Block from", idx, "to", idx2)
sys.stdout.buffer.write(text[idx:idx2].encode("utf-8"))
