import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

idx = text.find('function visualizeMic')
idx2 = text.find('function startMicTimer', idx)
print(text[max(0, idx+500):idx2])
