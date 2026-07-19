import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

idx = text.find('currentAudio.addEventListener(\'ended\'')
while idx != -1:
    print("Found at", idx)
    print(text[idx:idx+1500])
    idx = text.find('currentAudio.addEventListener(\'ended\'', idx+1)
