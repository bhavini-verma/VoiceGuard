import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

idx = text.find("change'")
while idx != -1:
    print(text[max(0, idx-100):idx+500])
    idx = text.find("change'", idx+1)
