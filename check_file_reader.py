import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

idx = text.find("FileReader")
while idx != -1:
    print(text[max(0, idx-50):idx+1000])
    idx = text.find("FileReader", idx+1)
