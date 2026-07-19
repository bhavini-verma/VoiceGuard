import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

idx = 209821
print("Context for 209821:")
print(text[max(0, idx-200):idx+200])

idx = 215351
print("Context for 215351:")
print(text[max(0, idx-200):idx+200])
