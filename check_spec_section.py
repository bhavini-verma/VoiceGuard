import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Find the spec-canvas HTML element and see what containers it lives in
idx = text.find('id="spec-canvas"')
print("spec-canvas at:", idx)
print(text[max(0, idx-2000):idx+100])
