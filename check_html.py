import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

idx = text.find("floating-chat-widget")
print("floating-chat-widget found at:", idx)

# let's look at the Javascript around the button again
idx2 = text.find("badge.addEventListener('click'")
if idx2 != -1:
    print(text[max(0, idx2-300):idx2+500])
