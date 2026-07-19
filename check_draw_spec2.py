import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Check the drawSpectrogram function - specifically how sCtx is referenced
idx = text.find("const specCanvas=document.getElementById")
print("specCanvas init:")
print(text[idx:idx+200])

idx2 = text.find("const sCtx=specCanvas")
print("\nsCtx init:")
print(text[idx2:idx2+100])

# Also check if spec-canvas has a width attribute in HTML (could be 0)
idx3 = text.find("spec-canvas")
while idx3 != -1 and idx3 < 100000:
    print("\nspec-canvas HTML at", idx3, ":", text[max(0,idx3-50):idx3+200])
    idx3 = text.find("spec-canvas", idx3+1)
    if idx3 > 100000:
        break
