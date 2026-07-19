import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Find where the file is sent to API
idx = text.find("/api/v1/analyze")
print("Found /api/v1/analyze at:", idx)
print(text[max(0, idx-500):idx+500])

print("\n\n---\n")
idx2 = text.find("formData")
print("Found formData at:", idx2)
print(text[max(0, idx2-200):idx2+500])

print("\n\n---\n")
idx3 = text.find("start-btn")
while idx3 != -1:
    print(text[max(0, idx3-30):idx3+100])
    print("---")
    idx3 = text.find("start-btn", idx3+1)
    if idx3 > len(text) - 100:
        break
