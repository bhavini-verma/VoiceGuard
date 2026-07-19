import codecs, sys

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\fastapi_app.py"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Find simulate_phone_codec
idx = text.find("simulate_phone_codec")
print(text[max(0, idx-100):idx+500])

print("---")
# Find y_for_deep
idx2 = text.find("y_for_deep")
print(text[max(0, idx2-100):idx2+500])
