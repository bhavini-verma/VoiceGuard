import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Check the CSS for .spec class
import re
spec_css = re.findall(r'\.spec\s*\{[^}]*\}', text, re.DOTALL)
print("CSS for .spec:")
for s in spec_css:
    print(s)

# Check the parent container of spec-canvas - look for the 130px height div
idx = text.find("spec-canvas")
print("\nParent structure:")
print(text[max(0, idx-600):idx+50])
