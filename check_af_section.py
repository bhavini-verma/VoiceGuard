import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Find the enclosing section element that might be hidden
idx = text.find('id="spec-canvas"')
# Walk back to find parent section/div with id or class
parent_search = text[:idx]
# look for section or id near 73000-76000
print("Looking for enclosing nav panel...")
for keyword in ["audio-forensics", "waveform-section", "section-", "id=\"af-", "id=\"main-content", "view-detect", "detect-page"]:
    kidx = parent_search.rfind(keyword)
    if kidx != -1:
        print(f"Found {keyword} at {kidx}: {text[kidx:kidx+200]}")
