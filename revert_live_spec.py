import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Locate the injected block
start_marker = "// ==== LIVE SPECTROGRAM WATERFALL ===="
end_marker = "// ===================================="

idx_start = text.find(start_marker)
idx_end = text.find(end_marker)

if idx_start != -1 and idx_end != -1:
    text = text[:idx_start] + text[idx_end + len(end_marker):]
    with codecs.open(path, "w", "utf-8") as f:
        f.write(text)
    print("Reverted live spectrogram waterfall logic.")
else:
    print("Could not find the injected block to revert.")
