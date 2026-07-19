import codecs, sys

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Find STAGED_FILE and what happens to it
idx = text.find("function startStagedAnalysis")
idx_end = text.find("\nfunction ", idx+1)
print("=== startStagedAnalysis ===")
sys.stdout.buffer.write(text[idx:idx_end].encode("utf-8"))
print()
print()

# Find how micChunks WAV is assembled
idx2 = text.find("var wavBlob=audioBufferToWav")
print("=== WAV creation in mr.onstop ===")
print(text[max(0, idx2-100):idx2+300])
