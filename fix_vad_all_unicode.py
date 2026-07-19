import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\dataset_tools\vad_split_genuine.py"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

text = text.replace(u"\u2192", "->")

with codecs.open(path, "w", "utf-8") as f:
    f.write(text)

print("Fixed ALL Unicode arrows in VAD script")
