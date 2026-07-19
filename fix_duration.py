import codecs
import re

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Fix timeupdate listener in mr.onstop
pattern1 = r"wavePos = audio\.currentTime / audio\.duration;"
replacement1 = "const dur = (audio.duration && audio.duration !== Infinity) ? audio.duration : buf.duration;\n              wavePos = audio.currentTime / dur;"

# Set global audioDuration in mr.onstop
pattern2 = r"waveData=wd;drawWave\(waveData,0\);"
replacement2 = "waveData=wd;drawWave(waveData,0);\n          audioDuration = buf.duration;"

text = re.sub(pattern1, replacement1, text)
text = text.replace("waveData=wd;drawWave(waveData,0);", "waveData=wd;drawWave(waveData,0);\n          audioDuration = buf.duration;")

# Fix startLiveSweepLoop
pattern3 = r"const resumeX = Math\.floor\(\(currentAudio\.currentTime / currentAudio\.duration\) \* w\);"
replacement3 = "const dur = (currentAudio.duration && currentAudio.duration !== Infinity) ? currentAudio.duration : audioDuration;\n    const resumeX = Math.floor((currentAudio.currentTime / dur) * w);"
text = re.sub(pattern3, replacement3, text)

# Fix timeupdate in handleFile just in case
# Wait, pattern1 already replaced ALL occurrences in the file! (which is good)

with codecs.open(path, "w", "utf-8") as f:
    f.write(text)
print("SUCCESS")
