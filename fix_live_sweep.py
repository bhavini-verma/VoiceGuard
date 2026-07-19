import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

old_draw = """    playbackAnalyser.getByteFrequencyData(dataArray);
    const ratio = currentAudio.currentTime / currentAudio.duration;
    if (isNaN(ratio)) return;"""

new_draw = """    playbackAnalyser.getByteFrequencyData(dataArray);
    const dur = (currentAudio.duration && currentAudio.duration !== Infinity) ? currentAudio.duration : audioDuration;
    const ratio = currentAudio.currentTime / dur;
    if (isNaN(ratio)) return;"""

if old_draw in text:
    text = text.replace(old_draw, new_draw)
    with codecs.open(path, "w", "utf-8") as f:
        f.write(text)
    print("SUCCESS")
else:
    print("FAILED")
