import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

target = "currentAudio = new Audio(URL.createObjectURL(blob));"

injection = """
          currentAudio.crossOrigin = "anonymous";
          if (!playbackCtx) {
            playbackCtx = new (window.AudioContext || window.webkitAudioContext)();
            playbackAnalyser = playbackCtx.createAnalyser();
            playbackAnalyser.fftSize = 256;
          }
          try {
            playbackSource = playbackCtx.createMediaElementSource(currentAudio);
            playbackSource.connect(playbackAnalyser);
            playbackAnalyser.connect(playbackCtx.destination);
          } catch(e) { console.error("WebAudio playback setup failed", e); }
"""

idx = text.find(target)
if idx != -1:
    text = text[:idx + len(target)] + injection + text[idx + len(target):]
    with codecs.open(path, "w", "utf-8") as f:
        f.write(text)
    print("SUCCESS")
else:
    print("Target not found")
