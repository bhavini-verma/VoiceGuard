import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

idx1 = text.find("function startLiveSweepLoop() {")
idx2 = text.find("\n// === END FORENSIC OVERLAY STATE ===")

if idx1 != -1 and idx2 != -1:
    old_func = text[idx1:idx2]
    
    new_func = """function startLiveSweepLoop() {
  if (!playbackAnalyser || !currentAudio) return;
  const c = document.getElementById('spec-live-canvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  
  // On resume from pause, preserve existing pixels by NOT resizing.
  // We only set canvas dimensions if it is uninitialized (default 300 width) or a fresh playback.
  const w = Math.floor(c.offsetWidth * devicePixelRatio);
  const h = Math.floor(c.offsetHeight * devicePixelRatio);
  
  if (c.width === 0 || c.width === 300 || currentAudio.currentTime < 0.1) {
    c.width = w;
    c.height = h;
    lastDrawX = 0;
  }
  
  const dataArray = new Uint8Array(playbackAnalyser.frequencyBinCount);
  const binH = h / dataArray.length;
  
  // On resume or scrub, sync lastDrawX to where we currently are in time
  if (currentAudio.currentTime > 0 && currentAudio.duration > 0) {
    const dur = (currentAudio.duration && currentAudio.duration !== Infinity) ? currentAudio.duration : audioDuration;
    const resumeX = Math.floor((currentAudio.currentTime / dur) * w);
    if (lastDrawX === 0 || Math.abs(lastDrawX - resumeX) > 10) {
      lastDrawX = resumeX;
    }
  }
  
  function draw() {
    if (currentAudio.paused || currentAudio.ended) return;
    liveAnimId = requestAnimationFrame(draw);
    
    playbackAnalyser.getByteFrequencyData(dataArray);
    const dur = (currentAudio.duration && currentAudio.duration !== Infinity) ? currentAudio.duration : audioDuration;
    const ratio = currentAudio.currentTime / dur;
    if (isNaN(ratio)) return;
    const currentX = Math.floor(ratio * w);
    
    // Draw columns between lastDrawX and currentX to prevent gaps
    const startX = Math.max(lastDrawX + 1, 0);
    if (currentX >= startX) {
      for (let x = startX; x <= currentX; x++) {
        for (let i = 0; i < dataArray.length; i++) {
          const v = dataArray[i] / 255.0;
          if (v > 0.05) {
            const rgb = getSpecColor(v);
            ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.9)`;
            ctx.fillRect(x, h - (i+1)*binH, 2, Math.ceil(binH));
          }
        }
      }
    }
    lastDrawX = currentX;
  }
  draw();
}"""
    
    text = text[:idx1] + new_func + text[idx2:]
    with codecs.open(path, "w", "utf-8") as f:
        f.write(text)
    print("SUCCESS")
else:
    print("FAILED")
