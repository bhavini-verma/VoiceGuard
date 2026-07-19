import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

live_spec_code = """
    // ==== LIVE SPECTROGRAM WATERFALL ====
    const liveC = document.getElementById('spec-live-canvas');
    if (liveC) {
       if (liveC.width !== liveC.offsetWidth * devicePixelRatio) {
           liveC.width = liveC.offsetWidth * devicePixelRatio;
           liveC.height = liveC.offsetHeight * devicePixelRatio;
       }
       const lCtx = liveC.getContext('2d');
       analyser.getByteFrequencyData(freqData);
       
       const w = liveC.width;
       const h = liveC.height;
       const sliceW = 2 * devicePixelRatio;
       
       // Scroll left
       const imgData = lCtx.getImageData(sliceW, 0, w - sliceW, h);
       lCtx.putImageData(imgData, 0, 0);
       lCtx.clearRect(w - sliceW, 0, sliceW, h);
       
       // Draw new slice on right edge
       const bins = freqData.length;
       const binH = h / bins;
       for (let i = 0; i < bins; i++) {
           const v = freqData[i] / 255.0;
           if (v > 0.02) {
               lCtx.fillStyle = `rgba(15, 23, 42, ${Math.pow(v, 1.5)})`; // Non-linear curve to match static spectrogram density
               lCtx.fillRect(w - sliceW, h - (i * binH) - binH, sliceW, Math.ceil(binH));
           }
       }
    }
    // ====================================
"""

# Find where to inject it in visualizeMic
idx = text.find('analyser.getByteTimeDomainData(dataArray);')
if idx != -1:
    text = text[:idx] + live_spec_code + "\n    " + text[idx:]
    with codecs.open(path, "w", "utf-8") as f:
        f.write(text)
    print("SUCCESS")
else:
    print("Could not find injection point")
