import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Old inner loop - shows ALL bins including inaudible ultra-high freqs
old_inner = """  const frames=Math.max(1, numFrames||Math.floor(audioData.length/256));
  const fftSize=256,hopSize=Math.max(1, Math.floor((audioData.length-fftSize)/frames));
  const freqBins=fftSize/2;
  const frameW=w/frames,binH=h/freqBins;
  
  for(let f=0;f<frames;f++){
    const offset=f*hopSize;
    for(let b=0;b<freqBins;b++){
      let re=0, im=0;
      for(let i=0; i<fftSize; i++) {
        if (offset+i >= audioData.length) break;
        const window = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (fftSize - 1));
        const val = audioData[offset+i] * window;
        const angle = 2 * Math.PI * b * i / fftSize;
        re += val * Math.cos(angle);
        im -= val * Math.sin(angle);
      }
      const mag = Math.sqrt(re*re + im*im) / (fftSize/2);
      const norm=Math.min(1,Math.pow(mag*3.5,0.5)); 
      
      if (norm > 0.02) {
        sCtx.fillStyle=`rgba(15, 23, 42, ${norm * 0.8})`; 
        sCtx.fillRect(f*frameW,(freqBins-1-b)*binH,frameW+1,binH+1);
      }
    }
  }"""

# New version - caps at 8000 Hz max, fills the full canvas height regardless of sample rate
new_inner = """  const frames=Math.max(1, numFrames||Math.floor(audioData.length/256));
  const fftSize=256,hopSize=Math.max(1, Math.floor((audioData.length-fftSize)/frames));
  const allBins=fftSize/2;
  // Cap display at 8000 Hz (covers all speech), regardless of sample rate.
  // This fixes 48kHz live mic appearing blank (speech was in bottom 12% of canvas).
  const hzPerBin = sampleRate / fftSize;
  const maxDisplayHz = 8000;
  const freqBins = Math.min(allBins, Math.max(1, Math.ceil(maxDisplayHz / hzPerBin)));
  const frameW=w/frames,binH=h/freqBins;
  
  for(let f=0;f<frames;f++){
    const offset=f*hopSize;
    for(let b=0;b<freqBins;b++){
      let re=0, im=0;
      for(let i=0; i<fftSize; i++) {
        if (offset+i >= audioData.length) break;
        const window = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (fftSize - 1));
        const val = audioData[offset+i] * window;
        const angle = 2 * Math.PI * b * i / fftSize;
        re += val * Math.cos(angle);
        im -= val * Math.sin(angle);
      }
      const mag = Math.sqrt(re*re + im*im) / (fftSize/2);
      const norm=Math.min(1,Math.pow(mag*3.5,0.5)); 
      
      if (norm > 0.02) {
        sCtx.fillStyle=`rgba(15, 23, 42, ${norm * 0.8})`; 
        sCtx.fillRect(f*frameW,(freqBins-1-b)*binH,frameW+1,binH+1);
      }
    }
  }"""

if old_inner in text:
    text = text.replace(old_inner, new_inner)
    with codecs.open(path, "w", "utf-8") as f:
        f.write(text)
    print("SUCCESS - Fixed spectrogram frequency scaling")
else:
    print("FAILED - old block not found")
