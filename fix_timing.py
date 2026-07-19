import codecs
import re

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# The problem: drawSpectrogram is called BEFORE setUploadZoneState('loaded') which
# triggers UI changes. Move drawSpectrogram AFTER setUploadZoneState with a setTimeout
# to ensure the canvas has proper dimensions once the DOM has settled.

# Find the mr.onstop block and replace the drawSpectrogram call location
old_block = """          waveData=wd;drawWave(waveData,0);
          audioDuration = buf.duration;
          drawSpectrogram(ch,buf.sampleRate,Math.min(150,Math.floor(ch.length/256)));
          
          document.getElementById('wv-dur').textContent=buf.duration.toFixed(1)+'s';
          document.getElementById('wv-sr').textContent=buf.sampleRate+'Hz';
          document.getElementById('wv-ch').textContent=buf.numberOfChannels;
          const specDurLabel = document.getElementById('spec-dur-label');
          if(specDurLabel) specDurLabel.textContent=buf.duration.toFixed(1)+'s';
          
          document.getElementById('wv-stat').textContent='STAGED - READY';
          document.getElementById('wv-stat').style.color='var(--success)';"""

new_block = """          waveData=wd;drawWave(waveData,0);
          audioDuration = buf.duration;
          
          document.getElementById('wv-dur').textContent=buf.duration.toFixed(1)+'s';
          document.getElementById('wv-sr').textContent=buf.sampleRate+'Hz';
          document.getElementById('wv-ch').textContent=buf.numberOfChannels;
          const specDurLabel = document.getElementById('spec-dur-label');
          if(specDurLabel) specDurLabel.textContent=buf.duration.toFixed(1)+'s';
          
          document.getElementById('wv-stat').textContent='STAGED - READY';
          document.getElementById('wv-stat').style.color='var(--success)';"""

if old_block in text:
    text = text.replace(old_block, new_block)
    print("Removed early drawSpectrogram call")
else:
    print("OLD BLOCK NOT FOUND, trying partial match")

# Now find the STAGED_FILE assignment (just before end of onstop) and add the deferred draw there
old_end = """          STAGED_FILE = f;
          
          setUploadZoneState('loaded', f.name, formatSize(f.size) + ' \\ufffd PCM/WAV \\ufffd ' + buf.duration.toFixed(1) + 's');
          setHdrChip('ready');
          document.getElementById('clear-btn').style.display='flex';
          document.getElementById('start-btn').style.display='flex';"""

if old_end in text:
    new_end = old_end + """
          // Defer spectrogram draw until AFTER the UI has settled (canvas must have pixel dimensions)
          setTimeout(() => {
            resizeSpec();
            drawSpectrogram(ch, buf.sampleRate, Math.min(150, Math.floor(ch.length / 256)));
          }, 100);"""
    text = text.replace(old_end, new_end)
    print("Added deferred spectrogram draw after setUploadZoneState")
else:
    # Try without the special chars
    search = "STAGED_FILE = f;\n          \n          setUploadZoneState"
    idx = text.find(search)
    if idx != -1:
        insert_pos = text.find("document.getElementById('start-btn').style.display='flex';", idx) + len("document.getElementById('start-btn').style.display='flex';")
        deferred = """
          // Defer spectrogram draw until AFTER the UI has settled (canvas must have pixel dimensions)
          setTimeout(() => {
            resizeSpec();
            drawSpectrogram(ch, buf.sampleRate, Math.min(150, Math.floor(ch.length / 256)));
          }, 100);"""
        text = text[:insert_pos] + deferred + text[insert_pos:]
        print("Added deferred spectrogram draw (fallback path)")
    else:
        print("FALLBACK ALSO FAILED")

with codecs.open(path, "w", "utf-8") as f:
    f.write(text)
