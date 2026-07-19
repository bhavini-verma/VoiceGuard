import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Inject a visible debug panel into the DOM
debug_html = """
<div id="vg-debug" style="position:fixed;bottom:80px;right:20px;z-index:99998;background:#1e293b;color:#f8fafc;font-family:monospace;font-size:11px;padding:10px 14px;border-radius:8px;max-width:320px;display:none;border:2px solid #ef4444;">
  <div style="font-weight:700;margin-bottom:6px;color:#ef4444;">&#128270; MIC DEBUG LOG</div>
  <div id="vg-debug-log" style="max-height:200px;overflow-y:auto;"></div>
  <button onclick="document.getElementById('vg-debug').style.display='none'" style="margin-top:6px;background:#ef4444;border:none;color:#fff;padding:2px 8px;border-radius:4px;cursor:pointer;">Close</button>
</div>
"""

debug_js = """
<script>
function vgDebug(msg, color) {
  const panel = document.getElementById('vg-debug');
  const log = document.getElementById('vg-debug-log');
  if (panel && log) {
    panel.style.display = 'block';
    const el = document.createElement('div');
    el.style.color = color || '#a5f3fc';
    el.style.marginBottom = '2px';
    el.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
  }
}
</script>
"""

# Insert debug panel + js right before </body>
body_end = text.rfind("</body>")
if body_end != -1:
    text = text[:body_end] + debug_html + debug_js + text[body_end:]
    print("Inserted debug panel")
else:
    print("No </body> found!")

# Now inject debug calls into mr.onstop
# After mr.onstop=()=>{
target = "mr.onstop=()=>{"
text = text.replace(target, target + "\n      vgDebug('STEP 1: onstop fired, micChunks=' + micChunks.length, '#facc15');", 1)

# After blob creation
target2 = "const blob=new Blob(micChunks,{type:'audio/webm'});"
text = text.replace(target2, target2 + "\n      vgDebug('STEP 2: blob created size=' + blob.size, '#facc15');", 1)

# After reader.onload
target3 = "reader.onload=ev=>{"
text = text.replace(target3, target3 + "\n        vgDebug('STEP 3: FileReader loaded', '#facc15');", 1)

# After decodeAudioData success
target4 = "audioCtx.decodeAudioData(ev.target.result,buf=>{"
text = text.replace(target4, target4 + "\n          vgDebug('STEP 4: decodeAudioData SUCCESS, dur=' + buf.duration.toFixed(2) + 's, sr=' + buf.sampleRate, '#4ade80');", 1)

# After wavBlob
target5 = "const wavBlob = audioBufferToWav(buf);"
text = text.replace(target5, target5 + "\n          vgDebug('STEP 5: WAV blob created size=' + wavBlob.size, '#4ade80');", 1)

# Before setTimeout drawSpectrogram
target6 = "// Defer spectrogram draw so canvas has proper pixel dimensions after DOM settles"
text = text.replace(target6, "vgDebug('STEP 6: about to draw spectrogram, offsetWidth=' + specCanvas.offsetWidth + ' h=' + specCanvas.offsetHeight, '#f472b6');\n          " + target6, 1)

# In the setTimeout
target7 = "resizeSpec();\n            drawSpectrogram(_ch, _sr, _nf);"
text = text.replace(target7, target7 + "\n            vgDebug('STEP 7: spectrogram drawn! w=' + specCanvas.width + ' h=' + specCanvas.height, '#4ade80');", 1)

# decode error callback
target8 = "},()=>{document.getElementById('wv-stat').textContent='DECODE ERR';})"
text = text.replace(target8, "},()=>{document.getElementById('wv-stat').textContent='DECODE ERR'; vgDebug('STEP 4 FAILED: decodeAudioData ERROR!', '#ef4444');})", 1)

with codecs.open(path, "w", "utf-8") as f:
    f.write(text)

print("SUCCESS - debug instrumentation injected!")
