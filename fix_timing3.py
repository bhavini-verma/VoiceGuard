import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Find exact position of start-btn display line inside mr.onstop
idx_onstop = text.find("mr.onstop")
idx_fn_stop = text.find("function stopMic()", idx_onstop)

# Within the block, find the start-btn line
block = text[idx_onstop:idx_fn_stop]
start_btn_idx = block.rfind("document.getElementById('start-btn').style.display='flex';")
if start_btn_idx != -1:
    # Absolute position
    abs_pos = idx_onstop + start_btn_idx + len("document.getElementById('start-btn').style.display='flex';")
    
    deferred = """
          // Defer spectrogram draw so canvas has proper pixel dimensions after DOM settles
          const _ch = ch, _sr = buf.sampleRate, _nf = Math.min(150, Math.floor(ch.length / 256));
          setTimeout(function() {
            resizeSpec();
            drawSpectrogram(_ch, _sr, _nf);
          }, 100);"""
    
    text = text[:abs_pos] + deferred + text[abs_pos:]
    with codecs.open(path, "w", "utf-8") as f:
        f.write(text)
    print("SUCCESS - injected deferred spectrogram draw at pos", abs_pos)
else:
    print("FAILED to find start-btn line in mr.onstop block")
