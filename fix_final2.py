import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

old_block = "// Defer spectrogram draw so canvas has proper pixel dimensions after DOM settles\n          const _ch = ch, _sr = buf.sampleRate, _nf = Math.min(150, Math.floor(ch.length / 256));\n          setTimeout(function() {\n            resizeSpec();\n            drawSpectrogram(_ch, _sr, _nf);\n            vgDebug('STEP 7: spectrogram drawn! w=' + specCanvas.width + ' h=' + specCanvas.height, '#4ade80');\n          }, 100);"

new_block = """// Defer spectrogram draw so canvas has proper dimensions.
          // page-intel is display:none when user is on Dashboard tab, so specCanvas.offsetWidth = 0.
          // Fix: briefly make it measurable (off-screen, invisible) then restore.
          const _ch = ch, _sr = buf.sampleRate, _nf = Math.min(150, Math.floor(ch.length / 256));
          setTimeout(function() {
            const pageIntel = document.getElementById('page-intel');
            const wasHidden = pageIntel && pageIntel.style.display === 'none';
            if (wasHidden) {
              pageIntel.style.visibility = 'hidden';
              pageIntel.style.display = 'block';
              pageIntel.style.position = 'absolute';
              pageIntel.style.top = '-99999px';
            }
            resizeSpec();
            drawSpectrogram(_ch, _sr, _nf);
            if (wasHidden) {
              pageIntel.style.display = 'none';
              pageIntel.style.visibility = '';
              pageIntel.style.position = '';
              pageIntel.style.top = '';
            }
          }, 100);"""

if old_block in text:
    text = text.replace(old_block, new_block)
    with codecs.open(path, "w", "utf-8") as f:
        f.write(text)
    print("SUCCESS")
else:
    print("FAILED - not found")
    # show the exact text there
    idx = text.find("Defer spectrogram draw so canvas")
    print(repr(text[idx:idx+50]))
