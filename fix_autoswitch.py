import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Replace the setTimeout block with a proper fix:
# 1. Remove the hacky off-screen approach
# 2. Instead, just call switchPage('page-intel') which already handles canvas resize+redraw

old_timeout = """          // Defer spectrogram draw so canvas has proper dimensions.
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

new_timeout = """          // switchPage('page-intel') already has a canvas-resize+redraw handler built in.
          // Navigate to Detection Intel so the spectrogram is rendered on a visible canvas.
          const navIntel = document.getElementById('nav-intel');
          switchPage('page-intel', navIntel);"""

if old_timeout in text:
    text = text.replace(old_timeout, new_timeout)
    with codecs.open(path, "w", "utf-8") as f:
        f.write(text)
    print("SUCCESS")
else:
    print("FAILED - block not found")
    idx = text.find("// Defer spectrogram draw so canvas has proper dimensions.")
    print("Marker found at:", idx)
    print(repr(text[idx:idx+200]))
