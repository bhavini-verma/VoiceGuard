import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Replace the deferred spectrogram draw with one that temporarily makes the
# canvas measurable by briefly setting visibility:hidden on the parent page
old_timeout = """          // Defer spectrogram draw so canvas has proper pixel dimensions after DOM settles
          const _ch = ch, _sr = buf.sampleRate, _nf = Math.min(150, Math.floor(ch.length / 256));
          setTimeout(function() {
            resizeSpec();
            drawSpectrogram(_ch, _sr, _nf);
          }, 100);"""

new_timeout = """          // Defer spectrogram draw so canvas has proper pixel dimensions.
          // page-intel may be display:none when user is on dashboard -- temporarily show it
          // as visibility:hidden (invisible but measurable) so offsetWidth > 0.
          const _ch = ch, _sr = buf.sampleRate, _nf = Math.min(150, Math.floor(ch.length / 256));
          setTimeout(function() {
            const pageIntel = document.getElementById('page-intel');
            const wasHidden = pageIntel && pageIntel.style.display === 'none';
            if (wasHidden) {
              pageIntel.style.visibility = 'hidden';
              pageIntel.style.display = 'block';
              pageIntel.style.position = 'absolute';
              pageIntel.style.top = '-9999px';
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

if old_timeout in text:
    text = text.replace(old_timeout, new_timeout)
    print("SUCCESS - replaced timeout block")
else:
    print("FAILED - old block not found!")
    # Try to find it
    idx = text.find("// Defer spectrogram draw so canvas has proper pixel dimensions after DOM settles")
    print("Marker at:", idx)

with codecs.open(path, "w", "utf-8") as f:
    f.write(text)
