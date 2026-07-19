import codecs
import re

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# The block to remove:
old_block = """          // switchPage('page-intel') already has a canvas-resize+redraw handler built in.
          // Navigate to Detection Intel so the spectrogram is rendered on a visible canvas.
          const navIntel = document.getElementById('nav-intel');
          switchPage('page-intel', navIntel);"""

if old_block in text:
    text = text.replace(old_block, "")
    with codecs.open(path, "w", "utf-8") as f:
        f.write(text)
    print("SUCCESS")
else:
    print("FAILED TO FIND BLOCK")
    idx = text.find("switchPage('page-intel'")
    print(text[max(0, idx-200):idx+200])
