import codecs
import re

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Remove the debug panel HTML
text = re.sub(r'\n<div id="vg-debug".*?</div>\n', '', text, flags=re.DOTALL)

# Remove the vgDebug script tag
text = re.sub(r'\n<script>\nfunction vgDebug.*?</script>\n', '', text, flags=re.DOTALL)

# Remove all vgDebug calls from mr.onstop
text = re.sub(r"\n\s*vgDebug\('[^']*'[^;]*;", '', text)

# Also remove window.onerror debug panel from earlier
text = re.sub(r'\n<script>\n  window\.onerror = function\(msg.*?</script>\n', '', text, flags=re.DOTALL)

with codecs.open(path, "w", "utf-8") as f:
    f.write(text)
print("Cleaned up debug code")
