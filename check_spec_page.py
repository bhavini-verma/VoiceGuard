import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Find spec-canvas and walk up to find the nearest page-xxx id
idx = text.find('id="spec-canvas"')
# Look for page-xxx in the preceding 50000 chars
preceding = text[:idx]
import re
pages = list(re.finditer(r'id="page-([^"]+)"', preceding))
if pages:
    last_page = pages[-1]
    print("Spectrogram canvas is inside:", last_page.group(0), "at pos", last_page.start())
    print("Context:", preceding[last_page.start():last_page.start()+100])

# Also check the waveform canvas
idx2 = text.find('id="wv-canvas"')
if idx2 == -1:
    idx2 = text.find('class="wave"')
print("\nWaveform canvas at:", idx2)
preceding2 = text[:idx2]
pages2 = list(re.finditer(r'id="page-([^"]+)"', preceding2))
if pages2:
    last_page2 = pages2[-1]
    print("Waveform canvas is inside:", last_page2.group(0))
