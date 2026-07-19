import codecs
import re

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Fix truncation at the end of the file
if text.strip().endswith("</script"):
    text = text.rstrip() + ">\n</body>\n</html>\n"

# Fix API_KEY reference errors
text = text.replace("'Authorization': 'Bearer ' + API_KEY", "'x-api-key': 'voiceguard123'")

with codecs.open(path, "w", "utf-8") as f:
    f.write(text)

print("SUCCESS")
