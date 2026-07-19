import codecs
import re

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Replace any > that is sitting on its own line surrounded by whitespace
new_text = re.sub(r'</div>\s*>\s*</div>\s*</div> <!-- end grid2 -->', r'</div>\n        </div>\n\n      </div> <!-- end grid2 -->', text)

if text != new_text:
    print("Replaced!")
    with codecs.open(path, "w", "utf-8") as f:
        f.write(new_text)
else:
    print("Not matched!")
