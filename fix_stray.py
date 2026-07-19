import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

text = text.replace("          </div>\n\n          >\n\n        </div>", "          </div>\n\n        </div>")

with codecs.open(path, "w", "utf-8") as f:
    f.write(text)

print("Fixed stray >")
