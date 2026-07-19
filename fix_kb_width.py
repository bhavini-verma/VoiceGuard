import codecs

html_path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"

with codecs.open(html_path, "r", "utf-8") as f:
    content = f.read()

# Fix search bar width
content = content.replace('<div style="position:relative; width:100%; max-width:680px;">',
                          '<div style="position:relative; width:100%;">')

# Fix KB body container width
content = content.replace('<div style="padding:40px; max-width:800px; margin:0 auto;">',
                          '<div style="padding:40px; width:100%;">')

with codecs.open(html_path, "w", "utf-8") as f:
    f.write(content)

print("SUCCESS")
