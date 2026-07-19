import codecs

html_path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"

with codecs.open(html_path, "r", "utf-8") as f:
    content = f.read()

content = content.replace(
    "const chat_session_id = crypto.randomUUID();",
    "const chat_session_id = (crypto && crypto.randomUUID) ? crypto.randomUUID() : Math.random().toString(36).substring(2, 15);"
)

with codecs.open(html_path, "w", "utf-8") as f:
    f.write(content)

print("SUCCESS")
