# -*- coding: utf-8 -*-
import codecs

html_path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"

with codecs.open(html_path, "r", "utf-8") as f:
    content = f.read()

# Replace KB_DATA
start_idx = content.find("const KB_DATA = [")
end_idx = content.find("let currentKBCat = 'all';", start_idx)

replacement = """let KB_DATA = [];
const chat_session_id = crypto.randomUUID();

// Fetch KB_DATA
fetch('/static/kb_data.json')
  .then(res => res.json())
  .then(data => {
    KB_DATA = data;
    renderKB();
  })
  .catch(err => console.error("Failed to load KB Data:", err));
"""

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + replacement + "\n" + content[end_idx:]

# Update the chat payload to include session_id
content = content.replace("const payload = { message: text };", "const payload = { message: text, session_id: chat_session_id };")

with codecs.open(html_path, "w", "utf-8") as f:
    f.write(content)

print("SUCCESS")
