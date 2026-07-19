import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Check how specCanvas is defined
idx = text.find("specCanvas")
while idx != -1 and idx < 200000:
    line_start = max(0, text.rfind("\n", 0, idx))
    line_end = text.find("\n", idx)
    line = text[line_start:line_end].strip()
    if "let " in line or "const " in line or "var " in line or "=" in line and "spec" in line.lower():
        print(f"At {idx}: {line[:120]}")
    idx = text.find("specCanvas", idx+1)
