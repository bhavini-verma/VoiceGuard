import re
content = open('static/voiceguard_uco_bank_platform.html', encoding='utf-8').read()
ids = re.findall(r'id="([a-zA-Z0-9_-]+)"', content)
for i in ids:
    print(i)
