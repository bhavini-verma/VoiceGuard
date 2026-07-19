import re
content = open('static/voiceguard_uco_bank_platform.html', encoding='utf-8').read()
tab_start = content.find('id="tab-intel"')
tab_end = content.find('id="tab-stats"')
html = content[tab_start:tab_end]
print(re.findall(r'id="([a-zA-Z0-9_-]+)"', html))
