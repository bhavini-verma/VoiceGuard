with open('static/voiceguard_uco_bank_platform.html', 'r', encoding='utf-8') as f:
    content = f.read()

start = content.find('id="page-investigator"')
if start != -1:
    end = content.find('id="page-', start + 50)
    if end == -1:
        end = start + 8000
    with open('scratch/investigator_section.txt', 'w', encoding='utf-8') as out:
        out.write(content[max(0, start-100):end])
    print('Extracted')
else:
    print('Not found')
