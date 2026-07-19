with open('static/voiceguard_uco_bank_platform.html', 'r', encoding='utf-8') as f:
    content = f.read()

start = content.find('<div id="page-investigator"')
if start != -1:
    end = content.find('</div> <!-- end page-investigator -->', start)
    if end == -1:
        end = content.find('<div id="page-', start + 10)
    
    with open('scratch/investigator_section.txt', 'w', encoding='utf-8') as out:
        out.write(content[start:end+50])
    print('Extracted to scratch/investigator_section.txt')
else:
    print('Not found')
