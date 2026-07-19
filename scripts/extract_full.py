def extract_div(filepath, div_id):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    start_str = f'id="{div_id}"'
    start_idx = content.find(start_str)
    if start_idx == -1:
        return "Not found"
    
    # find the <div before the id
    div_start = content.rfind('<div', 0, start_idx)
    
    balance = 0
    i = div_start
    while i < len(content):
        if content[i:i+4] == '<div':
            balance += 1
            i += 4
        elif content[i:i+5] == '</div':
            balance -= 1
            i += 5
            if balance == 0:
                return content[div_start:i+1] # Include the >
        else:
            i += 1
    return "Unbalanced"

html = extract_div('static/voiceguard_uco_bank_platform.html', 'page-investigator')
with open('scratch/investigator_full.txt', 'w', encoding='utf-8') as f:
    f.write(html)
