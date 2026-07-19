import codecs

path = r'c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html'
with codecs.open(path, 'r', 'utf-8') as f:
    text = f.read()

idx = text.find('id="chat-window"')
if idx != -1:
    print('Found chat window at', idx)
    print(text[max(0, idx-50):idx+1500])

idx_js = text.find('async function sendMessage(')
if idx_js != -1:
    print('\nFound JS at', idx_js)
    print(text[idx_js:idx_js+2000])
