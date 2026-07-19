import codecs

path = r'c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html'
with codecs.open(path, 'r', 'utf-8') as f:
    text = f.read()

idx = text.find('async function sendFloatingMessage(')
if idx != -1:
    print('\nFound sendFloatingMessage at', idx)
    print(text[idx:idx+2500])
else:
    print('Could not find sendFloatingMessage, trying just "function send"')
    matches = [i for i in range(len(text)) if text.startswith('function send', i)]
    for i in matches:
        print(text[i:i+100])
