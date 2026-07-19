import codecs

html_path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"

with codecs.open(html_path, "r", "utf-8") as f:
    content = f.read()

# Fix CSS sizes
content = content.replace('.kb-pill { padding:8px 16px;', '.kb-pill { padding:6px 12px;')
content = content.replace('.kb-pill { padding:6px 12px; display:flex; align-items:center; gap:6px; font-size:12px;', 
                          '.kb-pill { padding:6px 12px; display:flex; align-items:center; gap:6px; font-size:11px;')

content = content.replace('.kb-section-hdr { font-size:11px; font-weight:700; color:var(--text3); text-transform:uppercase; letter-spacing:0.5px; padding:22px 24px 10px 24px;',
                          '.kb-section-hdr { font-size:10px; font-weight:700; color:var(--text3); text-transform:uppercase; letter-spacing:0.5px; padding:14px 16px 8px 16px;')

content = content.replace('.kb-acc-hdr { padding:18px 24px;', '.kb-acc-hdr { padding:12px 16px;')
content = content.replace('.kb-acc-title { font-size:15px;', '.kb-acc-title { font-size:12px;')
content = content.replace('.kb-acc-icon { font-size:16px;', '.kb-acc-icon { font-size:14px;')

content = content.replace('.kb-acc-body { padding:0 24px 24px 24px;', '.kb-acc-body { padding:0 16px 16px 16px;')
content = content.replace('.kb-acc-ans { font-size:13px; color:var(--text2); line-height:1.6; margin-bottom:16px;',
                          '.kb-acc-ans { font-size:11px; color:var(--text2); line-height:1.6; margin-bottom:12px;')

content = content.replace('.kb-sop-list { display:flex; flex-direction:column; gap:12px; margin-bottom:20px;',
                          '.kb-sop-list { display:flex; flex-direction:column; gap:8px; margin-bottom:14px;')
content = content.replace('.kb-sop-num { width:20px; height:20px;', '.kb-sop-num { width:16px; height:16px;')
content = content.replace('font-size:10px; font-weight:700; flex-shrink:0; font-family:var(--mono); margin-top:1px; }',
                          'font-size:9px; font-weight:700; flex-shrink:0; font-family:var(--mono); margin-top:1px; }')
content = content.replace('.kb-sop-text { font-size:13px;', '.kb-sop-text { font-size:11px;')

content = content.replace('.kb-acc-meta { margin-left:28px; font-family:var(--mono); font-size:10px;',
                          '.kb-acc-meta { margin-left:28px; font-family:var(--mono); font-size:9px;')
content = content.replace('.kb-acc-feedback { margin-left:28px; font-size:11px;',
                          '.kb-acc-feedback { margin-left:28px; font-size:10px;')

# Fix HTML layout paddings and search bar
content = content.replace('<div style="background:#fff; border-bottom:1px solid var(--border); padding:30px 40px; text-align:center;">',
                          '<div style="background:#fff; border-bottom:1px solid var(--border); padding:20px 24px; text-align:center;">')

content = content.replace('<div style="padding:40px; width:100%;">',
                          '<div style="padding:24px; width:100%;">')

content = content.replace('padding:14px 14px 14px 44px; border:1px solid var(--border); border-radius:8px; font-size:14px;',
                          'padding:10px 10px 10px 34px; border:1px solid var(--border); border-radius:6px; font-size:12px;')

content = content.replace('font-size:18px;"></i>\n            <input',
                          'font-size:15px; left:12px;"></i>\n            <input')

with codecs.open(html_path, "w", "utf-8") as f:
    f.write(content)

print("SUCCESS")
