import codecs

html_path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"

with codecs.open(html_path, "r", "utf-8") as f:
    content = f.read()

# Fix the switchPage logic to use 'block' for page-kb so it stacks vertically
content = content.replace("if(el) el.style.display = (p === pageId) ? (p === 'page-kb' ? 'flex' : 'block') : 'none';",
                          "if(el) el.style.display = (p === pageId) ? 'block' : 'none';")

# Fix the pill labels in JS
content = content.replace("label: 'Fraud Threat Vectors',", "label: 'Threat vectors',")
content = content.replace("label: 'Detection & Forensics',", "label: 'Forensics',")
content = content.replace("label: 'Security Procedures',", "label: 'Procedures',")
content = content.replace("label: 'RBI & Compliance',", "label: 'Compliance',")
content = content.replace("<span>All Articles</span>", "<span>All</span>")

# Fix the left icon logic in JS: definitions get NO icon on the left. SOP gets blue circle on the left.
old_left_icon = "const leftIcon = art.isSop ? `<div class=\"kb-sop-num\" style=\"background:var(--warn); margin-right:4px;\">${art.steps.length}</div>` : `<i class=\"ti ti-chevron-right\" style=\"color:var(--text3); font-size:16px;\"></i>`;"
new_left_icon = "const leftIcon = art.isSop ? `<div class=\"kb-sop-num\" style=\"margin-right:12px;\">${art.steps.length}</div>` : ``;"
content = content.replace(old_left_icon, new_left_icon)

# Fix the title HTML to remove the gap if there's no left icon
# We'll just keep the gap, it looks fine, but let's change kb-acc-title CSS slightly
content = content.replace(".kb-acc-title { font-size:14px; font-weight:500; color:var(--text); display:flex; align-items:center; gap:12px; }",
                          ".kb-acc-title { font-size:15px; font-weight:500; color:var(--text); display:flex; align-items:center; }")

# Change the Search Bar placeholder to match screenshot 2
content = content.replace('placeholder="How can we help you today? Search threats, SOPs, compliance..."',
                          'placeholder="How can we help you today?"')

# Remove the breadcrumb since Screenshot 2 just shows the search bar and pills
breadcrumb = '<div style="font-size:11px; color:var(--text2); margin-bottom:16px; font-weight:600;"><i class="ti ti-chevron-right" style="font-size:10px; margin-right:4px;"></i>Support / Knowledge Base</div>'
content = content.replace(breadcrumb, "")

# The "Contact fraud team" footer looks different in screenshot 2. It's just a simple text line.
old_footer = """<!-- Footer Strip -->
        <div style="margin-top:30px; padding:20px; background:#fff; border:1px solid var(--border); border-radius:8px; display:flex; align-items:center; justify-content:space-between;">
          <div>
            <div style="font-weight:600; font-size:14px; color:var(--text); margin-bottom:4px;">Still didn't find it?</div>
            <div style="font-size:12px; color:var(--text2);">Our specialized fraud intelligence teams are available 24/7.</div>
          </div>
          <div style="display:flex; gap:10px;">
            <button class="primary-btn" style="background:var(--bank); border:none; padding:8px 16px; border-radius:4px; color:#fff; font-size:12px; font-weight:600; cursor:pointer;"><i class="ti ti-headset"></i> Contact Fraud Cell</button>
            <button style="background:#fff; border:1px solid var(--border); padding:8px 16px; border-radius:4px; color:var(--text); font-size:12px; font-weight:600; cursor:pointer;"><i class="ti ti-building-bank"></i> Escalate to Manager</button>
          </div>
        </div>"""
        
new_footer = """<!-- Footer Strip -->
        <div style="margin-top:30px; padding:20px 24px; font-size:13px; color:var(--text2);">
          Can't find what you're looking for? <a href="#" style="color:var(--bank); font-weight:600; text-decoration:none;">Contact the fraud team &rarr;</a>
        </div>"""
content = content.replace(old_footer, new_footer)

# Fix section header text to uppercase and ensure it matches screenshot 2
content = content.replace("container.innerHTML += `<div class=\"kb-section-hdr\">${cat.label}</div>`;",
                          "let label = cat.label === 'Threat vectors' ? 'Fraud threat vectors' : (cat.label === 'Procedures' ? 'Security procedures' : cat.label);\n      container.innerHTML += `<div class=\"kb-section-hdr\">${label}</div>`;")

# Fix background color of the header section. In screenshot 2, it is very clean, looks like white.
# In my HTML: <div style="background:#fff; border-bottom:1px solid var(--border); padding:30px 40px; text-align:center;">
# That's fine.

with codecs.open(html_path, "w", "utf-8") as f:
    f.write(content)

print("SUCCESS")
