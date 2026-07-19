import codecs

html_path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"

with codecs.open(html_path, "r", "utf-8") as f:
    content = f.read()

# CSS Update
css_new = """
.kb-pill { padding:8px 16px; display:flex; align-items:center; gap:6px; font-size:12px; color:var(--text2); cursor:pointer; background:#fff; border:1px solid var(--border); border-radius:20px; transition:all 0.2s; font-weight:500; }
.kb-pill:hover { background:var(--bg2); color:var(--bank); border-color:#b5cde8; }
.kb-pill.active { background:#F0F7FF; color:var(--bank); border-color:var(--bank); font-weight:600; }
.kb-pill-count { font-size:10px; color:var(--text3); font-weight:600; }
.kb-pill.active .kb-pill-count { color:var(--bank); opacity:0.8; }

.kb-section-hdr { font-size:11px; font-weight:700; color:var(--text3); text-transform:uppercase; letter-spacing:0.5px; padding:22px 24px 10px 24px; border-bottom:1px solid var(--border); background:var(--bg); }

.kb-acc-row { border-bottom:1px solid var(--border); background:#fff; transition:background 0.2s; }
.kb-acc-row:last-child { border-bottom:none; }
.kb-acc-row.expanded { background:#FAFAFA; }

.kb-acc-hdr { padding:18px 24px; display:flex; justify-content:space-between; align-items:flex-start; cursor:pointer; transition:background 0.2s; }
.kb-acc-hdr:hover { background:var(--bg2); }
.kb-acc-title { font-size:14px; font-weight:500; color:var(--text); display:flex; align-items:center; gap:12px; }
.kb-acc-icon { font-size:16px; color:var(--text3); transition:transform 0.3s cubic-bezier(0.4, 0, 0.2, 1); margin-top:2px; }
.kb-acc-body { padding:0 24px 24px 24px; display:none; }
.kb-acc-ans { font-size:13px; color:var(--text2); line-height:1.6; margin-bottom:16px; padding-left:28px; }

.kb-sop-list { display:flex; flex-direction:column; gap:12px; margin-bottom:20px; padding-left:28px; }
.kb-sop-item { display:flex; align-items:flex-start; gap:12px; }
.kb-sop-num { width:20px; height:20px; border-radius:50%; background:var(--bank); color:#fff; display:flex; align-items:center; justify-content:center; font-size:10px; font-weight:700; flex-shrink:0; font-family:var(--mono); margin-top:1px; }
.kb-sop-text { font-size:13px; color:var(--text); line-height:1.5; font-weight:500; }

.kb-acc-meta { margin-left:28px; font-family:var(--mono); font-size:10px; color:var(--text3); display:flex; align-items:center; gap:6px; border-top:1px dashed var(--border); padding-top:12px; }
.kb-acc-feedback { margin-left:28px; font-size:11px; color:var(--text3); display:flex; align-items:center; gap:8px; margin-top:12px; }
.kb-acc-feedback i { cursor:pointer; padding:4px; border-radius:4px; transition:all 0.2s; }
.kb-acc-feedback i:hover { background:var(--bg2); color:var(--bank); }
</style>
"""

start_css = content.find('.kb-nav-item')
end_css = content.find('</style>', start_css)
if start_css != -1 and end_css != -1:
    content = content[:start_css] + css_new[1:] + content[end_css+8:]

# HTML Update
html_new = """    <div id="page-kb" class="page-section" style="display:none; height:calc(100vh - 60px); overflow-y:auto; padding:0; background:var(--bg);">
      
      <!-- KB Header / Search & Pills -->
      <div style="background:#fff; border-bottom:1px solid var(--border); padding:30px 40px; text-align:center;">
        <div style="font-size:11px; color:var(--text2); margin-bottom:16px; font-weight:600;"><i class="ti ti-chevron-right" style="font-size:10px; margin-right:4px;"></i>Support / Knowledge Base</div>
        
        <div style="display:flex; justify-content:center; align-items:center; margin-bottom:20px;">
          <div style="position:relative; width:100%; max-width:680px;">
            <i class="ti ti-search" style="position:absolute; left:16px; top:50%; transform:translateY(-50%); color:var(--text3); font-size:18px;"></i>
            <input type="text" id="kb-search-input" placeholder="How can we help you today? Search threats, SOPs, compliance..." style="width:100%; padding:14px 14px 14px 44px; border:1px solid var(--border); border-radius:8px; font-size:14px; font-family:var(--sans); color:var(--text); outline:none; transition:all 0.2s; box-shadow:0 2px 8px rgba(0,0,0,0.02);" onkeyup="filterKB()">
          </div>
        </div>
        
        <!-- Horizontal Category Pills -->
        <div id="kb-nav-list" style="display:flex; justify-content:center; gap:8px; flex-wrap:wrap;">
           <!-- Injected by JS -->
        </div>
      </div>

      <!-- KB Body: Single Pane -->
      <div style="padding:40px; max-width:800px; margin:0 auto;">
        <div id="kb-accordions-container" style="background:#fff; border:1px solid var(--border); border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.02);">
          <!-- Accordions injected by JS -->
        </div>
        
        <!-- Footer Strip -->
        <div style="margin-top:30px; padding:20px; background:#fff; border:1px solid var(--border); border-radius:8px; display:flex; align-items:center; justify-content:space-between;">
          <div>
            <div style="font-weight:600; font-size:14px; color:var(--text); margin-bottom:4px;">Still didn't find it?</div>
            <div style="font-size:12px; color:var(--text2);">Our specialized fraud intelligence teams are available 24/7.</div>
          </div>
          <div style="display:flex; gap:10px;">
            <button class="primary-btn" style="background:var(--bank); border:none; padding:8px 16px; border-radius:4px; color:#fff; font-size:12px; font-weight:600; cursor:pointer;"><i class="ti ti-headset"></i> Contact Fraud Cell</button>
            <button style="background:#fff; border:1px solid var(--border); padding:8px 16px; border-radius:4px; color:var(--text); font-size:12px; font-weight:600; cursor:pointer;"><i class="ti ti-building-bank"></i> Escalate to Manager</button>
          </div>
        </div>
      </div>
    </div>
"""
start_kb = content.find('<div id="page-kb"')
end_kb = content.find('<div id="page-audit"', start_kb)
if start_kb != -1 and end_kb != -1:
    content = content[:start_kb] + html_new + content[end_kb:]


# JS Update
js_new = """function renderKBNav() {
  const nav = document.getElementById('kb-nav-list');
  if(!nav) return;
  
  const total = KB_DATA.reduce((sum, cat) => sum + cat.articles.length, 0);
  
  let html = `
    <div class="kb-pill ${currentKBCat === 'all' ? 'active' : ''}" onclick="setKBCategory('all')">
      <span>All Articles</span>
      <span class="kb-pill-count">${total}</span>
    </div>
  `;
  
  KB_DATA.forEach(cat => {
    html += `
      <div class="kb-pill ${currentKBCat === cat.category ? 'active' : ''}" onclick="setKBCategory('${cat.category}')">
        <span>${cat.label}</span>
        <span class="kb-pill-count">${cat.articles.length}</span>
      </div>
    `;
  });
  
  nav.innerHTML = html;
}

window.setKBCategory = function(catId) {
  currentKBCat = catId;
  document.getElementById('kb-search-input').value = '';
  renderKBNav();
  renderKBContent();
};

window.toggleAccordion = function(el) {
  const row = el.closest('.kb-acc-row');
  const body = el.nextElementSibling;
  const icon = el.querySelector('.ti-chevron-down');
  
  if(body.style.display === 'block') {
    body.style.display = 'none';
    row.classList.remove('expanded');
    if(icon) icon.style.transform = 'rotate(0deg)';
  } else {
    body.style.display = 'block';
    row.classList.add('expanded');
    if(icon) icon.style.transform = 'rotate(180deg)';
  }
};

function renderKBContent(query = '') {
  const container = document.getElementById('kb-accordions-container');
  if(!container) return;
  
  container.innerHTML = '';
  const q = query.toLowerCase().trim();
  
  let matchCount = 0;
  
  KB_DATA.forEach(cat => {
    if(currentKBCat !== 'all' && cat.category !== currentKBCat && q === '') return;
    
    // Filter articles in this category
    const filteredArts = cat.articles.filter(art => {
      const matchQ = art.q.toLowerCase().includes(q);
      const matchA = art.a.toLowerCase().includes(q);
      const matchSteps = art.steps ? art.steps.some(s => s.toLowerCase().includes(q)) : false;
      return (q === '' || matchQ || matchA || matchSteps) && (q === '' || currentKBCat === 'all' || cat.category === currentKBCat);
    });
    
    if (filteredArts.length === 0) return;
    
    // Add section header if in "all" view
    if (currentKBCat === 'all') {
      container.innerHTML += `<div class="kb-section-hdr">${cat.label}</div>`;
    }
    
    filteredArts.forEach(art => {
      matchCount++;
      
      let bodyHtml = `<div class="kb-acc-ans">${art.a}</div>`;
      if(art.isSop && art.steps) {
        bodyHtml += `<div class="kb-sop-list">`;
        art.steps.forEach((step, idx) => {
          bodyHtml += `
            <div class="kb-sop-item">
              <div class="kb-sop-num">${idx + 1}</div>
              <div class="kb-sop-text">${step}</div>
            </div>
          `;
        });
        bodyHtml += `</div>`;
      }
      
      bodyHtml += `
        <div class="kb-acc-meta">
          <i class="ti ti-clock"></i> Last reviewed: ${art.lastReviewed} &nbsp;&nbsp;&middot;&nbsp;&nbsp; <i class="ti ti-file-text"></i> Source: ${art.sourceRef}
        </div>
        <div class="kb-acc-feedback">
          Was this helpful? <i class="ti ti-thumb-up"></i> <i class="ti ti-thumb-down"></i>
        </div>
      `;
      
      // Left icon/badge: plain chevron for definition, numbered badge for SOP
      const leftIcon = art.isSop ? `<div class="kb-sop-num" style="background:var(--warn); margin-right:4px;">${art.steps.length}</div>` : `<i class="ti ti-chevron-right" style="color:var(--text3); font-size:16px;"></i>`;
      
      // Right icon: expand chevron (for all)
      const rightIcon = `<i class="ti ti-chevron-down kb-acc-icon"></i>`;
      
      container.innerHTML += `
        <div class="kb-acc-row">
          <div class="kb-acc-hdr" onclick="toggleAccordion(this)">
            <div class="kb-acc-title">
               ${leftIcon}
               ${art.q}
            </div>
            ${rightIcon}
          </div>
          <div class="kb-acc-body">
            ${bodyHtml}
          </div>
        </div>
      `;
    });
  });
  
  if(matchCount === 0) {
    container.innerHTML = `<div style="padding: 40px; text-align:center; color:var(--text3); font-size:13px;">No matching articles found.</div>`;
  }
}
"""
start_js = content.find('function renderKBNav() {')
end_js = content.find('window.filterKB = function()', start_js)
if start_js != -1 and end_js != -1:
    content = content[:start_js] + js_new + content[end_js:]
    
with codecs.open(html_path, "w", "utf-8") as f:
    f.write(content)
print("SUCCESS")
