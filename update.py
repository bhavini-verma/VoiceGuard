import codecs

with codecs.open('static/voiceguard_uco_bank_platform.html', 'r', 'utf-8') as f:
    content = f.read()

css = '''
.kb-cat-btn { padding: 6px 14px; border-radius: 20px; font-size: 11px; font-weight: 600; color: var(--text2); background: #fff; border: 1px solid var(--border); cursor: pointer; display: flex; align-items: center; gap: 6px; transition: all 0.2s; }
.kb-cat-btn:hover { border-color: var(--bank); color: var(--bank); }
.kb-cat-btn.active { background: var(--bank); color: #fff; border-color: var(--bank); }
.kb-card-rich { background: #fff; border: 1px solid var(--border); border-radius: 8px; padding: 14px; transition: all 0.2s; cursor: pointer; }
.kb-card-rich:hover { border-color: var(--bank); box-shadow: 0 4px 12px rgba(0,91,172,0.08); transform: translateY(-2px); }
.kb-card-icon { width: 32px; height: 32px; border-radius: 6px; background: #F0F7FF; color: var(--bank); display: flex; align-items: center; justify-content: center; font-size: 16px; margin-bottom: 10px; }
.kb-card-title { font-size: 13px; font-weight: 700; color: var(--text); margin-bottom: 4px; }
.kb-card-desc { font-size: 11px; color: var(--text2); line-height: 1.4; }
.kb-severity { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 9px; font-weight: 700; margin-top: 10px; }
.kb-sev-high { background: rgba(239,68,68,0.1); color: #DC2626; border: 1px solid rgba(239,68,68,0.2); }
.kb-sev-med { background: rgba(245,158,11,0.1); color: #D97706; border: 1px solid rgba(245,158,11,0.2); }

.audit-filter { display: inline-flex; padding: 4px 12px; border-radius: 20px; font-size: 10px; font-weight: 600; color: var(--text2); border: 1px solid var(--border); cursor: pointer; transition: all 0.15s; margin-right: 6px; background: #fff; }
.audit-filter.active { background: var(--text); color: #fff; border-color: var(--text); }
.audit-filter:hover:not(.active) { background: #F4F8FC; }
.audit-row-new { animation: highlight-row 2s ease-out; }
@keyframes highlight-row { 0% { background-color: rgba(20,184,166,0.2); } 100% { background-color: transparent; } }
</style>
'''
content = content.replace('</style>', css, 1)

new_kb = '''    <div id="page-kb" class="page-section" style="display:none">
      <div class="section-hdr" style="margin-bottom:16px;">
        <div class="section-title"><i class="ti ti-shield-lock"></i>Threat Intelligence & SOP Hub</div>
        <div style="display:flex;gap:10px;align-items:center;background:#fff;border:1px solid var(--border);border-radius:6px;padding:6px 12px;width:300px;box-shadow:0 1px 3px rgba(0,0,0,0.02)">
           <i class="ti ti-search" style="color:var(--text3);font-size:14px"></i>
           <input type="text" id="kb-search-input" placeholder="Search threats, guidelines, SOPs..." style="border:none;outline:none;font-size:12px;width:100%;font-family:var(--sans);color:var(--text)" onkeyup="filterKB()">
        </div>
      </div>

      <div style="display:flex; gap:10px; margin-bottom:20px;" id="kb-categories">
        <div class="kb-cat-btn active" onclick="setKBCategory('all', this)"><i class="ti ti-category"></i> All Articles</div>
        <div class="kb-cat-btn" onclick="setKBCategory('threat', this)"><i class="ti ti-bug"></i> Threat Vectors</div>
        <div class="kb-cat-btn" onclick="setKBCategory('forensic', this)"><i class="ti ti-cpu"></i> Forensic Methods</div>
        <div class="kb-cat-btn" onclick="setKBCategory('sop', this)"><i class="ti ti-clipboard-list"></i> Compliance & SOP</div>
      </div>

      <div class="grid3" id="kb-articles-grid" style="gap:16px">
      </div>
    </div>
'''

start_kb = content.find('<div id="page-kb"')
end_kb = content.find('<div id="page-audit"', start_kb)
if start_kb != -1 and end_kb != -1:
    content = content[:start_kb] + new_kb + content[end_kb:]

new_audit = '''    <div id="page-audit" class="page-section" style="display:none">
      <div class="section-hdr">
        <div class="section-title"><i class="ti ti-clipboard-list"></i>System Activity & Security Ledger</div>
        <div style="display:flex;gap:8px;align-items:center">
          <span class="section-badge" id="audit-log-count">0 entries</span>
          <button onclick="clearAuditLogs()" style="padding:4px 12px;border:1px solid var(--border);border-radius:4px;background:#fff;color:var(--text2);font-size:11px;font-weight:600;cursor:pointer;display:flex;align-items:center;gap:5px"><i class="ti ti-trash"></i> Clear Logs</button>
        </div>
      </div>
      
      <div style="margin-bottom:14px; display:flex; gap:6px;" id="audit-filters">
        <div class="audit-filter active" onclick="setAuditFilter('all', this)">All Events</div>
        <div class="audit-filter" onclick="setAuditFilter('security', this)">Security Alerts</div>
        <div class="audit-filter" onclick="setAuditFilter('analysis', this)">Analysis Runs</div>
        <div class="audit-filter" onclick="setAuditFilter('system', this)">System Logs</div>
      </div>

      <div class="card" style="padding:0;overflow:hidden">
        <div style="overflow-y:auto;max-height:550px">
          <table class="audit-table" id="audit-table">
            <thead><tr><th style="width:15%">Timestamp</th><th style="width:15%">Event ID</th><th style="width:12%">Category</th><th style="width:38%">Description / Metadata</th><th style="width:20%">Status</th></tr></thead>
            <tbody id="audit-tbody">
            </tbody>
          </table>
        </div>
      </div>
    </div>
'''

start_audit = content.find('<div id="page-audit"')
end_audit = content.find('<!-- ======== AI INVESTIGATOR PAGE ======== -->', start_audit)
if start_audit != -1 and end_audit != -1:
    content = content[:start_audit] + new_audit + content[end_audit:]

js = '''
// ================= KB & AUDIT OVERHAUL ================= //

const KB_ARTICLES = [
  { id: 'kb1', cat: 'threat', title: 'AI Voice Cloning', desc: 'Synthetic voices generated using deep learning models from small audio samples of legitimate customers.', sev: 'HIGH', sevCls: 'kb-sev-high', icon: 'ti-robot' },
  { id: 'kb2', cat: 'threat', title: 'Replay Attacks', desc: 'Pre-recorded audio of genuine customer voices played back through phone speakers or digital pipelines.', sev: 'HIGH', sevCls: 'kb-sev-high', icon: 'ti-player-play' },
  { id: 'kb3', cat: 'forensic', title: 'Deep Spectral Analysis', desc: 'Examining spectrogram frequencies to detect synthetic artifacts and missing respiratory cadences.', sev: 'MED', sevCls: 'kb-sev-med', icon: 'ti-wave-sine' },
  { id: 'kb4', cat: 'forensic', title: 'Jitter & Shimmer', desc: 'Metrics calculating variations in voice pitch and amplitude that synthetic voices fail to replicate naturally.', sev: 'MED', sevCls: 'kb-sev-med', icon: 'ti-chart-line' },
  { id: 'kb5', cat: 'sop', title: 'RBI Compliance 2026', desc: 'Mandatory verification protocols for transactions > 1 Lakh INR and handling flagged calls.', sev: 'HIGH', sevCls: 'kb-sev-high', icon: 'ti-building-bank' },
  { id: 'kb6', cat: 'sop', title: 'Escalation Flow', desc: 'Steps to securely transfer high-risk calls to the Tier-2 Fraud Intelligence Team without alerting perpetrators.', sev: 'HIGH', sevCls: 'kb-sev-high', icon: 'ti-alert-triangle' }
];

function renderKB(filterCat = 'all', query = '') {
  const grid = document.getElementById('kb-articles-grid');
  if(!grid) return;
  grid.innerHTML = '';
  const filtered = KB_ARTICLES.filter(a => {
    const matchCat = filterCat === 'all' || a.cat === filterCat;
    const matchQuery = a.title.toLowerCase().includes(query.toLowerCase()) || a.desc.toLowerCase().includes(query.toLowerCase());
    return matchCat && matchQuery;
  });

  if(filtered.length === 0) {
    grid.innerHTML = <div style="grid-column: span 3; text-align:center; padding: 40px; color: var(--text3); font-size:12px;">No articles found matching criteria.</div>;
    return;
  }

  filtered.forEach(a => {
    grid.innerHTML += 
      <div class="kb-card-rich" onclick="openKBArticle('')">
        <div class="kb-card-icon"><i class="ti "></i></div>
        <div class="kb-card-title"></div>
        <div class="kb-card-desc"></div>
        <div class="kb-severity "> RISK</div>
      </div>
    ;
  });
}

window.setKBCategory = function(cat, el) {
  document.querySelectorAll('.kb-cat-btn').forEach(b => b.classList.remove('active'));
  if(el) el.classList.add('active');
  const query = document.getElementById('kb-search-input').value;
  renderKB(cat, query);
};

window.filterKB = function() {
  const activeBtn = document.querySelector('.kb-cat-btn.active');
  const cat = activeBtn ? (activeBtn.innerText.includes('All') ? 'all' : (activeBtn.innerText.includes('Threat') ? 'threat' : (activeBtn.innerText.includes('Forensic') ? 'forensic' : 'sop'))) : 'all';
  const query = document.getElementById('kb-search-input').value;
  renderKB(cat, query);
};

window.openKBArticle = function(id) {
  const a = KB_ARTICLES.find(x => x.id === id);
  if(a) alert('Opening detailed view for: ' + a.title);
};

const AUDIT_LOGS = [
  { ts: new Date(Date.now() - 50000).toLocaleTimeString('en-IN'), id: 'EVT-9941', cat: 'system', desc: 'Model synchronized with Central DB', status: 'SUCCESS', statusCls: 'v-clean' },
  { ts: new Date(Date.now() - 3600000).toLocaleTimeString('en-IN'), id: 'EVT-9940', cat: 'analysis', desc: 'Case VG-A7B2 processed. Score: 12%', status: 'CLEARED', statusCls: 'v-clean' },
  { ts: new Date(Date.now() - 7200000).toLocaleTimeString('en-IN'), id: 'EVT-9939', cat: 'security', desc: 'Failed login attempt from Node-08', status: 'WARNING', statusCls: 'v-susp' },
  { ts: new Date(Date.now() - 8600000).toLocaleTimeString('en-IN'), id: 'EVT-9938', cat: 'analysis', desc: 'Case VG-X9F1 processed. Score: 89%', status: 'BLOCKED', statusCls: 'v-fraud' },
];

let currentAuditFilter = 'all';

function renderAudit() {
  const tbody = document.getElementById('audit-tbody');
  const cnt = document.getElementById('audit-log-count');
  if(!tbody || !cnt) return;
  tbody.innerHTML = '';
  
  const filtered = AUDIT_LOGS.filter(l => currentAuditFilter === 'all' || l.cat === currentAuditFilter);
  cnt.textContent = ${filtered.length} entries;

  if(filtered.length === 0) {
    tbody.innerHTML = <tr><td colspan="5" style="text-align:center;color:var(--text3);padding:24px;font-size:12px">No audit entries found.</td></tr>;
    return;
  }

  filtered.forEach(l => {
    tbody.innerHTML += 
      <tr class="">
        <td style="font-family:var(--mono); color:var(--text2); font-size:10px;"></td>
        <td style="font-family:var(--mono); color:var(--bank); font-weight:600; font-size:11px;"></td>
        <td><span style="font-size:9px;text-transform:uppercase;color:var(--text3);font-weight:600"></span></td>
        <td style="font-size:11px;color:var(--text);"></td>
        <td><span class="verdict-tag "></span></td>
      </tr>
    ;
    l.isNew = false;
  });
}

window.setAuditFilter = function(cat, el) {
  document.querySelectorAll('.audit-filter').forEach(b => b.classList.remove('active'));
  if(el) el.classList.add('active');
  currentAuditFilter = cat;
  renderAudit();
};

window.pushAuditLog = function(cat, desc, status, statusCls) {
  AUDIT_LOGS.unshift({
    ts: new Date().toLocaleTimeString('en-IN'),
    id: 'EVT-' + Math.floor(1000 + Math.random() * 9000),
    cat, desc, status, statusCls, isNew: true
  });
  renderAudit();
};

window.clearAuditLogs = function() {
  AUDIT_LOGS.length = 0;
  renderAudit();
};

const origProcess = window.processAnalysis;
if (typeof origProcess === 'function') {
  window.processAnalysis = function() {
    origProcess();
    setTimeout(() => {
      const sc = document.getElementById('hero-score').innerText;
      const sNum = parseInt(sc);
      const isHigh = !isNaN(sNum) && sNum > 50;
      pushAuditLog('analysis', Case  processed. Score: , isHigh ? 'HIGH RISK' : 'CLEARED', isHigh ? 'v-fraud' : 'v-clean');
    }, 4500);
  };
}

document.addEventListener('DOMContentLoaded', () => {
  renderKB();
  renderAudit();
});
</script>
'''

content = content.replace('</script>', js, 1)

with codecs.open('static/voiceguard_uco_bank_platform.html', 'w', 'utf-8') as f:
    f.write(content)

print("SUCCESS")
