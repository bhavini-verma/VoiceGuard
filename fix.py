import codecs

with codecs.open("static/voiceguard_uco_bank_platform.html", "r", "utf-8") as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, l in enumerate(lines):
    if "// ================= KB & AUDIT OVERHAUL ================= //" in l:
        start_idx = i
    if "document.addEventListener('DOMContentLoaded', () => {" in l:
        end_idx = i + 3
        break

js = """// ================= KB & AUDIT OVERHAUL ================= //

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
    grid.innerHTML = `<div style="grid-column: span 3; text-align:center; padding: 40px; color: var(--text3); font-size:12px;">No articles found matching criteria.</div>`;
    return;
  }

  filtered.forEach(a => {
    grid.innerHTML += `
      <div class="kb-card-rich" onclick="openKBArticle('${a.id}')">
        <div class="kb-card-icon"><i class="ti ${a.icon}"></i></div>
        <div class="kb-card-title">${a.title}</div>
        <div class="kb-card-desc">${a.desc}</div>
        <div class="kb-severity ${a.sevCls}">${a.sev} RISK</div>
      </div>
    `;
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
  cnt.textContent = `${filtered.length} entries`;

  if(filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--text3);padding:24px;font-size:12px">No audit entries found.</td></tr>`;
    return;
  }

  filtered.forEach(l => {
    tbody.innerHTML += `
      <tr class="${l.isNew ? 'audit-row-new' : ''}">
        <td style="font-family:var(--mono); color:var(--text2); font-size:10px;">${l.ts}</td>
        <td style="font-family:var(--mono); color:var(--bank); font-weight:600; font-size:11px;">${l.id}</td>
        <td><span style="font-size:9px;text-transform:uppercase;color:var(--text3);font-weight:600">${l.cat}</span></td>
        <td style="font-size:11px;color:var(--text);">${l.desc}</td>
        <td><span class="verdict-tag ${l.statusCls}">${l.status}</span></td>
      </tr>
    `;
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
      window.pushAuditLog('analysis', `Case ${typeof SESSION_ID !== 'undefined' ? SESSION_ID : 'VG-NEW'} processed. Score: ${sc}`, isHigh ? 'HIGH RISK' : 'CLEARED', isHigh ? 'v-fraud' : 'v-clean');
    }, 4500);
  };
}

document.addEventListener('DOMContentLoaded', () => {
  renderKB();
  renderAudit();
});
</script>
"""

if start_idx != -1 and end_idx != -1:
    new_lines = lines[:start_idx] + [js + "\n"] + lines[end_idx+2:]
    with codecs.open("static/voiceguard_uco_bank_platform.html", "w", "utf-8") as f:
        f.writelines(new_lines)
    print("SUCCESS")
else:
    print("FAILED TO FIND TARGET", start_idx, end_idx)
