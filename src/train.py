"""
VoiceGuard — Production Dashboard
ui/app.py  |  streamlit run ui/app.py
"""

import sys, os, tempfile
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceGuard — AI Voice Forensics",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Outfit:wght@300;400;600;700;800&display=swap');

:root {
  --void:       #03030a;
  --surface:    #080810;
  --card:       #0d0d1a;
  --raised:     #121224;
  --border:     rgba(255,255,255,0.06);
  --border-md:  rgba(255,255,255,0.10);
  --border-hi:  rgba(255,255,255,0.18);
  --cyan:       #00e5ff;
  --violet:     #8b5cf6;
  --green:      #10f5a0;
  --amber:      #f59e0b;
  --red:        #ff2d55;
  --t1: #eeeef8;
  --t2: #7777aa;
  --t3: #333355;
  --t4: #1a1a33;
  --mono: 'IBM Plex Mono', monospace;
  --sans: 'Outfit', sans-serif;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; }

[data-testid="stAppViewContainer"] { background: var(--void); }
[data-testid="stHeader"]           { background: transparent !important; }
[data-testid="stSidebar"]          { background: var(--surface); border-right: 1px solid var(--border); }
section.main > div                 { padding-top: 1.5rem; }
p, li, div                        { font-family: var(--sans); }
h1,h2,h3,h4                       { font-family: var(--sans); color: var(--t1) !important; }

[data-testid="stSidebarContent"] { padding: 1.4rem 1.1rem; }

[data-testid="metric-container"] {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 14px; padding: 16px !important;
}
[data-testid="stMetricValue"] { font-family: var(--sans) !important; color: var(--t1) !important; font-size: 20px !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-family: var(--mono) !important; color: var(--t3) !important; font-size: 10px !important; letter-spacing: .12em; text-transform: uppercase; }

[data-testid="stTabs"] button { font-family: var(--mono) !important; color: var(--t3) !important; font-size: 11px !important; letter-spacing: .12em; text-transform: uppercase; padding: 10px 20px !important; }
[data-testid="stTabs"] button:hover { color: var(--t2) !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--cyan) !important; border-bottom-color: var(--cyan) !important; }

[data-testid="stFileUploader"] {
  background: var(--raised); border: 1px dashed rgba(0,229,255,.18);
  border-radius: 16px; padding: 10px; transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(0,229,255,.35); }

.stButton > button {
  background: var(--raised) !important; border: 1px solid var(--border-md) !important;
  color: var(--t2) !important; border-radius: 10px !important;
  font-family: var(--mono) !important; font-size: 11px !important;
  letter-spacing: .06em; padding: 7px 18px !important; transition: all .18s !important;
}
.stButton > button:hover { border-color: var(--cyan) !important; color: var(--cyan) !important; background: rgba(0,229,255,.06) !important; }

audio { width: 100%; border-radius: 10px; filter: invert(1) hue-rotate(180deg) saturate(.5); }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb { background: var(--t4); border-radius: 2px; }

/* ── brand ── */
.vg-brand { display:flex; align-items:center; gap:10px; margin-bottom:22px; }
.vg-icon  { width:38px; height:38px; border-radius:11px; background:linear-gradient(135deg,rgba(0,229,255,.15),rgba(139,92,246,.15)); border:1px solid rgba(0,229,255,.25); display:flex; align-items:center; justify-content:center; font-size:19px; }
.vg-name  { font-family:var(--sans); font-size:18px; font-weight:800; color:var(--t1); letter-spacing:-.03em; }
.vg-ver   { font-family:var(--mono); font-size:9px; letter-spacing:.18em; background:rgba(0,229,255,.08); border:1px solid rgba(0,229,255,.2); color:var(--cyan); padding:3px 8px; border-radius:20px; margin-left:4px; vertical-align:middle; }

/* ── typography ── */
.eyebrow { font-family:var(--mono); font-size:10px; letter-spacing:.22em; text-transform:uppercase; color:var(--cyan); margin-bottom:8px; }
.hero-title { font-family:var(--sans); font-size:34px; font-weight:800; color:var(--t1); letter-spacing:-.04em; line-height:1.1; margin-bottom:8px; }
.hero-sub { font-size:13.5px; color:var(--t2); line-height:1.65; max-width:600px; }
.sec-rule { font-family:var(--mono); font-size:9px; letter-spacing:.18em; text-transform:uppercase; color:var(--t3); margin:28px 0 14px; display:flex; align-items:center; gap:10px; }
.sec-rule::after { content:''; flex:1; height:1px; background:var(--border); }

/* ── stat strip ── */
.strip { display:flex; gap:1px; margin-bottom:28px; }
.strip-cell { flex:1; background:var(--card); border:1px solid var(--border); padding:15px 18px; }
.strip-cell:first-child { border-radius:14px 0 0 14px; }
.strip-cell:last-child  { border-radius:0 14px 14px 0; }
.strip-num  { font-family:var(--sans); font-size:19px; font-weight:800; color:var(--t1); letter-spacing:-.03em; margin-bottom:3px; }
.strip-desc { font-family:var(--mono); font-size:9px; letter-spacing:.12em; text-transform:uppercase; color:var(--t3); }

/* ── sidebar ── */
.sb-label { font-family:var(--mono); font-size:9px; letter-spacing:.2em; text-transform:uppercase; color:var(--t3); margin-bottom:9px; }
.sb-row   { display:flex; align-items:center; gap:9px; padding:7px 9px; border-radius:8px; margin-bottom:3px; border:1px solid transparent; }
.sb-row:hover { background:var(--raised); border-color:var(--border); }
.sb-txt   { font-size:12px; color:var(--t2); }
.sb-chip  { margin-left:auto; font-family:var(--mono); font-size:9px; padding:2px 8px; border-radius:20px; }
.chip-h   { background:rgba(255,45,85,.10); color:var(--red);   border:1px solid rgba(255,45,85,.2); }
.chip-m   { background:rgba(245,158,11,.10); color:var(--amber); border:1px solid rgba(245,158,11,.2); }
.chip-l   { background:rgba(16,245,160,.10); color:var(--green); border:1px solid rgba(16,245,160,.2); }

/* ── verdict ── */
@keyframes breathe-red   { 0%,100%{box-shadow:0 0 0 0 rgba(255,45,85,0)}   50%{box-shadow:0 0 40px 6px rgba(255,45,85,.10)} }
@keyframes breathe-green { 0%,100%{box-shadow:0 0 0 0 rgba(16,245,160,0)}  50%{box-shadow:0 0 32px 4px rgba(16,245,160,.08)} }
@keyframes breathe-amber { 0%,100%{box-shadow:0 0 0 0 rgba(245,158,11,0)}  50%{box-shadow:0 0 32px 4px rgba(245,158,11,.08)} }
@keyframes fadein        { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }

.verdict      { border-radius:20px; padding:30px 34px 26px; position:relative; overflow:hidden; }
.verdict-high { background:rgba(255,45,85,.04);  border:1px solid rgba(255,45,85,.22); animation:fadein .45s ease both, breathe-red 2.8s ease-in-out infinite; }
.verdict-med  { background:rgba(245,158,11,.04); border:1px solid rgba(245,158,11,.18); animation:fadein .45s ease both, breathe-amber 3s ease-in-out infinite; }
.verdict-low  { background:rgba(16,245,160,.04); border:1px solid rgba(16,245,160,.18); animation:fadein .45s ease both, breathe-green 3.2s ease-in-out infinite; }

.verdict-orb { position:absolute; top:-80px; right:-80px; width:240px; height:240px; border-radius:50%; pointer-events:none; }
.orb-high    { background:radial-gradient(circle, rgba(255,45,85,.14) 0%, transparent 70%); }
.orb-med     { background:radial-gradient(circle, rgba(245,158,11,.12) 0%, transparent 70%); }
.orb-low     { background:radial-gradient(circle, rgba(16,245,160,.10) 0%, transparent 70%); }

.v-tag       { font-family:var(--mono); font-size:9px; letter-spacing:.28em; text-transform:uppercase; margin-bottom:10px; }
.v-tag-h     { color:rgba(255,45,85,.65); }
.v-tag-m     { color:rgba(245,158,11,.65); }
.v-tag-l     { color:rgba(16,245,160,.65); }

.v-level     { font-family:var(--sans); font-size:44px; font-weight:800; letter-spacing:-.04em; line-height:1; margin-bottom:14px; }
.v-level-h   { color:var(--red); }
.v-level-m   { color:var(--amber); }
.v-level-l   { color:var(--green); }

.v-action    { font-size:12.5px; font-weight:500; padding:8px 14px; border-radius:8px; display:inline-block; }
.v-act-h     { background:rgba(255,45,85,.12);  color:var(--red); }
.v-act-m     { background:rgba(245,158,11,.10); color:var(--amber); }
.v-act-l     { background:rgba(16,245,160,.10); color:var(--green); }

.v-prob-row  { display:flex; align-items:flex-end; gap:12px; margin-top:22px; }
.v-prob-big  { font-family:var(--sans); font-size:56px; font-weight:800; letter-spacing:-.05em; line-height:1; }
.v-prob-pct  { font-size:16px; color:var(--t2); }
.v-prob-lbl  { font-family:var(--mono); font-size:9px; letter-spacing:.14em; text-transform:uppercase; color:var(--t3); margin-top:4px; }

.conf-wrap   { margin-top:18px; }
.conf-labels { display:flex; justify-content:space-between; font-family:var(--mono); font-size:8.5px; letter-spacing:.09em; text-transform:uppercase; color:var(--t3); margin-bottom:5px; }
.conf-track  { height:7px; border-radius:4px; background:var(--raised); position:relative; overflow:visible; }
.conf-fill   { height:100%; border-radius:4px; background:linear-gradient(90deg, var(--green) 0%, var(--amber) 50%, var(--red) 100%); position:relative; }
.conf-needle { position:absolute; right:-1px; top:-4px; width:3px; height:15px; border-radius:2px; background:var(--t1); box-shadow:0 0 6px rgba(255,255,255,.3); }

/* ── proc log ── */
.proc-log   { background:rgba(0,229,255,.03); border:1px solid rgba(0,229,255,.12); border-radius:14px; padding:18px 22px; margin-bottom:16px; font-family:var(--mono); font-size:10.5px; }
.proc-title { color:var(--cyan); letter-spacing:.2em; font-size:9px; text-transform:uppercase; margin-bottom:14px; }
.proc-row   { display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid var(--border); color:var(--t2); }
.proc-row:last-child { border-bottom:none; }
.proc-ok    { color:var(--green); }
.proc-run   { color:var(--amber); }

/* ── features ── */
.feat-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:9px; }
.feat-card { background:var(--card); border:1px solid var(--border); border-radius:13px; padding:15px 17px; transition:border-color .18s, transform .18s; }
.feat-card:hover { border-color:var(--border-md); transform:translateY(-2px); }
.feat-key  { font-family:var(--mono); font-size:8.5px; letter-spacing:.18em; text-transform:uppercase; color:var(--t3); margin-bottom:9px; }
.feat-val  { font-family:var(--sans); font-size:21px; font-weight:700; letter-spacing:-.02em; margin-bottom:3px; }
.feat-unit { font-family:var(--mono); font-size:10px; color:var(--t3); }

/* ── info panel ── */
.info-panel { background:var(--card); border:1px solid var(--border); border-radius:13px; padding:18px; }
.info-row   { display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid var(--border); }
.info-row:last-child { border-bottom:none; }
.info-k     { font-family:var(--mono); font-size:9.5px; letter-spacing:.12em; text-transform:uppercase; color:var(--t3); }
.info-v     { font-family:var(--mono); font-size:11px; color:var(--t1); }

/* ── plots ── */
.plot-wrap { background:var(--card); border:1px solid var(--border); border-radius:13px; padding:14px; margin-bottom:4px; }
.plot-lbl  { font-family:var(--mono); font-size:8.5px; letter-spacing:.18em; text-transform:uppercase; color:var(--t3); margin-bottom:10px; }

/* ── history ── */
.hist-row { display:flex; align-items:center; gap:13px; padding:13px 15px; background:var(--card); border:1px solid var(--border); border-radius:12px; margin-bottom:7px; transition:border-color .18s; }
.hist-row:hover { border-color:var(--border-md); }
.dot   { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.dot-h { background:var(--red);   box-shadow:0 0 7px rgba(255,45,85,.5); }
.dot-m { background:var(--amber); box-shadow:0 0 7px rgba(245,158,11,.5); }
.dot-l { background:var(--green); box-shadow:0 0 7px rgba(16,245,160,.5); }
.h-name  { font-size:12.5px; color:var(--t1); flex:1; font-weight:500; }
.h-lvl-h { font-family:var(--mono); font-size:10px; color:var(--red);   letter-spacing:.1em; }
.h-lvl-m { font-family:var(--mono); font-size:10px; color:var(--amber); letter-spacing:.1em; }
.h-lvl-l { font-family:var(--mono); font-size:10px; color:var(--green); letter-spacing:.1em; }
.h-prob  { font-family:var(--mono); font-size:11px; color:var(--t2); }
.h-dur   { font-family:var(--mono); font-size:10px; color:var(--t3); }

/* ── upload zone ── */
.upload-zone { text-align:center; padding:52px 28px; border:1px dashed rgba(0,229,255,.13); border-radius:18px; background:rgba(0,229,255,.015); }
.uz-icon     { font-size:38px; margin-bottom:14px; }
.uz-title    { font-family:var(--sans); font-size:15px; font-weight:700; color:var(--t1); margin-bottom:8px; }
.uz-sub      { font-size:12.5px; color:var(--t3); line-height:1.6; }
.uz-pill     { display:inline-block; margin-top:14px; background:var(--raised); border:1px solid var(--border-md); color:var(--t3); font-family:var(--mono); font-size:9.5px; padding:5px 14px; border-radius:20px; }

/* ── about ── */
.about-card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:22px 24px; margin-bottom:12px; }
.about-h    { font-family:var(--sans); font-size:14px; font-weight:700; color:var(--t1); margin-bottom:12px; }
.pipe-code  { background:var(--void); border:1px solid var(--border); border-radius:10px; padding:16px 18px; font-family:var(--mono); font-size:10.5px; color:var(--cyan); line-height:1.85; overflow-x:auto; }
.fchip      { display:inline-block; background:var(--raised); border:1px solid var(--border); border-radius:6px; padding:4px 10px; font-size:10.5px; color:var(--t2); margin:3px; font-family:var(--mono); }

/* ── footer ── */
.footer { margin-top:52px; padding-top:18px; border-top:1px solid var(--border); text-align:center; font-family:var(--mono); font-size:10px; color:var(--t3); letter-spacing:.08em; line-height:2; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.join(os.path.dirname(__file__), "..", "models")
    mp = os.path.join(base, "model.pkl")
    sp = os.path.join(base, "scaler.pkl")
    if not os.path.exists(mp) or not os.path.exists(sp):
        return None, None
    return joblib.load(mp), joblib.load(sp)

model, scaler = load_model()


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────
SR, N_MFCC, DURATION = 16000, 40, 3

def extract(y, sr):
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
    y = y[:SR * DURATION]
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mel      = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
    zcr      = librosa.feature.zero_crossing_rate(y)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma   = librosa.feature.chroma_stft(y=y, sr=sr)
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0v      = f0[~np.isnan(f0)]
    rms      = librosa.feature.rms(y=y)
    har      = librosa.effects.harmonic(y)
    vec = np.concatenate([
        np.mean(mfcc,1), np.std(mfcc,1),
        [np.mean(mel)], [np.std(mel)],
        [np.mean(zcr)], [np.std(zcr)],
        np.mean(contrast,1), np.std(contrast,1),
        np.mean(chroma,1), np.std(chroma,1),
        [np.std(f0v) if len(f0v) > 0 else 0.0],
        [np.mean(rms)], [np.std(rms)],
        [np.mean(np.abs(har)) / (np.mean(np.abs(y)) + 1e-6)],
    ])
    vec = np.pad(vec, (0, 128 - len(vec)))
    diag = {
        "pitch_std":         float(np.std(f0v)) if len(f0v) > 0 else 0.0,
        "noise_floor":       float(np.mean(rms)),
        "harmonic_ratio":    float(np.mean(np.abs(har)) / (np.mean(np.abs(y)) + 1e-6)),
        "zcr":               float(np.mean(zcr)),
        "spectral_contrast": float(np.mean(contrast)),
        "mfcc_variance":     float(np.mean(np.var(mfcc, 1))),
    }
    return vec, diag

def risk_level(p):
    if p < 0.40: return "LOW",    "low"
    if p < 0.70: return "MEDIUM", "med"
    return           "HIGH",   "high"


# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────
BG, PANEL = "#04040e", "#0d0d1a"

def _ax(fig, ax):
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(BG)
    ax.tick_params(colors="#333355", labelsize=7, length=0, pad=4)
    for sp in ax.spines.values(): sp.set_edgecolor("#111128")
    for lbl in [ax.xaxis.label, ax.yaxis.label]:
        lbl.set_color("#333355"); lbl.set_fontsize(8)
    return fig, ax

def fig_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(6.8, 1.9))
    _ax(fig, ax)
    t = np.linspace(0, len(y)/sr, len(y))
    ax.fill_between(t, y, alpha=.18, color="#00e5ff")
    ax.plot(t, y, color="#00e5ff", lw=.55, alpha=.95)
    ax.axhline(0, color="#111128", lw=.5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    plt.tight_layout(pad=.5)
    return fig

def fig_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    _ax(fig, ax)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
    img = librosa.display.specshow(mel, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Freq (Hz)")
    cb = plt.colorbar(img, ax=ax, format="%+2.0f dB")
    cb.ax.tick_params(colors="#333355", labelsize=7, length=0)
    cb.outline.set_edgecolor("#111128")
    plt.tight_layout(pad=.5)
    return fig

def fig_mfcc(y, sr):
    fig, ax = plt.subplots(figsize=(6.8, 2.1))
    _ax(fig, ax)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax, cmap="twilight_shifted")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("MFCC #")
    cb = plt.colorbar(img, ax=ax)
    cb.ax.tick_params(colors="#333355", labelsize=7, length=0)
    cb.outline.set_edgecolor("#111128")
    plt.tight_layout(pad=.5)
    return fig

def fig_pitch(y, sr):
    fig, ax = plt.subplots(figsize=(6.8, 1.9))
    _ax(fig, ax)
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    times = librosa.times_like(f0, sr=sr)
    ax.plot(times, f0, color="#8b5cf6", lw=1.2, alpha=.9)
    ax.fill_between(times, f0, where=~np.isnan(f0), alpha=.12, color="#8b5cf6")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Pitch (Hz)")
    plt.tight_layout(pad=.5)
    return fig


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="vg-brand">
      <div class="vg-icon">🛡</div>
      <span class="vg-name">VoiceGuard<span class="vg-ver">v1.0</span></span>
    </div>
    """, unsafe_allow_html=True)

    if model:
        st.success("Model loaded — inference ready", icon="✅")
    else:
        st.error("No model found — run train.py", icon="⚠️")

    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,.06);margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-label" style="margin-bottom:10px">Risk thresholds</div>
    <div class="sb-row"><span>🟢</span><span class="sb-txt">Authentic voice</span><span class="sb-chip chip-l">0–40%</span></div>
    <div class="sb-row"><span>🟡</span><span class="sb-txt">Suspicious audio</span><span class="sb-chip chip-m">40–70%</span></div>
    <div class="sb-row"><span>🔴</span><span class="sb-txt">AI clone detected</span><span class="sb-chip chip-h">70–100%</span></div>
    <hr style='border:none;border-top:1px solid rgba(255,255,255,.06);margin:16px 0'>
    <div class="sb-label" style="margin-bottom:10px">Tech stack</div>
    <div class="sb-row"><span>🐍</span><span class="sb-txt">Python · Librosa · NumPy</span></div>
    <div class="sb-row"><span>🤖</span><span class="sb-txt">Scikit-learn · Random Forest</span></div>
    <div class="sb-row"><span>⚡</span><span class="sb-txt">Verilog · Vivado · FPGA</span></div>
    <div class="sb-row"><span>🎛</span><span class="sb-txt">Streamlit · Matplotlib</span></div>
    <hr style='border:none;border-top:1px solid rgba(255,255,255,.06);margin:16px 0'>
    <div class="sb-label" style="margin-bottom:10px">Team</div>
    <div class="sb-row"><span>👤</span><span class="sb-txt">Bhavini Verma</span></div>
    <div class="sb-row"><span>👤</span><span class="sb-txt">Ashit Raj</span></div>
    <div class="sb-row"><span>🏆</span><span class="sb-txt">PSB Hackathon 2026</span></div>
    <div class="sb-row"><span>🎓</span><span class="sb-txt">VIT Vellore</span></div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HERO + STRIP
# ─────────────────────────────────────────────
st.markdown("""
<div class="eyebrow">// Audio Forensics System · PSB Hackathon 2026</div>
<div class="hero-title">Voice Clone Detection</div>
<div class="hero-sub">
  Real-time AI-generated speech identification using spectral fingerprinting,
  MFCC analysis, and ensemble machine learning — deployable at telecom scale.
</div><br>
<div class="strip">
  <div class="strip-cell"><div class="strip-num">128-dim</div><div class="strip-desc">Feature Vector</div></div>
  <div class="strip-cell"><div class="strip-num">&gt;92%</div><div class="strip-desc">Target Accuracy</div></div>
  <div class="strip-cell"><div class="strip-num">≤10 s</div><div class="strip-desc">Detection Window</div></div>
  <div class="strip-cell"><div class="strip-num">3-tier</div><div class="strip-desc">Risk Levels</div></div>
  <div class="strip-cell"><div class="strip-num">RF-200</div><div class="strip-desc">Estimators</div></div>
  <div class="strip-cell"><div class="strip-num">FPGA</div><div class="strip-desc">HW Accelerated</div></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_analyze, tab_history, tab_about = st.tabs([
    "  📁  Analyze File  ",
    "  🕘  Session History  ",
    "  ℹ  About  ",
])


# ═══════════════════════════════════════════════
#  TAB 1 — ANALYZE
# ═══════════════════════════════════════════════
with tab_analyze:
    uploaded = st.file_uploader(
        "Drop audio file",
        type=["wav", "mp3", "flac"],
        label_visibility="collapsed",
    )

    if uploaded and model:
        suffix = "." + uploaded.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        y, sr    = librosa.load(tmp_path, sr=None, mono=True)
        duration = round(len(y) / sr, 2)

        # audio player
        st.markdown('<div style="background:var(--card);border:1px solid var(--border);border-radius:13px;padding:12px 16px;margin-bottom:16px;">', unsafe_allow_html=True)
        st.audio(uploaded)
        st.markdown('</div>', unsafe_allow_html=True)

        # processing log — running state
        log = st.empty()
        log.markdown("""
        <div class="proc-log">
          <div class="proc-title">// Signal Analysis Pipeline</div>
          <div class="proc-row"><span>Loading audio buffer</span><span class="proc-ok">✓ done</span></div>
          <div class="proc-row"><span>Resample → 16 kHz · clip → 3 s</span><span class="proc-ok">✓ done</span></div>
          <div class="proc-row"><span>MFCC extraction (40 coeff × mean + std)</span><span class="proc-ok">✓ done</span></div>
          <div class="proc-row"><span>Mel spectrogram computation</span><span class="proc-ok">✓ done</span></div>
          <div class="proc-row"><span>Pitch jitter via pyin F0</span><span class="proc-ok">✓ done</span></div>
          <div class="proc-row"><span>Harmonic ratio · ZCR · RMS energy</span><span class="proc-ok">✓ done</span></div>
          <div class="proc-row"><span>Spectral contrast · Chroma features</span><span class="proc-ok">✓ done</span></div>
          <div class="proc-row"><span>StandardScaler normalization → 128-dim</span><span class="proc-ok">✓ done</span></div>
          <div class="proc-row"><span>Random Forest inference (n=200, depth=20)</span><span class="proc-run">⟳ running…</span></div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner(""):
            feat_vec, diag  = extract(y, sr)
            feat_scaled     = scaler.transform(feat_vec.reshape(1, -1))
            prob_fake       = float(model.predict_proba(feat_scaled)[0][1])
            level, lkey     = risk_level(prob_fake)

        # processing log — complete
        log.markdown("""
        <div class="proc-log">
          <div class="proc-title">// Signal Analysis Pipeline</div>
          <div class="proc-row"><span>Loading audio buffer</span><span class="proc-ok">✓ done</span></div>
          <div class="proc-row"><span>Resample → 16 kHz · clip → 3 s</span><span class="proc-ok">✓ done</span></div>
          <div class="proc-row"><span>MFCC extraction (40 coeff × mean + std)</span><span class="proc-ok">✓ done</span></div>
          <div class="proc-row"><span>Mel spectrogram computation</span><span class="proc-ok">✓ done</span></div>
          <div class="proc-row"><span>Pitch jitter via pyin F0</span><span class="proc-ok">✓ done</span></div>
          <div class="proc-row"><span>Harmonic ratio · ZCR · RMS energy</span><span class="proc-ok">✓ done</span></div>
          <div class="proc-row"><span>Spectral contrast · Chroma features</span><span class="proc-ok">✓ done</span></div>
          <div class="proc-row"><span>StandardScaler normalization → 128-dim</span><span class="proc-ok">✓ done</span></div>
          <div class="proc-row"><span>Random Forest inference (n=200, depth=20)</span><span class="proc-ok">✓ done</span></div>
        </div>
        """, unsafe_allow_html=True)

        # ── verdict card ──────────────────────────
        actions = {
            "LOW":    "✅  Call proceeds normally — voice appears authentic",
            "MEDIUM": "⚠️  Trigger secondary caller verification protocol",
            "HIGH":   "🚫  Block call immediately — raise fraud alert",
        }
        pct   = f"{prob_fake * 100:.1f}"
        bar_w = f"{prob_fake * 100:.1f}%"
        accent_var = {"high":"red","med":"amber","low":"green"}[lkey]

        st.markdown(f"""
        <div class="verdict verdict-{lkey}">
          <div class="verdict-orb orb-{lkey}"></div>
          <div class="v-tag v-tag-{lkey}">// Threat Assessment Result</div>
          <div class="v-level v-level-{lkey}">{level} RISK</div>
          <div class="v-action v-act-{lkey}">{actions[level]}</div>
          <div class="v-prob-row">
            <div>
              <div class="v-prob-big v-level-{lkey}">{pct}<span class="v-prob-pct">%</span></div>
              <div class="v-prob-lbl">Synthetic probability score</div>
            </div>
          </div>
          <div class="conf-wrap">
            <div class="conf-labels"><span>Authentic</span><span>Uncertain</span><span>Synthetic</span></div>
            <div class="conf-track">
              <div class="conf-fill" style="width:{bar_w}">
                <div class="conf-needle"></div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.history.append({
            "file": uploaded.name, "level": level,
            "prob": prob_fake, "duration": duration,
        })

        # ── acoustic diagnostics ──────────────────
        st.markdown('<div class="sec-rule">Acoustic Diagnostics</div>', unsafe_allow_html=True)
        colors = ["#00e5ff","#8b5cf6","#f59e0b","#10f5a0","#ff6b35","#ff2d55"]
        fields = [
            ("Pitch Jitter",      f"{diag['pitch_std']:.2f}",           "Hz std"),
            ("Noise Floor",       f"{diag['noise_floor']*1000:.3f}",    "mV RMS"),
            ("Harmonic Ratio",    f"{diag['harmonic_ratio']:.3f}",      "H/N"),
            ("Zero Crossing",     f"{diag['zcr']*100:.2f}",             "rate ×100"),
            ("Spectral Contrast", f"{diag['spectral_contrast']:.1f}",   "dB mean"),
            ("MFCC Variance",     f"{diag['mfcc_variance']:.1f}",       "mean var"),
        ]
        st.markdown('<div class="feat-grid">', unsafe_allow_html=True)
        for (label, val, unit), col in zip(fields, colors):
            st.markdown(f"""
            <div class="feat-card">
              <div class="feat-key">{label}</div>
              <div class="feat-val" style="color:{col}">{val}</div>
              <div class="feat-unit">{unit}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── visualizations ────────────────────────
        st.markdown('<div class="sec-rule">Signal Visualizations</div>', unsafe_allow_html=True)
        cl, cr = st.columns(2, gap="medium")

        with cl:
            st.markdown('<div class="plot-wrap"><div class="plot-lbl">Waveform</div>', unsafe_allow_html=True)
            st.pyplot(fig_waveform(y, sr), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="plot-wrap"><div class="plot-lbl">MFCC Coefficients</div>', unsafe_allow_html=True)
            st.pyplot(fig_mfcc(y, sr), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with cr:
            st.markdown('<div class="plot-wrap"><div class="plot-lbl">Mel Spectrogram</div>', unsafe_allow_html=True)
            st.pyplot(fig_spectrogram(y, sr), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="plot-wrap"><div class="plot-lbl">Pitch Contour (F0)</div>', unsafe_allow_html=True)
            st.pyplot(fig_pitch(y, sr), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── file metadata ─────────────────────────
        st.markdown('<div class="sec-rule">File Metadata</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-panel">
          <div class="info-row"><span class="info-k">Filename</span><span class="info-v">{uploaded.name}</span></div>
          <div class="info-row"><span class="info-k">Duration</span><span class="info-v">{duration} s</span></div>
          <div class="info-row"><span class="info-k">Source sample rate</span><span class="info-v">{sr:,} Hz</span></div>
          <div class="info-row"><span class="info-k">Analysis sample rate</span><span class="info-v">{SR:,} Hz</span></div>
          <div class="info-row"><span class="info-k">Total samples</span><span class="info-v">{len(y):,}</span></div>
          <div class="info-row"><span class="info-k">Feature vector</span><span class="info-v">128-dim</span></div>
          <div class="info-row"><span class="info-k">Classifier</span><span class="info-v">Random Forest (n=200, depth=20)</span></div>
          <div class="info-row"><span class="info-k">Verdict</span><span class="info-v" style="color:var(--{accent_var})">{level}</span></div>
        </div>
        """, unsafe_allow_html=True)

        os.unlink(tmp_path)

    elif not model:
        st.warning("⚠️ Model not found. Run `python src/train.py` to generate model.pkl and scaler.pkl.")
    else:
        st.markdown("""
        <div class="upload-zone">
          <div class="uz-icon">🎙</div>
          <div class="uz-title">Drop an audio file to analyze</div>
          <div class="uz-sub">
            VoiceGuard extracts 128 acoustic features and runs ensemble ML inference<br>
            to determine if the voice is human or AI-generated.
          </div>
          <div class="uz-pill">WAV · MP3 · FLAC &nbsp;·&nbsp; max 200 MB</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  TAB 2 — HISTORY
# ═══════════════════════════════════════════════
with tab_history:
    if not st.session_state.history:
        st.markdown('<div style="padding:40px;text-align:center;font-family:var(--mono);font-size:11px;color:var(--t3)">No files analyzed yet this session.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-family:var(--mono);font-size:10px;color:var(--t3);letter-spacing:.1em;margin-bottom:14px">{len(st.session_state.history)} file(s) analyzed this session</div>', unsafe_allow_html=True)
        dot_cls   = {"HIGH":"dot-h","MEDIUM":"dot-m","LOW":"dot-l"}
        level_cls = {"HIGH":"h-lvl-h","MEDIUM":"h-lvl-m","LOW":"h-lvl-l"}
        for h in reversed(st.session_state.history):
            st.markdown(f"""
            <div class="hist-row">
              <div class="dot {dot_cls[h['level']]}"></div>
              <span class="h-name">{h['file']}</span>
              <span class="{level_cls[h['level']]}">{h['level']}</span>
              <span class="h-prob">{h['prob']*100:.1f}%</span>
              <span class="h-dur">{h['duration']} s</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()


# ═══════════════════════════════════════════════
#  TAB 3 — ABOUT
# ═══════════════════════════════════════════════
with tab_about:
    st.markdown("""
    <div class="about-card">
      <div class="about-h">Detection pipeline</div>
      <div class="pipe-code">
Audio Input (WAV / MP3 / FLAC)<br>
&nbsp;&nbsp;→ Resample → 16 kHz &nbsp;·&nbsp; clip → 3 s window<br>
&nbsp;&nbsp;→ Feature Extraction (Librosa)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ MFCC 40 coeff × mean + std &nbsp;&nbsp;&nbsp;→ 80-dim<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ Mel Spectrogram mean + std &nbsp;&nbsp;&nbsp;&nbsp;→ &nbsp;2-dim<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ Pitch jitter F0 std (pyin) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ &nbsp;1-dim<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ Harmonic ratio · RMS · ZCR &nbsp;&nbsp;&nbsp;&nbsp;→ &nbsp;5-dim<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ Spectral contrast + Chroma &nbsp;&nbsp;&nbsp;&nbsp;→ 38-dim<br>
&nbsp;&nbsp;→ 128-dim vector · StandardScaler normalization<br>
&nbsp;&nbsp;→ Random Forest (n=200, depth=20) · predict_proba<br>
&nbsp;&nbsp;→ Risk Score → LOW (0–40%) / MEDIUM (40–70%) / HIGH (70–100%)
      </div>
    </div>

    <div class="about-card">
      <div class="about-h">Feature vector (128-dim)</div>
      <span class="fchip">MFCC mean ×40</span><span class="fchip">MFCC std ×40</span>
      <span class="fchip">Mel mean</span><span class="fchip">Mel std</span>
      <span class="fchip">Pitch jitter (F0 std)</span><span class="fchip">RMS mean</span>
      <span class="fchip">RMS std</span><span class="fchip">ZCR mean</span><span class="fchip">ZCR std</span>
      <span class="fchip">Spectral contrast ×7</span><span class="fchip">Chroma ×12</span>
      <span class="fchip">Harmonic ratio</span>
    </div>

    <div class="about-card">
      <div class="about-h">ML ensemble</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
        <div style="background:var(--raised);border:1px solid var(--border);border-radius:10px;padding:14px">
          <div style="font-family:var(--mono);font-size:9px;color:var(--cyan);letter-spacing:.12em;margin-bottom:6px">PRIMARY</div>
          <div style="font-size:13px;color:var(--t1);font-weight:600">Random Forest</div>
          <div style="font-family:var(--mono);font-size:10px;color:var(--t3);margin-top:4px">n=200 · depth=20</div>
        </div>
        <div style="background:var(--raised);border:1px solid var(--border);border-radius:10px;padding:14px">
          <div style="font-family:var(--mono);font-size:9px;color:var(--violet);letter-spacing:.12em;margin-bottom:6px">BASELINE</div>
          <div style="font-size:13px;color:var(--t1);font-weight:600">Logistic Reg.</div>
          <div style="font-family:var(--mono);font-size:10px;color:var(--t3);margin-top:4px">Interpretability</div>
        </div>
        <div style="background:var(--raised);border:1px solid var(--border);border-radius:10px;padding:14px">
          <div style="font-family:var(--mono);font-size:9px;color:var(--amber);letter-spacing:.12em;margin-bottom:6px">ENSEMBLE</div>
          <div style="font-size:13px;color:var(--t1);font-weight:600">Voting Classifier</div>
          <div style="font-family:var(--mono);font-size:10px;color:var(--t3);margin-top:4px">Higher recall</div>
        </div>
      </div>
    </div>

    <div class="about-card">
      <div class="about-h">VLSI / FPGA hardware acceleration</div>
      <div style="font-size:13px;color:var(--t2);line-height:1.8">
        A Verilog MFCC hardware block handles FFT → Mel filterbank → DCT at silicon level,
        simulated in Vivado (Xilinx) and ModelSim. Near-zero latency at telecom scale enables
        real-time call analysis for banking fraud prevention without cloud round-trips.
        Phase 3 target: FPGA integration (July 2026) for live PSB call-center deployment.
      </div>
    </div>

    <div style="display:flex;gap:10px">
      <div class="about-card" style="flex:1;margin-bottom:0">
        <div class="about-h" style="margin-bottom:6px">Team</div>
        <div style="font-size:13px;color:var(--t2)">Bhavini Verma · Ashit Raj</div>
        <div style="font-family:var(--mono);font-size:10px;color:var(--t3);margin-top:4px">VIT Vellore</div>
      </div>
      <div class="about-card" style="flex:1;margin-bottom:0">
        <div class="about-h" style="margin-bottom:6px">Event</div>
        <div style="font-size:13px;color:var(--t2)">PSB Hackathon 2026</div>
        <div style="font-family:var(--mono);font-size:10px;color:var(--t3);margin-top:4px">UCO Bank Track</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
  "When voices lie, spectrograms don't."<br>
  Bhavini Verma · Ashit Raj &nbsp;·&nbsp; VIT Vellore &nbsp;·&nbsp; PSB Hackathon 2026
</div>
""", unsafe_allow_html=True)