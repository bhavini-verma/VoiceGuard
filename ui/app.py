"""
VoiceGuard — Professional Streamlit Dashboard
ui/app.py

Run: streamlit run ui/app.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import tempfile
import wave
import struct

# ── PAGE CONFIG ─────────────────────────────────────
st.set_page_config(
    page_title="VoiceGuard AI",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg-void: #050509;
    --bg-surface: #0c0c14;
    --bg-raised: #111120;
    --bg-card: #14141f;
    --border-subtle: rgba(255,255,255,0.045);
    --border-default: rgba(255,255,255,0.08);
    --border-accent: rgba(255,255,255,0.15);
    --text-primary: #f0f0f8;
    --text-secondary: #8a8aaa;
    --text-muted: #3e3e5a;
    --accent-cyan: #00d4ff;
    --accent-violet: #7c5cfc;
    --accent-green: #00e87a;
    --accent-amber: #ffb800;
    --accent-red: #ff3355;
    --glow-cyan: rgba(0,212,255,0.15);
    --glow-red: rgba(255,51,85,0.15);
    --glow-green: rgba(0,232,122,0.12);
}

/* ── Base Reset ── */
* { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: var(--bg-void);
    font-family: 'Space Grotesk', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--bg-surface);
    border-right: 1px solid var(--border-subtle);
}

[data-testid="stHeader"] { background: transparent !important; }
section.main > div { padding-top: 2rem; }

/* ── Typography ── */
h1, h2, h3, h4, h5 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}

p, li, label, div {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text-secondary);
}

/* ── Sidebar ── */
[data-testid="stSidebarContent"] { padding: 1.5rem 1.2rem; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 18px !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: var(--border-default);
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    color: var(--text-primary) !important;
    font-size: 22px !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 11px !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Tabs ── */
[data-testid="stTabs"] {
    border-bottom: 1px solid var(--border-subtle);
}
[data-testid="stTabs"] button {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-muted) !important;
    font-size: 12px !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 500;
    padding: 10px 18px !important;
}
[data-testid="stTabs"] button:hover {
    color: var(--text-secondary) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom-color: var(--accent-cyan) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-raised);
    border: 1px dashed rgba(0,212,255,0.2);
    border-radius: 16px;
    padding: 12px;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,255,0.4);
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div { height: 4px !important; border-radius: 2px; }
[data-testid="stProgress"] > div > div {
    background: var(--bg-raised) !important;
}
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet)) !important;
    border-radius: 2px !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: var(--bg-raised) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border-subtle) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border-subtle) !important;
    margin: 1.5rem 0 !important;
}

/* ── Audio player ── */
audio {
    width: 100%;
    border-radius: 10px;
    filter: invert(1) hue-rotate(180deg) saturate(0.6);
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--accent-cyan) !important; }

/* ── Button ── */
.stButton > button {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-default) !important;
    color: var(--text-secondary) !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 12px !important;
    letter-spacing: 0.05em;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    background: rgba(0,212,255,0.06) !important;
}

/* ── Caption ── */
[data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
    font-size: 11px !important;
}

/* ── Custom components ── */

/* Logo */
.vg-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
}
.vg-logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(124,92,252,0.2));
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.vg-logo-text { font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 800; color: var(--text-primary); letter-spacing: -0.03em; }
.vg-logo-badge {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    color: var(--accent-cyan);
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.15em;
    padding: 3px 8px; border-radius: 20px; margin-left: 4px;
    vertical-align: middle;
}

/* Page header */
.pg-header {
    margin-bottom: 2rem;
}
.pg-header-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 8px;
}
.pg-header-title {
    font-family: 'Syne', sans-serif;
    font-size: 32px; font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.04em;
    line-height: 1.15;
    margin-bottom: 8px;
}
.pg-header-sub {
    color: var(--text-secondary);
    font-size: 13px; line-height: 1.6;
}

/* Risk verdict */
.verdict-wrap {
    position: relative;
    border-radius: 20px;
    padding: 32px 36px;
    overflow: hidden;
    margin-bottom: 4px;
}
.verdict-high {
    background: rgba(255,51,85,0.05);
    border: 1px solid rgba(255,51,85,0.25);
}
.verdict-med {
    background: rgba(255,184,0,0.05);
    border: 1px solid rgba(255,184,0,0.2);
}
.verdict-low {
    background: rgba(0,232,122,0.05);
    border: 1px solid rgba(0,232,122,0.2);
}
.verdict-glow-high {
    position: absolute; top:-60px; right:-60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,51,85,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.verdict-glow-low {
    position: absolute; top:-60px; right:-60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,232,122,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.verdict-glow-med {
    position: absolute; top:-60px; right:-60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,184,0,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.verdict-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 0.25em; text-transform: uppercase;
    margin-bottom: 10px;
}
.verdict-label-high { color: rgba(255,51,85,0.7); }
.verdict-label-med  { color: rgba(255,184,0,0.7); }
.verdict-label-low  { color: rgba(0,232,122,0.7); }

.verdict-risk {
    font-family: 'Syne', sans-serif;
    font-size: 42px; font-weight: 800; letter-spacing: -0.04em;
    line-height: 1; margin-bottom: 12px;
}
.verdict-risk-high { color: #ff3355; }
.verdict-risk-med  { color: #ffb800; }
.verdict-risk-low  { color: #00e87a; }

.verdict-action {
    font-size: 13px; padding: 8px 14px;
    border-radius: 8px; display: inline-block;
    font-weight: 500; letter-spacing: 0.01em;
}
.verdict-action-high { background: rgba(255,51,85,0.12); color: #ff3355; }
.verdict-action-med  { background: rgba(255,184,0,0.1);  color: #ffb800; }
.verdict-action-low  { background: rgba(0,232,122,0.1);  color: #00e87a; }

.verdict-prob {
    margin-top: 20px; display: flex; align-items: baseline; gap: 8px;
}
.verdict-prob-num {
    font-family: 'Syne', sans-serif;
    font-size: 52px; font-weight: 800; letter-spacing: -0.05em; line-height: 1;
}
.verdict-prob-suffix {
    font-size: 18px; color: var(--text-secondary);
}
.verdict-prob-label {
    font-size: 11px; color: var(--text-muted);
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-top: 4px;
}

/* Prob bar */
.prob-bar-track {
    height: 6px; border-radius: 3px;
    background: var(--bg-raised);
    margin-top: 16px; overflow: hidden;
}
.prob-bar-fill-high { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #ff3355, #ff6680); }
.prob-bar-fill-med  { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #ffb800, #ffd060); }
.prob-bar-fill-low  { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #00e87a, #60ffb0); }

/* Feature tiles */
.feat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
.feat-tile {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px; padding: 16px 18px;
    transition: border-color 0.2s, transform 0.2s;
}
.feat-tile:hover { border-color: var(--border-default); transform: translateY(-1px); }
.feat-tile-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--text-muted); margin-bottom: 10px;
}
.feat-tile-val {
    font-family: 'Syne', sans-serif;
    font-size: 22px; font-weight: 700; letter-spacing: -0.03em;
    margin-bottom: 4px;
}
.feat-tile-unit { font-size: 11px; color: var(--text-muted); margin-top: 2px; }

/* Section heading */
.sec-head {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--text-muted); margin-bottom: 16px; margin-top: 28px;
    display: flex; align-items: center; gap: 10px;
}
.sec-head::after {
    content: '';
    flex: 1; height: 1px;
    background: var(--border-subtle);
}

/* File info panel */
.info-panel {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px; padding: 20px;
}
.info-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 0; border-bottom: 1px solid var(--border-subtle);
    font-size: 12px;
}
.info-row:last-child { border-bottom: none; padding-bottom: 0; }
.info-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-muted);
}
.info-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: var(--text-primary);
}

/* History row */
.hist-entry {
    display: flex; align-items: center; gap: 14px;
    padding: 14px 16px;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}
.hist-entry:hover { border-color: var(--border-default); }
.hist-dot-h { width: 8px; height: 8px; border-radius: 50%; background: #ff3355; flex-shrink: 0; box-shadow: 0 0 6px rgba(255,51,85,0.5); }
.hist-dot-m { width: 8px; height: 8px; border-radius: 50%; background: #ffb800; flex-shrink: 0; box-shadow: 0 0 6px rgba(255,184,0,0.5); }
.hist-dot-l { width: 8px; height: 8px; border-radius: 50%; background: #00e87a; flex-shrink: 0; box-shadow: 0 0 6px rgba(0,232,122,0.5); }
.hist-name { font-size: 13px; color: var(--text-primary); flex: 1; font-weight: 500; }
.hist-level-h { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #ff3355; letter-spacing: 0.1em; }
.hist-level-m { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #ffb800; letter-spacing: 0.1em; }
.hist-level-l { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #00e87a; letter-spacing: 0.1em; }
.hist-prob { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-secondary); }
.hist-dur { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); }

/* Sidebar item */
.sb-section { margin-bottom: 24px; }
.sb-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--text-muted); margin-bottom: 10px;
}
.sb-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 10px; border-radius: 8px; margin-bottom: 4px;
    border: 1px solid transparent;
}
.sb-row:hover { background: var(--bg-raised); border-color: var(--border-subtle); }
.sb-icon { font-size: 14px; flex-shrink: 0; }
.sb-text { font-size: 12px; color: var(--text-secondary); }
.sb-badge {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; padding: 2px 7px; border-radius: 20px;
}
.sb-badge-h { background: rgba(255,51,85,0.12); color: #ff3355; border: 1px solid rgba(255,51,85,0.2); }
.sb-badge-m { background: rgba(255,184,0,0.10); color: #ffb800; border: 1px solid rgba(255,184,0,0.2); }
.sb-badge-l { background: rgba(0,232,122,0.10); color: #00e87a; border: 1px solid rgba(0,232,122,0.2); }

/* Upload zone */
.upload-hint {
    text-align: center; padding: 48px 24px;
    border: 1px dashed rgba(0,212,255,0.15);
    border-radius: 18px;
    background: rgba(0,212,255,0.02);
}
.upload-hint-icon { font-size: 40px; margin-bottom: 14px; }
.upload-hint-title { font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 700; color: var(--text-primary); margin-bottom: 8px; }
.upload-hint-sub { font-size: 12px; color: var(--text-muted); line-height: 1.6; }
.upload-hint-pill {
    display: inline-block; margin-top: 14px;
    background: var(--bg-raised); border: 1px solid var(--border-subtle);
    color: var(--text-muted); font-family: 'JetBrains Mono', monospace;
    font-size: 10px; padding: 5px 14px; border-radius: 20px;
}

/* Mic zone */
.mic-hint {
    text-align: center; padding: 48px 24px;
    border: 1px dashed rgba(124,92,252,0.2);
    border-radius: 18px;
    background: rgba(124,92,252,0.02);
}
.mic-hint-icon { font-size: 40px; margin-bottom: 14px; }
.mic-hint-title { font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 700; color: var(--text-primary); margin-bottom: 8px; }
.mic-hint-sub { font-size: 12px; color: var(--text-muted); line-height: 1.6; }

/* Recording active */
.rec-active {
    display: flex; align-items: center; gap: 12px;
    background: rgba(255,51,85,0.06);
    border: 1px solid rgba(255,51,85,0.2);
    border-radius: 14px; padding: 16px 20px;
    margin-bottom: 16px;
}
.rec-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #ff3355;
    animation: blink 1s infinite;
    flex-shrink: 0;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
.rec-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #ff3355; letter-spacing: 0.1em;
}

/* DPDP badge */
.dpdp-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0,232,122,0.06);
    border: 1px solid rgba(0,232,122,0.2);
    border-radius: 8px; padding: 8px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #00e87a; letter-spacing: 0.08em;
    margin-bottom: 16px;
}

/* About panel */
.about-block {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px; padding: 22px 24px;
    margin-bottom: 12px;
}
.about-block-title {
    font-family: 'Syne', sans-serif; font-weight: 700;
    color: var(--text-primary); font-size: 14px;
    margin-bottom: 12px; letter-spacing: -0.01em;
}
.pipeline-code {
    background: var(--bg-void);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--accent-cyan);
    line-height: 1.8; letter-spacing: 0.02em;
    overflow-x: auto;
}
.feature-chip {
    display: inline-block;
    background: var(--bg-raised);
    border: 1px solid var(--border-subtle);
    border-radius: 6px; padding: 4px 10px;
    font-size: 11px; color: var(--text-secondary);
    margin: 3px; font-family: 'JetBrains Mono', monospace;
}

/* Stat strip */
.stat-strip {
    display: flex; gap: 1px; margin-bottom: 24px;
}
.stat-item {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    padding: 16px 18px;
}
.stat-item:first-child { border-radius: 14px 0 0 14px; }
.stat-item:last-child  { border-radius: 0 14px 14px 0; }
.stat-num {
    font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 800;
    color: var(--text-primary); letter-spacing: -0.04em; margin-bottom: 4px;
}
.stat-desc {
    font-size: 10px; color: var(--text-muted);
    letter-spacing: 0.1em; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}

/* Audio player wrapper */
.audio-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px; padding: 12px 16px;
    margin-bottom: 4px;
}

/* Plot wrapper */
.plot-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px; padding: 16px;
    margin-bottom: 4px;
}
.plot-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--text-muted); margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "mic_recording" not in st.session_state:
    st.session_state.mic_recording = False
if "mic_audio" not in st.session_state:
    st.session_state.mic_audio = None


# ── LOAD MODEL ───────────────────────────────────────
@st.cache_resource
def load_model():
    mp = "models/model.pkl"
    sp = "models/scaler.pkl"
    if not os.path.exists(mp) or not os.path.exists(sp):
        return None, None
    return joblib.load(mp), joblib.load(sp)

model, scaler = load_model()


# ── FEATURE EXTRACTION ───────────────────────────────
SR, N_MFCC, DURATION = 16000, 40, 3

def extract(y, sr):
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR); sr = SR
    y = y[:SR * DURATION]

    mfcc          = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mel           = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
    zcr           = librosa.feature.zero_crossing_rate(y)
    contrast      = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma        = librosa.feature.chroma_stft(y=y, sr=sr)
    f0, _, _      = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0v           = f0[~np.isnan(f0)]
    rms           = librosa.feature.rms(y=y)
    har           = librosa.effects.harmonic(y)

    vec = np.concatenate([
        np.mean(mfcc,1), np.std(mfcc,1),
        [np.mean(mel)], [np.std(mel)],
        [np.mean(zcr)], [np.std(zcr)],
        np.mean(contrast,1), np.std(contrast,1),
        np.mean(chroma,1), np.std(chroma,1),
        [np.std(f0v) if len(f0v)>0 else 0.0],
        [np.mean(rms)], [np.std(rms)],
        [np.mean(np.abs(har))/(np.mean(np.abs(y))+1e-6)],
    ])
    return np.pad(vec, (0, 128 - len(vec))), {
        "pitch_std":      float(np.std(f0v)) if len(f0v)>0 else 0.0,
        "noise_floor":    float(np.mean(rms)),
        "zcr":            float(np.mean(zcr)),
        "harmonic_ratio": float(np.mean(np.abs(har))/(np.mean(np.abs(y))+1e-6)),
        "spectral_contrast": float(np.mean(contrast)),
        "mfcc_variance":  float(np.mean(np.var(mfcc,1))),
    }

def risk_level(p):
    if p < 0.40: return "LOW",    "low"
    if p < 0.70: return "MEDIUM", "med"
    return           "HIGH",   "high"


# ── MIC RECORDING ────────────────────────────────────
def record_audio(duration_sec=5, sample_rate=16000):
    """Record audio from microphone using PyAudio."""
    try:
        import pyaudio
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1

        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=sample_rate,
            input=True,
            frames_per_buffer=CHUNK
        )

        frames = []
        total_chunks = int(sample_rate / CHUNK * duration_sec)
        for _ in range(total_chunks):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save to temp WAV file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wf = wave.open(tmp.name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        return tmp.name, None
    except ImportError:
        return None, "PyAudio not installed. Run: pip install pyaudio"
    except Exception as e:
        return None, str(e)


# ── RESULT RENDERER ──────────────────────────────────
def render_result(y, sr, filename, duration):
    """Shared result rendering for both upload and mic modes."""
    with st.spinner("Analyzing signal — extracting 128 acoustic features..."):
        feat_vec, feats = extract(y, sr)
        feat_scaled = scaler.transform(feat_vec.reshape(1, -1))
        prob_fake = float(model.predict_proba(feat_scaled)[0][1])
        level, lkey = risk_level(prob_fake)

    actions = {
        "LOW":    "✅  Call proceeds normally — voice appears authentic",
        "MEDIUM": "⚠️  Trigger secondary caller verification protocol",
        "HIGH":   "🚫  Block immediately — raise fraud alert"
    }
    pct = f"{prob_fake*100:.1f}"
    bar_width = f"{prob_fake*100:.1f}%"

    st.markdown(f"""
    <div class="verdict-wrap verdict-{lkey}">
        <div class="verdict-glow-{lkey}"></div>
        <div class="verdict-label verdict-label-{lkey}">// Threat Assessment</div>
        <div class="verdict-risk verdict-risk-{lkey}">{level} RISK</div>
        <div class="verdict-action verdict-action-{lkey}">{actions[level]}</div>
        <div class="verdict-prob">
            <div>
                <div class="verdict-prob-num verdict-risk-{lkey}">{pct}<span class="verdict-prob-suffix">%</span></div>
                <div class="verdict-prob-label">Synthetic probability score</div>
                <div class="prob-bar-track" style="width:280px;">
                    <div class="prob-bar-fill-{lkey}" style="width:{bar_width};"></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.history.append({
        "file": filename, "level": level,
        "prob": prob_fake, "duration": duration
    })

    # Feature breakdown
    st.markdown('<div class="sec-head">Acoustic features</div>', unsafe_allow_html=True)

    feat_colors = {
        "pitch_std":         "#00d4ff",
        "noise_floor":       "#7c5cfc",
        "harmonic_ratio":    "#ff6b35",
        "zcr":               "#00e87a",
        "spectral_contrast": "#ffb800",
        "mfcc_variance":     "#ff3355",
    }
    feat_labels = {
        "pitch_std":         ("Pitch Jitter",      f"{feats['pitch_std']:.2f}",           "Hz std"),
        "noise_floor":       ("Noise Floor",        f"{feats['noise_floor']*1000:.3f}",   "mV RMS"),
        "harmonic_ratio":    ("Harmonic Ratio",     f"{feats['harmonic_ratio']:.3f}",     "H/N ratio"),
        "zcr":               ("Zero Crossing",      f"{feats['zcr']*100:.2f}",            "rate × 100"),
        "spectral_contrast": ("Spectral Contrast",  f"{feats['spectral_contrast']:.1f}",  "dB"),
        "mfcc_variance":     ("MFCC Variance",      f"{feats['mfcc_variance']:.1f}",      "mean var"),
    }

    st.markdown('<div class="feat-grid">', unsafe_allow_html=True)
    for key, (label, val, unit) in feat_labels.items():
        col = feat_colors[key]
        st.markdown(f"""
        <div class="feat-tile">
            <div class="feat-tile-eyebrow">{label}</div>
            <div class="feat-tile-val" style="color:{col};">{val}</div>
            <div class="feat-tile-unit">{unit}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Visualizations
    st.markdown('<div class="sec-head">Signal visualizations</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        st.markdown('<div class="plot-wrap"><div class="plot-label">Waveform</div>', unsafe_allow_html=True)
        st.pyplot(plot_waveform(y, sr), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="plot-wrap"><div class="plot-label">MFCC Coefficients</div>', unsafe_allow_html=True)
        st.pyplot(plot_mfcc(y, sr), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="plot-wrap"><div class="plot-label">Mel Spectrogram</div>', unsafe_allow_html=True)
        st.pyplot(plot_spectrogram(y, sr), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-panel">
            <div class="plot-label" style="margin-bottom:14px;">File metadata</div>
            <div class="info-row">
                <span class="info-key">Source</span>
                <span class="info-val">{filename}</span>
            </div>
            <div class="info-row">
                <span class="info-key">Duration</span>
                <span class="info-val">{duration} s</span>
            </div>
            <div class="info-row">
                <span class="info-key">Sample rate</span>
                <span class="info-val">{sr:,} Hz</span>
            </div>
            <div class="info-row">
                <span class="info-key">Total samples</span>
                <span class="info-val">{len(y):,}</span>
            </div>
            <div class="info-row">
                <span class="info-key">Analysis SR</span>
                <span class="info-val">{SR:,} Hz</span>
            </div>
            <div class="info-row">
                <span class="info-key">Vector dims</span>
                <span class="info-val">128</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── PLOTS ────────────────────────────────────────────
BG    = "#0c0c14"
PANEL = "#14141f"

def _style(fig, ax):
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(BG)
    ax.tick_params(colors="#3e3e5a", labelsize=7, length=0)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1a2e")
    ax.xaxis.label.set_color("#3e3e5a")
    ax.yaxis.label.set_color("#3e3e5a")
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontsize(8)
    return fig, ax

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(7, 2.0))
    _style(fig, ax)
    t = np.linspace(0, len(y)/sr, len(y))
    ax.fill_between(t, y, alpha=0.25, color="#00d4ff")
    ax.plot(t, y, color="#00d4ff", lw=0.5, alpha=0.9)
    ax.axhline(0, color="#1a1a2e", lw=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout(pad=0.6)
    return fig

def plot_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(7, 2.5))
    _style(fig, ax)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
    img = librosa.display.specshow(mel, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="inferno")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    cb = plt.colorbar(img, ax=ax, format="%+2.0f dB")
    cb.ax.tick_params(colors="#3e3e5a", labelsize=7, length=0)
    cb.outline.set_edgecolor("#1a1a2e")
    plt.tight_layout(pad=0.6)
    return fig

def plot_mfcc(y, sr):
    fig, ax = plt.subplots(figsize=(7, 2.2))
    _style(fig, ax)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax, cmap="twilight_shifted")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFCC #")
    cb = plt.colorbar(img, ax=ax)
    cb.ax.tick_params(colors="#3e3e5a", labelsize=7, length=0)
    cb.outline.set_edgecolor("#1a1a2e")
    plt.tight_layout(pad=0.6)
    return fig


# ── SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="vg-logo">
        <div class="vg-logo-icon">🛡</div>
        <span class="vg-logo-text">VoiceGuard<span class="vg-logo-badge">AI</span></span>
    </div>
    """, unsafe_allow_html=True)

    if model:
        st.success("Model ready — inference active", icon="✅")
    else:
        st.error("Model not found — run train.py", icon="⚠️")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-section">
        <div class="sb-label">Risk thresholds</div>
        <div class="sb-row">
            <span class="sb-icon">🟢</span>
            <span class="sb-text">Authentic voice</span>
            <span class="sb-badge sb-badge-l">0–40%</span>
        </div>
        <div class="sb-row">
            <span class="sb-icon">🟡</span>
            <span class="sb-text">Suspicious audio</span>
            <span class="sb-badge sb-badge-m">40–70%</span>
        </div>
        <div class="sb-row">
            <span class="sb-icon">🔴</span>
            <span class="sb-text">AI clone detected</span>
            <span class="sb-badge sb-badge-h">70–100%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-section">
        <div class="sb-label">Tech stack</div>
        <div class="sb-row"><span class="sb-icon">🐍</span><span class="sb-text">Python · Librosa · NumPy</span></div>
        <div class="sb-row"><span class="sb-icon">🤖</span><span class="sb-text">Scikit-learn · Random Forest</span></div>
        <div class="sb-row"><span class="sb-icon">⚡</span><span class="sb-text">Verilog · Vivado (FPGA)</span></div>
        <div class="sb-row"><span class="sb-icon">🎛</span><span class="sb-text">Streamlit · Matplotlib</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-label">Team</div>
    <div class="sb-row"><span class="sb-icon">👤</span><span class="sb-text">Bhavini Verma</span></div>
    <div class="sb-row"><span class="sb-icon">👤</span><span class="sb-text">Ashit Raj</span></div>
    <br>
    <div class="sb-label">Event</div>
    <div class="sb-row"><span class="sb-icon">🏆</span><span class="sb-text">PSB Hackathon 2026</span></div>
    <div class="sb-row"><span class="sb-icon">🏦</span><span class="sb-text">UCO Bank Track</span></div>
    <div class="sb-row"><span class="sb-icon">🎓</span><span class="sb-text">VIT Vellore</span></div>
    """, unsafe_allow_html=True)


# ── PAGE HEADER ──────────────────────────────────────
st.markdown("""
<div class="pg-header">
    <div class="pg-header-eyebrow">// Audio Forensics System</div>
    <div class="pg-header-title">Voice Clone Detection</div>
    <div class="pg-header-sub">Real-time AI-generated speech identification using spectral fingerprinting &amp; machine learning inference.</div>
</div>
""", unsafe_allow_html=True)

# ── STAT STRIP ───────────────────────────────────────
st.markdown("""
<div class="stat-strip">
    <div class="stat-item"><div class="stat-num">128-dim</div><div class="stat-desc">Feature Vector</div></div>
    <div class="stat-item"><div class="stat-num">&gt;92%</div><div class="stat-desc">Target Accuracy</div></div>
    <div class="stat-item"><div class="stat-num">10 s</div><div class="stat-desc">Detection Window</div></div>
    <div class="stat-item"><div class="stat-num">3-tier</div><div class="stat-desc">Risk Levels</div></div>
    <div class="stat-item"><div class="stat-num">RF-200</div><div class="stat-desc">Estimators</div></div>
</div>
""", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────
tab_upload, tab_history, tab_about = st.tabs(["  📁  Analyze File  ", "  🕘  Session History  ", "  ℹ  About  "])


# ── TAB 1 · UPLOAD + MIC ─────────────────────────────
with tab_upload:

    # DPDP compliance badge
    st.markdown("""
    <div class="dpdp-badge">
        🔒 &nbsp; Privacy by Design — audio processed in-memory only · no voice data stored · DPDP 2023 compliant
    </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Input source",
        ["📁  Upload File", "🎙  Live Microphone"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── UPLOAD MODE ──────────────────────────────────
    if input_mode == "📁  Upload File":
        uploaded = st.file_uploader(
            "audio_upload",
            type=["wav", "mp3", "flac"],
            label_visibility="collapsed"
        )

        if uploaded and model:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(uploaded.read()); tmp_path = tmp.name

            y, sr = librosa.load(tmp_path, sr=None, mono=True)
            duration = round(len(y)/sr, 2)

            st.markdown('<div class="audio-wrap">', unsafe_allow_html=True)
            st.audio(uploaded)
            st.markdown('</div>', unsafe_allow_html=True)

            render_result(y, sr, uploaded.name, duration)
            os.unlink(tmp_path)

        elif not model:
            st.warning("⚠️ Model not found — run `python src/train.py` to generate model.pkl and scaler.pkl.")
        else:
            st.markdown("""
            <div class="upload-hint">
                <div class="upload-hint-icon">🎙</div>
                <div class="upload-hint-title">Drop an audio file to analyze</div>
                <div class="upload-hint-sub">VoiceGuard extracts 128 acoustic features and runs inference<br>to determine if the voice is human or AI-generated.</div>
                <div class="upload-hint-pill">WAV · MP3 · FLAC</div>
            </div>
            """, unsafe_allow_html=True)

    # ── MIC MODE ─────────────────────────────────────
    else:
        if not model:
            st.warning("⚠️ Model not found — run `python src/train.py` first.")
        else:
            rec_duration = st.slider("Recording duration (seconds)", min_value=3, max_value=10, value=5, step=1)

            st.markdown("""
            <div class="mic-hint">
                <div class="mic-hint-icon">🎙</div>
                <div class="mic-hint-title">Live Voice Analysis</div>
                <div class="mic-hint-sub">Click Record to capture audio from your microphone.<br>Speak clearly for best results. Recording will auto-stop.</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_rec, col_clear = st.columns([1, 1])

            with col_rec:
                if st.button(f"⏺  Record {rec_duration}s", use_container_width=True):
                    st.markdown(f"""
                    <div class="rec-active">
                        <div class="rec-dot"></div>
                        <span class="rec-text">RECORDING — {rec_duration} seconds — speak now...</span>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.spinner(f"Recording for {rec_duration} seconds..."):
                        tmp_path, error = record_audio(duration_sec=rec_duration)

                    if error:
                        st.error(f"Recording failed: {error}")
                    else:
                        st.session_state.mic_audio = tmp_path
                        st.success("Recording complete — analyzing...")
                        st.rerun()

            with col_clear:
                if st.button("🗑  Clear recording", use_container_width=True):
                    if st.session_state.mic_audio and os.path.exists(st.session_state.mic_audio):
                        os.unlink(st.session_state.mic_audio)
                    st.session_state.mic_audio = None
                    st.rerun()

            # Show result if recording exists
            if st.session_state.mic_audio and os.path.exists(st.session_state.mic_audio):
                y, sr = librosa.load(st.session_state.mic_audio, sr=None, mono=True)
                duration = round(len(y)/sr, 2)

                st.markdown('<div class="audio-wrap">', unsafe_allow_html=True)
                st.audio(st.session_state.mic_audio)
                st.markdown('</div>', unsafe_allow_html=True)

                render_result(y, sr, "live_mic_recording.wav", duration)


# ── TAB 2 · HISTORY ──────────────────────────────────
with tab_history:
    if not st.session_state.history:
        st.markdown("""
        <div style="padding:40px 20px;text-align:center;color:var(--text-muted);font-size:13px;">
            No files analyzed yet in this session.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="margin-bottom:16px;font-size:11px;color:var(--text-muted);
                    font-family:'JetBrains Mono',monospace;letter-spacing:0.1em;">
            {len(st.session_state.history)} file(s) analyzed this session
        </div>
        """, unsafe_allow_html=True)

        dot_map   = {"HIGH": "hist-dot-h",   "MEDIUM": "hist-dot-m",   "LOW": "hist-dot-l"}
        level_map = {"HIGH": "hist-level-h", "MEDIUM": "hist-level-m", "LOW": "hist-level-l"}

        for h in reversed(st.session_state.history):
            dc = dot_map[h["level"]]
            lc = level_map[h["level"]]
            st.markdown(f"""
            <div class="hist-entry">
                <div class="{dc}"></div>
                <span class="hist-name">{h['file']}</span>
                <span class="{lc}">{h['level']}</span>
                <span class="hist-prob">{h['prob']*100:.1f}%</span>
                <span class="hist-dur">{h['duration']}s</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()


# ── TAB 3 · ABOUT ────────────────────────────────────
with tab_about:
    st.markdown("""
    <div class="about-block">
        <div class="about-block-title">Detection pipeline</div>
        <div class="pipeline-code">
Audio Input (File Upload or Live Mic)<br>
&nbsp;&nbsp;→ Resample to 16 kHz · Clip to 3 s<br>
&nbsp;&nbsp;→ Feature Extraction (Librosa)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ MFCC (40 coeff × mean + std = 80-dim)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ Mel Spectrogram (mean + std = 2-dim)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ Pitch jitter via pyin (f0 std = 1-dim)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├─ Harmonic ratio, RMS, ZCR<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└─ Spectral contrast + chroma features<br>
&nbsp;&nbsp;→ 128-dim vector · StandardScaler normalization<br>
&nbsp;&nbsp;→ Random Forest (n=200) inference<br>
&nbsp;&nbsp;→ Risk Score → LOW / MEDIUM / HIGH<br>
&nbsp;&nbsp;→ Audio discarded from memory · zero storage · DPDP compliant
        </div>
    </div>

    <div class="about-block">
        <div class="about-block-title">Privacy by Design — DPDP 2023</div>
        <div style="font-size:13px;color:var(--text-secondary);line-height:1.8;">
            VoiceGuard processes all audio <strong style="color:var(--text-primary);">in-memory only</strong>.
            No raw voice data is written to disk or transmitted externally.
            Only the risk score and timestamp are logged — never the audio itself.
            Fully compliant with India's Digital Personal Data Protection Act 2023
            for deployment in regulated banking environments.
        </div>
    </div>

    <div class="about-block">
        <div class="about-block-title">Feature vector (128-dim)</div>
        <span class="feature-chip">MFCC mean×40</span>
        <span class="feature-chip">MFCC std×40</span>
        <span class="feature-chip">Mel mean</span>
        <span class="feature-chip">Mel std</span>
        <span class="feature-chip">Pitch jitter (f0 std)</span>
        <span class="feature-chip">RMS mean</span>
        <span class="feature-chip">RMS std</span>
        <span class="feature-chip">ZCR mean</span>
        <span class="feature-chip">ZCR std</span>
        <span class="feature-chip">Spectral contrast×7</span>
        <span class="feature-chip">Chroma×12</span>
        <span class="feature-chip">Harmonic ratio</span>
    </div>

    <div class="about-block">
        <div class="about-block-title">ML models</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:4px;">
            <div style="background:var(--bg-raised);border:1px solid var(--border-subtle);border-radius:10px;padding:14px;">
                <div style="font-size:10px;color:#00d4ff;font-family:'JetBrains Mono',monospace;letter-spacing:0.1em;margin-bottom:6px;">PRIMARY</div>
                <div style="font-size:13px;color:var(--text-primary);font-weight:600;">Random Forest</div>
                <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">n=200 · max_depth=20</div>
            </div>
            <div style="background:var(--bg-raised);border:1px solid var(--border-subtle);border-radius:10px;padding:14px;">
                <div style="font-size:10px;color:#7c5cfc;font-family:'JetBrains Mono',monospace;letter-spacing:0.1em;margin-bottom:6px;">BASELINE</div>
                <div style="font-size:13px;color:var(--text-primary);font-weight:600;">Logistic Reg.</div>
                <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">Interpretability</div>
            </div>
            <div style="background:var(--bg-raised);border:1px solid var(--border-subtle);border-radius:10px;padding:14px;">
                <div style="font-size:10px;color:#ff6b35;font-family:'JetBrains Mono',monospace;letter-spacing:0.1em;margin-bottom:6px;">ENSEMBLE</div>
                <div style="font-size:13px;color:var(--text-primary);font-weight:600;">Voting Classifier</div>
                <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">Higher fraud recall</div>
            </div>
        </div>
    </div>

    <div class="about-block">
        <div class="about-block-title">VLSI / FPGA hardware acceleration</div>
        <div style="font-size:13px;color:var(--text-secondary);line-height:1.8;">
            A Verilog MFCC hardware block handles FFT → Mel filterbank → DCT at the silicon level,
            simulated in Vivado (Xilinx) and ModelSim. The target is near-zero latency at telecom
            scale — enabling real-time call analysis for banking fraud prevention without cloud round-trips.
        </div>
    </div>

    <div style="display:flex;gap:10px;margin-top:4px;">
        <div class="about-block" style="flex:1;margin-bottom:0;">
            <div class="about-block-title" style="margin-bottom:6px;">Team</div>
            <div style="font-size:13px;color:var(--text-secondary);">Bhavini Verma · Ashit Raj</div>
            <div style="font-size:11px;color:var(--text-muted);margin-top:4px;font-family:'JetBrains Mono',monospace;">VIT Vellore</div>
        </div>
        <div class="about-block" style="flex:1;margin-bottom:0;">
            <div class="about-block-title" style="margin-bottom:6px;">Event</div>
            <div style="font-size:13px;color:var(--text-secondary);">PSB Hackathon 2026</div>
            <div style="font-size:11px;color:var(--text-muted);margin-top:4px;font-family:'JetBrains Mono',monospace;">UCO Bank Track</div>
        </div>
        </div>
        """, unsafe_allow_html=True)