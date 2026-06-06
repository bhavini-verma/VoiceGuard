#  VoiceGuard — Audio Forensics for Voice Security

> **"When voices lie, spectrograms don't."**

AI voice clone detection using MFCC + Random Forest — real-time 10s risk scoring with Streamlit UI and VLSI hardware acceleration.

---

##  Problem Statement

AI-generated voice clones are enabling a new wave of financial fraud:
- **$5.8B** lost annually to voice-based fraud
- **83%** of financial institutions report increased voice spoofing attacks
- Existing systems verify *identity* — VoiceGuard verifies *authenticity*

Attackers use tools like ElevenLabs, Play.ht, Murf.ai, and RVC to clone voices and bypass KYC/IVR systems at banks and call centers.

---

##  System Pipeline

```
 AUDIO INPUT →  PRE-PROCESS → FEATURE EXTRACT →  ML CLASSIFIER →  RISK SCORE
  (PCM/WAV)       (Noise filter +    (Spectrogram +        (Random Forest +    (≤10 seconds)
                   normalization)      MFCC, Librosa)        Ensemble)
```

**Output Layer:**
| Risk Level | Action |
|---|---|
| 🟢 LOW | Call proceeds normally |
| 🟡 MEDIUM | Secondary verification triggered |
| 🔴 HIGH | Call blocked + alert raised |

---

##  Features Extracted

128-dimensional normalized feature vector per audio clip:

| Feature | Method |
|---|---|
| MFCC (40 coefficients) | `librosa.feature.mfcc(n_mfcc=40)` → mean + std |
| Mel Spectrogram | `librosa.stft()` → `power_to_db()` |
| Pitch Jitter & Shimmer | `librosa.pyin()` |
| Noise Floor | Estimation from silent frames |
| Harmonic Ratio | Harmonic-to-noise ratio |
| Zero Crossing Rate | `librosa.feature.zero_crossing_rate()` |
| Spectral Contrast | `librosa.feature.spectral_contrast()` |
| Chroma Features | `librosa.feature.chroma_stft()` |

---

##  ML Models

| Model | Role | Target |
|---|---|---|
| Random Forest (`n_estimators=200`) | Primary classifier | >92% accuracy |
| Logistic Regression | Baseline + interpretability | — |
| Ensemble Voting Classifier | Higher fraud recall | Minimize FN |

**Key metric: Equal Error Rate (EER)** — minimizing false negatives (AI voice passing as real) is the priority.

---

##  Repository Structure

```
VoiceGuard/
├── data/
│   ├── real/              # Bonafide audio clips (WAV, 16kHz)
│   └── fake/              # AI-generated / spoofed audio clips
├── features/
│   └── features.csv       # 128-dim vectors, label: 0=real / 1=fake
├── models/
│   ├── model.pkl          # Trained Random Forest classifier
│   └── scaler.pkl         # StandardScaler for inference
├── src/
│   ├── extract.py         # Feature extraction pipeline
│   ├── train.py           # Model training + evaluation
│   └── predict.py         # audio_path → Low / Med / High risk score
├── ui/
│   └── app.py             # Streamlit dashboard (live + upload mode)
├── verilog/
│   └── mfcc_block.v       # Verilog MFCC: FFT → Mel filterbank → DCT
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── test_report.csv    # Accuracy, FP, FN per voice source
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+

```bash
# Clone repo
git clone https://github.com/<your-username>/VoiceGuard.git
cd VoiceGuard

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
librosa
numpy
scipy
scikit-learn
pyaudio
streamlit
matplotlib
seaborn
joblib
imbalanced-learn
```

---

##  Usage

### 1. Feature Extraction
```bash
python src/extract.py
# Output: features/features.csv
```

### 2. Train Model
```bash
python src/train.py
# Output: models/model.pkl, models/scaler.pkl
# Prints: accuracy, F1, EER, confusion matrix
```

### 3. Predict on a file
```bash
python src/predict.py --audio path/to/audio.wav
# Output: RISK: HIGH (0.87 confidence)
```

### 4. Launch Streamlit UI
```bash
streamlit run ui/app.py
```
Live mic mode: captures 10-second window → displays risk badge + spectrogram  
Upload mode: drop any WAV file → instant analysis

---

##  Test Results

| Voice Source | Expected | Result |
|---|---|---|
| Real human voices (10+) | 🟢 LOW | ✅ |
| ElevenLabs generated | 🔴 HIGH | ✅ |
| Play.ht / Murf.ai | 🔴 HIGH | ✅ |
| Google TTS | 🔴 HIGH | ✅ |
| RVC voice clone | 🔴 HIGH | ✅ |
| Noisy environment | 🟡 MEDIUM | ✅ |
| Short clips (<5s) | Edge case | Tested |

---

##  VLSI / FPGA Hardware Acceleration

The MFCC feature extraction block is implemented in Verilog for near-zero latency at telecom scale:

```
FFT Module → Mel Filterbank → DCT → 40-coefficient MFCC output
```

- Simulated in **Vivado (Xilinx)** / **ModelSim**
- Output verified against Python `librosa` MFCC reference
- Timing analysis: latency vs software baseline documented in `results/`

---

##  Roadmap

| Phase | Timeline | Scope |
|---|---|---|
| Phase 1 — MVP | Now | Live capture, MFCC, Random Forest, Streamlit UI |
| Phase 2 — Enhanced | May 2026 | CNN on spectrograms, 10K+ dataset, banking API |
| Phase 3 — Production | July 2026 | FPGA integration, telecom plugin, RBI compliance |
| Phase 4 — Scale | Dec 2026 | Multilingual (Hindi, Tamil), video deepfake, SaaS |

---

##  Competitive Advantage

| Feature | VoiceGuard | Traditional Systems |
|---|---|---|
| Detection window | ≤10 seconds | Minutes / manual |
| Hardware acceleration | VLSI / FPGA | CPU only |
| Real-time risk scoring | ✅ 3-tier | ❌ |
| Open-source datasets | ASVspoof 2019 + WaveFake | Proprietary |
| Regulatory alignment | RBI-ready (Phase 3) | Varies |

---

##  Datasets

- [ASVspoof 2019 LA](https://www.asvspoof.org/) — Primary (bonafide + spoofed, ~15GB)
- [WaveFake](https://github.com/RUB-SysSec/WaveFake) — Secondary diversity dataset

---
