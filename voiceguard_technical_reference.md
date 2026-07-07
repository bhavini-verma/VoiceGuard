# 📘 VoiceGuard: Deep Technical Reference & System Architecture Manual

This document provides a comprehensive, micro-level technical reference for the VoiceGuard platform. It details the system architecture, file index, mathematical formulations, feature extraction pipelines, and machine learning structures to prepare you for any advanced evaluation panel questions.

---

## 🗂️ Section 1: Complete Workspace File Index

### 1. Root & Core Scripts
*   **[fastapi_app.py](file:///C:/Users/rajas/.gemini/antigravity-ide/scratch/VoiceGaurd_TelephonyExp/fastapi_app.py):** The main production FastAPI application server. It loads models, handles audio preprocessing, routes calls to feature extractors, evaluates risk classification verdicts, supports active learning feedback, and hosts the UI static assets.
*   **[fastapi_app_meta.py](file:///C:/Users/rajas/.gemini/antigravity-ide/scratch/VoiceGaurd_TelephonyExp/fastapi_app_meta.py):** Development reference FastAPI app server implementing the calibrated metadata pipeline.
*   **[run_pipeline.py](file:///C:/Users/rajas/.gemini/antigravity-ide/scratch/VoiceGaurd_TelephonyExp/run_pipeline.py):** Orchestrates the full model retraining and evaluation suite. Sequentially runs XGBoost base training (`train.py`), calibration calibration core (`train_meta.py`), and error analysis checks (`diagnose_failures.py`).

### 2. Core Source Directory (`src/`)
*   **[extract_bio.py](file:///C:/Users/rajas/.gemini/antigravity-ide/scratch/VoiceGaurd_TelephonyExp/src/extract_bio.py):** Implements physics-based digital signal processing (DSP) to extract 345 biological acoustic features (pitch, jitter, shimmer, HNR, formants) from standard audio files.
*   **[extract_deep.py](file:///C:/Users/rajas/.gemini/antigravity-ide/scratch/VoiceGaurd_TelephonyExp/src/extract_deep.py):** Handles the deep learning feature extraction pipeline. Standardizes audio and passes it through the Wav2Vec2 transformer network to extract 4096D Multi-Condition Training (MCT) features.
*   **[contrastive_head.py](file:///C:/Users/rajas/.gemini/antigravity-ide/scratch/VoiceGaurd_TelephonyExp/src/contrastive_head.py):** Defines the PyTorch neural network class `ContrastiveProjectionHead` and triplet dataset loaders. Trains this projection head using Triplet Loss.
*   **[train.py](file:///C:/Users/rajas/.gemini/antigravity-ide/scratch/VoiceGaurd_TelephonyExp/src/train.py):** Trains the base biological (`xgb_bio`) and deep (`xgb_deep`) XGBoost classifiers.
*   **[train_meta.py](file:///C:/Users/rajas/.gemini/antigravity-ide/scratch/VoiceGaurd_TelephonyExp/src/train_meta.py):** Fits the Isotonic Regression calibration wrappers and computes the optimal fusion weights.
*   **[format_dataset.py](file:///C:/Users/rajas/.gemini/antigravity-ide/scratch/VoiceGaurd_TelephonyExp/src/format_dataset.py):** Ingests raw audio and resamples/standardizes it into 16kHz mono PCM_16 WAV format.

### 3. Scratch & Diagnostics Directory (`scratch/`)
*   **[diagnose_failures.py](file:///C:/Users/rajas/.gemini/antigravity-ide/scratch/VoiceGaurd_TelephonyExp/scratch/diagnose_failures.py):** Test harness that runs model predictions over out-of-sample Hindi and English datasets to measure category-wise EER.
*   **[cleanup_telugu.py](file:///C:/Users/rajas/.gemini/antigravity-ide/scratch/VoiceGaurd_TelephonyExp/scratch/cleanup_telugu.py):** Cleans up incorrect duplicates and purges corrupted database entries.
*   **[import_telugu_real.py](file:///C:/Users/rajas/.gemini/antigravity-ide/scratch/VoiceGaurd_TelephonyExp/scratch/import_telugu_real.py):** Imports and resamples genuine Telugu voice notes.

---

## 🔊 Section 2: Dual-Stream Extraction Pipeline (The Mechanics)

### 1. Biological Stream (345 Dimensions)
This stream extracts physical characteristics of human speech production (vocal cords, larynx length, nasal resonance). Fakes lack natural physical irregularities.

*   **Pitch (F0) Tracking:** Computed using the Probabilistic YIN (`pyin`) algorithm. We extract the mean, standard deviation, minimum, and maximum of the fundamental frequency ($F_0$).
*   **Jitter (Cycle-to-Cycle Periodicity Variation):** Measures pitch frequency instability:
    $$\text{Jitter(Absolute)} = \frac{1}{N-1} \sum_{i=1}^{N-1} |T_i - T_{i+1}|$$
    where $T_i$ is the period duration of pitch cycle $i$.
*   **Shimmer (Cycle-to-Cycle Amplitude Variation):** Measures amplitude instability of vocal fold vibration:
    $$\text{Shimmer(dB)} = \frac{1}{N-1} \sum_{i=1}^{N-1} 20 \log_{10} \left( \frac{A_{i+1}}{A_i} \right)$$
    where $A_i$ is the peak amplitude of pitch cycle $i$.
*   **Harmonics-to-Noise Ratio (HNR):** Quantifies the ratio of periodic harmonic energy to noise. Breathiness or AI generation artifacts lower this ratio.
*   **Formant Tracking:** Estimating resonant frequencies ($F_1, F_2, F_3$) of the vocal tract using Linear Predictive Coding (LPC) coefficients.

### 2. Deep Stream (4096 Dimensions - Multi-Condition Training)
Captures high-level semantic and channel representation features using deep speech models.

*   **Wav2Vec2 Speech Backbone:** Uses `indicwav2vec-hindi` (a convolutional + transformer architecture). The raw audio is passed through, and we extract hidden states from the **12th layer** (the final encoder layer outputting the most refined contextual speech embeddings).
*   **Pooling:** Mean pooling (1024D) and Standard Deviation pooling (1024D) are computed across the temporal frame length, yielding a **2048-dimensional** representation.
*   **Multi-Condition Training (MCT) Simulation:** To handle telephony channel degradation:
    1.  **Clean Pass:** Runs inference on clean 16kHz audio.
    2.  **Degraded Pass:** Simulates G.711/AMR narrow-band telephony line by applying an 80Hz highpass filter, resampling to 8kHz, applying a 4kHz lowpass filter (Butterworth 4th order), and resampling back to 16kHz.
    3.  **Concatenation:** Stacks clean (2048D) and degraded (2048D) features to form a **4,096-dimensional** vector. This makes the model robust to mobile noise.

---

## 🧠 Section 3: Machine Learning Models & Mathematics

```
                ┌───────────────────────────────────┐
                │        Raw 16kHz WAV Audio        │
                └─────────────────┬─────────────────┘
                                  ▼
         ┌─────────────────────────────────────────────────┐
         │              Dual-Stream Extraction             │
         └────────┬───────────────────────────────┬────────┘
                  │ (Biological)                  │ (Deep MCT)
                  ▼                               ▼
     ┌────────────────────────┐      ┌─────────────────────────┐
     │ 345D Acoustic Features │      │  4096D Wav2Vec2 Vector  │
     └────────────┬───────────┘      └────────────┬────────────┘
                  │                               ▼
                  │                  ┌─────────────────────────┐
                  │                  │ PyTorch Contrastive Head│
                  │                  └────────────┬────────────┘
                  │                               │ (128D Projected)
                  ▼                               ▼
     ┌────────────────────────┐      ┌─────────────────────────┐
     │  xgb_bio Classifier    │      │   xgb_deep Classifier   │
     └────────────┬───────────┘      └────────────┬────────────┘
                  │ (Probability)                 │ (Probability)
                  ▼                               ▼
     ┌─────────────────────────────────────────────────────────┐
     │        Calibrated Fusion Core (Isotonic / Platt)        │
     │      p_fused = 0.764 * p_bio + 0.236 * p_deep           │
     └────────────────────────────┬────────────────────────────┘
                                  ▼
                   [ Risk Verdict & RBI SOP ]
```

### 1. Custom PyTorch Contrastive Head
The raw 4096D deep feature vectors are high-dimensional and prone to overfitting. We train a multi-layer perceptron (MLP) to project this data into a lower-dimensional space.
*   **Architecture:**
    *   `Linear (4096 -> 512)` ➔ `Batch Normalization` ➔ `ReLU` ➔ `Dropout (p=0.2)` ➔ `Linear (512 -> 128)`.
    *   Output vector is L2 normalized onto a hypersphere: $\hat{\mathbf{z}} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2}$.
*   **Triplet Loss Function:**
    $$L(A, P, N) = \max(d(a, p) - d(a, n) + \alpha, 0)$$
    where $d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2$ is the Euclidean distance, $A$ is the anchor (genuine), $P$ is positive (another genuine), $N$ is negative (synthetic), and $\alpha = 0.2$ is the margin. This clusters human voices close together and pushes AI fakes far away.

### 2. XGBoost Base Classifiers
*   We train two separate Extreme Gradient Boosting (`XGBClassifier`) models on the tabular features:
    1.  `xgb_bio`: Input = 345D biological features.
    2.  `xgb_deep`: Input = 128D projected embeddings from the contrastive head.
*   XGBoost builds sequential decision trees using gradient descent on the log-loss objective function, handling feature correlation and high dimensional variance much better than linear classifiers.

### 3. Platt Scaling & Isotonic Regression (Calibration Core)
Classifier raw prediction scores are often uncalibrated (they don't represent real-world probability). To calibrate predictions:
*   **Platt Scaling:** Fits a logistic regression model over the validation logit predictions:
    $$P(y=1 | s) = \frac{1}{1 + \exp(A \cdot s + B)}$$
*   **Isotonic Regression:** Fits a non-decreasing piecewise constant function to minimize mean squared error:
    $$\min \sum (y_i - f(s_i))^2 \quad \text{subject to } f(s_i) \le f(s_j) \text{ if } s_i \le s_j$$
*   **Fusion Core Formula:**
    $$P_{\text{fused}} = w_{\text{bio}} \cdot P_{\text{calibrated\_bio}} + w_{\text{deep}} \cdot P_{\text{calibrated\_deep}}$$
    where the optimal weights calculated from dataset calibration are:
    $$w_{\text{bio}} = 0.764, \quad w_{\text{deep}} = 0.236$$

---

## 🎨 Section 4: Advanced Frontend & System Logic

### 1. Client-Side Fuzzy Token Matcher (Levenshtein Distance)
The floating AI chatbot widget features a client-side search engine. To process search queries containing spelling mistakes, it calculates the **Levenshtein Distance** between search tokens and database query keywords.
*   **Levenshtein Distance Algorithm:** Given string $A$ of length $m$ and string $B$ of length $n$, the distance $D(i, j)$ is computed dynamically:
    $$D(i, j) = \min \begin{cases} 
      D(i-1, j) + 1 \\
      D(i, j-1) + 1 \\
      D(i-1, j-1) + \text{cost}
    \end{cases}$$
    where $\text{cost} = 0$ if $A[i] = B[j]$ else $1$.
*   **Word Similarity Score:**
    $$\text{Similarity}(w_1, w_2) = 1.0 - \frac{\text{Levenshtein}(w_1, w_2)}{\max(\text{len}(w_1), \text{len}(w_2))}$$
    Any query with similarity $> 0.8$ triggers a match in the knowledge base.

### 2. Disagreement Alarm Override Logic
To catch spoofing attacks targeting specific streams, the backend implements override rules:
*   **Deep Stream Disagreement (Synthetic Telephony):** If $P_{\text{deep}} > 0.70$ and $P_{\text{bio}} < 0.30$, the overall fused risk score is overwritten to the maximum value:
    $$P_{\text{fused}} = \max(P_{\text{deep}}, P_{\text{fused}})$$
    This overrides the biological stream and flags the transaction as **SUSPICIOUS (Medium Risk)** with the label *"Deep Stream Disagreement (AI Voice)"*.
*   **Bio Stream Disagreement:** If $P_{\text{bio}} > 0.70$ and $0.15 < P_{\text{deep}} < 0.30$, the overall fused score is overwritten:
    $$P_{\text{fused}} = \max(P_{\text{bio}}, P_{\text{fused}})$$
    This flags the transaction as **SUSPICIOUS (Medium Risk)** with the label *"Bio Stream Disagreement (AI Voice)"*.

### 3. Active Learning Workflow
```
[User Analyst Flags Failure] 
       │
       ▼
[Save features to hard_val_bio.csv / hard_val_deep.csv]
       │
       ▼
[Trigger /retrain endpoint] ➔ [Run train_meta.py] ➔ [Reload Calibrated Wrappers]
```
The active learning system ensures the models are updated with analyst feedback:
1.  When an analyst flags a classification discrepancy on the dashboard, the backend appends the extracted features to `hard_val_bio.csv` and `hard_val_deep.csv` along with their correct labels.
2.  Calling the `/retrain` endpoint triggers `src/train_meta.py` on the server.
3.  The Isotonic regression calibrators are re-trained on this updated feature dataset, and the new models are reloaded without restarting the FastAPI server.
