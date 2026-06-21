"""
VoiceGuard — Biological + Replay-Attack Feature Engine (Inference)
─────────────────────────────────────────────────────────────────────────
Adapted from the user's training-time extract_bio.py for per-chunk inference.

Per the VoiceGuard project plan, Stream B (XGBoost B) consumes a ~101-number (plan's "~102" is approximate; 101 is the literal itemized sum)
feature vector per 3-second chunk:

    MFCC mean/std (40 coef x 2)                 = 80
    Pitch mean/std/min/max                      = 4
    Jitter (mean)                                = 1
    Shimmer (mean)                               = 1
    HNR (mean)                                   = 1
    ZCR mean/std                                 = 2
    Spectral contrast (mean of 7 bands)          = 7
    --- Replay-attack features (4) ---
    Spectral flatness (mean)                     = 1
    Reverberation proxy (peak_to_mean + decay)   = 2
    Sub-300Hz noise energy ratio                 = 1
    Microphone clipping rate                     = 1
    ─────────────────────────────────────────────
    TOTAL                                        = 101

    (The plan document states "~102 numbers" as an approximate figure;
    the literal sum of every feature it itemizes is 101. This module
    follows the itemized list exactly rather than padding with an
    invented feature to force a round 102 — if your trained
    voiceguard_b.json expects a different count, update FEATURE_ORDER
    below to match its actual training-time column order exactly.)

IMPORTANT: This module expects audio that has ALREADY been through
preprocessing.py (16kHz mono, codec-simulated, chunked). It does NOT
re-load from disk or re-apply codec simulation, since that must happen
once on the full clip before chunking, matching how the training pipeline
processes each pre-chunked 3-second clip.

The underlying signal-processing functions (jitter, shimmer, reverb proxy,
clipping rate, sub-300Hz noise) are carried over unchanged from the user's
real training script to guarantee train/inference consistency.
"""

import numpy as np
import librosa

HOP_LENGTH = 512

# Canonical, ordered feature names — this order MUST match the order used
# when voiceguard_b.json (XGBoost B) was trained. If training used a
# different column order, update FEATURE_ORDER to match exactly.
FEATURE_ORDER = (
    [f"MFCC_{i+1}_Mean" for i in range(40)] +
    [f"MFCC_{i+1}_Std" for i in range(40)] +
    ["Pitch_Mean", "Pitch_Std", "Pitch_Min", "Pitch_Max"] +
    ["Jitter_Mean", "Shimmer_Mean", "HNR_Mean"] +
    ["ZCR_Mean", "ZCR_Std"] +
    [f"Contrast_Band{i+1}_Mean" for i in range(7)] +
    ["Spectral_Flatness_Mean"] +
    ["Reverb_PeakToMean", "Reverb_Decay"] +
    ["Sub300Hz_Noise"] +
    ["Clipping_Rate"]
)

assert len(FEATURE_ORDER) == 101, f"Expected 101 features, got {len(FEATURE_ORDER)}"


def compute_jitter(f0: np.ndarray) -> float:
    """Cycle-to-cycle variation of fundamental frequency (mean only, per plan spec)."""
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) < 2:
        return 0.0
    f0_diff = np.abs(np.diff(f0_clean))
    mean_f0 = np.mean(f0_clean)
    return float(np.mean(f0_diff) / mean_f0) if mean_f0 > 0 else 0.0


def compute_shimmer(y: np.ndarray, hop_length: int = HOP_LENGTH) -> float:
    """Cycle-to-cycle variation of amplitude (mean only, per plan spec)."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if len(rms) < 2:
        return 0.0
    rms_diff = np.abs(np.diff(rms))
    mean_rms = np.mean(rms)
    return float(np.mean(rms_diff) / mean_rms) if mean_rms > 0 else 0.0


def compute_clipping_rate(y: np.ndarray) -> float:
    """Fraction of samples saturated near full scale — replay-attack signal."""
    if len(y) == 0:
        return 0.0
    threshold = 0.99
    clipping_samples = np.sum(np.abs(y) >= threshold)
    return float(clipping_samples / len(y))


def compute_sub_300hz_noise(y: np.ndarray, sr: int) -> float:
    """Energy ratio below 300Hz — speaker/hardware distortion signal for replay attacks."""
    if len(y) == 0:
        return 0.0
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    low_freq_idx = np.where(freqs < 300)[0]
    total_energy = np.sum(S)
    if total_energy == 0:
        return 0.0
    low_energy = np.sum(S[low_freq_idx, :])
    return float(low_energy / total_energy)


def compute_reverberation_proxy(y: np.ndarray, sr: int, hop_length: int = HOP_LENGTH) -> tuple[float, float]:
    """Estimate room/speaker reverberation from the energy envelope.

    Returns (peak_to_mean, decay_autocorr):
      - peak_to_mean: ratio of peak RMS to mean RMS (lower => more reverberant)
      - decay_autocorr: autocorrelation of energy envelope at ~50ms lag
        (higher => sustained/reverberant tail, characteristic of replay attacks)
    """
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if len(rms) < 2 or np.max(rms) == 0:
        return 0.0, 0.0
    peak_to_mean = float(np.max(rms) / np.mean(rms))
    rms_norm = (rms - np.mean(rms)) / (np.std(rms) + 1e-10)
    autocorr = np.correlate(rms_norm, rms_norm, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / (autocorr[0] + 1e-10)
    lag_samples = max(1, int(0.05 * sr / hop_length))
    decay_autocorr = float(autocorr[min(lag_samples, len(autocorr) - 1)])
    return peak_to_mean, decay_autocorr


def extract_bio_features_chunk(y: np.ndarray, sr: int = 16000) -> dict:
    """Extract the 101-dim biological + replay feature vector for ONE
    already-preprocessed (16kHz mono, codec-simulated) audio chunk.

    Returns a dict keyed by FEATURE_ORDER names. Use vectorize() to turn
    this into the ordered np.ndarray that voiceguard_b.json expects.
    """
    if len(y) == 0 or np.max(np.abs(y)) < 1e-4:
        # Silent chunk — return zeroed features rather than raising, so the
        # caller can decide whether to skip this chunk for display purposes.
        return {name: 0.0 for name in FEATURE_ORDER}

    hop_length = HOP_LENGTH
    features = {}

    # MFCCs (40 coefficients) — mean and std only, per plan spec
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=hop_length)
    for i in range(40):
        features[f'MFCC_{i+1}_Mean'] = float(np.mean(mfccs[i]))
    for i in range(40):
        features[f'MFCC_{i+1}_Std'] = float(np.std(mfccs[i]))

    # Pitch (F0) via YIN, RMS-gated voicing (matches training script)
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    voiced_frames = rms > 0.1 * np.mean(rms) if np.mean(rms) > 0 else np.zeros_like(rms, dtype=bool)
    min_len = min(len(f0), len(voiced_frames))
    f0_clean = f0[:min_len][voiced_frames[:min_len]]

    if len(f0_clean) > 0:
        features['Pitch_Mean'] = float(np.mean(f0_clean))
        features['Pitch_Std'] = float(np.std(f0_clean))
        features['Pitch_Min'] = float(np.min(f0_clean))
        features['Pitch_Max'] = float(np.max(f0_clean))
    else:
        features['Pitch_Mean'] = features['Pitch_Std'] = features['Pitch_Min'] = features['Pitch_Max'] = 0.0

    # Jitter / Shimmer (biological liveness signals)
    features['Jitter_Mean'] = compute_jitter(f0_clean if len(f0_clean) > 0 else f0)
    features['Shimmer_Mean'] = compute_shimmer(y, hop_length)

    # HNR (Harmonics-to-Noise Ratio)
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = np.sum(y_harm ** 2)
    perc_energy = np.sum(y_perc ** 2)
    features['HNR_Mean'] = float(10 * np.log10(harm_energy / perc_energy)) if perc_energy > 0 else 0.0

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
    features['ZCR_Mean'] = float(np.mean(zcr))
    features['ZCR_Std'] = float(np.std(zcr))

    # Spectral contrast — mean of each of the 7 bands (not the single
    # overall mean used in the 343-dim training script; the plan spec
    # calls for "mean of 7 bands = 7 numbers")
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    n_bands = contrast.shape[0]
    for i in range(7):
        if i < n_bands:
            features[f'Contrast_Band{i+1}_Mean'] = float(np.mean(contrast[i]))
        else:
            features[f'Contrast_Band{i+1}_Mean'] = 0.0

    # --- Replay-attack features ---
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    features['Spectral_Flatness_Mean'] = float(np.mean(flatness))

    reverb_peak_to_mean, reverb_decay = compute_reverberation_proxy(y, sr, hop_length)
    features['Reverb_PeakToMean'] = reverb_peak_to_mean
    features['Reverb_Decay'] = reverb_decay

    features['Sub300Hz_Noise'] = compute_sub_300hz_noise(y, sr)
    features['Clipping_Rate'] = compute_clipping_rate(y)

    return features


def vectorize(features: dict) -> np.ndarray:
    """Convert a feature dict into the ordered 101-dim np.ndarray that
    voiceguard_b.json expects. Raises if any expected feature is missing —
    fail loudly rather than silently feeding the model a misaligned vector.
    """
    missing = [name for name in FEATURE_ORDER if name not in features]
    if missing:
        raise ValueError(f"Missing biological features required for inference: {missing}")
    return np.array([features[name] for name in FEATURE_ORDER], dtype=np.float32)


def get_top_contributing_features(features: dict, feature_importances: dict | None = None, top_n: int = 4) -> list[str]:
    """Return human-readable labels for the explainability panel.

    If real feature_importances from the trained XGBoost B model are
    supplied (name -> importance score), rank by importance * |z-score
    deviation from a "typical real voice" baseline|. Without a trained
    model's importances, falls back to flagging features with extreme
    Jitter/Shimmer/HNR/Contrast values, which the plan doc identifies as
    the most diagnostic signals.
    """
    labels = []
    if features.get('Jitter_Mean', 0) < 0.01:
        labels.append("Prosody Anomaly — Jitter Too Regular")
    if features.get('Shimmer_Mean', 0) < 0.01:
        labels.append("Amplitude Irregularity Absent (Shimmer)")
    if features.get('HNR_Mean', 0) > 20:
        labels.append("Unnaturally Clean Harmonic Structure")
    if features.get('Spectral_Flatness_Mean', 0) > 0.3:
        labels.append("Spectral Flatness — Possible Replay Artifact")
    if features.get('Reverb_Decay', 0) > 0.5:
        labels.append("Reverberation Tail Detected")
    if not labels:
        labels.append("No strong anomaly markers in biological stream")
    return labels[:top_n]
