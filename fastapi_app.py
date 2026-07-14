import os
import sys
import json
import time
import torch
import librosa
import logging
import numpy as np
import scipy.io.wavfile as wavfile
import joblib
import uuid
import pandas as pd
import xgboost as xgb
import threading
from fastapi import FastAPI, UploadFile, File, Request, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from auth import verify_api_key
import warnings

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger("voiceguard")

warnings.filterwarnings('ignore')

# Add src to path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'src'))
from extract_bio import extract_bio_features, simulate_phone_codec
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from sklearn.base import BaseEstimator, ClassifierMixin

class BinaryWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, multiclass_xgb, contrastive_head=None):
        self.multiclass_xgb = multiclass_xgb
        self.contrastive_head = contrastive_head
        self.classes_ = np.array([0, 1])
        if contrastive_head is not None:
            self.feature_names_in_ = [f'Proj_Deep_{i}' for i in range(128)]
            self.n_features_in_ = 128
        else:
            self.feature_names_in_ = getattr(multiclass_xgb, 'feature_names_in_', None)
            self.n_features_in_ = getattr(multiclass_xgb, 'n_features_in_', None)

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        if self.contrastive_head is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.contrastive_head.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                X_proj = self.contrastive_head(X_tensor).cpu().numpy()
            proj_cols = [f'Proj_Deep_{i}' for i in range(128)]
            X_df = pd.DataFrame(X_proj, columns=proj_cols)
            p_raw = self.multiclass_xgb.predict_proba(X_df)
        else:
            p_raw = self.multiclass_xgb.predict_proba(X)
            
        p_real = p_raw[:, 0]
        p_fake = 1.0 - p_real
        return np.column_stack([p_real, p_fake])

    def predict(self, X):
        p_proba = self.predict_proba(X)
        return (p_proba[:, 1] >= 0.5).astype(int)

app = FastAPI(
    title="VoiceGuard AI - Dual-Stream Voice Deepfake Detection API",
    description="Real-time multilingual voice spoofing detection for banking telephony & Video-KYC.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State for Models
class ModelState:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.w2v_model = None
        self.xgb_bio = None
        self.xgb_deep = None
        self.calibrated_bio = None
        self.calibrated_deep = None
        self.meta_clf = None
        self.meta_cfg = None
        self.bio_feature_cols = []
        self.deep_feature_cols = []
        self.contrastive_head = None
        self.is_loaded = False

model_state = ModelState()

def compute_snr(y):
    if len(y) == 0:
        return 0.0
    signal_power = np.mean(y**2)
    noise_power = np.var(y) - signal_power if np.var(y) > signal_power else np.var(y) * 0.1
    if noise_power <= 0:
        return 40.0
    snr = 10 * np.log10(signal_power / noise_power + 1e-9)
    return float(np.clip(snr, -10, 40))

def compute_silence_ratio(y, sr):
    if len(y) == 0:
        return 0.0
    rms = librosa.feature.rms(y=y)[0]
    if len(rms) == 0:
        return 0.0
    threshold = np.percentile(rms, 20)
    silence_frames = np.sum(rms < threshold)
    return float(silence_frames / len(rms))

@app.on_event("startup")
def load_models():
    print(f"Loading models on {model_state.device}...")
    
    # 1. Load Deep Model (Wav2Vec2 feature extractor)
    model_local_path = os.path.join(base_dir, 'models', 'indicwav2vec-hindi')
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_state.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_local_path, local_files_only=True)
    model_state.w2v_model = AutoModel.from_pretrained(model_local_path, output_hidden_states=True, local_files_only=True).to(model_state.device)
    model_state.w2v_model.eval()

    # 2. Load Base XGBoost Models (raw predictions for fixed weight)
    model_state.xgb_bio = xgb.XGBClassifier()
    try:
        model_state.xgb_bio.load_model(os.path.join(base_dir, 'models', 'xgb_bio.json'))
        if model_state.xgb_bio.get_booster().feature_names:
            model_state.bio_feature_cols = model_state.xgb_bio.get_booster().feature_names
    except Exception as e:
        print(f"XGB Bio load failed: {e}")
        
    model_state.xgb_deep = xgb.XGBClassifier()
    try:
        model_state.xgb_deep.load_model(os.path.join(base_dir, 'models', 'xgb_deep.json'))
        if model_state.xgb_deep.get_booster().feature_names:
            model_state.deep_feature_cols = model_state.xgb_deep.get_booster().feature_names
    except Exception as e:
        print(f"XGB Deep load failed: {e}")
        
    # Load Contrastive Projection Head
    try:
        from contrastive_head import ContrastiveProjectionHead
        model_state.contrastive_head = ContrastiveProjectionHead().to(model_state.device)
        model_state.contrastive_head.load_state_dict(torch.load(os.path.join(base_dir, 'models', 'contrastive_head.pt'), map_location=model_state.device))
        model_state.contrastive_head.eval()
        print("Contrastive projection head loaded successfully!")
    except Exception as e:
        print(f"Contrastive projection head load failed: {e}")

    # 3. Load Calibrated Wrappers
    try:
        model_state.calibrated_bio = joblib.load(os.path.join(base_dir, 'models', 'calibrated_bio.pkl'))
        model_state.calibrated_deep = joblib.load(os.path.join(base_dir, 'models', 'calibrated_deep.pkl'))
    except Exception as e:
        logger.warning(f"Calibrated classifiers load failed: {e}")

    # 4. Load Meta-Classifier (kept for reference)
    try:
        model_state.meta_clf = joblib.load(os.path.join(base_dir, 'models', 'meta_classifier.pkl'))
        with open(os.path.join(base_dir, 'models', 'meta_config.json'), 'r') as f:
            model_state.meta_cfg = json.load(f)
    except Exception as e:
        logger.warning(f"Meta-classifier load failed: {e}")

    # 5. Warm Up Deep Learning Models to prevent cold start latency
    if model_state.w2v_model is not None and model_state.processor is not None and model_state.contrastive_head is not None:
        try:
            logger.info("Warming up Deep Learning models on startup...")
            dummy_audio = np.zeros((16000,), dtype=np.float32)
            dummy_input = model_state.processor([dummy_audio], sampling_rate=16000, return_tensors="pt")
            dummy_input = {k: v.to(model_state.device) for k, v in dummy_input.items()}
            with torch.no_grad():
                if model_state.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        out = model_state.w2v_model(**dummy_input)
                else:
                    out = model_state.w2v_model(**dummy_input)
                h12 = out.hidden_states[12]
                
                # Warm up contrastive head
                dummy_deep_feats = torch.zeros((1, 4096), dtype=torch.float32).to(model_state.device)
                _ = model_state.contrastive_head(dummy_deep_feats)
            logger.info("Deep Learning models warmed up successfully!")
        except Exception as warmup_err:
            logger.warning(f"Model warmup failed: {warmup_err}")

    model_state.is_loaded = True
    logger.info("Models loaded successfully!")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "backend_api": "CONNECTED",
        "models_loaded": model_state.is_loaded,
        "device": str(model_state.device),
        "version": "2.0.0"
    }

@app.get("/model-status")
def model_status(auth: str = Depends(verify_api_key)):
    return {
        "feature_counts": {
            "biological_features": len(model_state.bio_feature_cols) if model_state.bio_feature_cols else 97,
            "deep_features": len(model_state.deep_feature_cols) if model_state.deep_feature_cols else 4096
        },
        "model_loaded": model_state.is_loaded,
        "inference_ready": model_state.is_loaded
    }

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...), auth: str = Depends(verify_api_key)):
    logger.info("Request Received")
    start_time = time.time()
    
    # Read the file content
    content = await file.read()
    logger.info("Audio Uploaded")
    
    # Clean the audio content from trailing junk/watermarks (e.g. OPPO watermark)
    oppo_idx = content.find(b"oppoMark")
    if oppo_idx != -1:
        logger.info(f"Found oppoMark at byte index {oppo_idx}. Truncating trailing metadata.")
        content = content[:oppo_idx]
        
    # Detect the correct extension based on magic bytes and filename
    ext = ".wav"
    if content.startswith(b"RIFF") and len(content) > 12 and content[8:12] == b"WAVE":
        ext = ".wav"
    elif content.startswith(b"ID3") or content.startswith(b"\xff\xfb") or content.startswith(b"\xff\xf3") or content.startswith(b"\xff\xf2"):
        ext = ".mp3"
    elif content.startswith(b"\x1a\x45\xdf\xa3"):
        ext = ".webm"
    elif content.startswith(b"fLaC"):
        ext = ".flac"
    elif content.startswith(b"OggS"):
        ext = ".ogg"
    else:
        # Fall back to original file extension
        _, file_ext = os.path.splitext(file.filename)
        if file_ext:
            ext = file_ext.lower()
            
    temp_path = os.path.join(base_dir, "scratch", f"temp_{uuid.uuid4().hex}{ext}")
    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        logger.info("Preprocessing Started")
        # Load audio robustly: Try librosa first (for MP3/OGG/FLAC), then fallback to scipy (for manual JS WAVs)
        try:
            y, sr = librosa.load(temp_path, sr=16000, mono=True)
        except Exception as e_librosa:
            try:
                orig_sr, y = wavfile.read(temp_path)
                y = y.astype(np.float32)
                if np.abs(y).max() > 1.5:
                    y = y / 32768.0
                if len(y.shape) > 1:
                    y = y.mean(axis=1)
                if orig_sr != 16000:
                    y = librosa.resample(y, orig_sr=orig_sr, target_sr=16000)
                sr = 16000
            except Exception as e_scipy:
                debug_path = os.path.join(base_dir, "scratch", f"failed_live_audio_{uuid.uuid4().hex}{ext}")
                import shutil
                shutil.copy(temp_path, debug_path)
                logger.error(f"FAILED TO READ AUDIO. Saved to {debug_path}")
                raise ValueError(f"Audio format not supported. librosa error: {e_librosa}. scipy error: {e_scipy}")
            
        orig_duration = len(y) / sr
        duration = orig_duration
        
        # TRUNCATE TO MAX 10 SECONDS to prevent extreme latency on long files
        max_duration = 10.0
        if duration > max_duration:
            logger.info(f"Audio is {duration:.1f}s. Truncating to {max_duration}s for real-time latency.")
            y = y[:int(max_duration * sr)]
        
        # Apply 80Hz high-pass filter to remove low-frequency room rumble & fan hum
        from scipy.signal import butter, lfilter
        import soundfile as sf
        
        def butter_highpass(cutoff, fs, order=1):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return b, a
            
        def highpass_filter(data, cutoff=80, fs=16000, order=1):
            b, a = butter_highpass(cutoff, fs, order=order)
            return lfilter(b, a, data)
            
        y = highpass_filter(y, cutoff=80, fs=sr)
        
        # Loudness/Peak normalization (scale peak to 0.95 to match training dataset levels)
        peak = np.max(np.abs(y))
        if peak > 1e-4:
            y = (y / peak) * 0.95
            
        # Save the filtered & normalized audio back to a WAV file so bio-extraction and librosa load it correctly
        processed_path = temp_path.replace(ext, "_processed.wav")
        sf.write(processed_path, y, sr, format='WAV', subtype='PCM_16')
        
        # Remove original raw file and point temp_path to the processed WAV path
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        temp_path = processed_path
        
        # Audio Quality Metrics
        signal_power = compute_snr(y) # SNR placeholder
        silence_ratio = compute_silence_ratio(y, sr)
        file_size = os.path.getsize(temp_path)
        
        # 1. Bio extraction
        logger.info("Extracting Biological Features")
        bio_feats = extract_bio_features(temp_path)
        if not bio_feats:
            raise ValueError("Bio feature extraction failed")
        bio_data = {c: [bio_feats.get(c, 0.0)] for c in model_state.bio_feature_cols}
        df_bio_input = pd.DataFrame(bio_data)
        
        # 2. Deep extraction
        logger.info("Extracting Deep Features")
        # Dual-pass Wav2Vec2 extraction (MCT)
        y_for_deep = y[:10*16000].astype(np.float32) if len(y) > 10*16000 else y.astype(np.float32)
        
        # Filter matching extract_deep.py
        b_lp, a_lp = butter(4, 4000.0 / (16000.0 / 2.0), btype='low')
        y_degraded = lfilter(b_lp, a_lp, y_for_deep).astype(np.float32)

        def wav2vec_pass(audio_array):
            inp = model_state.processor([audio_array], sampling_rate=16000, return_tensors="pt", padding=True)
            inp = {k: v.to(model_state.device) for k, v in inp.items()}
            with torch.no_grad():
                if model_state.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        out = model_state.w2v_model(**inp)
                else:
                    out = model_state.w2v_model(**inp)
            h12 = out.hidden_states[12]
            mean = torch.mean(h12, dim=1).float().cpu().numpy()[0]  # 1024D
            std  = torch.std(h12,  dim=1).float().cpu().numpy()[0]  # 1024D
            return np.concatenate([mean, std])  # 2048D

        feats_clean    = wav2vec_pass(y_for_deep)   # 2048D
        feats_degraded = wav2vec_pass(y_degraded)    # 2048D
        feats_mct      = np.concatenate([feats_clean, feats_degraded])  # 4096D

        # Build 4096D feature DataFrame
        deep_feats_dict = {}
        for k in range(4096):
            deep_feats_dict[f'Deep_{k}'] = [float(feats_mct[k])]
        df_deep_input = pd.DataFrame(deep_feats_dict)

        # 3. Base Stream Predictions (raw multiclass probabilities)
        if model_state.xgb_bio is not None:
            p_bio_classes = model_state.xgb_bio.predict_proba(df_bio_input)[0]
            p_bio = float(1.0 - p_bio_classes[0])
        else:
            p_bio_classes = np.array([0.5, 0.16, 0.16, 0.18])
            p_bio = 0.5

        if model_state.xgb_deep is not None:
            # Feed raw 4096D features aligned to model's feature names (primary route)
            deep_cols = model_state.deep_feature_cols if model_state.deep_feature_cols else [f'Deep_{k}' for k in range(4096)]
            for c in deep_cols:
                if c not in df_deep_input.columns:
                    df_deep_input[c] = 0.0
            df_deep_input_aligned = df_deep_input[deep_cols]
            p_deep_classes = model_state.xgb_deep.predict_proba(df_deep_input_aligned)[0]
            p_deep = float(1.0 - p_deep_classes[0])
        else:
            p_deep_classes = np.array([0.5, 0.16, 0.16, 0.18])
            p_deep = 0.5
        
        # Save temp features for feedback loop
        df_bio_input.to_csv(os.path.join(base_dir, "scratch", "temp_bio.csv"), index=False)
        df_deep_input.to_csv(os.path.join(base_dir, "scratch", "temp_deep.csv"), index=False)

        # 4. Decision Fusion (uses Robust 5-input Meta-Classifier if trained, otherwise falls back to Fixed-Weight)
        logger.info("Running Fusion Model")
        using_meta = False
        if model_state.meta_clf is not None and model_state.meta_cfg is not None:
            try:
                meta_features = pd.DataFrame([{
                    'bio_score': p_bio,
                    'deep_score': p_deep,
                    'disagreement': abs(p_bio - p_deep),
                    'max_score': max(p_bio, p_deep),
                    'min_score': min(p_bio, p_deep)
                }])
                p_fused = float(model_state.meta_clf.predict_proba(meta_features)[0][1])
                t_high = model_state.meta_cfg.get("threshold_high", 0.54)
                t_mid = model_state.meta_cfg.get("threshold_mid", 0.30)
                using_meta = True
                logger.info("Flipped prediction to Robust 5-input Meta Classifier")
            except Exception as meta_err:
                logger.warning(f"Meta-classifier inference failed: {meta_err}. Falling back to fixed weight.")

        if not using_meta:
            try:
                with open(os.path.join(base_dir, 'models', 'fusion_weights.json'), 'r') as f:
                    weights = json.load(f)
                w_deep = weights.get('w_deep', 0.236)
                w_bio = weights.get('w_bio', 0.764)
            except:
                w_deep = 0.236
                w_bio = 0.764
            p_fused = w_deep * p_deep + w_bio * p_bio
            t_high = 0.54
            t_mid = 0.30
        
        # ──────────────────────────────────────────────────────
        # 5-TIER RISK CLASSIFICATION WITH EXPLAINABILITY
        # ──────────────────────────────────────────────────────
        logger.info("Generating Risk Score & Explanation")
        disagreement = abs(p_bio - p_deep)

        # ── Disagreement override: elevate fused score when streams conflict ──
        if disagreement > 0.30:
            p_fused = max(p_bio, p_deep, p_fused)

        # ── 5-tier mapping ──
        def classify_risk(score):
            if score >= 0.75:
                return "CRITICAL", "HIGH", "🔴 Critical Risk — AI-Generated Audio Detected"
            elif score >= 0.50:
                return "HIGH_RISK", "HIGH", "🟠 High Risk — Strong Synthesis Indicators"
            elif score >= 0.25:
                return "MODERATE", "MEDIUM", "🟡 Moderate Risk — Analyst Review Required"
            elif score >= 0.10:
                return "LOW_RISK", "LOW", "🟢 Low Risk — Minor Anomalies Detected"
            else:
                return "CLEAR", "LOW", "🟢 Clear — Genuine Human Voice"

        verdict, risk_level, label = classify_risk(p_fused)

        # ── Dynamic confidence (clamped 70–99.9%) ──
        raw_conf = 70.0 + 30.0 * (abs(p_fused - 0.5) / 0.5)
        confidence = round(min(99.9, max(70.0, raw_conf)), 1)

        # ── Explainability flags ──
        flags = []
        risk_factors = []
        mitigating_factors = []

        # Stream agreement analysis
        if disagreement > 0.30:
            flags.append("HIGH_DISAGREEMENT")
            if p_bio > p_deep:
                risk_factors.append(f"High model disagreement ({disagreement*100:.1f}%) — Bio stream flags synthetic patterns the Deep stream missed")
            else:
                risk_factors.append(f"High model disagreement ({disagreement*100:.1f}%) — Deep stream detects vocoder artifacts the Bio stream missed")
        elif disagreement < 0.10:
            flags.append("STREAMS_AGREE")
            if p_fused < 0.25:
                mitigating_factors.append("Both analysis streams agree this is genuine human speech")
            else:
                risk_factors.append("Both analysis streams independently detect synthesis indicators")

        # Stream-specific borderline detection
        if 0.35 <= p_bio <= 0.65:
            flags.append("BIO_BORDERLINE")
            risk_factors.append(f"Bio stream score ({p_bio*100:.1f}%) is in the uncertain zone — biological features are inconclusive")
        elif p_bio > 0.65:
            flags.append("BIO_ALERT")
            risk_factors.append(f"Bio stream detects strong synthetic patterns ({p_bio*100:.1f}%)")
        elif p_bio < 0.15:
            mitigating_factors.append(f"Bio stream is confident this is human ({p_bio*100:.1f}%)")

        if 0.35 <= p_deep <= 0.65:
            flags.append("DEEP_BORDERLINE")
            risk_factors.append(f"Deep stream score ({p_deep*100:.1f}%) is in the uncertain zone — Wav2Vec2 embeddings are inconclusive")
        elif p_deep > 0.65:
            flags.append("DEEP_ALERT")
            risk_factors.append(f"Deep stream detects strong vocoder artifacts ({p_deep*100:.1f}%)")
        elif p_deep < 0.15:
            mitigating_factors.append(f"Deep stream is confident this is human ({p_deep*100:.1f}%)")

        # Bio-specific feature flags (analyze individual bio features for explainability)
        bio_feats_raw = bio_feats if bio_feats else {}
        jitter_val = bio_feats_raw.get('Jitter_Mean', None)
        shimmer_val = bio_feats_raw.get('Shimmer_Mean', None)
        hnr_val = bio_feats_raw.get('HNR_Mean', None)
        pitch_std_val = bio_feats_raw.get('Pitch_Std', None)
        flatness_val = bio_feats_raw.get('Flatness_Mean', None)

        if jitter_val is not None and jitter_val < 0.003:
            flags.append("LOW_JITTER")
            risk_factors.append(f"Jitter is unusually low ({jitter_val:.4f}) — synthetic voices often lack natural pitch perturbation")
        if shimmer_val is not None and shimmer_val < 0.02:
            flags.append("LOW_SHIMMER")
            risk_factors.append(f"Shimmer is unusually low ({shimmer_val:.4f}) — synthetic voices have unnaturally stable amplitude")
        if hnr_val is not None and hnr_val > 30:
            flags.append("HIGH_HNR")
            risk_factors.append(f"HNR is unusually high ({hnr_val:.1f} dB) — voice is unnaturally clean, typical of neural vocoders")
        if pitch_std_val is not None and pitch_std_val < 5.0:
            flags.append("FLAT_PITCH")
            risk_factors.append(f"Pitch variation is very low ({pitch_std_val:.1f} Hz) — monotone delivery typical of early TTS")
        if flatness_val is not None and flatness_val < 0.001:
            flags.append("LOW_SPECTRAL_FLATNESS")
            risk_factors.append("Spectral flatness is very low — unnaturally tonal, possible formant synthesis artifact")

        # Confidence qualifier
        if p_fused > 0.85 or p_fused < 0.10:
            flags.append("HIGH_CONFIDENCE")
        elif 0.30 <= p_fused <= 0.60:
            flags.append("LOW_CONFIDENCE")
            risk_factors.append("Fusion score is in the uncertain range — treat verdict with caution")

        # ── Stream assessment text ──
        def stream_assessment(score, stream_name):
            if score < 0.10:
                return f"Confident genuine — no {stream_name} anomalies detected"
            elif score < 0.25:
                return f"Likely genuine — minor {stream_name} anomalies present"
            elif score < 0.50:
                return f"Borderline — {stream_name} patterns are atypical but not conclusive"
            elif score < 0.75:
                return f"Likely synthetic — {stream_name} detects significant anomalies"
            else:
                return f"Confident synthetic — strong {stream_name} indicators of AI generation"

        def agreement_level(d):
            if d < 0.10:
                return f"HIGH ({d*100:.1f}% disagreement)"
            elif d < 0.20:
                return f"MODERATE ({d*100:.1f}% disagreement)"
            elif d < 0.30:
                return f"LOW ({d*100:.1f}% disagreement)"
            else:
                return f"CRITICAL ({d*100:.1f}% disagreement)"

        # ── Primary concern text ──
        if disagreement > 0.30 and p_bio > p_deep:
            primary_concern = "Bio stream detects potential synthesis artifacts that the Deep stream does not recognize — possible novel vocoder not in training data"
        elif disagreement > 0.30 and p_deep > p_bio:
            primary_concern = "Deep stream detects vocoder artifacts that biological features miss — possible high-quality clone with natural prosody"
        elif p_fused >= 0.50:
            primary_concern = "Both streams indicate this audio contains synthetic speech characteristics"
        elif p_fused >= 0.25:
            primary_concern = "Inconclusive analysis — audio shows some atypical patterns that warrant human review"
        else:
            primary_concern = "No significant synthesis indicators detected"

        # ── Recommendation ──
        RECOMMENDATIONS = {
            "CRITICAL": {"action": "BLOCK", "urgency": "CRITICAL", "detail": "Immediately block and escalate. Both analysis streams detect strong AI generation signatures. Flag source number for investigation."},
            "HIGH_RISK": {"action": "BLOCK", "urgency": "HIGH", "detail": "Block the call and queue for supervisor review. Significant synthesis indicators detected across the analysis pipeline."},
            "MODERATE": {"action": "REVIEW", "urgency": "MEDIUM", "detail": "Route to analyst queue for manual spectrogram inspection. Audio shows patterns that may indicate a synthesis method not yet fully characterized."},
            "LOW_RISK": {"action": "ALLOW_MONITOR", "urgency": "LOW", "detail": "Allow the call but log for pattern analysis. Minor anomalies detected that are likely benign but worth tracking."},
            "CLEAR": {"action": "ALLOW", "urgency": "NONE", "detail": "No action required. Audio is consistent with natural human speech across all analysis dimensions."}
        }
        rec = RECOMMENDATIONS[verdict]

        # Override recommendation for disagreement cases
        if "HIGH_DISAGREEMENT" in flags and verdict in ("CLEAR", "LOW_RISK"):
            rec = {"action": "REVIEW", "urgency": "MEDIUM", "detail": "Model disagreement exceeds safety threshold. Route to analyst for manual review despite low fusion score."}
            verdict = "MODERATE"
            risk_level = "MEDIUM"
            label = "🟡 Moderate Risk — High Model Disagreement"

        # ── Build explanation object ──
        explanation = {
            "summary": primary_concern,
            "flags": flags,
            "stream_analysis": {
                "bio_score": round(p_bio * 100, 1),
                "bio_assessment": stream_assessment(p_bio, "biological/prosodic"),
                "deep_score": round(p_deep * 100, 1),
                "deep_assessment": stream_assessment(p_deep, "vocoder/spectral"),
                "agreement": agreement_level(disagreement)
            },
            "risk_factors": risk_factors if risk_factors else ["No significant risk factors identified"],
            "mitigating_factors": mitigating_factors if mitigating_factors else ["No specific mitigating factors"]
        }

        # Chunk results (used by UI plots)
        chunk_results = [
            {"index": 1, "score": p_fused * 100, "confidence": confidence, "verdict": verdict, "start": 0, "end": round(duration, 1), "bio_score": p_bio * 100, "deep_score": p_deep * 100}
        ]

        result = {
            "case_id": f"VG-{uuid.uuid4().hex[:6].upper()}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fraud_score": round(p_fused * 100, 1),
            "confidence": confidence,
            "risk_level": risk_level,
            "risk_tier": verdict,
            "verdict": verdict,
            "verdict_label": label,
            "primary_trigger": "Robust 5-input Meta Classifier" if using_meta else "Fixed Weight Fusion",
            "secondary_trigger": "Wav2Vec2 Anomaly" if p_deep > p_bio else "Bio Feature Anomaly",
            "explanation": explanation,
            "recommendation": rec["detail"] if isinstance(rec, dict) else rec,
            "using_real_models": True,
            "metadata": {"filename": file.filename, "format": "WAV", "sample_rate": 16000, "channels": 1, "duration": round(duration, 2), "file_size_bytes": file_size},
            "performance": {"model_version": "v3.0-5TierExplainable" if using_meta else "v3.0-FixedWeight", "inference_time_ms": int((time.time() - start_time) * 1000), "audio_duration_sec": round(duration, 2), "chunks_processed": 1},
            "threat_intel": {
                "threat_type": "AI Voice Clone" if verdict not in ("CLEAR", "LOW_RISK") else "None",
                "sophistication": "Advanced" if verdict in ("CRITICAL", "HIGH_RISK") else ("Moderate" if verdict == "MODERATE" else "None"),
                "replay_indicators": "None",
                "synthetic_confidence": round(float(max(p_deep_classes[1:])) * 100, 1) if verdict not in ("CLEAR", "LOW_RISK") else 0.0
            },
            "chunk_results": chunk_results
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    logger.info("Returning JSON Response")
    return JSONResponse(content=result)

@app.post("/feedback")
async def process_feedback(is_correct: str = Form(...), true_label: str = Form(...), auth: str = Depends(verify_api_key)):
    is_correct = (is_correct.lower() == 'true')
    label = 0 if true_label == "Real" else 1
    
    if true_label == "Synthetic":
        prefix = "active_learning_synthetic"
    elif true_label == "Generic":
        prefix = "active_learning_generic"
    else:
        prefix = "active_learning_real"
    
    bio_temp = os.path.join(base_dir, "scratch", "temp_bio.csv")
    deep_temp = os.path.join(base_dir, "scratch", "temp_deep.csv")
    
    if os.path.exists(bio_temp) and os.path.exists(deep_temp):
        df_bio_input = pd.read_csv(bio_temp)
        df_deep_input = pd.read_csv(deep_temp)
        
        df_bio_input['Label'] = label
        df_deep_input['Label'] = label
        df_bio_input['Filename'] = f"{prefix}_{int(time.time())}.wav"
        df_deep_input['Filename'] = f"{prefix}_{int(time.time())}.wav"
        
        hard_val_bio = os.path.join(base_dir, 'features', 'hard_val_bio.csv')
        hard_val_deep = os.path.join(base_dir, 'features', 'hard_val_deep.csv')
        
        # Align columns to prevent format corruption when appending via mode='a'
        if os.path.exists(hard_val_bio):
            existing_bio = pd.read_csv(hard_val_bio, nrows=0)
            for col in existing_bio.columns:
                if col not in df_bio_input.columns:
                    df_bio_input[col] = 0.0
            df_bio_input = df_bio_input[existing_bio.columns]
            
        if os.path.exists(hard_val_deep):
            existing_deep = pd.read_csv(hard_val_deep, nrows=0)
            for col in existing_deep.columns:
                if col not in df_deep_input.columns:
                    df_deep_input[col] = 0.0
            df_deep_input = df_deep_input[existing_deep.columns]
            
        df_bio_input.to_csv(hard_val_bio, mode='a', header=not os.path.exists(hard_val_bio), index=False)
        df_deep_input.to_csv(hard_val_deep, mode='a', header=not os.path.exists(hard_val_deep), index=False)
        
        return {"status": "success", "message": f"Feedback received. Added to hard validation data as label {label}."}
    else:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No recent analysis found."})

@app.post("/retrain")
async def retrain_model(auth: str = Depends(verify_api_key)):
    logger.info("Retraining meta-classifier triggered...")
    # Call the training script
    os.system(f"{sys.executable} src/train_meta.py")
    
    # Reload the model
    try:
        load_models()
        return {"status": "success", "message": "Classifier retrained and reloaded successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/generate-report")
async def generate_report(auth: str = Depends(verify_api_key)):
    return {"status": "Placeholder for PDF report"}

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(base_dir, "static", "voiceguard_uco_bank_platform.html"))

# Serve the static UI files
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    
    # Open Chrome in a separate thread after a short delay
    def open_chrome():
        time.sleep(2)  # Wait 2 seconds for server to start
        try:
            # Windows native way to open URL in default browser
            os.startfile('http://127.0.0.1:8000')
        except Exception as e:
            print(f"Failed to open browser: {e}")
            print("Open http://127.0.0.1:8000 manually in your browser")
    
    chrome_thread = threading.Thread(target=open_chrome, daemon=True)
    chrome_thread.start()
    
    uvicorn.run("fastapi_app:app", host="127.0.0.1", port=8000, reload=True)
    # Trigger model reload: 2026-06-29T20:59
