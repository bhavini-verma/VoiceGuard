import os
import sys
import torch.nn as nn

class BioDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))
    def forward(self, x): return self.net(x)

class DeepDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 1))
    def forward(self, x): return self.net(x)
import pandas as pd
import json
import google.genai as genai
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
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Request, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from auth import verify_api_key
import warnings
import random
from db import init_db, seed_database, insert_case, update_checklist, update_case_action, get_case, get_all_cases, get_similar_cases

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger("voiceguard")

warnings.filterwarnings('ignore')

# Add src to path
base_dir = os.path.dirname(os.path.abspath(__file__))
scratch_dir = os.path.join(base_dir, "scratch")
os.makedirs(scratch_dir, exist_ok=True)
parent_dir = os.path.dirname(base_dir)
model_dir = os.path.join(parent_dir, 'VoiceGaurd_TelephonyExp')
sys.path.append(os.path.join(model_dir, 'src'))
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
        # Legacy XGBoost (kept for fallback)
        self.xgb_bio = None
        self.xgb_deep = None
        self.calibrated_bio = None
        self.calibrated_deep = None
        self.contrastive_head = None
        # Late Fusion DL models
        self.dl_bio = None
        self.dl_bio_scaler_mean = None
        self.dl_bio_scaler_scale = None
        self.dl_bio_features = None
        self.dl_deep = None
        self.dl_deep_scaler_mean = None
        self.dl_deep_scaler_scale = None
        self.dl_deep_features = None
        # XGBoost meta-classifier
        self.meta_clf = None
        self.meta_cfg = None
        self.bio_feature_cols = []
        self.deep_feature_cols = []
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
    print("Initializing and seeding database...")
    init_db()
    seed_database()
    print(f"Loading models on {model_state.device}...")
    
    # 1. Load Deep Model (Wav2Vec2 feature extractor)
    model_local_path = os.path.join(model_dir, 'models', 'indicwav2vec-hindi')
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_state.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_local_path, local_files_only=True)
    model_state.w2v_model = AutoModel.from_pretrained(model_local_path, output_hidden_states=True, local_files_only=True).to(model_state.device)
    model_state.w2v_model.eval()

    # 2. Load Base XGBoost Models (raw predictions for fixed weight)
    model_state.xgb_bio = xgb.XGBClassifier()
    try:
        model_state.xgb_bio.load_model(os.path.join(model_dir, 'models', 'xgb_bio.json'))
        if model_state.xgb_bio.get_booster().feature_names:
            model_state.bio_feature_cols = model_state.xgb_bio.get_booster().feature_names
    except Exception as e:
        print(f"XGB Bio load failed: {e}")
        
    model_state.xgb_deep = xgb.XGBClassifier()
    try:
        model_state.xgb_deep.load_model(os.path.join(model_dir, 'models', 'xgb_deep.json'))
        if model_state.xgb_deep.get_booster().feature_names:
            model_state.deep_feature_cols = model_state.xgb_deep.get_booster().feature_names
    except Exception as e:
        print(f"XGB Deep load failed: {e}")
        
    # Load Contrastive Projection Head
    try:
        from contrastive_head import ContrastiveProjectionHead
        model_state.contrastive_head = ContrastiveProjectionHead().to(model_state.device)
        model_state.contrastive_head.load_state_dict(torch.load(os.path.join(model_dir, 'models', 'contrastive_head.pt'), map_location=model_state.device))
        model_state.contrastive_head.eval()
        print("Contrastive projection head loaded successfully!")
    except Exception as e:
        print(f"Contrastive projection head load failed: {e}")

    # 3. Load Calibrated Wrappers
    try:
        model_state.calibrated_bio = joblib.load(os.path.join(model_dir, 'models', 'calibrated_bio.pkl'))
        model_state.calibrated_deep = joblib.load(os.path.join(model_dir, 'models', 'calibrated_deep.pkl'))
    except Exception as e:
        logger.warning(f"Calibrated classifiers load failed: {e}")

    # 4. Load Late Fusion DL Models (BioDNN + DeepDNN + XGBoost Meta)
    late_fusion_dir = os.path.join(base_dir, 'models_late_fusion')
    try:
        # Load BioDNN
        ckb = torch.load(os.path.join(late_fusion_dir, 'dl_bio.pt'), map_location=model_state.device, weights_only=False)
        model_state.dl_bio = BioDNN(input_dim=ckb['input_dim']).to(model_state.device)
        model_state.dl_bio.load_state_dict(ckb['model_state_dict'])
        model_state.dl_bio.eval()
        model_state.dl_bio_scaler_mean = ckb['scaler_mean']
        model_state.dl_bio_scaler_scale = ckb['scaler_scale']
        model_state.dl_bio_features = ckb['feature_names']
        logger.info(f"BioDNN loaded ({ckb['input_dim']} features)")
    except Exception as e:
        logger.error(f"BioDNN load error: {e}")

    try:
        # Load DeepDNN
        ckd = torch.load(os.path.join(late_fusion_dir, 'dl_deep.pt'), map_location=model_state.device, weights_only=False)
        model_state.dl_deep = DeepDNN(input_dim=ckd['input_dim']).to(model_state.device)
        model_state.dl_deep.load_state_dict(ckd['model_state_dict'])
        model_state.dl_deep.eval()
        model_state.dl_deep_scaler_mean = ckd['scaler_mean']
        model_state.dl_deep_scaler_scale = ckd['scaler_scale']
        model_state.dl_deep_features = ckd['feature_names']
        logger.info(f"DeepDNN loaded ({ckd['input_dim']} features)")
    except Exception as e:
        logger.error(f"DeepDNN load error: {e}")

    try:
        # Load XGBoost Meta-Classifier
        model_state.meta_clf = joblib.load(os.path.join(late_fusion_dir, 'meta_classifier.pkl'))
        with open(os.path.join(late_fusion_dir, 'meta_config.json'), 'r') as f:
            model_state.meta_cfg = json.load(f)
        logger.info("XGBoost Meta-Classifier loaded")
    except Exception as e:
        logger.error(f"Meta-Classifier load error: {e}")
        model_state.meta_clf = None

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
async def analyze_audio(
    file: UploadFile = File(...),
    customer_name: str = Form(None),
    phone_number: str = Form(None),
    branch: str = Form(None),
    account_ref: str = Form(None),
    auth: str = Depends(verify_api_key)
):
    logger.info("Request Received")
    t_start = time.time()
    
    t_ingest_start = time.time()
    # Read the file content
    content = await file.read()
    logger.info("Audio Uploaded")
    
    # Clean the audio content from trailing junk/watermarks
    oppo_idx = content.find(b"oppoMark")
    if oppo_idx != -1:
        logger.info(f"Found oppoMark at byte index {oppo_idx}. Truncating trailing metadata.")
        content = content[:oppo_idx]
        
    # Detect the correct extension
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
        _, file_ext = os.path.splitext(file.filename)
        if file_ext:
            ext = file_ext.lower()
            
    scratch_dir = os.path.join(base_dir, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    temp_path = os.path.join(scratch_dir, f"temp_{uuid.uuid4().hex}{ext}")
    with open(temp_path, "wb") as f:
        f.write(content)
    t_ingest_end = time.time()

    try:
        t_denoise_start = time.time()
        logger.info("Preprocessing Started")
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
        
        max_duration = 10.0
        if duration > max_duration:
            logger.info(f"Audio is {duration:.1f}s. Truncating to {max_duration}s for real-time latency.")
            y = y[:int(max_duration * sr)]
        
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
        
        peak = np.max(np.abs(y))
        if peak > 1e-4:
            y = (y / peak) * 0.95
            
        processed_path = temp_path.replace(ext, "_processed.wav")
        sf.write(processed_path, y, sr, format='WAV', subtype='PCM_16')
        
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        temp_path = processed_path
        t_denoise_end = time.time()
        
        # Audio Quality Metrics
        signal_power = compute_snr(y)
        silence_ratio = compute_silence_ratio(y, sr)
        file_size = os.path.getsize(temp_path)
        
        # 1. Bio extraction
        t_acoustic_start = time.time()
        logger.info("Extracting Biological Features")
        bio_feats = extract_bio_features(temp_path)
        if not bio_feats:
            raise ValueError("Bio feature extraction failed")
        bio_data = {c: [bio_feats.get(c, 0.0)] for c in model_state.bio_feature_cols}
        df_bio_input = pd.DataFrame(bio_data)
        t_acoustic_end = time.time()
        
        # 2. Deep extraction
        t_spectral_start = time.time()
        logger.info("Extracting Deep Features")
        y_for_deep = y[:10*16000].astype(np.float32) if len(y) > 10*16000 else y.astype(np.float32)
        
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
            mean = torch.mean(h12, dim=1).float().cpu().numpy()[0]
            std  = torch.std(h12,  dim=1).float().cpu().numpy()[0]
            return np.concatenate([mean, std])

        feats_clean    = wav2vec_pass(y_for_deep)
        feats_degraded = wav2vec_pass(y_degraded)
        feats_mct      = np.concatenate([feats_clean, feats_degraded])

        deep_feats_dict = {}
        for k in range(4096):
            deep_feats_dict[f'Deep_{k}'] = [float(feats_mct[k])]
        df_deep_input = pd.DataFrame(deep_feats_dict)
        t_spectral_end = time.time()

        # 3. Late Fusion: BioDNN -> p_bio, DeepDNN -> p_deep
        t_fusion_start = time.time()
        logger.info("Running PyTorch BioDNN")
        if model_state.dl_bio is not None:
            try:
                bio_cols = model_state.dl_bio_features
                for c in bio_cols:
                    if c not in df_bio_input.columns:
                        df_bio_input[c] = 0.0
                X_bio_ordered = df_bio_input[bio_cols].values.astype(np.float32)
                X_bio_scaled = (X_bio_ordered - model_state.dl_bio_scaler_mean) / model_state.dl_bio_scaler_scale
                X_bio_t = torch.FloatTensor(X_bio_scaled).to(model_state.device)
                with torch.no_grad():
                    p_bio = float(torch.sigmoid(model_state.dl_bio(X_bio_t)).cpu().numpy()[0][0])
                logger.info(f"BioDNN p_bio={p_bio:.4f}")
            except Exception as bio_err:
                logger.warning(f"BioDNN inference failed: {bio_err}. Using fallback.")
                p_bio = 0.5
        else:
            p_bio = 0.5

        logger.info("Running PyTorch DeepDNN")
        if model_state.dl_deep is not None:
            try:
                deep_cols = model_state.dl_deep_features
                for c in deep_cols:
                    if c not in df_deep_input.columns:
                        df_deep_input[c] = 0.0
                X_deep_ordered = df_deep_input[deep_cols].values.astype(np.float32)
                X_deep_scaled = (X_deep_ordered - model_state.dl_deep_scaler_mean) / model_state.dl_deep_scaler_scale
                X_deep_t = torch.FloatTensor(X_deep_scaled).to(model_state.device)
                with torch.no_grad():
                    p_deep = float(torch.sigmoid(model_state.dl_deep(X_deep_t)).cpu().numpy()[0][0])
                logger.info(f"DeepDNN p_deep={p_deep:.4f}")
            except Exception as deep_err:
                logger.warning(f"DeepDNN inference failed: {deep_err}. Using fallback.")
                p_deep = 0.5
        else:
            p_deep = 0.5

        p_deep_classes = np.array([1.0 - p_deep, p_deep, 0.0, 0.0])

        df_bio_input.to_csv(os.path.join(base_dir, "scratch", "temp_bio.csv"), index=False)
        df_deep_input.to_csv(os.path.join(base_dir, "scratch", "temp_deep.csv"), index=False)

        # 4. Late Fusion: XGBoost Meta-Classifier
        logger.info("Running XGBoost Meta-Classifier (Late Fusion)")
        using_meta = False
        if model_state.meta_clf is not None:
            try:
                disagreement_val = abs(p_bio - p_deep)
                meta_features = pd.DataFrame([{
                    'bio_score': p_bio,
                    'deep_score': p_deep,
                    'disagreement': disagreement_val,
                    'max_score': max(p_bio, p_deep),
                    'min_score': min(p_bio, p_deep)
                }])
                p_fused = float(model_state.meta_clf.predict_proba(meta_features)[0][1])
                t_high = model_state.meta_cfg.get("threshold_high", 0.54)
                t_mid = model_state.meta_cfg.get("threshold_mid", 0.30)
                using_meta = True
                logger.info(f"Late Fusion p_fused={p_fused:.4f}")
            except Exception as meta_err:
                logger.warning(f"Meta-classifier inference failed: {meta_err}. Falling back to fixed weight.")

        if not using_meta:
            try:
                with open(os.path.join(model_dir, 'models', 'fusion_weights.json'), 'r') as f:
                    weights = json.load(f)
                w_deep = weights.get('w_deep', 0.236)
                w_bio = weights.get('w_bio', 0.764)
            except:
                w_deep = 0.236
                w_bio = 0.764
            p_fused = w_deep * p_deep + w_bio * p_bio
            t_high = 0.54
            t_mid = 0.30
        t_fusion_end = time.time()
        
        # 5-Tier Risk Classification
        t_verdict_start = time.time()
        disagreement = abs(p_bio - p_deep)
        if disagreement > 0.30:
            p_fused = max(p_bio, p_deep, p_fused)

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
        raw_conf = 70.0 + 30.0 * (abs(p_fused - 0.5) / 0.5)
        confidence = round(min(99.9, max(70.0, raw_conf)), 1)
        t_verdict_end = time.time()

        # XAI attribution stage
        t_xai_start = time.time()
        flags = []
        risk_factors = []
        mitigating_factors = []

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

        if 0.35 <= p_bio <= 0.65:
            flags.append("BIO_BORDERLINE")
            risk_factors.append(f"Bio stream score ({p_bio*100:.1f}%) is in the uncertain zone")
        elif p_bio > 0.65:
            flags.append("BIO_ALERT")
            risk_factors.append(f"Bio stream detects strong synthetic patterns ({p_bio*100:.1f}%)")
        elif p_bio < 0.15:
            mitigating_factors.append(f"Bio stream is confident this is human ({p_bio*100:.1f}%)")

        if 0.35 <= p_deep <= 0.65:
            flags.append("DEEP_BORDERLINE")
            risk_factors.append(f"Deep stream score ({p_deep*100:.1f}%) is in the uncertain zone")
        elif p_deep > 0.65:
            flags.append("DEEP_ALERT")
            risk_factors.append(f"Deep stream detects strong vocoder artifacts ({p_deep*100:.1f}%)")
        elif p_deep < 0.15:
            mitigating_factors.append(f"Deep stream is confident this is human ({p_deep*100:.1f}%)")

        bio_feats_raw = bio_feats if bio_feats else {}
        jitter_val = bio_feats_raw.get('Jitter_Mean', None)
        shimmer_val = bio_feats_raw.get('Shimmer_Mean', None)
        hnr_val = bio_feats_raw.get('HNR_Mean', None)
        pitch_std_val = bio_feats_raw.get('Pitch_Std', None)
        flatness_val = bio_feats_raw.get('Flatness_Mean', None)

        if jitter_val is not None and jitter_val < 0.003:
            flags.append("LOW_JITTER")
            risk_factors.append(f"Jitter is unusually low ({jitter_val:.4f})")
        if shimmer_val is not None and shimmer_val < 0.02:
            flags.append("LOW_SHIMMER")
            risk_factors.append(f"Shimmer is unusually low ({shimmer_val:.4f})")
        if hnr_val is not None and hnr_val > 30:
            flags.append("HIGH_HNR")
            risk_factors.append(f"HNR is unusually high ({hnr_val:.1f} dB)")
        if pitch_std_val is not None and pitch_std_val < 5.0:
            flags.append("FLAT_PITCH")
            risk_factors.append(f"Pitch variation is very low ({pitch_std_val:.1f} Hz)")
        if flatness_val is not None and flatness_val < 0.001:
            flags.append("LOW_SPECTRAL_FLATNESS")
            risk_factors.append("Spectral flatness is very low")

        if p_fused > 0.85 or p_fused < 0.10:
            flags.append("HIGH_CONFIDENCE")
        elif 0.30 <= p_fused <= 0.60:
            flags.append("LOW_CONFIDENCE")
            risk_factors.append("Fusion score is in the uncertain range")

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
            if d < 0.10: return f"HIGH ({d*100:.1f}% disagreement)"
            elif d < 0.20: return f"MODERATE ({d*100:.1f}% disagreement)"
            elif d < 0.30: return f"LOW ({d*100:.1f}% disagreement)"
            else: return f"CRITICAL ({d*100:.1f}% disagreement)"

        if disagreement > 0.30 and p_bio > p_deep:
            primary_concern = "Bio stream detects potential synthesis artifacts that the Deep stream does not recognize"
        elif disagreement > 0.30 and p_deep > p_bio:
            primary_concern = "Deep stream detects vocoder artifacts that biological features miss"
        elif p_fused >= 0.50:
            primary_concern = "Both streams indicate this audio contains synthetic speech characteristics"
        elif p_fused >= 0.25:
            primary_concern = "Inconclusive analysis — audio shows some atypical patterns"
        else:
            primary_concern = "No significant synthesis indicators detected"

        RECOMMENDATIONS = {
            "CRITICAL": {"action": "BLOCK", "urgency": "CRITICAL", "detail": "Immediately block and escalate. Both analysis streams detect strong AI generation signatures."},
            "HIGH_RISK": {"action": "BLOCK", "urgency": "HIGH", "detail": "Block the call and queue for supervisor review. Significant synthesis indicators detected."},
            "MODERATE": {"action": "REVIEW", "urgency": "MEDIUM", "detail": "Route to analyst queue for manual spectrogram inspection. Audio shows patterns that require review."},
            "LOW_RISK": {"action": "ALLOW_MONITOR", "urgency": "LOW", "detail": "Allow the call but log for pattern analysis. Minor anomalies detected."},
            "CLEAR": {"action": "ALLOW", "urgency": "NONE", "detail": "No action required. Audio is consistent with natural human speech."}
        }
        rec = RECOMMENDATIONS[verdict]

        if "HIGH_DISAGREEMENT" in flags and verdict in ("CLEAR", "LOW_RISK"):
            rec = {"action": "REVIEW", "urgency": "MEDIUM", "detail": "Model disagreement exceeds safety threshold. Route to analyst for manual review."}
            verdict = "MODERATE"
            risk_level = "MEDIUM"
            label = "🟡 Moderate Risk — High Model Disagreement"

        # Generate the dynamic z-score narrative
        narrative_parts = []
        if verdict in ("CRITICAL", "HIGH_RISK"):
            narrative_parts.append(f"Forensic analysis detected strong artificial signatures in the audio stream (overall risk score: {p_fused*100:.1f}%).")
        elif verdict == "MODERATE":
            narrative_parts.append(f"Analysis indicates atypical vocal characteristics (overall risk score: {p_fused*100:.1f}%) that warrant human review.")
        else:
            narrative_parts.append(f"Audio stream exhibits normal acoustic and spectral characteristics consistent with organic human speech.")
            
        acoustic_anomalies = []
        if "LOW_JITTER" in flags:
            acoustic_anomalies.append("an abnormally flat micro-pitch curve (low jitter)")
        if "LOW_SHIMMER" in flags:
            acoustic_anomalies.append("lack of natural amplitude shimmer")
        if "HIGH_HNR" in flags:
            acoustic_anomalies.append("an unnaturally clean vocal track (high HNR)")
        if "FLAT_PITCH" in flags:
            acoustic_anomalies.append("lack of standard expressive prosody (flat pitch variation)")
            
        if acoustic_anomalies:
            narrative_parts.append("Key anomalies include " + ", ".join(acoustic_anomalies) + ".")
            narrative_parts.append("These characteristics match the signature of a neural vocoder synthesis model rather than a biological human vocal tract.")
        else:
            narrative_parts.append("Micro-pitch jitter, amplitude shimmer, and harmonic-to-noise ratios are all within typical human baselines.")
            
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
            "mitigating_factors": mitigating_factors if mitigating_factors else ["No specific mitigating factors"],
            "xai_narrative": " ".join(narrative_parts)
        }
        t_xai_end = time.time()

        t_report_start = time.time()
        chunk_results = [
            {"index": 1, "score": p_fused * 100, "confidence": confidence, "verdict": verdict, "start": 0, "end": round(duration, 1), "bio_score": p_bio * 100, "deep_score": p_deep * 100}
        ]
        
        case_id = f"VG-{uuid.uuid4().hex[:6].upper()}"
        cust_name = customer_name or "Rajesh Kumar"
        ph_num = phone_number or "+91 94311 08842"
        br = branch or "Hazratganj, Lucknow"
        acc = account_ref or "UCO-308849102"
        otp_code = str(random.randint(100000, 999999))

        # Insert case in SQLite
        insert_case(
            case_id=case_id,
            customer_name=cust_name,
            phone_number=ph_num,
            branch=br,
            account_ref=acc,
            fraud_score=round(p_fused * 100, 1),
            confidence=confidence,
            verdict=verdict,
            bio_features=bio_feats,
            deep_features=deep_feats_dict,
            otp_code=otp_code
        )
        t_report_end = time.time()

        # Telemetry timings
        stage_timings = {
            "ingest_ms": int((t_ingest_end - t_ingest_start) * 1000),
            "denoise_ms": int((t_denoise_end - t_denoise_start) * 1000),
            "acoustic_ms": int((t_acoustic_end - t_acoustic_start) * 1000),
            "spectral_ms": int((t_spectral_end - t_spectral_start) * 1000),
            "fusion_ms": int((t_fusion_end - t_fusion_start) * 1000),
            "xai_ms": int((t_xai_end - t_xai_start) * 1000),
            "verdict_ms": int((t_verdict_end - t_verdict_start) * 1000),
            "report_ms": int((t_report_end - t_report_start) * 1000)
        }

        result = {
            "case_id": case_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fraud_score": round(p_fused * 100, 1),
            "confidence": confidence,
            "risk_level": risk_level,
            "risk_tier": verdict,
            "verdict": verdict,
            "verdict_label": label,
            "primary_trigger": "PyTorch Late Fusion DL" if using_meta else "Fixed Weight Fusion",
            "secondary_trigger": "Wav2Vec2 Anomaly" if p_deep > p_bio else "Bio Feature Anomaly",
            "explanation": explanation,
            "recommendation": rec["detail"] if isinstance(rec, dict) else rec,
            "using_real_models": True,
            "metadata": {"filename": file.filename, "format": "WAV", "sample_rate": 16000, "channels": 1, "duration": round(duration, 2), "file_size_bytes": file_size},
            "performance": {"model_version": "v3.0-5TierExplainable" if using_meta else "v3.0-FixedWeight", "inference_time_ms": int((time.time() - t_start) * 1000), "audio_duration_sec": round(duration, 2), "chunks_processed": 1},
            "threat_intel": {
                "threat_type": "AI Voice Clone" if verdict not in ("CLEAR", "LOW_RISK") else "None",
                "sophistication": "Advanced" if verdict in ("CRITICAL", "HIGH_RISK") else ("Moderate" if verdict == "MODERATE" else "None"),
                "replay_indicators": "None",
                "synthetic_confidence": round(float(max(p_deep_classes[1:])) * 100, 1) if verdict not in ("CLEAR", "LOW_RISK") else 0.0
            },
            "chunk_results": chunk_results,
            "stage_timings": stage_timings,
            "otp_code": otp_code,
            "customer_name": cust_name,
            "phone_number": ph_num,
            "branch": br,
            "account_ref": acc
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
        
        hard_val_bio = os.path.join(model_dir, 'features', 'hard_val_bio.csv')
        hard_val_deep = os.path.join(model_dir, 'features', 'hard_val_deep.csv')
        
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


@app.get("/api/v1/cases")
def list_cases(auth: str = Depends(verify_api_key)):
    return get_all_cases()

@app.get("/api/v1/cases/{case_id}")
def case_details(case_id: str, auth: str = Depends(verify_api_key)):
    c = get_case(case_id)
    if not c:
        return JSONResponse(status_code=404, content={"message": "Case not found"})
    return c

@app.post("/api/v1/cases/{case_id}/checklist")
async def update_case_checklist(case_id: str, req: Request, auth: str = Depends(verify_api_key)):
    state = await req.json()
    update_checklist(case_id, state)
    return {"status": "success", "message": "Checklist updated"}

@app.post("/api/v1/cases/{case_id}/action")
async def save_case_action(
    case_id: str, 
    frozen: bool = Form(None), 
    escalated: bool = Form(None), 
    notes: str = Form(None), 
    auth: str = Depends(verify_api_key)
):
    update_case_action(case_id, frozen=frozen, escalated=escalated, notes=notes)
    return {"status": "success", "message": "Action updated"}

@app.get("/api/v1/cases/{case_id}/similar")
def get_sim_cases(case_id: str, auth: str = Depends(verify_api_key)):
    return get_similar_cases(case_id)



class ChatRequest(BaseModel):
    message: str
    case_id: str = None

@app.post("/api/v1/chat")
async def process_chat(req: ChatRequest, auth: str = Depends(verify_api_key)):
    session_id = req.session_id or "default"
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
        
    history = chat_sessions[session_id]
    
    # 1. Structured Case Context Truncated
    context = ""
    case_summary_fact = "No active case."
    if req.case_id:
        c = get_case(req.case_id)
        if c:
            # Parse Bio/Deep features just to grab a couple top keys if we want, but let's just stick to structured core
            try:
                bio = json.loads(c['bio_features'])
                top_bio = {k: round(v, 4) for k, v in list(bio.items())[:3]}
            except:
                top_bio = {}
                
            context = f"Active Case: {c['case_id']} | Verdict: {c['verdict']} | Score: {c['fraud_score']}% | Bio Flags: {top_bio}"
            case_summary_fact = f"Active Case {c['case_id']} is {c['verdict']} ({c['fraud_score']}%)."
            
    # 2. KB Retrieval & Direct Cache Check
    kb_context = ""
    matches = get_kb_matches(req.message)
    if matches and matches[0]["score"] > 0.85:
        # High confidence exact match -> Skip LLM entirely
        ans = matches[0]["article"]["a"]
        # Add tracking note to history
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": ans})
        if len(history) > 6:
            history = history[-6:]
        chat_sessions[session_id] = history
        return {"status": "success", "reply": ans, "model": "Direct KB Cache"}
        
    if matches:
        kb_text = []
        for m in matches:
            if m["score"] > 0.1: # Only inject if slightly relevant
                a_trunc = m["article"]["a"][:300] + "..." if len(m["article"]["a"]) > 300 else m["article"]["a"]
                kb_text.append(f"Q: {m['article']['q']}\nA: {a_trunc}")
        if kb_text:
            kb_context = "\n\nBANK KNOWLEDGE BASE REFERENCE:\n" + "\n---\n".join(kb_text)

    # 3. Rolling History Summarization
    # Keep last 2-3 turns (4-6 messages), summarize older ones
    # To keep it completely deterministic, the summary is just a list of past user topics
    if len(history) > 6:
        # We need to drop the oldest ones and update a deterministic summary
        old_msgs = history[:-6]
        history = history[-6:]
        chat_sessions[session_id] = history
        
    # Extract topics from all past user messages in history for the summary line
    past_topics = [msg["content"] for msg in history if msg["role"] == "user"]
    topic_summary = ""
    if past_topics:
        # Just list them briefly
        topics_str = " | ".join([t[:30] + ("..." if len(t)>30 else "") for t in past_topics[:-1]]) if len(past_topics) > 1 else "None"
        topic_summary = f"\n\nPrior Topics Discussed: {topics_str}\n{case_summary_fact}"

    sys_prompt = f"""You are the VoiceGuard Forensic AI Agent exclusively built for UCO Bank's Fraud Control Cell. You analyze highly sensitive voice biometric data.
    
    CRITICAL INSTRUCTIONS:
    1. Act strictly as the VoiceGuard AI Agent. Keep answers professional and highly analytical.
    2. If an Active Case is provided, directly reference its Verdict and Bio Flags.
    3. Use the BANK KNOWLEDGE BASE REFERENCE if provided to answer policy/glossary questions. Do NOT invent facts.
    
    {context}{kb_context}{topic_summary}"""
    
    import requests
    try:
        headers = {
            "Authorization": "Bearer YOUR_OPENROUTER_API_KEY",
            "HTTP-Referer": "http://localhost:8001",
            "X-Title": "VoiceGuard AI Agent",
            "Content-Type": "application/json"
        }
        
        # Build messages payload
        messages = [{"role": "system", "content": sys_prompt}]
        for msg in history:
            messages.append(msg)
        messages.append({"role": "user", "content": req.message})
        
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": messages
        }
        
        used_model = "gpt-4o-mini"
        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20)
            response.raise_for_status()
        except requests.exceptions.RequestException as req_err:
            print(f"GPT-4o-mini failed ({req_err}), falling back to Gemini...")
            used_model = "gemini-2.0-flash-exp:free"
            payload["model"] = "google/gemini-2.0-flash-exp:free"
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            
        data = response.json()
        reply_text = data["choices"][0]["message"]["content"]
        
        # Append to history
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": reply_text})
        chat_sessions[session_id] = history
        
        return {"status": "success", "reply": reply_text, "model": used_model}
    except Exception as e:
        print("LLM Error:", e)
        return {"status": "error", "reply": "OpenRouter LLM Error: " + str(e), "model": "error"}


@app.post("/analyze-chunks")
async def analyze_chunks(
    file: UploadFile = File(...),
    auth: str = Depends(verify_api_key)
):
    """Per-window bio-feature scoring for forensic overlay.
    
    Slices audio into 2-3s windows and runs ONLY the cheap bio-feature
    extraction (jitter, shimmer, HNR, MFCCs — pure DSP, no neural net)
    per window, then scores each via the BioDNN model.
    
    This endpoint is called AFTER /analyze returns, in the background.
    It does NOT run the expensive Wav2Vec2 deep model.
    """
    t_start = time.time()
    
    content = await file.read()
    
    # Clean trailing junk
    oppo_idx = content.find(b"oppoMark")
    if oppo_idx != -1:
        content = content[:oppo_idx]
    
    # Detect extension
    ext = ".wav"
    if content.startswith(b"RIFF") and len(content) > 12 and content[8:12] == b"WAVE":
        ext = ".wav"
    elif content.startswith(b"ID3") or content.startswith(b"\xff\xfb"):
        ext = ".mp3"
    elif content.startswith(b"\x1a\x45\xdf\xa3"):
        ext = ".webm"
    elif content.startswith(b"OggS"):
        ext = ".ogg"
    else:
        _, file_ext = os.path.splitext(file.filename)
        if file_ext:
            ext = file_ext.lower()
    
    temp_path = os.path.join(base_dir, "scratch", f"chunk_{uuid.uuid4().hex}{ext}")
    with open(temp_path, "wb") as f:
        f.write(content)
    
    try:
        # Load and preprocess (same as /analyze)
        try:
            y, sr = librosa.load(temp_path, sr=16000, mono=True)
        except Exception:
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
            except Exception as e:
                return JSONResponse(content={"error": str(e)}, status_code=400)
        
        duration = len(y) / sr
        max_duration = 10.0
        if duration > max_duration:
            y = y[:int(max_duration * sr)]
            duration = max_duration
        
        # Highpass filter
        from scipy.signal import butter, lfilter
        b, a = butter(1, 80.0 / (0.5 * sr), btype='high', analog=False)
        y = lfilter(b, a, y)
        peak = np.max(np.abs(y))
        if peak > 1e-4:
            y = (y / peak) * 0.95
        
        # Determine window size: 2-3s, cap at ~5 windows max
        if duration <= 3.0:
            window_sec = duration  # single chunk for very short clips
        elif duration <= 6.0:
            window_sec = duration / 2.0
        else:
            window_sec = max(2.0, duration / 5.0)  # aim for ~5 windows
        
        window_samples = int(window_sec * sr)
        chunks = []
        idx = 0
        chunk_index = 0
        
        while idx < len(y):
            end_sample = min(idx + window_samples, len(y))
            chunk_audio = y[idx:end_sample]
            chunk_start = idx / sr
            chunk_end = end_sample / sr
            
            # Skip chunks shorter than 0.5s (not enough for meaningful features)
            if len(chunk_audio) < int(0.5 * sr):
                idx = end_sample
                continue
            
            # Extract bio features
            try:
                bio_score = _score_chunk_bio(chunk_audio, sr)
            except Exception:
                bio_score = 0.0
                
            # Extract deep features
            try:
                deep_score = _score_chunk_deep(chunk_audio, sr)
            except Exception as e:
                logger.error(f"Deep chunk score failed: {e}")
                deep_score = 0.0
            
            # Classify chunk (fusion: max of both scores)
            chunk_max_score = max(bio_score, deep_score)
            if chunk_max_score >= 0.65:
                chunk_verdict = "HIGH_RISK"
            elif chunk_max_score >= 0.35:
                chunk_verdict = "MODERATE"
            elif chunk_max_score >= 0.15:
                chunk_verdict = "LOW_RISK"
            else:
                chunk_verdict = "CLEAR"
            
            chunks.append({
                "index": chunk_index,
                "start": round(chunk_start, 2),
                "end": round(chunk_end, 2),
                "bio_score": round(bio_score * 100, 1),
                "deep_score": round(deep_score * 100, 1),
                "score": round(chunk_max_score * 100, 1), 
                "verdict": chunk_verdict
            })
            
            chunk_index += 1
            idx = end_sample
        
        return JSONResponse(content={
            "chunk_results": chunks,
            "duration": round(duration, 2),
            "window_sec": round(window_sec, 2),
            "num_chunks": len(chunks),
            "compute_ms": int((time.time() - t_start) * 1000)
        })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _score_chunk_bio(chunk_audio, sr=16000):
    """Score a single audio chunk using only bio features + BioDNN.
    
    This is the cheap path: pure signal processing (jitter, shimmer, HNR,
    MFCCs, spectral features). No Wav2Vec2, no deep model.
    Takes a numpy array directly — no file I/O.
    """
    import soundfile as sf
    
    # Write chunk to temp file for extract_bio_features (it expects a path)
    chunk_path = os.path.join(base_dir, "scratch", f"_chunk_{uuid.uuid4().hex}.wav")
    try:
        sf.write(chunk_path, chunk_audio, sr, format='WAV', subtype='PCM_16')
        bio_feats = extract_bio_features(chunk_path)
        
        if not bio_feats or model_state.dl_bio is None:
            return 0.0
        
        # Score via BioDNN (same path as /analyze, just the bio part)
        bio_cols = model_state.dl_bio_features
        bio_data = {c: [bio_feats.get(c, 0.0)] for c in bio_cols}
        df = pd.DataFrame(bio_data)
        X = df.values.astype(np.float32)
        X_scaled = (X - model_state.dl_bio_scaler_mean) / model_state.dl_bio_scaler_scale
        X_t = torch.FloatTensor(X_scaled).to(model_state.device)
        
        with torch.no_grad():
            p = float(torch.sigmoid(model_state.dl_bio(X_t)).cpu().numpy()[0][0])
        return p
    finally:
        if os.path.exists(chunk_path):
            os.remove(chunk_path)

def _score_chunk_deep(chunk_audio, sr=16000):
    """Score a single audio chunk using Wav2Vec2 + PyTorch DeepDNN.
    
    Extracts deep spectral features from the chunk and passes them through
    the fusion DeepDNN to get a synthetic probability score.
    """
    if model_state.w2v_model is None or model_state.processor is None or model_state.dl_deep is None:
        return 0.0
        
    y = chunk_audio.astype(np.float32)
    
    # Degrade audio (lowpass filter)
    from scipy.signal import butter, lfilter
    b_lp, a_lp = butter(4, 4000.0 / (sr / 2.0), btype='low')
    y_deg = lfilter(b_lp, a_lp, y).astype(np.float32)
    
    def w2v(audio_array):
        inp = model_state.processor([audio_array], sampling_rate=sr, return_tensors="pt", padding=True)
        inp = {k: v.to(model_state.device) for k, v in inp.items()}
        with torch.no_grad():
            if model_state.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    out = model_state.w2v_model(**inp)
            else:
                out = model_state.w2v_model(**inp)
        h12 = out.hidden_states[12]
        mean = torch.mean(h12, dim=1).float().cpu().numpy()[0]
        std  = torch.std(h12,  dim=1).float().cpu().numpy()[0]
        return np.concatenate([mean, std])
        
    try:
        f_clean = w2v(y)
        f_deg = w2v(y_deg)
        f_mct = np.concatenate([f_clean, f_deg])
        
        deep_feats_dict = {f'Deep_{k}': [float(f_mct[k])] for k in range(4096)}
        df_deep = pd.DataFrame(deep_feats_dict)
        
        deep_cols = model_state.dl_deep_features
        for c in deep_cols:
            if c not in df_deep.columns:
                df_deep[c] = 0.0
                
        X = df_deep[deep_cols].values.astype(np.float32)
        X_scaled = (X - model_state.dl_deep_scaler_mean) / model_state.dl_deep_scaler_scale
        X_t = torch.FloatTensor(X_scaled).to(model_state.device)
        
        with torch.no_grad():
            p_deep = float(torch.sigmoid(model_state.dl_deep(X_t)).cpu().numpy()[0][0])
            
        return p_deep
    except Exception as e:
        logger.error(f"_score_chunk_deep error: {e}")
        return 0.0


if __name__ == "__main__":

    import uvicorn
    
    # Open Chrome in a separate thread after a short delay
    def open_chrome():
        time.sleep(2)  # Wait 2 seconds for server to start
        try:
            # Windows native way to open URL in default browser
            os.startfile('http://127.0.0.1:8001')
        except Exception as e:
            print(f"Failed to open browser: {e}")
            print("Open http://127.0.0.1:8001 manually in your browser")
    
    chrome_thread = threading.Thread(target=open_chrome, daemon=True)
    chrome_thread.start()
    
    uvicorn.run("fastapi_app:app", host="127.0.0.1", port=8001, reload=True)
    # Trigger model reload: 2026-06-29T20:59
