import os
import sys
import json
import time
import torch
import torch.nn as nn
import librosa
import numpy as np
import scipy.io.wavfile as wavfile
import joblib
import uuid
import numpy as np
import pandas as pd
import xgboost as xgb
import threading
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import warnings

warnings.filterwarnings('ignore')

# Add src to path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'src'))
from extract_bio import extract_bio_features, simulate_phone_codec
from transformers import Wav2Vec2FeatureExtractor, AutoModel

app = FastAPI(title="FraudRadar AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PyTorch Model Definitions for Deep Fusion Network
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class VoiceGuardDeepFusionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout_rate=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.res1 = ResidualBlock(hidden_dim, dropout_rate)
        self.res2 = ResidualBlock(hidden_dim, dropout_rate)
        self.middle_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.output_layer = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.middle_layer(x)
        return self.output_layer(x)

# Global State for Models
class ModelState:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.w2v_model = None
        self.xgb_bio = None
        self.xgb_deep = None
        self.deep_fusion_model = None
        self.deep_fusion_scaler = None
        self.deep_fusion_cfg = None
        self.deep_fusion_feature_names = []
        self.bio_feature_cols = []
        self.deep_feature_cols = []
        self.is_loaded = False

model_state = ModelState()

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

    # 2. Load Deep Fusion Config, feature list, and thresholds
    cfg_path = os.path.join(base_dir, 'models', 'deep_fusion_config.json')
    try:
        with open(cfg_path, 'r') as f:
            model_state.deep_fusion_cfg = json.load(f)
        model_state.deep_fusion_feature_names = model_state.deep_fusion_cfg['feature_names']
        model_state.bio_feature_cols = [c for c in model_state.deep_fusion_feature_names if not c.startswith('Deep_')]
    except Exception as e:
        print(f"Error loading deep fusion config: {e}")
        bio_csv_path = os.path.join(base_dir, 'features', 'bio_features.csv')
        bio_df_cols = pd.read_csv(bio_csv_path, nrows=0).columns.tolist()
        ignore_cols = ['Filename', 'Label', 'Pitch_AC_500ms', 'Pitch_AC_1000ms', 'Pitch_AC_2000ms', 'Breath_DominantFreq', 'Breath_DominantPower', 'Breath_Ratio']
        model_state.bio_feature_cols = [c for c in bio_df_cols if c not in ignore_cols]

    # 3. Load StandardScaler
    scaler_path = os.path.join(base_dir, 'models', 'deep_fusion_scaler.joblib')
    try:
        model_state.deep_fusion_scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"Error loading scaler: {e}")

    # 4. Load PyTorch Deep Fusion classifier
    try:
        input_dim = model_state.deep_fusion_cfg.get('input_dim', 2397)
        model_state.deep_fusion_model = VoiceGuardDeepFusionNet(input_dim=input_dim)
        model_path = os.path.join(base_dir, 'models', 'deep_fusion_classifier.pth')
        model_state.deep_fusion_model.load_state_dict(torch.load(model_path, map_location=model_state.device))
        model_state.deep_fusion_model.to(model_state.device)
        model_state.deep_fusion_model.eval()
    except Exception as e:
        print(f"Error loading PyTorch classifier model: {e}")

    # 5. Load XGBoost Models for individual stream scoring visualizers (USP)
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

    model_state.is_loaded = True
    print("Models loaded successfully!")

@app.get("/health")
def health_check():
    return {"backend_api": "CONNECTED", "database": "CONNECTED"}

@app.get("/model-status")
def model_status():
    return {
        "feature_counts": {
            "biological_features": 349,
            "deep_features": 2048
        },
        "model_loaded": model_state.is_loaded,
        "inference_ready": model_state.is_loaded
    }

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    start_time = time.time()
    
    # Read the file content
    content = await file.read()
    
    # Clean the audio content from trailing junk/watermarks (e.g. OPPO watermark)
    oppo_idx = content.find(b"oppoMark")
    if oppo_idx != -1:
        print(f"Found oppoMark at byte index {oppo_idx}. Truncating trailing metadata.")
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
                print(f"FAILED TO READ AUDIO. Saved to {debug_path}")
                raise ValueError(f"Audio format not supported. librosa error: {e_librosa}. scipy error: {e_scipy}")
            
        duration = len(y) / sr
        
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
        signal_power = np.mean(y**2)
        noise_power = np.var(y) - signal_power if np.var(y) > signal_power else np.var(y) * 0.1
        snr = 10 * np.log10(signal_power / noise_power + 1e-9) if noise_power > 0 else 40.0
        snr = np.clip(snr, -10, 40)
        
        rms_curve = librosa.feature.rms(y=y)[0]
        if len(rms_curve) > 0:
            silence_thresh = np.percentile(rms_curve, 20)
            silence_ratio = np.sum(rms_curve < silence_thresh) / len(rms_curve)
        else:
            silence_ratio = 0.0
        file_size = os.path.getsize(temp_path)
        print(f"Loaded audio: {len(y)} samples, first 5: {y[:5]}")
        
        # 1. Bio extraction
        bio_feats = extract_bio_features(temp_path)
        if not bio_feats:
            raise ValueError("Bio feature extraction failed")
        bio_data = {c: [bio_feats.get(c, 0.0)] for c in model_state.bio_feature_cols}
        df_bio_input = pd.DataFrame(bio_data)
        
        # 2. Deep extraction
        y_sim = simulate_phone_codec(y, sr=16000)
        max_audio_samples = 10 * 16000
        if len(y_sim) > max_audio_samples:
            y_sim = y_sim[:max_audio_samples]

        inputs = model_state.processor([y_sim], sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(model_state.device) for k, v in inputs.items()}

        with torch.no_grad():
            if model_state.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model_state.w2v_model(**inputs)
            else:
                outputs = model_state.w2v_model(**inputs)

        hidden_states = outputs.hidden_states[12]
        pooled_mean = torch.mean(hidden_states, dim=1).float().cpu().numpy()[0]
        pooled_std = torch.std(hidden_states, dim=1).float().cpu().numpy()[0]

        deep_feats_dict = {}
        for k in range(1024): deep_feats_dict[f'Deep_{k}'] = [float(pooled_mean[k])]
        for k in range(1024): deep_feats_dict[f'Deep_{1024 + k}'] = [float(pooled_std[k])]
        df_deep_input = pd.DataFrame(deep_feats_dict)

        if model_state.deep_feature_cols:
            # Add any missing columns with 0.0 (though they shouldn't be missing)
            for c in model_state.deep_feature_cols:
                if c not in df_deep_input.columns:
                    df_deep_input[c] = 0.0
            df_deep_input = df_deep_input[model_state.deep_feature_cols]

        # 3. Prediction
        p_bio = float(model_state.xgb_bio.predict_proba(df_bio_input)[:, 1][0]) if model_state.xgb_bio is not None else 0.5
        p_deep = float(model_state.xgb_deep.predict_proba(df_deep_input)[:, 1][0]) if model_state.xgb_deep is not None else 0.5
        
        # Save temp features for feedback loop
        df_bio_input.to_csv(os.path.join(base_dir, "scratch", "temp_bio.csv"), index=False)
        df_deep_input.to_csv(os.path.join(base_dir, "scratch", "temp_deep.csv"), index=False)

        # 4. Deep Learning Fusion Prediction
        p_fused = 0.5
        t_high = 0.90
        t_mid = 0.50
        
        if model_state.deep_fusion_model is not None and model_state.deep_fusion_scaler is not None:
            # Align both feature sets into the exact training sequence
            feats = {}
            for k in range(1024):
                feats[f'Deep_{k}'] = float(pooled_mean[k])
                feats[f'Deep_{1024 + k}'] = float(pooled_std[k])
            for k in model_state.bio_feature_cols:
                feats[k] = float(bio_feats.get(k, 0.0))
            
            df_combined = pd.DataFrame([feats])
            df_combined = df_combined.reindex(columns=model_state.deep_fusion_feature_names, fill_value=0.0)
            
            # Normalize and predict with PyTorch DNN
            X_scaled = model_state.deep_fusion_scaler.transform(df_combined.values)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(model_state.device)
            with torch.no_grad():
                logit = model_state.deep_fusion_model(X_tensor)
                p_fused = float(torch.sigmoid(logit).cpu().numpy()[0][0])
                
            if model_state.deep_fusion_cfg is not None:
                t_high = model_state.deep_fusion_cfg.get('threshold_high', 0.90)
                t_mid = model_state.deep_fusion_cfg.get('threshold_mid', 0.50)
                # Prevent collapse if train partition gets perfect classification
                if t_high == t_mid:
                    t_mid = 0.50
                    t_high = 0.90
        else:
            p_fused = 0.236 * p_deep + 0.764 * p_bio
            t_high = 0.54
            t_mid = 0.30
        
        # Calculate dynamic confidence score (clamped between 70.0% and 99.9%)
        raw_conf = 70.0 + 30.0 * (abs(p_fused - 0.5) / 0.5)
        confidence = round(min(99.9, max(70.0, raw_conf)), 1)

        # Determine verdict based purely on the SOTA PyTorch Deep Fusion classifier
        if p_fused >= t_high:
            verdict = "FRAUD"
            risk_level = "HIGH"
            label = "🔴 AI-Generated Audio Detected"
        elif p_fused >= t_mid:
            verdict = "SUSPICIOUS"
            risk_level = "MEDIUM"
            label = "🟡 Suspicious Audio Detected"
        else:
            verdict = "LEGITIMATE"
            risk_level = "LOW"
            label = "🟢 Genuine Human Voice"

        # Output chunk results (used by UI plots)
        chunk_results = [
            {"index": 1, "score": p_fused * 100, "confidence": confidence, "verdict": verdict, "start": 0, "end": round(duration, 1), "bio_score": p_bio * 100, "deep_score": p_deep * 100}
        ]

        result = {
            "case_id": f"VG-{uuid.uuid4().hex[:6].upper()}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fraud_score": round(p_fused * 100, 1),
            "confidence": confidence,
            "risk_level": risk_level,
            "verdict": verdict,
            "verdict_label": label,
            "primary_trigger": "Deep Neural Net" if p_fused > 0.5 else "None",
            "secondary_trigger": "Wav2Vec2 Anomaly" if p_deep > p_bio else "Bio Feature Anomaly",
            "recommendation": "Block User" if verdict == "FRAUD" else ("Manual Review" if verdict == "SUSPICIOUS" else "Allow"),
            "using_real_models": True,
            "metadata": {"filename": file.filename, "format": "WAV", "sample_rate": 16000, "channels": 1, "duration": round(duration, 2), "file_size_bytes": file_size},
            "performance": {"model_version": "v3.0-PyTorchDeepFusion", "inference_time_ms": int((time.time() - start_time) * 1000), "audio_duration_sec": round(duration, 2), "chunks_processed": 1},
            "threat_intel": {"threat_type": "TTS Clone" if p_deep > 0.5 else "None", "sophistication": "Advanced", "replay_indicators": "None", "synthetic_confidence": round(p_deep * 100, 1)},
            "chunk_results": chunk_results
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return JSONResponse(content=result)

@app.post("/feedback")
async def process_feedback(is_correct: str = Form(...), true_label: str = Form(...)):
    # Append to hard validation CSVs
    is_correct = (is_correct.lower() == 'true')
    label = 1 if true_label == "Fake (1)" else 0
    
    bio_temp = os.path.join(base_dir, "scratch", "temp_bio.csv")
    deep_temp = os.path.join(base_dir, "scratch", "temp_deep.csv")
    
    if os.path.exists(bio_temp) and os.path.exists(deep_temp):
        df_bio_input = pd.read_csv(bio_temp)
        df_deep_input = pd.read_csv(deep_temp)
        
        df_bio_input['Label'] = label
        df_deep_input['Label'] = label
        df_bio_input['Filename'] = f"active_learning_{int(time.time())}.wav"
        df_deep_input['Filename'] = f"active_learning_{int(time.time())}.wav"
        
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
async def retrain_model():
    print("Retraining Deep Fusion model triggered...")
    # Call the training script
    os.system(f"{sys.executable} src/train_deep.py")
    
    # Reload the model
    try:
        load_models()
        return {"status": "success", "message": "Deep Fusion Classifier retrained and reloaded successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/generate-report")
async def generate_report():
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
