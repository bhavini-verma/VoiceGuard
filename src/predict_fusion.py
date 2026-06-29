import os
import sys
import json
import torch
import librosa
import numpy as np
import pandas as pd
import xgboost as xgb
import argparse
import joblib

# Setup paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'src'))
from extract_bio import extract_bio_features
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

def compute_snr(y):
    if len(y) == 0:
        return 0.0
    signal_power = np.mean(y**2)
    noise_power = np.var(y) - signal_power if np.var(y) > signal_power else np.var(y) * 0.1
    if noise_power <= 0:
        return 40.0
    snr = 10 * np.log10(signal_power / noise_power + 1e-9)
    return np.clip(snr, -10, 40)

def compute_silence_ratio(y, sr):
    if len(y) == 0:
        return 0.0
    rms = librosa.feature.rms(y=y)[0]
    if len(rms) == 0:
        return 0.0
    threshold = np.percentile(rms, 20)
    silence_frames = np.sum(rms < threshold)
    return silence_frames / len(rms)

def main():
    parser = argparse.ArgumentParser(description="VoiceGuard Fusion Prediction")
    parser.add_argument("--audio", required=True, help="Path to wav file to analyze")
    args = parser.parse_args()

    audio_path = args.audio
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Audio and Audio Quality Metrics
    print("\n--- Loading Audio & Quality Metrics ---")
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio_dur = len(y) / sr
    snr = compute_snr(y)
    silence_ratio = compute_silence_ratio(y, sr)
    print(f"Audio Duration: {audio_dur:.2f}s")
    print(f"Signal-to-Noise Ratio (SNR): {snr:.1f} dB")
    print(f"Silence Ratio: {silence_ratio*100:.1f}%")

    # 2. Extract Biological Features
    print("\n--- [Stream A] Extracting Biological Features ---")
    bio_feats = extract_bio_features(audio_path)
    if bio_feats is None:
        print("Error: Could not extract biological features.")
        return
        
    xgb_bio = xgb.XGBClassifier()
    xgb_bio.load_model(os.path.join(base_dir, 'models', 'xgb_bio.json'))
    bio_cols = xgb_bio.get_booster().feature_names
    
    bio_data = {c: [bio_feats.get(c, 0.0)] for c in bio_cols}
    df_bio_input = pd.DataFrame(bio_data)[bio_cols]

    # 3. Extract Deep Features
    print("\n--- [Stream B] Extracting Deep Features (indicwav2vec-hindi) ---")
    model_local_path = os.path.join(base_dir, 'models', 'indicwav2vec-hindi')
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_local_path)
    w2v_model = AutoModel.from_pretrained(model_local_path).to(device)
    w2v_model.eval()

    inputs = processor(y, sampling_rate=sr, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = w2v_model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    layer_12 = hidden_states[12].squeeze(0).cpu().numpy()
    pooled_mean = np.mean(layer_12, axis=0)
    pooled_std = np.std(layer_12, axis=0)

    xgb_deep = xgb.XGBClassifier()
    xgb_deep.load_model(os.path.join(base_dir, 'models', 'xgb_deep.json'))
    
    # Build raw 2048D feature vector
    deep_feats_dict = {}
    for k in range(1024): deep_feats_dict[f'Deep_{k}'] = [float(pooled_mean[k])]
    for k in range(1024): deep_feats_dict[f'Deep_{1024 + k}'] = [float(pooled_std[k])]
    df_deep_raw = pd.DataFrame(deep_feats_dict)
    
    # Project through contrastive head if available
    contrastive_head_path = os.path.join(base_dir, 'models', 'contrastive_head.pt')
    if os.path.exists(contrastive_head_path):
        from contrastive_head import ContrastiveProjectionHead
        proj_head = ContrastiveProjectionHead().to(device)
        proj_head.load_state_dict(torch.load(contrastive_head_path, map_location=device, weights_only=True))
        proj_head.eval()
        
        raw_cols = [f'Deep_{k}' for k in range(2048)]
        X_raw = df_deep_raw[raw_cols].values.astype(np.float32)
        with torch.no_grad():
            X_proj = proj_head(torch.tensor(X_raw).to(device)).cpu().numpy()
        proj_cols = [f'Proj_Deep_{i}' for i in range(128)]
        df_deep_input = pd.DataFrame(X_proj, columns=proj_cols)
        print(f"  (Projected 2048D -> 128D via contrastive head)")
    else:
        # Fallback: use raw features with expected column alignment
        deep_cols = xgb_deep.get_booster().feature_names
        for c in deep_cols:
            if c not in df_deep_raw.columns:
                df_deep_raw[c] = 0.0
        df_deep_input = df_deep_raw[deep_cols]
        print(f"  (Using raw {len(deep_cols)}D features, no contrastive head)")

    # 4. Predictions
    p_bio_classes = xgb_bio.predict_proba(df_bio_input)[0]
    p_deep_classes = xgb_deep.predict_proba(df_deep_input)[0]
    p_bio = float(1.0 - p_bio_classes[0])
    p_deep = float(1.0 - p_deep_classes[0])
    
    print(f"\nBio Stream Fraud Prob  : {p_bio*100:.1f}%")
    print(f"Deep Stream Fraud Prob : {p_deep*100:.1f}%")
    
    # Show per-class breakdown
    class_names = ['Genuine', 'ElevenLabs', 'Resemble', 'Generic TTS']
    print(f"\n--- Per-Class Deep Probabilities ---")
    for i, name in enumerate(class_names):
        if i < len(p_deep_classes):
            print(f"  {name:15s}: {p_deep_classes[i]*100:.1f}%")
    print(f"--- Per-Class Bio Probabilities ---")
    for i, name in enumerate(class_names):
        if i < len(p_bio_classes):
            print(f"  {name:15s}: {p_bio_classes[i]*100:.1f}%")

    # 5. Meta-Classifier Fusion
    print("\n--- [Stream C] Fusion Meta-Classifier ---")
    meta_clf_path = os.path.join(base_dir, 'models', 'meta_classifier.pkl')
    meta_cfg_path = os.path.join(base_dir, 'models', 'meta_config.json')
    
    verdict = "UNKNOWN"
    final_score = 0.0
    
    try:
        meta_clf = joblib.load(meta_clf_path)
        with open(meta_cfg_path, 'r') as f:
            meta_cfg = json.load(f)
            
        threshold_high = meta_cfg.get('threshold_high', 0.54)
        threshold_mid = meta_cfg.get('threshold_mid', 0.30)
        
        # Load calibrated wrappers — these are what the meta-classifier was trained on.
        # calibrated_deep internally runs: raw 2048D -> contrastive_head -> 128D -> xgb_deep -> sigmoid
        calibrated_bio = joblib.load(os.path.join(base_dir, 'models', 'calibrated_bio.pkl'))
        calibrated_deep = joblib.load(os.path.join(base_dir, 'models', 'calibrated_deep.pkl'))
        
        # Feed raw feature arrays through calibrated wrappers
        raw_deep_cols = [f'Deep_{k}' for k in range(2048)]
        X_bio_raw = df_bio_input.values
        X_deep_raw = df_deep_raw[raw_deep_cols].values.astype(np.float32)
        
        p_bio_cal = float(calibrated_bio.predict_proba(X_bio_raw)[:, 1][0])
        p_deep_cal = float(calibrated_deep.predict_proba(X_deep_raw)[:, 1][0])
        
        print(f"Calibrated Bio Score   : {p_bio_cal*100:.1f}%")
        print(f"Calibrated Deep Score  : {p_deep_cal*100:.1f}%")
        
        meta_features = np.array([[
            p_bio_cal,
            p_deep_cal,
            abs(p_bio_cal - p_deep_cal),
            max(p_bio_cal, p_deep_cal),
            min(p_bio_cal, p_deep_cal),
            snr,
            audio_dur,
            silence_ratio
        ]])
        
        meta_prob = float(meta_clf.predict_proba(meta_features)[:, 1][0])
        print(f"Meta-Classifier Out    : {meta_prob*100:.1f}%")
        final_score = meta_prob
        
        # Override Rule
        if p_deep > 0.70 and p_bio < 0.30:
            verdict = "SUSPICIOUS (Deep Stream Disagreement)"
            final_score = max(p_deep, meta_prob)
        elif p_bio > 0.70 and p_deep < 0.30:
            verdict = "SUSPICIOUS (Bio Stream Disagreement)"
            final_score = max(p_bio, meta_prob)
        elif meta_prob >= threshold_high:
            verdict = "HIGH RISK (AI-GENERATED / FRAUD)"
        elif meta_prob >= threshold_mid:
            verdict = "SUSPICIOUS (Moderate Risk)"
        else:
            verdict = "LOW RISK (GENUINE HUMAN VOICE)"
            
    except Exception as e:
        print(f"Meta-classifier not found. Falling back to fixed weights. ({e})")
        final_score = 0.618 * p_deep + 0.382 * p_bio
        print(f"Fixed Fusion Score     : {final_score*100:.1f}%")
        
        if p_deep > 0.70 and p_bio < 0.30:
            verdict = "SUSPICIOUS (Deep Stream Disagreement)"
        elif p_bio > 0.70 and p_deep < 0.30:
            verdict = "SUSPICIOUS (Bio Stream Disagreement)"
        elif final_score >= 0.54:
            verdict = "HIGH RISK (AI-GENERATED / FRAUD)"
        else:
            verdict = "LOW RISK (GENUINE HUMAN VOICE)"

    print(f"\n====================================")
    print(f"FINAL VERDICT : {verdict}")
    print(f"FINAL SCORE   : {final_score*100:.1f}%")
    print(f"====================================\n")

if __name__ == "__main__":
    main()