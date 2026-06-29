import os
import sys
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import librosa
import scipy.io.wavfile as wavfile
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_curve
import joblib
import torch
from contrastive_head import ContrastiveProjectionHead
from tqdm import tqdm

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    return fpr[eer_index], thresholds[eer_index]

def extract_group_id(filename):
    fname = str(filename).lower()
    if 'elevenlabs_advanced' in fname:
        parts = fname.replace('.wav', '').split('_')
        return 'elevenlabs_' + parts[-1]
    return fname

def group_shuffle_split(df, test_size=0.15, val_size=0.15, random_state=42):
    df_temp = df.reset_index(drop=True)
    groups = list(df_temp['GroupID'].unique())
    np.random.seed(random_state)
    np.random.shuffle(groups)
    
    n = len(groups)
    test_idx = int(n * test_size)
    val_idx = int(n * val_size)
    
    test_groups = set(groups[:test_idx])
    val_groups = set(groups[test_idx : test_idx + val_idx])
    train_groups = set(groups[test_idx + val_idx:])
    
    test_idx_arr = df_temp[df_temp['GroupID'].isin(test_groups)].index.values
    val_idx_arr = df_temp[df_temp['GroupID'].isin(val_groups)].index.values
    train_idx_arr = df_temp[df_temp['GroupID'].isin(train_groups)].index.values
    
    return train_idx_arr, val_idx_arr, test_idx_arr

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

def load_audio_fast(path):
    try:
        if path.lower().endswith('.wav'):
            sr, y = wavfile.read(path)
            y = y.astype(np.float32)
            if np.abs(y).max() > 1.5:
                y = y / 32768.0  # normalize PCM16
            if len(y.shape) > 1:
                y = y.mean(axis=1)  # stereo to mono
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
            return y, sr
    except Exception as e:
        pass
    return librosa.load(path, sr=16000, mono=True)

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

def main():
    print("Loading base XGBoost models...")
    xgb_bio = xgb.XGBClassifier()
    xgb_bio.load_model(os.path.join(base_dir, 'models', 'xgb_bio.json'))
    bio_cols = xgb_bio.get_booster().feature_names
    
    xgb_deep = xgb.XGBClassifier()
    xgb_deep.load_model(os.path.join(base_dir, 'models', 'xgb_deep.json'))
    
    # Load contrastive head
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contrastive_head = ContrastiveProjectionHead().to(device)
    contrastive_head.load_state_dict(torch.load(os.path.join(base_dir, 'models', 'contrastive_head.pt'), map_location=device))
    contrastive_head.eval()
    
    deep_cols = [f'Deep_{i}' for i in range(2048)]

    print("Loading features CSVs...")
    features_dir = os.path.join(base_dir, 'features')
    df_bio = pd.read_csv(os.path.join(features_dir, 'bio_features.csv')).dropna()
    df_deep = pd.read_csv(os.path.join(features_dir, 'deep_features.csv')).dropna()
    
    df_bio = df_bio[df_bio['Label'].isin([0, 1])]
    df_deep = df_deep[df_deep['Label'].isin([0, 1])]

    df_hard_bio = pd.read_csv(os.path.join(features_dir, 'hard_val_bio.csv')).dropna()
    df_hard_deep = pd.read_csv(os.path.join(features_dir, 'hard_val_deep.csv')).dropna()
    
    df_hard_bio = df_hard_bio[df_hard_bio['Label'].isin([0, 1])]
    df_hard_deep = df_hard_deep[df_hard_deep['Label'].isin([0, 1])]
    
    # Merge on filename
    df_main = pd.merge(df_deep, df_bio, on=['Filename', 'Label'], suffixes=('_deep', '_bio'))
    df_hard = pd.merge(df_hard_deep, df_hard_bio, on=['Filename', 'Label'], suffixes=('_deep', '_bio'))
    
    print(f"Total main clips loaded: {len(df_main)}")
    print(f"Total hard validation clips loaded: {len(df_hard)}")

    # Fast directory index mapping required files
    print("\nBuilding filename lookup index (caching walk)...")
    required_filenames = set(df_main['Filename'].tolist() + df_hard['Filename'].tolist())
    file_paths = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # Check lower case and original case for robustness
            file_lower = file.lower()
            if file in required_filenames:
                file_paths[file] = os.path.join(root, file)
            elif file_lower in required_filenames:
                file_paths[file_lower] = os.path.join(root, file)

    print(f"Indexed {len(file_paths)} out of {len(required_filenames)} required WAV files.")

    # Extract audio quality features from files on disk
    aq_features = {}
    print("\nExtracting SNR, duration, and silence ratios from WAV files...")
    for filename in tqdm(required_filenames, desc="WAV Audio Quality Extraction"):
        path = file_paths.get(filename)
        if path is None or not os.path.exists(path):
            # Safe fallbacks if file is missing
            aq_features[filename] = {'snr': 20.0, 'audio_dur': 4.0, 'silence_ratio': 0.1}
            continue
            
        try:
            y, sr = load_audio_fast(path)
            dur = len(y) / sr
            snr_val = compute_snr(y)
            sil_ratio = compute_silence_ratio(y, sr)
            aq_features[filename] = {'snr': snr_val, 'audio_dur': dur, 'silence_ratio': sil_ratio}
        except Exception as e:
            aq_features[filename] = {'snr': 20.0, 'audio_dur': 4.0, 'silence_ratio': 0.1}

    # Group splitting main dataset
    df_main['GroupID'] = df_main['Filename'].apply(extract_group_id)
    hindi_mask = df_main['Filename'].str.contains('hindi', case=False) | \
                 df_main['Filename'].str.contains('indicsynth', case=False) | \
                 df_main['Filename'].str.contains('elevenlabs', case=False)
                 
    df_hindi = df_main[hindi_mask].reset_index(drop=True)
    df_eng = df_main[~hindi_mask].reset_index(drop=True)

    # Group split subsets
    eng_tr, eng_vl, eng_ts = group_shuffle_split(df_eng)
    hin_tr, hin_vl, hin_ts = group_shuffle_split(df_hindi)

    # Reconstruct dataframes
    train_df = pd.concat([df_eng.iloc[eng_tr], df_hindi.iloc[hin_tr]], ignore_index=True)
    val_df = pd.concat([df_eng.iloc[eng_vl], df_hindi.iloc[hin_vl]], ignore_index=True)
    test_df = pd.concat([df_eng.iloc[eng_ts], df_hindi.iloc[hin_ts]], ignore_index=True)

    # Append hard validation files to training split of meta-classifier
    train_df = pd.concat([train_df, df_hard], ignore_index=True)
    print(f"Splits -> Train: {len(train_df)} (includes hard val), Val: {len(val_df)}, Test: {len(test_df)}")

    # 1. Base Classifier Wrappers
    # We use BinaryWrapper directly to get raw probabilities on a 0.0-1.0 scale.
    # Sigmoid calibration is removed because the 0.00% EER validation set makes sigmoid fitting numerically unstable (complete separation).
    calibrated_bio = BinaryWrapper(xgb_bio)
    calibrated_deep = BinaryWrapper(xgb_deep, contrastive_head=contrastive_head)

    # 2. Build 8-column meta-features mapping function
    def build_meta_features(df_split):
        X_bio = df_split[bio_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
        X_deep = df_split[deep_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
        
        # Predicted probabilities
        p_bio = calibrated_bio.predict_proba(X_bio)[:, 1]
        p_deep = calibrated_deep.predict_proba(X_deep)[:, 1]
        
        # Audio quality features from lookup table
        snr = df_split['Filename'].map(lambda x: aq_features[x]['snr']).values
        audio_dur = df_split['Filename'].map(lambda x: aq_features[x]['audio_dur']).values
        silence_ratio = df_split['Filename'].map(lambda x: aq_features[x]['silence_ratio']).values
        
        meta_df = pd.DataFrame({
            'bio_score': p_bio,
            'deep_score': p_deep,
            'disagreement': np.abs(p_bio - p_deep),
            'max_score': np.maximum(p_bio, p_deep),
            'min_score': np.minimum(p_bio, p_deep),
            'snr': snr,
            'audio_dur': audio_dur,
            'silence_ratio': silence_ratio
        })
        return meta_df

    # Construct split matrices
    X_train_meta = build_meta_features(train_df)
    y_train = train_df['Label'].values
    
    X_val_meta = build_meta_features(val_df)
    y_val = val_df['Label'].values
    
    X_test_meta = build_meta_features(test_df)
    y_test = test_df['Label'].values

    # Train Logistic Regression meta-classifier
    print("\nTraining Logistic Regression Meta-Classifier...")
    meta_clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    meta_clf.fit(X_train_meta, y_train)

    # Calculate optimal thresholds on validation set
    val_probs = meta_clf.predict_proba(X_val_meta)[:, 1]
    
    # EER threshold (high threshold)
    val_eer, eer_threshold = compute_eer(y_val, val_probs)
    
    # 5% FPR threshold (mid threshold)
    fpr, tpr, thresholds = roc_curve(y_val, val_probs)
    idx_5pct = np.where(fpr <= 0.05)[0][-1]
    mid_threshold = thresholds[idx_5pct]

    print(f"Validation EER Threshold: {eer_threshold:.4f} (EER: {val_eer*100:.2f}%)")
    print(f"Validation 5% FPR Threshold: {mid_threshold:.4f}")

    # Evaluate on Test Set
    test_probs = meta_clf.predict_proba(X_test_meta)[:, 1]
    test_eer, _ = compute_eer(y_test, test_probs)

    # 5-input baseline comparison (scores only)
    X_train_5in = X_train_meta[['bio_score', 'deep_score', 'disagreement', 'max_score', 'min_score']]
    X_test_5in = X_test_meta[['bio_score', 'deep_score', 'disagreement', 'max_score', 'min_score']]
    meta_5in = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    meta_5in.fit(X_train_5in, y_train)
    test_eer_5in, _ = compute_eer(y_test, meta_5in.predict_proba(X_test_5in)[:, 1])

    # Fixed weight baseline comparison
    try:
        with open(os.path.join(base_dir, 'models', 'fusion_weights.json'), 'r') as f:
            weights = json.load(f)
        w_deep = weights.get('w_deep', 0.236)
        w_bio = weights.get('w_bio', 0.764)
    except:
        w_deep = 0.236
        w_bio = 0.764

    # Compute raw test predictions for fixed weight comparison
    X_test_bio_raw = test_df[bio_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    X_test_deep_raw = test_df[deep_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    p_bio_raw = 1.0 - xgb_bio.predict_proba(X_test_bio_raw)[:, 0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        X_test_deep_tensor = torch.tensor(X_test_deep_raw, dtype=torch.float32).to(device)
        X_test_deep_proj = contrastive_head(X_test_deep_tensor).cpu().numpy()
    proj_cols = [f'Proj_Deep_{i}' for i in range(128)]
    df_test_deep_proj = pd.DataFrame(X_test_deep_proj, columns=proj_cols)
    p_deep_raw = 1.0 - xgb_deep.predict_proba(df_test_deep_proj)[:, 0]
    
    p_fixed_weight = w_deep * p_deep_raw + w_bio * p_bio_raw
    fixed_eer, _ = compute_eer(y_test, p_fixed_weight)

    print("\n================ COMPARISON METRICS ================")
    print(f"Fixed weight EER (test):      {fixed_eer*100:.2f}% (deep weight: {w_deep})")
    print(f"5-input meta EER (test):      {test_eer_5in*100:.2f}%")
    print(f"8-input meta EER (test):      {test_eer*100:.2f}%")
    print("====================================================")

    # Save trained calibrated wrappers and meta classifier
    models_dir = os.path.join(base_dir, 'models')
    joblib.dump(calibrated_bio, os.path.join(models_dir, 'calibrated_bio.pkl'))
    joblib.dump(calibrated_deep, os.path.join(models_dir, 'calibrated_deep.pkl'))
    joblib.dump(meta_clf, os.path.join(models_dir, 'meta_classifier.pkl'))

    # Save configurations
    config = {
        "threshold_high": float(eer_threshold),
        "threshold_mid": float(mid_threshold),
        "feature_names": list(X_train_meta.columns),
        "val_eer": float(val_eer),
        "test_eer": float(test_eer),
        "fixed_weight_test_eer": float(fixed_eer)
    }
    with open(os.path.join(models_dir, 'meta_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print("\nRetraining complete and calibrated models/configs saved to models/")

if __name__ == "__main__":
    main()