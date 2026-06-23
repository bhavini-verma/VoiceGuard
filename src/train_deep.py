import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import joblib
from tqdm import tqdm

# Compute Equal Error Rate (EER)
def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    return fpr[eer_index], thresholds[eer_index]

# Base Group ID extraction logic to prevent data leakage
def extract_group_id(filename):
    fname = str(filename).lower()
    if 'elevenlabs_advanced' in fname:
        parts = fname.replace('.wav', '').split('_')
        return 'elevenlabs_' + parts[-1]
    return fname

# Group shuffle split logic consistent with audit/train.py
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

# Dataset Definition
class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Residual block for MLP
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

# Unified SOTA Deep Fusion Network
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

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bio_path = os.path.join(base_dir, 'features', 'bio_features.csv')
    deep_path = os.path.join(base_dir, 'features', 'deep_features.csv')

    print("Loading features CSVs...")
    df_bio = pd.read_csv(bio_path).dropna()
    df_deep = pd.read_csv(deep_path).dropna()
    
    # Filter valid labels (0 = Real, 1 = Fake)
    df_bio = df_bio[df_bio['Label'].isin([0, 1])]
    df_deep = df_deep[df_deep['Label'].isin([0, 1])]

    print(f"Loaded Bio features: {df_bio.shape}")
    print(f"Loaded Deep features: {df_deep.shape}")

    # Merge dataframes
    df = pd.merge(df_deep, df_bio, on=['Filename', 'Label'], suffixes=('_deep', '_bio'))
    print(f"Total merged samples: {len(df)}")
    
    # Setup GroupID for splitting
    df['GroupID'] = df['Filename'].apply(extract_group_id)

    # Separate English and Hindi subsets for stratified group splits
    hindi_mask = df['Filename'].str.contains('hindi', case=False) | \
                 df['Filename'].str.contains('indicsynth', case=False) | \
                 df['Filename'].str.contains('elevenlabs', case=False)
    
    df_hindi = df[hindi_mask].reset_index(drop=True)
    df_eng = df[~hindi_mask].reset_index(drop=True)

    print(f"English samples: {len(df_eng)}, Hindi/TwelveLabs samples: {len(df_hindi)}")

    # Perform group splits on subsets
    eng_tr, eng_vl, eng_ts = group_shuffle_split(df_eng)
    hin_tr, hin_vl, hin_ts = group_shuffle_split(df_hindi)

    # Map back to train/val/test dataframes
    train_df = pd.concat([df_eng.iloc[eng_tr], df_hindi.iloc[hin_tr]], ignore_index=True)
    val_df = pd.concat([df_eng.iloc[eng_vl], df_hindi.iloc[hin_vl]], ignore_index=True)
    test_df = pd.concat([df_eng.iloc[eng_ts], df_hindi.iloc[hin_ts]], ignore_index=True)

    # Double check leak prevention
    train_groups = set(train_df['GroupID'])
    val_groups = set(val_df['GroupID'])
    test_groups = set(test_df['GroupID'])
    assert len(train_groups.intersection(val_groups)) == 0, "Data leakage between Train and Val!"
    assert len(train_groups.intersection(test_groups)) == 0, "Data leakage between Train and Test!"
    assert len(val_groups.intersection(test_groups)) == 0, "Data leakage between Val and Test!"

    print(f"Final Partitions -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Feature column identification
    deep_cols = [c for c in df.columns if c.startswith('Deep_')]
    bio_cols = [c for c in df_bio.columns if c not in ['Filename', 'Label']]
    feature_cols = deep_cols + bio_cols
    print(f"Total features: {len(feature_cols)} ({len(deep_cols)} deep, {len(bio_cols)} biological)")

    # Prepare inputs and labels
    X_train_raw = train_df[feature_cols].values
    y_train = train_df['Label'].values
    
    X_val_raw = val_df[feature_cols].values
    y_val = val_df['Label'].values
    
    X_test_raw = test_df[feature_cols].values
    y_test = test_df['Label'].values

    # Fit and save StandardScaler
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, 'deep_fusion_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Datasets and Loaders
    train_dataset = FeatureDataset(X_train, y_train)
    val_dataset = FeatureDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_dim = len(feature_cols)
    model = VoiceGuardDeepFusionNet(input_dim=input_dim).to(device)

    # Calculate class weights for BCE loss
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32).to(device)
    print(f"Class imbalance weights: {pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training Loop
    epochs = 30
    best_val_loss = float('inf')
    best_model_state = None
    
    print("\nTraining Deep Fusion Network Classifier...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(batch_y.cpu().numpy())
                
        val_loss /= len(val_dataset)
        val_probs = np.array(val_probs).squeeze()
        val_targets = np.array(val_targets).squeeze()
        
        val_eer, _ = compute_eer(val_targets, val_probs)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val EER: {val_eer*100:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best validation loss model checkpoint.")

    # Evaluate on Test Set
    model.eval()
    test_probs = []
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        for i in range(0, len(X_test_tensor), 256):
            batch = X_test_tensor[i:i+256]
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.extend(probs)
            
    test_probs = np.array(test_probs).squeeze()
    test_eer, eer_threshold = compute_eer(y_test, test_probs)
    print(f"\n--- Final Test Set Results ---")
    print(f"Test Equal Error Rate (EER): {test_eer*100:.2f}% (Threshold: {eer_threshold:.4f})")

    # Calculate 5% FPR threshold (Suspicious threshold)
    fpr, tpr, thresholds = roc_curve(y_test, test_probs)
    idx_5pct = np.where(fpr <= 0.05)[0][-1]
    mid_threshold = thresholds[idx_5pct]
    print(f"5% FPR Threshold:            {mid_threshold:.4f}")

    # Save model checkpoint
    model_save_path = os.path.join(models_dir, 'deep_fusion_classifier.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")

    # Save configurations and thresholds
    config_save_path = os.path.join(models_dir, 'deep_fusion_config.json')
    config = {
        "threshold_high": float(eer_threshold),
        "threshold_mid": float(mid_threshold),
        "input_dim": input_dim,
        "deep_feature_count": len(deep_cols),
        "bio_feature_count": len(bio_cols),
        "feature_names": feature_cols
    }
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config JSON saved to {config_save_path}")

if __name__ == "__main__":
    main()
