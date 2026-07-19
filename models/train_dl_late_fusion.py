import os, json, joblib, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE_DIR, "VoiceGuard_DL_Exp")
FEATURES_DIR = os.path.join(BASE_DIR, "VoiceGaurd_TelephonyExp", "features")
MODELS_DIR = os.path.join(EXP_DIR, "models_late_fusion")
os.makedirs(MODELS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

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

def load_data(path):
    df = pd.read_csv(path)
    df = df[df["Label"].isin([0, 1])].fillna(0)
    y = df["Label"].values
    drop = [c for c in ["Label", "Filename"] if c in df.columns]
    X = df.drop(columns=drop).values
    return X, y, df.drop(columns=drop).columns.tolist()

def train_model(model, Xtr, ytr, Xvl, yvl, epochs=60, lr=1e-3, patience=12, save_path=None, name="Model"):
    tr_dl = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr).unsqueeze(1)), batch_size=64, shuffle=True)
    vl_dl = DataLoader(TensorDataset(torch.FloatTensor(Xvl), torch.FloatTensor(yvl).unsqueeze(1)), batch_size=64)
    model = model.to(DEVICE)
    crit = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    best_auc, pc, best_probs, best_targets = 0.0, 0, None, None
    for ep in range(1, epochs+1):
        model.train()
        tl = 0
        for bx, by in tr_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            tl += loss.item() * bx.size(0)
        tl /= len(tr_dl.dataset)
        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for bx, by in vl_dl:
                preds.extend(torch.sigmoid(model(bx.to(DEVICE))).cpu().numpy().flatten())
                tgts.extend(by.numpy().flatten())
        auc = roc_auc_score(tgts, preds)
        acc = accuracy_score(tgts, (np.array(preds)>0.5).astype(int))
        print(f"[{name}] Ep {ep:02d} | Loss:{tl:.4f} | AUC:{auc:.4f} | Acc:{acc:.4f}")
        sched.step(auc)
        if auc > best_auc:
            best_auc, pc = auc, 0
            best_probs, best_targets = np.array(preds), np.array(tgts)
            if save_path:
                torch.save({"model_state_dict": model.state_dict()}, save_path)
            print(f"  --> Saved best (AUC={best_auc:.4f})")
        else:
            pc += 1
            if pc >= patience:
                print(f"  --> Early stop ep {ep}")
                break
    print(f"[{name}] Best AUC: {best_auc:.4f}")
    return best_probs, best_targets

def main():
    print("="*60)
    print("STEP 1: BioDNN on 97 Biological Features")
    print("="*60)
    Xb, yb, bn = load_data(os.path.join(FEATURES_DIR, "bio_features.csv"))
    print(f"Bio dataset: {Xb.shape}")
    Xbtr, Xbvl, ybtr, ybvl = train_test_split(Xb, yb, test_size=0.2, random_state=42, stratify=yb)
    sb = StandardScaler()
    Xbtr_s = sb.fit_transform(Xbtr)
    Xbvl_s = sb.transform(Xbvl)
    bio_path = os.path.join(MODELS_DIR, "dl_bio.pt")
    bp, bt = train_model(BioDNN(Xb.shape[1]), Xbtr_s, ybtr, Xbvl_s, ybvl, save_path=bio_path, name="BioDNN")
    ckb = torch.load(bio_path, map_location="cpu", weights_only=False)
    ckb.update({"scaler_mean": sb.mean_, "scaler_scale": sb.scale_, "feature_names": bn,
                "input_dim": int(Xb.shape[1]), "architecture": "BioDNN"})
    torch.save(ckb, bio_path)
    print("Bio Classification Report:")
    print(classification_report(bt, (bp>0.5).astype(int)))

    print("="*60)
    print("STEP 2: DeepDNN on 4096 Wav2Vec2 Features")
    print("="*60)
    Xd, yd, dn = load_data(os.path.join(FEATURES_DIR, "deep_features.csv"))
    print(f"Deep dataset: {Xd.shape}")
    Xdtr, Xdvl, ydtr, ydvl = train_test_split(Xd, yd, test_size=0.2, random_state=42, stratify=yd)
    sd = StandardScaler()
    Xdtr_s = sd.fit_transform(Xdtr)
    Xdvl_s = sd.transform(Xdvl)
    deep_path = os.path.join(MODELS_DIR, "dl_deep.pt")
    dp, dt = train_model(DeepDNN(Xd.shape[1]), Xdtr_s, ydtr, Xdvl_s, ydvl, lr=5e-4, save_path=deep_path, name="DeepDNN")
    ckd = torch.load(deep_path, map_location="cpu", weights_only=False)
    ckd.update({"scaler_mean": sd.mean_, "scaler_scale": sd.scale_, "feature_names": dn,
                "input_dim": int(Xd.shape[1]), "architecture": "DeepDNN"})
    torch.save(ckd, deep_path)
    print("Deep Classification Report:")
    print(classification_report(dt, (dp>0.5).astype(int)))

    print("="*60)
    print("STEP 3: XGBoost Meta-Classifier")
    print("Features: [bio_score, deep_score, disagreement, max_score, min_score]")
    print("="*60)
    n = min(len(bp), len(dp))
    p_bio, p_deep, my = bp[:n], dp[:n], bt[:n]
    mX = np.column_stack([p_bio, p_deep, np.abs(p_bio-p_deep), np.maximum(p_bio,p_deep), np.minimum(p_bio,p_deep)])
    Xmtr, Xmvl, ymtr, ymvl = train_test_split(mX, my, test_size=0.2, random_state=7, stratify=my)
    meta = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                             random_state=42, n_jobs=-1)
    meta.fit(Xmtr, ymtr, eval_set=[(Xmvl, ymvl)], verbose=False)
    mp = meta.predict_proba(Xmvl)[:,1]
    mpred = meta.predict(Xmvl)
    auc = roc_auc_score(ymvl, mp)
    acc = accuracy_score(ymvl, mpred)
    print(f"Meta XGBoost | AUC:{auc:.4f} | Acc:{acc:.4f}")
    print(classification_report(ymvl, mpred))
    meta_path = os.path.join(MODELS_DIR, "meta_classifier.pkl")
    joblib.dump(meta, meta_path)
    cfg = {"threshold_high": 0.54, "threshold_mid": 0.30,
           "feature_order": ["bio_score","deep_score","disagreement","max_score","min_score"],
           "architecture": "PyTorch Late Fusion DL + XGBoost Meta",
           "meta_auc": round(float(auc),4), "meta_acc": round(float(acc),4)}
    with open(os.path.join(MODELS_DIR, "meta_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved dl_bio.pt, dl_deep.pt, meta_classifier.pkl to {MODELS_DIR}")
    print("TRAINING COMPLETE!")

if __name__ == "__main__":
    main()
