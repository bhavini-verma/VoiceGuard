import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE_DIR, "VoiceGuard_DL_Exp")
FEATURES_DIR = os.path.join(BASE_DIR, "VoiceGaurd_TelephonyExp", "features")

class EarlyFusionDNN(nn.Module):
    def __init__(self, input_dim=4193, hidden_dims=[1024, 256, 64]):
        super(EarlyFusionDNN, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.4))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def load_and_merge_data():
    print("Loading biological features...")
    df_bio = pd.read_csv(os.path.join(FEATURES_DIR, "bio_features.csv"))
    
    print("Loading deep features...")
    df_deep = pd.read_csv(os.path.join(FEATURES_DIR, "deep_features.csv"))
    
    # Filter labels to only 0 (genuine) and 1 (synthetic)
    df_bio = df_bio[df_bio['Label'].isin([0, 1])]
    df_deep = df_deep[df_deep['Label'].isin([0, 1])]
    
    print("Merging datasets...")
    # Drop Label from one before merge to avoid Label_x, Label_y
    df_deep = df_deep.drop(columns=['Label'], errors='ignore')
    
    df_merged = pd.merge(df_bio, df_deep, on='Filename', how='inner')
    print(f"Merged shape: {df_merged.shape}")
    
    # Fill NA values
    df_merged = df_merged.fillna(0)
    
    # Extract X and y
    y = df_merged['Label'].values
    
    # Drop non-feature columns
    cols_to_drop = ['Filename', 'Label']
    X = df_merged.drop(columns=cols_to_drop).values
    
    feature_names = df_merged.drop(columns=cols_to_drop).columns.tolist()
    
    return X, y, feature_names

def main():
    if not os.path.exists(EXP_DIR):
        os.makedirs(EXP_DIR)
        
    X, y, feature_names = load_and_merge_data()
    
    print(f"Total features: {X.shape[1]}")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    
    # DataLoaders
    batch_size = 64
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = EarlyFusionDNN(input_dim=X.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    epochs = 30
    best_auc = 0
    patience = 10
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(batch_y.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_auc = roc_auc_score(all_targets, all_preds)
        val_preds_bin = (np.array(all_preds) > 0.5).astype(int)
        val_acc = accuracy_score(all_targets, val_preds_bin)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_auc)
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_,
                'feature_names': feature_names
            }, os.path.join(EXP_DIR, "dl_early_fusion.pt"))
            print("  --> Saved best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
    print("\nTraining Complete.")
    print(f"Best Validation AUC: {best_auc:.4f}")
    
    # Final evaluation on val set
    checkpoint = torch.load(os.path.join(EXP_DIR, "dl_early_fusion.pt"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for batch_X, _ in val_loader:
            batch_X = batch_X.to(device)
            probs = torch.sigmoid(model(batch_X)).cpu().numpy()
            all_preds.extend(probs)
            
    val_preds_bin = (np.array(all_preds) > 0.5).astype(int)
    print("\nFinal Classification Report:")
    print(classification_report(all_targets, val_preds_bin))

if __name__ == '__main__':
    main()
