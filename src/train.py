import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve, confusion_matrix
from scipy.optimize import minimize_scalar

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    return fpr[eer_index]

def extract_group_id(filename):
    """
    Extracts the core base identity of a file to prevent data leakage.
    E.g. elevenlabs_advanced_codec_042.wav -> elevenlabs_042
    """
    fname = str(filename).lower()
    if 'elevenlabs_advanced' in fname:
        # e.g., elevenlabs_advanced_clean_042.wav
        parts = fname.replace('.wav', '').split('_')
        return 'elevenlabs_' + parts[-1]
    
    # For others, if we don't have explicit augmented versions in the dataset,
    # we can use the filename itself.
    return fname

def group_shuffle_split(df, test_size=0.15, val_size=0.15, random_state=42):
    groups = df['GroupID'].unique()
    np.random.seed(random_state)
    np.random.shuffle(groups)
    
    n = len(groups)
    test_idx = int(n * test_size)
    val_idx = int(n * val_size)
    
    test_groups = set(groups[:test_idx])
    val_groups = set(groups[test_idx : test_idx + val_idx])
    train_groups = set(groups[test_idx + val_idx:])
    
    test_idx_arr = df[df['GroupID'].isin(test_groups)].index.values
    val_idx_arr = df[df['GroupID'].isin(val_groups)].index.values
    train_idx_arr = df[df['GroupID'].isin(train_groups)].index.values
    
    return train_idx_arr, val_idx_arr, test_idx_arr

def get_multiclass_label(filename, binary_label):
    if int(binary_label) == 0:
        return 0
    fn_lower = filename.lower()
    if 'elevenlabs' in fn_lower:
        return 1
    elif 'resemble' in fn_lower or 'ashit' in fn_lower or fn_lower.startswith('clip-') or 'voice clone' in fn_lower:
        return 2
    else:
        return 3

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bio_path = os.path.join(base_dir, 'features', 'bio_features.csv')
    deep_path = os.path.join(base_dir, 'features', 'deep_features.csv')

    print("Loading features...")
    df_bio = pd.read_csv(bio_path).dropna()
    df_deep = pd.read_csv(deep_path).dropna()
    
    df_bio = df_bio[df_bio['Label'].isin([0, 1])]
    df_deep = df_deep[df_deep['Label'].isin([0, 1])]

    df = pd.merge(df_deep, df_bio, on=['Filename', 'Label'], suffixes=('_deep', '_bio'))

    # Load hard validation/user feedback features if present and combine
    hard_bio_path = os.path.join(base_dir, 'features', 'hard_val_bio.csv')
    hard_deep_path = os.path.join(base_dir, 'features', 'hard_val_deep.csv')
    if os.path.exists(hard_bio_path) and os.path.exists(hard_deep_path):
        df_hard_bio = pd.read_csv(hard_bio_path).dropna()
        df_hard_deep = pd.read_csv(hard_deep_path).dropna()
        df_hard_bio = df_hard_bio[df_hard_bio['Label'].isin([0.0, 1.0, 0, 1])]
        df_hard_deep = df_hard_deep[df_hard_deep['Label'].astype(str).str.strip().isin(['0.0', '1.0', '0', '1', '0.000000', '1.000000'])]
        df_hard_deep['Label'] = df_hard_deep['Label'].astype(float).astype(int)
        df_hard = pd.merge(df_hard_deep, df_hard_bio, on=['Filename', 'Label'], suffixes=('_deep', '_bio'))
        print(f"Loaded {len(df_hard)} hard validation/user feedback samples.")
        
        df = pd.concat([df, df_hard], ignore_index=True)
        print(f"Total combined features count: {len(df)}")
    
    # Assign Multiclass Labels
    df['MulticlassLabel'] = df.apply(lambda r: get_multiclass_label(r['Filename'], r['Label']), axis=1)
    
    # Assign Group IDs
    df['GroupID'] = df['Filename'].apply(extract_group_id)
    print(f"Total merged samples: {len(df)}")
    print(f"Total unique groups: {df['GroupID'].nunique()}")

    # Identify Hindi subset
    hindi_mask = df['Filename'].str.contains('hindi', case=False) | df['Filename'].str.contains('indicsynth', case=False) | df['Filename'].str.contains('elevenlabs', case=False)
    df_hindi = df[hindi_mask]
    df_eng = df[~hindi_mask]

    # Group split English
    eng_tr, eng_vl, eng_ts = group_shuffle_split(df_eng)
    
    # Group split Hindi
    hin_tr, hin_vl, hin_ts = group_shuffle_split(df_hindi)

    # Combine
    train_idx = np.concatenate([eng_tr, hin_tr])
    val_idx = np.concatenate([eng_vl, hin_vl])
    test_idx = np.concatenate([eng_ts, hin_ts])

    train_df = df.loc[train_idx]
    val_df = df.loc[val_idx]
    test_df = df.loc[test_idx]

    # Verify no leakage
    train_groups = set(train_df['GroupID'])
    test_groups = set(test_df['GroupID'])
    val_groups = set(val_df['GroupID'])
    assert len(train_groups.intersection(test_groups)) == 0, "Data leakage detected between Train/Test!"
    assert len(train_groups.intersection(val_groups)) == 0, "Data leakage detected between Train/Val!"

    print(f"Final Partitions -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    deep_cols = [c for c in df.columns if c.startswith('Deep_')]
    bio_cols = [c for c in df_bio.columns if c not in ['Filename', 'Label']]

    print("\nTraining Contrastive Projection Head on Deep features...")
    from contrastive_head import train_contrastive_head, ContrastiveProjectionHead
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_deep = train_df[deep_cols].values
    y_train_mc = train_df['MulticlassLabel'].values
    
    # Train the projection head
    contrastive_head = train_contrastive_head(X_train_deep, y_train_mc, epochs=20, batch_size=128)
    
    # Save the PyTorch model weights
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    torch.save(contrastive_head.state_dict(), os.path.join(models_dir, 'contrastive_head.pt'))
    print("Contrastive projection head saved to models/contrastive_head.pt")
    
    # Project features for XGB_Deep
    contrastive_head.eval()
    with torch.no_grad():
        X_train_deep_proj = contrastive_head(torch.tensor(X_train_deep, dtype=torch.float32).to(device)).cpu().numpy()
        X_val_deep_proj = contrastive_head(torch.tensor(val_df[deep_cols].values, dtype=torch.float32).to(device)).cpu().numpy()
        X_test_deep_proj = contrastive_head(torch.tensor(test_df[deep_cols].values, dtype=torch.float32).to(device)).cpu().numpy()
        
    proj_cols = [f'Proj_Deep_{i}' for i in range(128)]
    train_df_proj = pd.DataFrame(X_train_deep_proj, columns=proj_cols, index=train_df.index)
    val_df_proj = pd.DataFrame(X_val_deep_proj, columns=proj_cols, index=val_df.index)
    test_df_proj = pd.DataFrame(X_test_deep_proj, columns=proj_cols, index=test_df.index)
    
    print("\nTraining XGB_Deep (Wav2Vec2 stream in multiclass space on 128D projected features)...")
    xgb_deep = xgb.XGBClassifier(objective='multi:softprob', num_class=4, n_estimators=200, max_depth=5, learning_rate=0.05, n_jobs=-1, random_state=42)
    xgb_deep.fit(train_df_proj, train_df['MulticlassLabel'])

    print("Training XGB_Bio (Biological stream in multiclass)...")
    xgb_bio = xgb.XGBClassifier(objective='multi:softprob', num_class=4, n_estimators=200, max_depth=5, learning_rate=0.05, n_jobs=-1, random_state=42)
    xgb_bio.fit(train_df[bio_cols], train_df['MulticlassLabel'])

    # Optimize Fusion
    p_val_deep = 1.0 - xgb_deep.predict_proba(val_df_proj)[:, 0]
    p_val_bio = 1.0 - xgb_bio.predict_proba(val_df[bio_cols])[:, 0]
    y_val = val_df['Label'].values

    def fusion_eer(w):
        return compute_eer(y_val, w * p_val_deep + (1 - w) * p_val_bio)

    res = minimize_scalar(fusion_eer, bounds=(0, 1), method='bounded')
    optimal_w = res.x
    print(f"\nOptimization complete. Optimal Deep Weight: {optimal_w:.3f}")

    # Test Eval
    p_test_deep = 1.0 - xgb_deep.predict_proba(test_df_proj)[:, 0]
    p_test_bio = 1.0 - xgb_bio.predict_proba(test_df[bio_cols])[:, 0]
    p_test_fused = optimal_w * p_test_deep + (1 - optimal_w) * p_test_bio
    y_test = test_df['Label'].values

    print("\n--- Test Set Results (Group Split) ---")
    print(f"Overall Fusion EER: {compute_eer(y_test, p_test_fused):.4f}")
    
    hin_test_df = df.loc[hin_ts]
    if len(hin_test_df) > 0:
        with torch.no_grad():
            X_hin_deep = contrastive_head(torch.tensor(hin_test_df[deep_cols].values, dtype=torch.float32).to(device)).cpu().numpy()
        hin_test_df_proj = pd.DataFrame(X_hin_deep, columns=proj_cols, index=hin_test_df.index)
        p_hin_deep = 1.0 - xgb_deep.predict_proba(hin_test_df_proj)[:, 0]
        p_hin_bio = 1.0 - xgb_bio.predict_proba(hin_test_df[bio_cols])[:, 0]
        p_hin_fused = optimal_w * p_hin_deep + (1 - optimal_w) * p_hin_bio
        y_hin = hin_test_df['Label'].values
        try:
            print(f"Hindi Subset EER:   {compute_eer(y_hin, p_hin_fused):.4f}")
        except:
            pass

    eleven_mask = test_df['Filename'].str.contains('elevenlabs', case=False)
    eleven_test_df = test_df[eleven_mask]
    if len(eleven_test_df) > 0:
        with torch.no_grad():
            X_el_deep = contrastive_head(torch.tensor(eleven_test_df[deep_cols].values, dtype=torch.float32).to(device)).cpu().numpy()
        eleven_test_df_proj = pd.DataFrame(X_el_deep, columns=proj_cols, index=eleven_test_df.index)
        p_el_deep = 1.0 - xgb_deep.predict_proba(eleven_test_df_proj)[:, 0]
        p_el_bio = 1.0 - xgb_bio.predict_proba(eleven_test_df[bio_cols])[:, 0]
        p_el_fused = optimal_w * p_el_deep + (1 - optimal_w) * p_el_bio
        
        real_mask = test_df['Label'] == 0
        real_test_df = test_df[real_mask].sample(n=min(len(eleven_test_df), len(test_df[real_mask])), random_state=42)
        
        with torch.no_grad():
            X_real_deep = contrastive_head(torch.tensor(real_test_df[deep_cols].values, dtype=torch.float32).to(device)).cpu().numpy()
        real_test_df_proj = pd.DataFrame(X_real_deep, columns=proj_cols, index=real_test_df.index)
        p_real_deep = 1.0 - xgb_deep.predict_proba(real_test_df_proj)[:, 0]
        
        eval_df = pd.concat([eleven_test_df, real_test_df])
        p_eval_fused = np.concatenate([p_el_fused, (optimal_w * p_real_deep + (1 - optimal_w) * (1.0 - xgb_bio.predict_proba(real_test_df[bio_cols])[:, 0]))])
        try:
            print(f"ElevenLabs Subset EER: {compute_eer(eval_df['Label'].values, p_eval_fused):.4f}")
        except:
            pass

    # Save models in JSON format for the UI to load
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    xgb_deep.save_model(os.path.join(base_dir, 'models', 'xgb_deep.json'))
    xgb_bio.save_model(os.path.join(base_dir, 'models', 'xgb_bio.json'))
    with open(os.path.join(base_dir, 'models', 'fusion_weights.json'), 'w') as f:
        json.dump({'w_deep': optimal_w, 'w_bio': 1 - optimal_w}, f)

    print("\nTraining complete and models saved to models/ (.json format).")

if __name__ == "__main__":
    main()