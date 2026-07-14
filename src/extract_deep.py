import os
import glob
import numpy as np
import pandas as pd
import librosa
import torch
import traceback
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

def simulate_phone_codec(y, sr=16000):
    """Simulate GSM/AMR phone codec degradation by downsampling to 8kHz and back."""
    y_8k = librosa.resample(y, orig_sr=sr, target_sr=8000)
    y_degraded = librosa.resample(y_8k, orig_sr=8000, target_sr=sr)
    return y_degraded

def get_audio_files(data_dir):
    audio_files = []
    
    # Define limits per subfolder
    subfolder_limits = {
        'ljspeech': 2000,
        'wavefake': 2000,
        'asvspoof_2021_df': 1000,
        'hindi_common_voice': 1000,
        'indicsynth_hindi': 1000,
        'hard_negatives': 1000,
        'real_replays': 1000
    }
    
    # Track counts per subfolder category to enforce limits
    counts = {}
    
    # Search for wav, flac, and mp3 files
    for ext in ['*.wav', '*.flac', '*.mp3', '*.ogg']:
        for file in glob.glob(os.path.join(data_dir, '**', ext), recursive=True):
            if 'DEMONSTRATION' in file:
                continue
                
            norm_path = os.path.normpath(file).lower()
            path_parts = norm_path.split(os.sep)
            
            label = None
            category = None
            
            if 'ljspeech' in path_parts:
                category = 'ljspeech'
                label = 0
            elif 'wavefake' in path_parts:
                category = 'wavefake'
                label = 1
            elif 'asvspoof_2021_df' in path_parts:
                category = 'asvspoof_2021_df'
                # ASVspoof has both real and fake subfolders under raw_real and raw_fake
                if 'raw_real' in norm_path:
                    label = 0
                else:
                    label = 1
            elif 'hindi_common_voice' in path_parts:
                category = 'hindi_common_voice'
                label = 0
            elif 'indicsynth_hindi' in path_parts:
                category = 'indicsynth_hindi'
                label = 1
            elif 'hard_negatives' in path_parts:
                category = 'hard_negatives'
                label = 0  # Hard negatives are human voices (REAL)
            elif 'real_replays' in path_parts:
                category = 'real_replays'
                label = 2  # Replays are REPLAY
            elif '1_genuine' in path_parts:
                # Use the subfolder name as category
                idx = path_parts.index('1_genuine')
                if idx + 1 < len(path_parts):
                    category = path_parts[idx + 1]
                else:
                    category = '1_genuine'
                label = 0
            elif '2_synthetic' in path_parts:
                # Use the subfolder name as category
                idx = path_parts.index('2_synthetic')
                if idx + 1 < len(path_parts):
                    category = path_parts[idx + 1]
                else:
                    category = '2_synthetic'
                label = 1
            elif 'real' in path_parts:
                category = 'real'
                label = 0
            elif 'fake' in path_parts:
                category = 'fake'
                label = 1
                
            if category and label is not None:
                limit = subfolder_limits.get(category, 1000)
                current_count = counts.get(category, 0)
                if current_count < limit:
                    audio_files.append((file, label))
                    counts[category] = current_count + 1
                    
    return audio_files

def main():
    print("Starting Deep Engine Feature Extraction (Batched)...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    features_dir = os.path.join(base_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_local_path = os.path.join(base_dir, 'models', 'indicwav2vec-hindi')
    from transformers import Wav2Vec2FeatureExtractor
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_local_path, local_files_only=True)
    model = Wav2Vec2Model.from_pretrained(model_local_path, use_safetensors=True, output_hidden_states=True, local_files_only=True).to(device)
    model.eval()
    
    audio_files = get_audio_files(data_dir)

    if not audio_files:
        print("No audio files found! Please place some .wav or .flac files in data/raw_real, data/raw_fake.")
        return

    print(f"Found {len(audio_files)} audio files to process.")
    
    output_csv = os.path.join(features_dir, 'deep_features.csv')
    
    # Check if we are resuming
    processed_files = set()
    if os.path.exists(output_csv):
        print(f"Found existing {output_csv}. Resuming...")
        try:
            existing_df = pd.read_csv(output_csv, usecols=['Filename'])
            processed_files = set(existing_df['Filename'].tolist())
            print(f"Already processed {len(processed_files)} files.")
        except Exception as e:
            print(f"Error reading existing CSV: {e}")
            
    # Filter out already processed files
    audio_files = [(fp, label) for fp, label in audio_files if os.path.basename(fp) not in processed_files]
    print(f"Remaining files to process: {len(audio_files)}")
    
    if not audio_files:
        print("All files already processed.")
        return

    batch_size = 4

    # === FILTER PARAMS — identical to fastapi_app_meta.py and extract_hard_val_mct.py ===
    from scipy.signal import butter, lfilter
    b_lp, a_lp = butter(4, 4000.0 / (16000.0 / 2.0), btype='low')
    MAX_SAMPLES = 10 * 16000  # 10-second truncation limit

    def extract_single_file_mct(file_path, label):
        """
        Extract 4096D MCT features for a single file.
        Runs two Wav2Vec2 passes: clean and lowpass-degraded, concatenates to 4096D.
        Returns a dict row or raises an exception.
        """
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        duration = len(y) / sr
        if duration < 1.0 or np.max(np.abs(y)) < 1e-4:
            raise ValueError(f"File too short or silent: {os.path.basename(file_path)}")

        y_clean = y[:MAX_SAMPLES].astype(np.float32)
        y_degraded = lfilter(b_lp, a_lp, y_clean).astype(np.float32)

        # lfilter is length-preserving — assert to catch any unexpected divergence
        assert len(y_clean) == len(y_degraded), \
            f"Length mismatch after lfilter: {len(y_clean)} vs {len(y_degraded)}"

        def wav2vec_pass(audio_array):
            inp = processor([audio_array], sampling_rate=16000, return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                out = model(**inp)
            h12 = out.hidden_states[12]
            mean = torch.mean(h12, dim=1).float().cpu().numpy()[0]  # 1024D
            std  = torch.std(h12,  dim=1).float().cpu().numpy()[0]  # 1024D
            return np.concatenate([mean, std])  # 2048D

        feats_clean    = wav2vec_pass(y_clean)    # 2048D
        feats_degraded = wav2vec_pass(y_degraded)  # 2048D
        feats_mct      = np.concatenate([feats_clean, feats_degraded])  # 4096D

        row = {'Filename': os.path.basename(file_path), 'Label': label}
        for k in range(4096):
            row[f'Deep_{k}'] = float(feats_mct[k])
        return row

    def process_batch_mct(batch_clean, batch_degraded, batch_meta):
        """
        Run two batched Wav2Vec2 passes (clean + degraded) for a prepared batch.
        Returns list of result rows. Both batch_clean and batch_degraded have same lengths.
        """
        def batched_wav2vec(audio_list):
            inputs = processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
            h12 = outputs.hidden_states[12]
            mean = torch.mean(h12, dim=1).float().cpu().numpy()  # (B, 1024)
            std  = torch.std(h12,  dim=1).float().cpu().numpy()  # (B, 1024)
            return np.concatenate([mean, std], axis=1)  # (B, 2048)

        feats_c = batched_wav2vec(batch_clean)    # (B, 2048)
        feats_d = batched_wav2vec(batch_degraded)  # (B, 2048)
        feats_mct = np.concatenate([feats_c, feats_d], axis=1)  # (B, 4096)

        results = []
        for j, meta in enumerate(batch_meta):
            row = dict(meta)
            for k in range(4096):
                row[f'Deep_{k}'] = float(feats_mct[j, k])
            results.append(row)
        return results

    # Process in batches
    for i in tqdm(range(0, len(audio_files), batch_size), desc="Extracting MCT Deep Features (4096D)"):
        batch_files = audio_files[i:i+batch_size]

        # --- Per-file audio loading phase ---
        batch_clean    = []
        batch_degraded = []
        batch_meta     = []

        for file_path, label in batch_files:
            try:
                y, sr = librosa.load(file_path, sr=16000, mono=True)
                duration = len(y) / sr
                if duration < 1.0 or np.max(np.abs(y)) < 1e-4:
                    continue

                y_clean    = y[:MAX_SAMPLES].astype(np.float32)
                y_degraded = lfilter(b_lp, a_lp, y_clean).astype(np.float32)

                assert len(y_clean) == len(y_degraded), \
                    f"Length mismatch: {len(y_clean)} vs {len(y_degraded)}"

                batch_clean.append(y_clean)
                batch_degraded.append(y_degraded)
                batch_meta.append({'Filename': os.path.basename(file_path), 'Label': label})
            except Exception as e:
                print(f"\n  Skipping corrupt/invalid file: {os.path.basename(file_path)} — {e}")
                continue

        if not batch_clean:
            continue

        # --- Batched GPU forward pass (with per-file fallback on failure) ---
        batch_results = []
        try:
            batch_results = process_batch_mct(batch_clean, batch_degraded, batch_meta)
        except Exception as batch_err:
            print(f"\n  Batch GPU error at index {i}: {batch_err}")
            print("  Falling back to per-file processing for this batch...")
            for file_path, label in batch_files:
                try:
                    row = extract_single_file_mct(file_path, label)
                    batch_results.append(row)
                except Exception as single_err:
                    print(f"    Per-file fallback failed: {os.path.basename(file_path)} — {single_err}")
                    continue

        if not batch_results:
            continue

        # --- Append to CSV ---
        df = pd.DataFrame(batch_results)
        deep_col_order = ['Filename', 'Label'] + [f'Deep_{k}' for k in range(4096)]
        df = df[deep_col_order]

        write_header = not os.path.exists(output_csv)
        df.to_csv(output_csv, mode='a', header=write_header, index=False)

        # Periodic VRAM clear
        if device.type == 'cuda' and (i // batch_size) % 10 == 0:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
