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
    
    # Search for wav and flac files
    for ext in ['*.wav', '*.flac']:
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
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", use_safetensors=True).to(device)
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

    batch_size = 8
    
    # Process in batches
    for i in tqdm(range(0, len(audio_files), batch_size), desc="Extracting Deep Features"):
        batch_files = audio_files[i:i+batch_size]
        
        batch_audio = []
        batch_meta = []
        
        for file_path, label in batch_files:
            try:
                y, sr = librosa.load(file_path, sr=16000, mono=True)
                duration = librosa.get_duration(y=y, sr=sr)
                if duration < 1.0 or np.max(np.abs(y)) < 1e-4:
                    continue
                
                # Simulate phone codec degradation
                y = simulate_phone_codec(y, sr=16000)
                batch_audio.append(y)
                batch_meta.append({'Filename': os.path.basename(file_path), 'Label': label})
            except Exception as e:
                # skip corrupt files silently during batch
                continue
                
        if not batch_audio:
            continue
            
        try:
            inputs = processor(batch_audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            hidden_states = outputs.last_hidden_state
            pooled_features = torch.mean(hidden_states, dim=1).cpu().numpy()
            
            batch_results = []
            for j in range(len(batch_meta)):
                feats = batch_meta[j]
                for k in range(pooled_features.shape[1]):
                    feats[f'Deep_{k}'] = float(pooled_features[j, k])
                batch_results.append(feats)
                
            # Append to CSV
            df = pd.DataFrame(batch_results)
            cols = ['Filename', 'Label'] + [c for c in df.columns if c not in ['Filename', 'Label']]
            df = df[cols]
            
            df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
            
        except Exception as e:
            print(f"\nError processing batch starting at index {i}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
