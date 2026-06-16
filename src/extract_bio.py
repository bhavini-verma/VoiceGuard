import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import numpy as np
import pandas as pd
import librosa
import traceback
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def simulate_phone_codec(y, sr=16000):
    """Simulate GSM/AMR phone codec degradation by downsampling to 8kHz and back.

    Real bank calls arrive compressed at 8kHz through GSM or AMR codecs.
    This smears fine timing variations (especially Jitter) that exist in clean 16kHz audio.
    By simulating this degradation during training, we ensure the model learns
    features that are robust to codec compression.
    """
    y_8k = librosa.resample(y, orig_sr=sr, target_sr=8000)
    y_degraded = librosa.resample(y_8k, orig_sr=8000, target_sr=sr)
    return y_degraded

def augment_noise(y, noise_level=0.005):
    """Add random Gaussian noise to simulate noisy call environments.

    This function is defined here but should ONLY be called from the training
    script on training-split clips (not validation/test). Applying noise to
    ~50% of training clips improves model robustness to background noise on
    real bank calls.
    """
    noise = np.random.randn(len(y)) * noise_level
    return y + noise

def compute_reverberation_proxy(y, sr, hop_length=512):
    """Estimate reverberation characteristics from the energy envelope.

    Returns two values:
    - peak_to_mean: Ratio of peak RMS to mean RMS. Lower values suggest
      reverberant environments where energy is spread more evenly.
    - decay_autocorr: Normalized autocorrelation of the energy envelope at
      ~50ms lag. High values indicate sustained/reverberant energy.
    """
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if len(rms) < 2 or np.max(rms) == 0:
        return 0.0, 0.0
    # Peak-to-mean energy ratio
    peak_to_mean = float(np.max(rms) / np.mean(rms))
    # Autocorrelation of energy envelope at ~50ms lag
    rms_norm = (rms - np.mean(rms)) / (np.std(rms) + 1e-10)
    autocorr = np.correlate(rms_norm, rms_norm, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # positive lags only
    autocorr = autocorr / (autocorr[0] + 1e-10)  # normalize
    lag_samples = max(1, int(0.05 * sr / hop_length))
    decay_autocorr = float(autocorr[min(lag_samples, len(autocorr) - 1)])
    return peak_to_mean, decay_autocorr

def compute_jitter(f0):
    """Calculate Jitter: cycle-to-cycle variation of fundamental frequency"""
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) < 2:
        return 0.0, 0.0
    f0_diff = np.abs(np.diff(f0_clean))
    mean_f0 = np.mean(f0_clean)
    jitter_mean = np.mean(f0_diff) / mean_f0 if mean_f0 > 0 else 0
    jitter_std = np.std(f0_diff) / mean_f0 if mean_f0 > 0 else 0
    return jitter_mean, jitter_std

def compute_shimmer(y, hop_length=512):
    """Calculate Shimmer: cycle-to-cycle variation of amplitude"""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if len(rms) < 2:
        return 0.0, 0.0
    rms_diff = np.abs(np.diff(rms))
    mean_rms = np.mean(rms)
    shimmer_mean = np.mean(rms_diff) / mean_rms if mean_rms > 0 else 0
    shimmer_std = np.std(rms_diff) / mean_rms if mean_rms > 0 else 0
    return shimmer_mean, shimmer_std

def compute_clipping_rate(y):
    """Calculate percentage of samples that are near the maximum possible value"""
    # Assuming audio is normalized or float32 in [-1.0, 1.0]
    threshold = 0.99
    clipping_samples = np.sum(np.abs(y) >= threshold)
    return clipping_samples / len(y) if len(y) > 0 else 0.0

def compute_sub_300hz_noise(y, sr):
    """Calculate energy ratio below 300Hz"""
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    low_freq_idx = np.where(freqs < 300)[0]
    
    total_energy = np.sum(S)
    if total_energy == 0:
        return 0.0
        
    low_energy = np.sum(S[low_freq_idx, :])
    return low_energy / total_energy

def extract_bio_features(audio_path):
    """Extract biological and environmental features from an audio file."""
    # 1. Quality Checks
    try:
        # Load audio at 16kHz mono
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

    duration = librosa.get_duration(y=y, sr=sr)
    if duration < 1.0:
        print(f"Skipping {audio_path}: Duration ({duration:.2f}s) is less than 1 second.")
        return None

    if np.max(np.abs(y)) < 1e-4:
        print(f"Skipping {audio_path}: Audio is completely silent.")
        return None

    # Simulate phone codec degradation (8kHz GSM/AMR compression)
    y = simulate_phone_codec(y, sr=16000)

    hop_length = 512
    features = {}
    
    # 2. Pitch (f0) using fast YIN algorithm + RMS voicing check
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    voiced_frames = rms > 0.1 * np.mean(rms)
    min_len = min(len(f0), len(voiced_frames))
    f0_clean = f0[:min_len][voiced_frames[:min_len]]
    
    if len(f0_clean) > 0:
        features['Pitch_Mean'] = float(np.mean(f0_clean))
        features['Pitch_Std'] = float(np.std(f0_clean))
        features['Pitch_Min'] = float(np.min(f0_clean))
        features['Pitch_Max'] = float(np.max(f0_clean))
    else:
        features['Pitch_Mean'], features['Pitch_Std'], features['Pitch_Min'], features['Pitch_Max'] = 0.0, 0.0, 0.0, 0.0

    # 3. Jitter and Shimmer
    j_mean, j_std = compute_jitter(f0_clean)
    features['Jitter_Mean'] = float(j_mean)
    features['Jitter_Std'] = float(j_std)

    s_mean, s_std = compute_shimmer(y, hop_length)
    features['Shimmer_Mean'] = float(s_mean)
    features['Shimmer_Std'] = float(s_std)
    
    # 4. HNR (Harmonics-to-Noise Ratio)
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = np.sum(y_harm**2)
    perc_energy = np.sum(y_perc**2)
    features['HNR_Mean'] = float(10 * np.log10(harm_energy / perc_energy)) if perc_energy > 0 else 0.0

    # 5. MFCCs (40 coefficients for finer spectral texture detail across entire spectrum)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=hop_length)
    for i in range(40):
        features[f'MFCC_{i+1}_Mean'] = float(np.mean(mfccs[i]))
        features[f'MFCC_{i+1}_Std'] = float(np.std(mfccs[i]))
        features[f'MFCC_{i+1}_Min'] = float(np.min(mfccs[i]))
        features[f'MFCC_{i+1}_Max'] = float(np.max(mfccs[i]))

    # 5b. Delta MFCCs (rate of spectral change — captures phoneme transition smoothness)
    delta_mfccs = librosa.feature.delta(mfccs)
    for i in range(40):
        features[f'Delta_MFCC_{i+1}_Mean'] = float(np.mean(delta_mfccs[i]))
        features[f'Delta_MFCC_{i+1}_Std'] = float(np.std(delta_mfccs[i]))

    # 5c. Delta-Delta MFCCs (acceleration of spectral change)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    for i in range(40):
        features[f'Delta2_MFCC_{i+1}_Mean'] = float(np.mean(delta2_mfccs[i]))
        features[f'Delta2_MFCC_{i+1}_Std'] = float(np.std(delta2_mfccs[i]))

    # 6. Spectral Roll-Off (0.95 to target high-frequency vocoder artifact region)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95, hop_length=hop_length)[0]
    features['RollOff_Mean'] = float(np.mean(rolloff))
    features['RollOff_Std'] = float(np.std(rolloff))
    features['RollOff_Min'] = float(np.min(rolloff))
    features['RollOff_Max'] = float(np.max(rolloff))

    # 7. Spectral Flatness & Contrast
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    features['Flatness_Mean'] = float(np.mean(flatness))
    features['Flatness_Std'] = float(np.std(flatness))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    features['Contrast_Mean'] = float(np.mean(contrast))
    features['Contrast_Std'] = float(np.std(contrast))

    # 8. ZCR
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
    features['ZCR_Mean'] = float(np.mean(zcr))
    features['ZCR_Std'] = float(np.std(zcr))
    
    # 9. Hardware / Room artifacts
    features['Clipping_Rate'] = float(compute_clipping_rate(y))
    features['Sub300Hz_Noise'] = float(compute_sub_300hz_noise(y, sr))

    # 10. Reverberation proxy
    reverb_peak_to_mean, reverb_decay = compute_reverberation_proxy(y, sr, hop_length)
    features['Reverb_PeakToMean'] = reverb_peak_to_mean
    features['Reverb_Decay'] = reverb_decay

    return features

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

def process_single_file(args):
    file_path, label = args
    try:
        feats = extract_bio_features(file_path)
        if feats is not None:
            feats['Filename'] = os.path.basename(file_path)
            feats['Label'] = label
            return feats
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return None

def main():
    print("Starting Biological Engine Feature Extraction (Parallel)...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    features_dir = os.path.join(base_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)
    
    audio_files = get_audio_files(data_dir)
    
    if not audio_files:
        print("No audio files found! Please place some audio files in the data subdirectories.")
        return
        
    print(f"Found {len(audio_files)} audio files to process.")
    
    output_csv = os.path.join(features_dir, 'bio_features.csv')
    
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
        
    all_features = []
    max_workers = max(1, os.cpu_count() - 2)
    print(f"Using {max_workers} parallel workers on {os.cpu_count()} CPU cores...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, item): item for item in audio_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Biological Features (Parallel)"):
            result = future.result()
            if result is not None:
                all_features.append(result)
                
            # Write batches of 100 to prevent losing progress if interrupted
            if len(all_features) >= 100:
                df = pd.DataFrame(all_features)
                cols = ['Filename', 'Label'] + [c for c in df.columns if c not in ['Filename', 'Label']]
                df = df[cols]
                df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
                all_features = []
                
    # Write any remaining features
    if all_features:
        df = pd.DataFrame(all_features)
        cols = ['Filename', 'Label'] + [c for c in df.columns if c not in ['Filename', 'Label']]
        df = df[cols]
        df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
        
    # Print summary of the final file
    if os.path.exists(output_csv):
        df_final = pd.read_csv(output_csv)
        print(f"\nExtraction completed successfully.")
        print(f"Saved to: {output_csv}")
        print(f"CSV Shape: {df_final.shape}")
        print(f"Label Distribution:\n{df_final['Label'].value_counts().to_string()}")
        print(f"NaN Count: {df_final.isna().sum().sum()}")

if __name__ == "__main__":
    main()

