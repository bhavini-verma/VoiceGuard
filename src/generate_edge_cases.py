import os
import io
import glob
import numpy as np
import soundfile as sf
import scipy.signal as signal
from tqdm import tqdm

def add_noise(y, snr_db):
    sig_power = np.mean(y ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise

def simulate_weak_mic(y, sr):
    # Bandpass filter simulating cheap microphone capsule (e.g., 200 Hz to 4000 Hz)
    nyq = 0.5 * sr
    low = 200 / nyq
    high = 4000 / nyq
    b, a = signal.butter(4, [low, high], btype='band')
    y_filtered = signal.lfilter(b, a, y)
    
    # Boost gain to cause heavy clipping (saturation)
    y_clipped = y_filtered * 2.5
    y_clipped = np.clip(y_clipped, -0.98, 0.98)
    return y_clipped

def simulate_hoarse_sick(y, sr):
    # Hoarseness is characterized by high jitter (frequency instability) and shimmer (amplitude instability)
    # Plus high-frequency turbulent noise (breathiness)
    
    # 1. Shimmer (amplitude modulation with random noise)
    mod_amp = 1.0 + 0.15 * np.random.normal(0, 1, len(y))
    y_shimmer = y * mod_amp
    
    # 2. Add turbulent breathy noise (high-passed noise mixed into the signal)
    nyq = 0.5 * sr
    b, a = signal.butter(4, 3000 / nyq, btype='high')
    noise = np.random.normal(0, 0.05, len(y))
    breathy_noise = signal.lfilter(b, a, noise)
    
    # Mix breathy noise
    y_sick = y_shimmer + breathy_noise * np.std(y_shimmer) * 0.4
    return np.clip(y_sick, -0.99, 0.99)

def simulate_elderly_voice(y, sr):
    # Elderly voice exhibits tremor (6Hz LFO amplitude modulation) and low tension (lowpass filter)
    t = np.arange(len(y)) / sr
    
    # 1. Tremor: 6Hz amplitude modulation
    tremor = 1.0 + 0.25 * np.sin(2 * np.pi * 6.0 * t)
    y_tremor = y * tremor
    
    # 2. Lowpass filter (loss of high-frequency energy due to vocal fold thinning)
    nyq = 0.5 * sr
    b, a = signal.butter(4, 1800 / nyq, btype='low')
    y_elderly = signal.lfilter(b, a, y_tremor)
    
    return np.clip(y_elderly, -0.99, 0.99)

def simulate_replay(y, sr, device_type):
    t = np.arange(len(y)) / sr
    
    # 1. Speaker frequency response curves (bandpass filters)
    nyq = 0.5 * sr
    if device_type == 'mobile':
        # Mobile speaker: restricted bass, peak mids (400 Hz to 3500 Hz)
        b, a = signal.butter(4, [400 / nyq, 3500 / nyq], btype='band')
        y_spk = signal.lfilter(b, a, y)
        # Add a faint 50Hz electrical hum
        hum = 0.015 * np.sin(2 * np.pi * 50.0 * t)
        y_spk += hum
    elif device_type == 'laptop':
        # Laptop speaker: thin bass, flat mids (250 Hz to 6000 Hz)
        b, a = signal.butter(4, [250 / nyq, 6000 / nyq], btype='band')
        y_spk = signal.lfilter(b, a, y)
    else: # bluetooth
        # Bluetooth speaker: better bass, flat treble (100 Hz to 10000 Hz)
        b, a = signal.butter(4, [100 / nyq, 10000 / nyq], btype='band')
        y_spk = signal.lfilter(b, a, y)

    # 2. Room reverberation (simple feedforward/feedback delay comb filter)
    # Delay of 35ms (0.035s) and 60ms (0.06s) representing room reflection boundaries
    delay_samples_1 = int(0.035 * sr)
    delay_samples_2 = int(0.060 * sr)
    
    y_reverb = np.copy(y_spk)
    # Feed reflections back into the signal
    for i in range(delay_samples_1, len(y_reverb)):
        y_reverb[i] += 0.35 * y_reverb[i - delay_samples_1]
    for i in range(delay_samples_2, len(y_reverb)):
        y_reverb[i] += 0.20 * y_reverb[i - delay_samples_2]
        
    # Standardise level and clip
    y_reverb = y_reverb / np.max(np.abs(y_reverb) + 1e-6) * 0.90
    return y_reverb

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ljspeech_dir = os.path.join(base_dir, 'data', 'raw_real', 'ljspeech')
    hard_neg_dir = os.path.join(base_dir, 'data', 'hard_negatives')
    replay_dir = os.path.join(base_dir, 'data', 'real_replays')
    
    os.makedirs(hard_neg_dir, exist_ok=True)
    os.makedirs(replay_dir, exist_ok=True)
    
    # Gather clean files to use as templates
    clean_files = sorted(glob.glob(os.path.join(ljspeech_dir, "*.wav")))
    if len(clean_files) == 0:
        print("Error: No clean LJSpeech wav files found in data/raw_real/ljspeech.")
        return
        
    print(f"Found {len(clean_files)} clean files. Generating edge cases...")
    
    # Target Counts:
    # Hard Negatives: 15 per category (total 60)
    # Replays: 25 per device (total 75)
    
    # 1. Generate Hard Negatives
    print("\n=== Generating Hard Negatives ===")
    categories = ['noisy_line', 'weak_mic', 'hoarse_sick', 'elderly']
    samples_per_cat = 15
    
    src_idx = 0
    for cat in categories:
        print(f"Generating {cat} hard negatives...")
        for i in tqdm(range(samples_per_cat)):
            src_file = clean_files[src_idx % len(clean_files)]
            src_idx += 1
            
            y, sr = sf.read(src_file)
            
            if cat == 'noisy_line':
                y_mod = add_noise(y, snr_db=10.0) # SNR 10dB
            elif cat == 'weak_mic':
                y_mod = simulate_weak_mic(y, sr)
            elif cat == 'hoarse_sick':
                y_mod = simulate_hoarse_sick(y, sr)
            elif cat == 'elderly':
                y_mod = simulate_elderly_voice(y, sr)
                
            out_path = os.path.join(hard_neg_dir, f"{cat}_{i:03d}.wav")
            sf.write(out_path, y_mod, sr)
            
    # 2. Generate Replay Attacks
    print("\n=== Generating Replay Attacks ===")
    devices = ['mobile', 'laptop', 'bluetooth']
    samples_per_device = 25
    
    for dev in devices:
        print(f"Generating {dev} replays...")
        for i in tqdm(range(samples_per_device)):
            src_file = clean_files[src_idx % len(clean_files)]
            src_idx += 1
            
            y, sr = sf.read(src_file)
            y_replay = simulate_replay(y, sr, dev)
            
            out_path = os.path.join(replay_dir, f"replay_{dev}_{i:03d}.wav")
            sf.write(out_path, y_replay, sr)
            
    print("\nEdge cases successfully generated!")
    print(f"Hard negatives count: {len(glob.glob(os.path.join(hard_neg_dir, '*.wav')))}")
    print(f"Replay attacks count: {len(glob.glob(os.path.join(replay_dir, '*.wav')))}")

if __name__ == "__main__":
    main()
