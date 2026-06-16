import os
import numpy as np
import soundfile as sf

def generate_tone(freq, duration, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), False)
    # Adding some harmonics to make it pitch-detectable by pyin
    tone = np.sin(freq * t * 2 * np.pi) + 0.5 * np.sin(freq * 2 * t * 2 * np.pi)
    return tone

def generate_noise(duration, sr=16000):
    return np.random.randn(int(sr * duration))

def generate_dummy_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    demo_dir = os.path.join(base_dir, 'DEMONSTRATION')
    os.makedirs(demo_dir, exist_ok=True)
    
    sr = 16000
    print(f"Generating dummy audio in {demo_dir}...")
    
    # 1. Valid File: 2 seconds of a 440Hz tone (should pass)
    y1 = generate_tone(440, 2.0, sr) * 0.5
    sf.write(os.path.join(demo_dir, 'dummy_valid_tone.wav'), y1, sr)
    print("Created dummy_valid_tone.wav (2.0s tone)")
    
    # 2. Too Short File: 0.5 seconds of a 440Hz tone (should be rejected)
    y2 = generate_tone(440, 0.5, sr) * 0.5
    sf.write(os.path.join(demo_dir, 'dummy_short_tone.wav'), y2, sr)
    print("Created dummy_short_tone.wav (0.5s tone)")
    
    # 3. Silent File: 2 seconds of silence (should be rejected)
    y3 = np.zeros(int(sr * 2.0))
    sf.write(os.path.join(demo_dir, 'dummy_silent.wav'), y3, sr)
    print("Created dummy_silent.wav (2.0s silence)")
    
    # 4. Noisy File: 2 seconds of white noise (valid length, might fail pitch but should pass quality checks)
    y4 = generate_noise(2.0, sr) * 0.1
    sf.write(os.path.join(demo_dir, 'dummy_noise.wav'), y4, sr)
    print("Created dummy_noise.wav (2.0s noise)")

if __name__ == "__main__":
    generate_dummy_data()
