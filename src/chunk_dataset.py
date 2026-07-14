import os
import glob
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

def chunk_audio(input_file, output_dir, chunk_duration=10.0, overlap=0.0):
    try:
        y, sr = librosa.load(input_file, sr=16000)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return

    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap * sr)
    step_samples = chunk_samples - overlap_samples

    base_name = os.path.basename(input_file).replace('.mp3', '').replace('.wav', '')
    
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    # Process full chunks
    for start in range(0, len(y) - chunk_samples + 1, step_samples):
        chunk = y[start:start + chunk_samples]
        out_path = os.path.join(output_dir, f"{base_name}_chunk_{count:03d}.wav")
        sf.write(out_path, chunk, sr)
        count += 1
        
    # Process remaining tail if it's longer than 2 seconds
    tail_start = count * step_samples
    if len(y) - tail_start > 2 * sr:
        chunk = y[tail_start:]
        out_path = os.path.join(output_dir, f"{base_name}_chunk_{count:03d}.wav")
        sf.write(out_path, chunk, sr)
        count += 1

    print(f"Created {count} chunks for {base_name}")

if __name__ == '__main__':
    real_dir = os.path.join("data", "1_genuine", "English Real")
    fake_dir = os.path.join("data", "2_synthetic", "English Fake (Elevenlabs)")
    
    real_out = os.path.join("data", "1_genuine", "english_real_chunked")
    fake_out = os.path.join("data", "2_synthetic", "english_fake_chunked")
    
    print("Chunking Real English...")
    for f in glob.glob(os.path.join(real_dir, '*.*')):
        chunk_audio(f, real_out)
        
    print("Chunking Fake English...")
    for f in glob.glob(os.path.join(fake_dir, '*.*')):
        chunk_audio(f, fake_out)
        
    print("Done!")
