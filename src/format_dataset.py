import os
import glob
import librosa
import soundfile as sf
import traceback

def format_audio_file(input_path, output_path, target_sr=16000):
    """Loads an audio file of any format, resamples it to target_sr mono, and saves as PCM_16 WAV."""
    try:
        # librosa.load will automatically resample to 16000Hz and mix down to mono.
        # On Windows, it leverages Windows Media Foundation to read .mp3, .m4a, and WhatsApp .ogg/.opus.
        y, sr = librosa.load(input_path, sr=target_sr, mono=True)
        
        # Ensure the parent directory of the output file exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as 16-bit PCM WAV
        sf.write(output_path, y, target_sr, subtype='PCM_16')
        return True
    except Exception as e:
        print(f"Failed to process {os.path.basename(input_path)}: {e}")
        traceback.print_exc()
        return False

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_real_dir = os.path.join(base_dir, 'data', 'raw_real')
    raw_fake_dir = os.path.join(base_dir, 'data', 'raw_fake')
    
    target_real_dir = os.path.join(base_dir, 'data', 'real')
    target_fake_dir = os.path.join(base_dir, 'data', 'fake')
    
    print("=== VoiceGuard Dataset Formatter ===")
    
    # Create the raw folders if they don't exist yet so the user knows where to put files
    os.makedirs(raw_real_dir, exist_ok=True)
    os.makedirs(raw_fake_dir, exist_ok=True)
    
    # Supported extensions
    extensions = ['*.wav', '*.mp3', '*.m4a', '*.ogg', '*.opus', '*.mp4', '*.3gp']
    
    # Process Real files
    real_files = []
    for ext in extensions:
        real_files.extend(glob.glob(os.path.join(raw_real_dir, ext)))
        real_files.extend(glob.glob(os.path.join(raw_real_dir, ext.upper())))
    
    print(f"\nFound {len(real_files)} raw files in 'data/raw_real/'. Processing...")
    real_success = 0
    for idx, file_path in enumerate(real_files):
        filename = os.path.basename(file_path)
        name, _ = os.path.splitext(filename)
        output_name = f"{name}_formatted.wav"
        output_path = os.path.join(target_real_dir, output_name)
        
        print(f"[{idx+1}/{len(real_files)}] Formatting {filename} -> {output_name}")
        if format_audio_file(file_path, output_path):
            real_success += 1

    # Process Fake / Replay files
    fake_files = []
    for ext in extensions:
        fake_files.extend(glob.glob(os.path.join(raw_fake_dir, ext)))
        fake_files.extend(glob.glob(os.path.join(raw_fake_dir, ext.upper())))
        
    print(f"\nFound {len(fake_files)} raw files in 'data/raw_fake/'. Processing...")
    fake_success = 0
    for idx, file_path in enumerate(fake_files):
        filename = os.path.basename(file_path)
        name, _ = os.path.splitext(filename)
        output_name = f"{name}_formatted.wav"
        output_path = os.path.join(target_fake_dir, output_name)
        
        print(f"[{idx+1}/{len(fake_files)}] Formatting {filename} -> {output_name}")
        if format_audio_file(file_path, output_path):
            fake_success += 1
            
    print("\n=== Processing Summary ===")
    print(f"Real Files: Successful {real_success}/{len(real_files)} -> Saved in data/real/")
    print(f"Fake Files: Successful {fake_success}/{len(fake_files)} -> Saved in data/fake/")
    print("\nIf you have added new files, run this script again to format them.")

if __name__ == "__main__":
    main()
