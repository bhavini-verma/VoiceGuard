import os
import io
import subprocess
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

def download_file(url, dest_dir, dest_filename):
    aria2c_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'aria2', 'aria2-1.36.0-win-64bit-build1', 'aria2c.exe')
    dest_path = os.path.join(dest_dir, dest_filename)
    
    # If the file is already downloaded, check if it's there
    if os.path.exists(dest_path):
        print(f"{dest_filename} already exists, skipping download.")
        return dest_path
        
    print(f"Downloading {url} to {dest_path} using aria2c...")
    cmd = [
        aria2c_path,
        "-c",                   # Continue partial download
        "-x", "16",             # 16 connections per server
        "-s", "16",             # 16 concurrent splits
        "--max-connection-per-server=16",
        "--retry-wait=3",
        "--max-tries=100",
        "-d", dest_dir,         # Directory to save
        "-o", dest_filename,    # Filename
        url
    ]
    
    max_retries = 10
    retries = 0
    while retries < max_retries:
        try:
            subprocess.run(cmd, check=True)
            print(f"Finished downloading {dest_filename}")
            return dest_path
        except subprocess.CalledProcessError as e:
            print(f"Download dropped (Error {e.returncode}). Retrying... ({retries}/{max_retries})")
            retries += 1
            import time
            time.sleep(5)
    raise Exception(f"Failed to download {dest_filename} after {max_retries} attempts.")

def extract_and_save_parquet(parquet_paths, target_dir, max_files=2000, prefix="hindi"):
    os.makedirs(target_dir, exist_ok=True)
    
    total_extracted = 0
    
    for parquet_path in parquet_paths:
        if total_extracted >= max_files:
            break
            
        print(f"Reading local parquet file: {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} rows. Extracting audio...")
        
        # Use tqdm to show extraction progress
        remaining_slots = max_files - total_extracted
        for idx, row in tqdm(df.iterrows(), total=min(len(df), remaining_slots)):
            if total_extracted >= max_files:
                break
                
            try:
                audio_data = row['audio']
                audio_bytes = audio_data['bytes']
                
                # Decode using soundfile
                y, sr = sf.read(io.BytesIO(audio_bytes))
                
                # Resample to 16000Hz immediately to save space and standardise
                if sr != 16000:
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                    sr = 16000
                    
                out_path = os.path.join(target_dir, f"{prefix}_{total_extracted:05d}.wav")
                sf.write(out_path, y, sr)
                total_extracted += 1
            except Exception as e:
                print(f"Failed to process row {idx} in {os.path.basename(parquet_path)}: {e}")
                continue
                
    print(f"Successfully saved {total_extracted} files to {target_dir}")
    return total_extracted

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(base_dir, 'data', 'temp_download')
    real_dir = os.path.join(base_dir, 'data', 'raw_real', 'hindi_common_voice')
    fake_dir = os.path.join(base_dir, 'data', 'raw_fake', 'indicsynth_hindi')
    
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. Download Google FLEURS Hindi (REAL) Validation & Test splits (approx 690 MB total)
    fleurs_val_url = "https://huggingface.co/api/datasets/google/fleurs/parquet/hi_in/validation/0.parquet"
    fleurs_test_url = "https://huggingface.co/api/datasets/google/fleurs/parquet/hi_in/test/0.parquet"
    
    print("=== Downloading FLEURS Hindi Validation split (~199MB) ===")
    val_parquet = download_file(fleurs_val_url, temp_dir, "fleurs_hindi_val.parquet")
    
    print("\n=== Downloading FLEURS Hindi Test split (~495MB) ===")
    test_parquet = download_file(fleurs_test_url, temp_dir, "fleurs_hindi_test.parquet")
    
    # 2. Extract FLEURS Real Hindi clips (Validation + Test)
    print("\n=== Extracting FLEURS Real Hindi ===")
    real_count = extract_and_save_parquet(
        [val_parquet, test_parquet], 
        real_dir, 
        max_files=1500, # We grab up to 1500 clips
        prefix="hindi_real"
    )
    
    # 3. Download IndicSynth Hindi (FAKE) Parquet file (~450MB)
    indicsynth_url = "https://huggingface.co/api/datasets/vdivyasharma/IndicSynth/parquet/Hindi/train/0.parquet"
    print("\n=== Downloading IndicSynth Hindi (~450MB) ===")
    indicsynth_parquet = download_file(indicsynth_url, temp_dir, "indicsynth_hindi_train.parquet")
    
    # 4. Extract IndicSynth Fake Hindi clips (Match the real count to maintain balance!)
    print(f"\n=== Extracting IndicSynth Fake Hindi (matching count: {real_count}) ===")
    extract_and_save_parquet(
        [indicsynth_parquet], 
        fake_dir, 
        max_files=real_count, 
        prefix="hindi_fake"
    )
    
    # 5. Clean up downloaded parquet files to save disk space
    print("\n=== Cleaning Up Parquet Archives ===")
    try:
        os.remove(val_parquet)
        os.remove(test_parquet)
        os.remove(indicsynth_parquet)
        print("Cleaned up temporary parquet files.")
    except Exception as e:
        print(f"Failed to delete parquet files: {e}")

if __name__ == "__main__":
    main()


