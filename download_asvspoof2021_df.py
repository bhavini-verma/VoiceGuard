import os
import subprocess
import tarfile
import shutil
import urllib.request
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'data', 'temp_download')
RAW_REAL_DIR = os.path.join(BASE_DIR, 'data', 'raw_real', 'asvspoof_2021_df')
RAW_FAKE_DIR = os.path.join(BASE_DIR, 'data', 'raw_fake', 'asvspoof_2021_df')

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RAW_REAL_DIR, exist_ok=True)
os.makedirs(RAW_FAKE_DIR, exist_ok=True)

def download_file(url, dest_path):
    print(f"Starting turbo multi-threaded download of {url} to {dest_path} using aria2c...")
    aria2c_path = os.path.join(BASE_DIR, 'aria2', 'aria2-1.36.0-win-64bit-build1', 'aria2c.exe')
    dest_dir = os.path.dirname(dest_path)
    dest_filename = os.path.basename(dest_path)
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
    max_retries = 100
    retries = 0
    while retries < max_retries:
        try:
            subprocess.run(cmd, check=True)
            print(f"Finished downloading {dest_path}")
            return
        except subprocess.CalledProcessError as e:
            print(f"Curl connection dropped (Error {e.returncode}). Restarting auto-resume... ({retries}/{max_retries})")
            retries += 1
            import time
            time.sleep(5)
    raise Exception(f"Failed to download {dest_path} after {max_retries} attempts.")

def get_zenodo_files():
    api_url = "https://zenodo.org/api/records/4835108"
    req = urllib.request.Request(api_url)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
    return [(f['key'], f['links']['self']) for f in data['files'] if f['key'].endswith('.tar.gz')]

def process_asvspoof_2021():
    keys_url = "https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz"
    keys_tar = os.path.join(TEMP_DIR, 'DF-keys-full.tar.gz')
    
    # Download keys
    if not os.path.exists(keys_tar):
        download_file(keys_url, keys_tar)
        
    print("Parsing ASVSpoof 2021 DF protocols...")
    file_labels = {}
    
    with tarfile.open(keys_tar, "r:gz") as tar:
        for member in tar.getmembers():
            if 'CM/trial_metadata.txt' in member.name:
                f = tar.extractfile(member)
                if f is not None:
                    for line in f:
                        parts = line.decode('utf-8').strip().split()
                        if len(parts) >= 6:
                            # ASVspoof 2021 DF CM keys format:
                            # speaker_id filename - - - label
                            filename = parts[1]
                            label = parts[5] # 'bonafide' or 'spoof'
                            file_labels[filename] = label
                break
                
    print(f"Loaded {len(file_labels)} label mappings.")
    
    # Download and extract each part from Zenodo
    files = get_zenodo_files()
    for filename, url in files:
        done_marker = os.path.join(TEMP_DIR, f"{filename}.done")
        if os.path.exists(done_marker):
            print(f"Skipping {filename}, already fully extracted.")
            continue
            
        dest_path = os.path.join(TEMP_DIR, filename)
        download_file(url, dest_path)
        
        print(f"Extracting {filename}...")
        with tarfile.open(dest_path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith('.flac')]
            for i, member in enumerate(members):
                flac_name = os.path.basename(member.name).replace('.flac', '')
                label = file_labels.get(flac_name)
                
                if label == 'bonafide':
                    target_dir = RAW_REAL_DIR
                elif label == 'spoof':
                    target_dir = RAW_FAKE_DIR
                else:
                    continue
                    
                target_path = os.path.join(target_dir, os.path.basename(member.name))
                if not os.path.exists(target_path):
                    source = tar.extractfile(member)
                    if source:
                        with open(target_path, "wb") as target:
                            shutil.copyfileobj(source, target)
                            
                if i > 0 and i % 5000 == 0:
                    print(f"  Extracted {i} files from {filename}...")
        
        print(f"Deleting {dest_path} to save space.")
        os.remove(dest_path)
        # Mark as done
        with open(done_marker, 'w') as f:
            f.write('done')
        
    print("Deleting keys file.")
    os.remove(keys_tar)

if __name__ == '__main__':
    print("Starting ASVSpoof 2021 DF Dataset Download and Extraction...")
    process_asvspoof_2021()
    print("All tasks completed successfully!")
