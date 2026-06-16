import os
import json
import urllib.request
import tarfile
import zipfile
import shutil
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'data', 'temp_download')
RAW_REAL_DIR = os.path.join(BASE_DIR, 'data', 'raw_real', 'ljspeech')
RAW_FAKE_DIR = os.path.join(BASE_DIR, 'data', 'raw_fake', 'wavefake')

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RAW_REAL_DIR, exist_ok=True)
os.makedirs(RAW_FAKE_DIR, exist_ok=True)

import subprocess

def download_file(url, dest_path):
    print(f"Starting robust download of {url} to {dest_path} using curl...")
    # -C - tells curl to automatically resume if the file exists
    # -L tells curl to follow redirects (Zenodo uses them)
    # --retry 10 to retry on connection drops
    cmd = ["curl.exe", "-C", "-", "-L", "--retry", "10", "--retry-delay", "3", "-o", dest_path, url]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Finished downloading {dest_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {dest_path}: {e}")
        raise e

def get_wavefake_url():
    api_url = "https://zenodo.org/api/records/5642694"
    req = urllib.request.Request(api_url, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        for f in data.get('files', []):
            if f.get('key') == 'generated_audio.zip':
                return f.get('links', {}).get('self')
    return None

def process_ljspeech():
    ljspeech_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    ljspeech_tar = os.path.join(TEMP_DIR, 'LJSpeech-1.1.tar.bz2')
    
    if not os.path.exists(os.path.join(RAW_REAL_DIR, "LJ001-0001.wav")):
        download_file(ljspeech_url, ljspeech_tar)
        
        print("Extracting LJSpeech...")
        with tarfile.open(ljspeech_tar, "r:bz2") as tar:
            members = [m for m in tar.getmembers() if m.name.startswith("LJSpeech-1.1/wavs/") and m.isfile()]
            for i, member in enumerate(members):
                member.name = member.name.replace("LJSpeech-1.1/wavs/", "")
                tar.extract(member, RAW_REAL_DIR)
                if i > 0 and i % 5000 == 0:
                    print(f"  Extracted {i} LJSpeech files...")
        print("Finished extracting LJSpeech.")
        
        print(f"Deleting {ljspeech_tar} to save space.")
        os.remove(ljspeech_tar)
    else:
        print("LJSpeech already extracted.")

def process_wavefake():
    wavefake_zip = os.path.join(TEMP_DIR, 'generated_audio.zip')
    
    if not os.path.exists(os.path.join(RAW_FAKE_DIR, "ljspeech_melgan")):
        url = get_wavefake_url()
        download_file(url, wavefake_zip)
            
        print("Extracting WaveFake English sets...")
        with zipfile.ZipFile(wavefake_zip, 'r') as zf:
            namelist = zf.namelist()
            english_files = [n for n in namelist if n.startswith("generated_audio/ljspeech_") or n.startswith("generated_audio/common_voices_")]
            
            for i, name in enumerate(english_files):
                if name.endswith("/"):
                    continue
                target_path = os.path.join(RAW_FAKE_DIR, name.replace("generated_audio/", ""))
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                with zf.open(name) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                    
                if i > 0 and i % 10000 == 0:
                    print(f"  Extracted {i} WaveFake files...")
        print("Finished extracting WaveFake English fakes.")
        
        print(f"Deleting {wavefake_zip} to save space.")
        os.remove(wavefake_zip)
    else:
        print("WaveFake already extracted.")

if __name__ == '__main__':
    print("Starting Phase 2 Dataset Download and Extraction...")
    process_ljspeech()
    process_wavefake()
    print("All tasks completed successfully!")
