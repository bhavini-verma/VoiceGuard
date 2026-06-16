import os
import subprocess
import zipfile
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'data', 'temp_download')
RAW_REAL_DIR = os.path.join(BASE_DIR, 'data', 'raw_real', 'asvspoof')
RAW_FAKE_DIR = os.path.join(BASE_DIR, 'data', 'raw_fake', 'asvspoof')

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RAW_REAL_DIR, exist_ok=True)
os.makedirs(RAW_FAKE_DIR, exist_ok=True)

def download_file(url, dest_path):
    print(f"Starting robust download of {url} to {dest_path} using curl...")
    cmd = ["curl.exe", "-C", "-", "-L", "--retry", "10", "--retry-delay", "3", "-o", dest_path, url]
    try:
        subprocess.run(cmd, check=True)
        print(f"Finished downloading {dest_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {dest_path}: {e}")
        raise e

def process_asvspoof():
    asv_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
    asv_zip = os.path.join(TEMP_DIR, 'LA.zip')
    
    if not os.path.exists(os.path.join(RAW_REAL_DIR, "LA_T_1138215.flac")):
        download_file(asv_url, asv_zip)
        
        print("Parsing ASVSpoof LA protocols...")
        # Dictionary to store file mapping: filename -> 'bonafide' or 'spoof'
        file_labels = {}
        
        with zipfile.ZipFile(asv_zip, 'r') as zf:
            namelist = zf.namelist()
            # Find protocol files
            protocol_files = [n for n in namelist if 'cm_protocols' in n and n.endswith('.txt')]
            
            for p_file in protocol_files:
                with zf.open(p_file) as f:
                    for line in f:
                        parts = line.decode('utf-8').strip().split()
                        if len(parts) >= 5:
                            # format: speaker_id filename - - label
                            filename = parts[1]
                            label = parts[4] # 'bonafide' or 'spoof'
                            file_labels[filename] = label
                            
            print(f"Loaded {len(file_labels)} label mappings.")
            
            # Extract files based on label
            flac_files = [n for n in namelist if n.endswith('.flac')]
            print(f"Found {len(flac_files)} flac files to extract...")
            
            for i, name in enumerate(flac_files):
                filename = os.path.basename(name).replace('.flac', '')
                label = file_labels.get(filename)
                
                if label == 'bonafide':
                    target_dir = RAW_REAL_DIR
                elif label == 'spoof':
                    target_dir = RAW_FAKE_DIR
                else:
                    continue # Skip files without labels
                    
                target_path = os.path.join(target_dir, os.path.basename(name))
                
                if not os.path.exists(target_path):
                    with zf.open(name) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                        
                if i > 0 and i % 5000 == 0:
                    print(f"  Extracted {i} ASVSpoof files...")
                    
        print("Finished extracting ASVSpoof LA.")
        print(f"Deleting {asv_zip} to save space.")
        os.remove(asv_zip)
    else:
        print("ASVSpoof already extracted.")

if __name__ == '__main__':
    print("Starting ASVSpoof Dataset Download and Extraction...")
    process_asvspoof()
    print("All tasks completed successfully!")
