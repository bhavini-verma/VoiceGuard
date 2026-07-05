import os
import subprocess
import sys

project_dir = r"C:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGaurd_TelephonyExp"
venv_python = os.path.join(project_dir, 'venv', 'Scripts', 'python.exe')

def run_cmd(args):
    print(f"\n=========================================")
    print(f"Running: {' '.join(args)}")
    print(f"=========================================")
    res = subprocess.run(args, cwd=project_dir)
    if res.returncode != 0:
        print(f"Error: Command failed with code {res.returncode}")
        sys.exit(res.returncode)

def main():
    # 1. Train base models
    run_cmd([venv_python, 'src/train.py'])
    
    # 2. Train meta-classifier
    run_cmd([venv_python, 'src/train_meta.py'])
    
    # 3. Run diagnostics
    run_cmd([venv_python, 'scratch/diagnose_failures.py'])
    
    print("\nTelephony experiment pipeline completed successfully!")

if __name__ == '__main__':
    main()
