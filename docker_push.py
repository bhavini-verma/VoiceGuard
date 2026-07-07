import os
import subprocess
import sys

# Define Docker Hub target
DOCKER_HUB_IMAGE = "ashitraj/voiceguard-ai:latest"
LOCAL_IMAGE = "voiceguard-ai"

# Force standard compatible API version to prevent 500 Internal Server Error
os.environ["DOCKER_API_VERSION"] = "1.41"

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False
    except FileNotFoundError:
        print("Docker executable not found. Make sure Docker Desktop is installed and running, and 'docker' is added to your system PATH.")
        return False

def main():
    print("=== VOICEGUARD DOCKER EXPORT & COMPATIBILITY SCRIPT ===")
    
    # 1. Build
    print("\n--- Step 1: Building local image ---")
    if not run_command(["docker", "build", "-t", LOCAL_IMAGE, "."]):
        sys.exit(1)
        
    # 2. Tag
    print("\n--- Step 2: Tagging image for Docker Hub ---")
    if not run_command(["docker", "tag", LOCAL_IMAGE, DOCKER_HUB_IMAGE]):
        # Try alternate API version if mismatch occurs
        print("Retrying with older API version fallback...")
        os.environ["DOCKER_API_VERSION"] = "1.40"
        if not run_command(["docker", "tag", LOCAL_IMAGE, DOCKER_HUB_IMAGE]):
            sys.exit(1)
            
    # 3. Push
    print("\n--- Step 3: Pushing image to Docker Hub ---")
    print(f"Pushing to: {DOCKER_HUB_IMAGE}")
    print("Make sure you ran 'docker login' first if authentication is required.")
    if not run_command(["docker", "push", DOCKER_HUB_IMAGE]):
        sys.exit(1)
        
    print("\n=== SUCCESS ===")
    print(f"Image successfully pushed! Your friend can now run:")
    print(f"docker run -p 8000:8000 {DOCKER_HUB_IMAGE}")

if __name__ == "__main__":
    main()
