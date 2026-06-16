import urllib.request
import json

def discover():
    url = "https://zenodo.org/api/records/5642694"
    print(f"Querying Zenodo API: {url}")
    
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            print(f"\nTitle: {data.get('metadata', {}).get('title')}")
            files = data.get('files', [])
            print(f"Found {len(files)} files.\n")
            
            total_size = 0
            for f in files:
                name = f.get('key')
                size_mb = f.get('size', 0) / (1024 * 1024)
                total_size += size_mb
                print(f"File: {name} | Size: {size_mb:.2f} MB")
                
            print(f"\nTotal Size: {total_size / 1024:.2f} GB")
    except Exception as e:
        print(f"Error querying API: {e}")

if __name__ == '__main__':
    discover()
