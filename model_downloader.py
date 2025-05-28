
import os
import requests
import torch
from tqdm import tqdm
import hashlib
import json

# Define model URLs and checksums
SAM_MODELS = {
    "sam_vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "md5": "4b8939a88964f0f4ff5f5b2642c598a6",
        "size_mb": 2564
    },
    "sam_vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "md5": "0b3195507c641ddb6910d2bb5adee89c",
        "size_mb": 1249
    },
    "sam_vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "md5": "01ec64d29a2fca3f0661936605ae66f8",
        "size_mb": 375
    },
    "mobile_sam": {
        "url": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
        "md5": "151d5d21443d7562266b50e9c93eb2eb",
        "size_mb": 40
    }
}

DINO_MODELS = {
    "groundingdino_swinb": {
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "md5": "5bb6c46c8bb2f5e5c1ab7c117e6f0a5e",
        "size_mb": 1600
    }
}

def get_model_dir(model_type):
    """Get the directory for storing models."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if model_type == "sam":
        return os.path.join(base_dir, "models", "sam")
    elif model_type == "dino":
        return os.path.join(base_dir, "models", "grounding-dino")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def calculate_md5(file_path):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, file_path, expected_md5=None, expected_size_mb=None):
    """Download a file with progress bar and verification."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Check if file already exists and is valid
    if os.path.exists(file_path):
        if expected_md5:
            file_md5 = calculate_md5(file_path)
            if file_md5 == expected_md5:
                print(f"File already exists and MD5 matches: {file_path}")
                return file_path
            else:
                print(f"File exists but MD5 doesn't match. Re-downloading: {file_path}")
        else:
            print(f"File already exists (no MD5 check): {file_path}")
            return file_path
    
    # Download the file
    print(f"Downloading {url} to {file_path}")
    if expected_size_mb:
        print(f"Expected file size: {expected_size_mb} MB")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get file size from headers if available
    file_size = int(response.headers.get('content-length', 0))
    
    # Create progress bar
    progress_bar = tqdm(
        total=file_size, 
        unit='B', 
        unit_scale=True, 
        desc=os.path.basename(file_path)
    )
    
    # Download with progress
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))
    
    progress_bar.close()
    
    # Verify MD5 if provided
    if expected_md5:
        print("Verifying file integrity...")
        file_md5 = calculate_md5(file_path)
        if file_md5 != expected_md5:
            raise ValueError(f"MD5 verification failed for {file_path}. Expected: {expected_md5}, Got: {file_md5}")
        print("MD5 verification successful!")
    
    return file_path

def download_sam_model(model_name):
    """Download a SAM model by name."""
    if model_name not in SAM_MODELS:
        raise ValueError(f"Unknown SAM model: {model_name}. Available models: {list(SAM_MODELS.keys())}")
    
    model_info = SAM_MODELS[model_name]
    model_dir = get_model_dir("sam")
    file_path = os.path.join(model_dir, f"{model_name}.pth")
    
    return download_file(
        url=model_info["url"],
        file_path=file_path,
        expected_md5=model_info["md5"],
        expected_size_mb=model_info["size_mb"]
    )

def download_dino_model(model_name):
    """Download a DINO model by name."""
    if model_name not in DINO_MODELS:
        raise ValueError(f"Unknown DINO model: {model_name}. Available models: {list(DINO_MODELS.keys())}")
    
    model_info = DINO_MODELS[model_name]
    model_dir = get_model_dir("dino")
    file_path = os.path.join(model_dir, f"{model_name}.pth")
    
    return download_file(
        url=model_info["url"],
        file_path=file_path,
        expected_md5=model_info["md5"],
        expected_size_mb=model_info["size_mb"]
    )

def download_all_models():
    """Download all available models."""
    print("Downloading all SAM models...")
    for model_name in SAM_MODELS:
        try:
            print(f"\nDownloading {model_name}...")
            download_sam_model(model_name)
            print(f"{model_name} downloaded successfully!")
        except Exception as e:
            print(f"Error downloading {model_name}: {str(e)}")
    
    print("\nDownloading all DINO models...")
    for model_name in DINO_MODELS:
        try:
            print(f"\nDownloading {model_name}...")
            download_dino_model(model_name)
            print(f"{model_name} downloaded successfully!")
        except Exception as e:
            print(f"Error downloading {model_name}: {str(e)}")
    
    print("\nAll models downloaded!")

def list_available_models():
    """List all available models with their details."""
    print("Available SAM models:")
    for name, info in SAM_MODELS.items():
        print(f"  - {name} ({info['size_mb']} MB)")
    
    print("\nAvailable DINO models:")
    for name, info in DINO_MODELS.items():
        print(f"  - {name} ({info['size_mb']} MB)")

def get_downloaded_models():
    """Get a list of already downloaded models."""
    sam_dir = get_model_dir("sam")
    dino_dir = get_model_dir("dino")
    
    downloaded = {
        "sam": [],
        "dino": []
    }
    
    # Check SAM models
    for model_name in SAM_MODELS:
        file_path = os.path.join(sam_dir, f"{model_name}.pth")
        if os.path.exists(file_path):
            # Verify MD5
            expected_md5 = SAM_MODELS[model_name]["md5"]
            file_md5 = calculate_md5(file_path)
            is_valid = file_md5 == expected_md5
            
            downloaded["sam"].append({
                "name": model_name,
                "path": file_path,
                "valid": is_valid
            })
    
    # Check DINO models
    for model_name in DINO_MODELS:
        file_path = os.path.join(dino_dir, f"{model_name}.pth")
        if os.path.exists(file_path):
            # Verify MD5
            expected_md5 = DINO_MODELS[model_name]["md5"]
            file_md5 = calculate_md5(file_path)
            is_valid = file_md5 == expected_md5
            
            downloaded["dino"].append({
                "name": model_name,
                "path": file_path,
                "valid": is_valid
            })
    
    return downloaded

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for Forge Anything extension")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--download-all", action="store_true", help="Download all available models")
    parser.add_argument("--sam", type=str, help="Download specific SAM model")
    parser.add_argument("--dino", type=str, help="Download specific DINO model")
    parser.add_argument("--check", action="store_true", help="Check downloaded models")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    elif args.download_all:
        download_all_models()
    elif args.sam:
        try:
            path = download_sam_model(args.sam)
            print(f"Downloaded SAM model to: {path}")
        except ValueError as e:
            print(f"Error: {str(e)}")
    elif args.dino:
        try:
            path = download_dino_model(args.dino)
            print(f"Downloaded DINO model to: {path}")
        except ValueError as e:
            print(f"Error: {str(e)}")
    elif args.check:
        downloaded = get_downloaded_models()
        
        print("Downloaded SAM models:")
        for model in downloaded["sam"]:
            status = "Valid" if model["valid"] else "Invalid (MD5 mismatch)"
            print(f"  - {model['name']}: {status}")
        
        print("\nDownloaded DINO models:")
        for model in downloaded["dino"]:
            status = "Valid" if model["valid"] else "Invalid (MD5 mismatch)"
            print(f"  - {model['name']}: {status}")
    else:
        parser.print_help()
