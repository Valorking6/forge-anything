
import os
import sys
import subprocess
import argparse

def is_package_installed(package_name):
    """Check if a package is installed."""
    try:
        __import__(package_name.split('==')[0].split('>=')[0].strip())
        return True
    except ImportError:
        return False

def install_requirements():
    """Install required packages from requirements.txt."""
    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    
    if not os.path.exists(req_file):
        print("Error: requirements.txt not found!")
        return False
    
    print("Installing requirements...")
    try:
        # Try to use the launch module from A1111 webui if available
        try:
            import launch
            with open(req_file) as file:
                for package in file:
                    package = package.strip()
                    if not package or package.startswith('#'):
                        continue
                    
                    # Check if the package is already installed
                    if not launch.is_installed(package.split('==')[0].split('>=')[0].strip()):
                        # Install the package
                        launch.run_pip(f"install {package}", f"forge-anything requirement: {package}")
            
            # Try to install groundingdino if not already installed
            if not launch.is_installed("groundingdino"):
                print("Installing groundingdino from git...")
                launch.run_pip("install git+https://github.com/IDEA-Research/GroundingDINO.git", "forge-anything requirement: groundingdino")
            
            return True
        except ImportError:
            # If launch module is not available, use subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
            
            # Try to install groundingdino if not already installed
            try:
                import groundingdino
                print("groundingdino already installed")
            except ImportError:
                print("Installing groundingdino from git...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/IDEA-Research/GroundingDINO.git"])
            
            return True
    except Exception as e:
        print(f"Error installing requirements: {str(e)}")
        print("Note: Some dependencies may need to be installed manually.")
        print("For groundingdino, run: pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
        return False

def create_directories():
    """Create necessary directories for models."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create model directories
    sam_model_dir = os.path.join(base_dir, "models", "sam")
    dino_model_dir = os.path.join(base_dir, "models", "grounding-dino")
    
    os.makedirs(sam_model_dir, exist_ok=True)
    os.makedirs(dino_model_dir, exist_ok=True)
    
    print(f"Created model directories:")
    print(f"  - SAM models: {sam_model_dir}")
    print(f"  - DINO models: {dino_model_dir}")
    
    return True

def download_default_models():
    """Download default models if they don't exist."""
    try:
        from model_downloader import download_sam_model, download_dino_model
        
        # Download the smallest SAM model by default
        print("Checking for default SAM model (mobile_sam)...")
        sam_model_path = download_sam_model("mobile_sam")
        
        # Download the default DINO model
        print("Checking for default DINO model (groundingdino_swinb)...")
        dino_model_path = download_dino_model("groundingdino_swinb")
        
        return True
    except Exception as e:
        print(f"Error downloading default models: {str(e)}")
        print("You can download models later using the UI or by running model_downloader.py")
        return False

def main():
    parser = argparse.ArgumentParser(description="Install Forge Anything extension")
    parser.add_argument("--check", action="store_true", help="Check installation without downloading models")
    parser.add_argument("--download-all", action="store_true", help="Download all available models")
    args = parser.parse_args()
    
    print("Setting up Forge Anything extension...")
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please check your internet connection and try again.")
        return
    
    # Create directories
    if not create_directories():
        print("Failed to create directories. Please check permissions and try again.")
        return
    
    if args.check:
        print("Installation check completed successfully.")
        return
    
    # Download models
    if args.download_all:
        try:
            from model_downloader import download_all_models
            download_all_models()
        except Exception as e:
            print(f"Error downloading all models: {str(e)}")
    else:
        download_default_models()
    
    print("Forge Anything extension setup complete!")

if __name__ == "__main__":
    main()
