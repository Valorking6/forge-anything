import os
import sys
import launch

# Check if required packages are installed
req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")

if os.path.exists(req_file):
    with open(req_file) as file:
        for package in file:
            package = package.strip()
            if not package or package.startswith('#'):
                continue
            
            # Check if the package is already installed
            if not launch.is_installed(package.split('==')[0].strip()):
                # Install the package
                launch.run_pip(f"install {package}", f"sd-webui-segment-anything-forge requirement: {package}")

# Create model directories if they don't exist
sam_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sam")
dino_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "grounding-dino")

os.makedirs(sam_model_dir, exist_ok=True)
os.makedirs(dino_model_dir, exist_ok=True)

print("Segment Anything Forge extension setup complete!")