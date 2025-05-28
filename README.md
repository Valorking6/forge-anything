
# Forge Anything - Segment Anything Model for Stable Diffusion WebUI
Currently in Alpha - some UI elements missing, errors when running generate.
This extension integrates Meta's Segment Anything Model (SAM) into the Stable Diffusion WebUI with a dedicated tab interface for easier access and improved workflow.

## Features

- Dedicated tab interface for Segment Anything functionality
- Automatic model downloading - no manual downloads required!
- Support for multiple segmentation methods:
  - Everything mode (automatic segmentation)
  - Text-based segmentation (using GroundingDINO)
  - Point-based segmentation
  - Bounding box segmentation
- Model selection for both SAM and GroundingDINO
- CPU/GPU support
- Mask export functionality

## Installation

### Method 1: Install from WebUI Extensions Tab (Recommended)

1. Open the Stable Diffusion WebUI
2. Go to the "Extensions" tab
3. Click on "Install from URL"
4. Enter: `https://github.com/Valorking6/forge-anything`
5. Click "Install"
6. Restart the WebUI

### Method 2: Manual Installation

1. Clone the repository into your extensions folder:
   ```bash
   cd extensions
   git clone https://github.com/Valorking6/forge-anything
   ```
2. Restart the WebUI

## First Run

On first run, the extension will:

1. Install all required dependencies
2. Create necessary directories for models
3. Download the default models (mobile_sam and groundingdino_swinb) if they don't exist

This process happens automatically - no manual intervention required!

## Available Models

### SAM Models
- `sam_vit_h`: The largest and most accurate model (~2.5GB)
- `sam_vit_l`: Medium-sized model (~1.2GB)
- `sam_vit_b`: Base model, good balance of accuracy and size (~375MB)
- `mobile_sam`: Lightweight model for faster inference (~40MB)

### DINO Models
- `groundingdino_swinb`: Model for text-based object detection (~1.6GB)

## Usage

1. Navigate to the "Segment Anything" tab in the WebUI
2. Select a SAM model and DINO model (if using text prompts)
   - If the models aren't downloaded yet, click the "Download" buttons
3. Upload an image
4. Choose a segmentation method:
   - **Everything mode**: Automatically segments all objects
   - **Text prompt**: Enter text to find specific objects
   - **Points**: Click on objects to segment
   - **Bounding box**: Draw a box around objects
5. Click "Generate Mask" to create the segmentation
6. Export the mask if needed

## Troubleshooting

### Models Not Downloading
- Check your internet connection
- Ensure you have enough disk space
- Try running the model_downloader.py script directly:
  ```bash
  cd extensions/forge-anything
  python model_downloader.py
  ```

### GroundingDINO Installation Issues
The GroundingDINO package is installed from GitHub during the extension setup. If you encounter issues:
1. Try installing it manually:
   ```bash
   pip install git+https://github.com/IDEA-Research/GroundingDINO.git
   ```
2. If you still have issues, check the [GroundingDINO repository](https://github.com/IDEA-Research/GroundingDINO) for specific installation instructions.

### Import Errors
- Make sure all dependencies are installed:
  ```bash
  cd extensions/forge-anything
  pip install -r requirements.txt
  ```

### CUDA Out of Memory
- Try using a smaller SAM model (mobile_sam is the smallest)
- Enable the "Use CPU" option (slower but uses less VRAM)
- Close other applications using GPU memory

## Advanced Configuration

### Using Custom Models

You can place custom SAM or DINO models in the following directories:
- SAM models: `extensions/forge-anything/models/sam/`
- DINO models: `extensions/forge-anything/models/grounding-dino/`

The models will be automatically detected and appear in the dropdown menus.

### Command Line Options

When running the WebUI, you can use these options:
- `--forge-anything-download-all`: Download all available models during startup
- `--forge-anything-cpu-only`: Force CPU mode for all models

## Credits

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI Research
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for text-based object detection
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) for the lightweight SAM model
- [AUTOMATIC1111's Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
