# Segment Anything Forge Extension with Tab Interface

This extension integrates Meta's Segment Anything Model (SAM) into the Stable Diffusion WebUI with a dedicated tab interface for easier access and improved workflow.

## Features

- Dedicated tab interface for Segment Anything functionality
- Support for multiple segmentation methods:
  - Everything mode (automatic segmentation)
  - Text-based segmentation (using GroundingDINO)
  - Point-based segmentation
  - Bounding box segmentation
- Model selection for both SAM and GroundingDINO
- CPU/GPU support
- Mask export functionality

## Installation

1. Install the extension through the WebUI's extension tab:
   - Open the "Extensions" tab in the WebUI
   - Click on "Install from URL"
   - Enter: `https://github.com/Valorking6/forge-anything`
   - Click "Install"

2. Alternatively, clone the repository manually:
   ```bash
   cd extensions
   git clone https://github.com/Valorking6/forge-anything
   ```

3. Restart the WebUI

## Usage

1. Navigate to the "Segment Anything" tab in the WebUI
2. Select a SAM model and DINO model (if using text prompts)
3. Upload an image
4. Choose a segmentation method:
   - Everything mode: Automatically segments all objects
   - Text prompt: Enter text to find specific objects
   - Points: Click on objects to segment (not fully implemented yet)
   - Bounding box: Draw a box around objects (not fully implemented yet)
5. Click "Generate Mask" to create the segmentation
6. Export the mask if needed

## Models

The extension supports various SAM models:
- SAM (original)
- SAM-HQ
- MobileSAM
- And more...

For text-based segmentation, it uses GroundingDINO models.

## Requirements

All requirements are automatically installed when you run the WebUI with this extension.

## Credits

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI Research
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for text-based object detection
- [AUTOMATIC1111's Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [Original sd-webui-segment-anything extension](https://github.com/continue-revolution/sd-webui-segment-anything)

## License

This project is licensed under the MIT License - see the LICENSE file for details.