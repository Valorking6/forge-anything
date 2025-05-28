
import os
import torch
import numpy as np
from PIL import Image
import cv2
from segment_anything import sam_model_registry, SamPredictor
import sys

# Add the parent directory to the path to import model_downloader
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from model_downloader import download_sam_model

class SAMSegmenter:
    def __init__(self, model_name="mobile_sam", device=None):
        """
        Initialize the SAM segmenter.
        
        Args:
            model_name (str): Name of the SAM model to use.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.predictor = None
    
    def load_model(self):
        """Load the SAM model."""
        try:
            # Download the model if it doesn't exist
            model_path = download_sam_model(self.model_name)
            
            # Determine model type
            if self.model_name == "mobile_sam":
                model_type = "vit_t"
            elif self.model_name == "sam_vit_h":
                model_type = "vit_h"
            elif self.model_name == "sam_vit_l":
                model_type = "vit_l"
            elif self.model_name == "sam_vit_b":
                model_type = "vit_b"
            else:
                raise ValueError(f"Unknown model type for {self.model_name}")
            
            # Load the model
            sam = sam_model_registry[model_type](checkpoint=model_path)
            sam.to(device=self.device)
            
            # Create predictor
            predictor = SamPredictor(sam)
            
            self.model = sam
            self.predictor = predictor
            
            print(f"SAM model {self.model_name} loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"Error loading SAM model: {str(e)}")
            return False
    
    def set_image(self, image):
        """
        Set the image for segmentation.
        
        Args:
            image (PIL.Image or numpy.ndarray): Input image.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Load model if not loaded
            if self.predictor is None:
                if not self.load_model():
                    return False
            
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
            
            # Convert to RGB if grayscale
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            
            # Convert to RGB if RGBA
            if image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
            # Set image in predictor
            self.predictor.set_image(image_np)
            
            return True
        except Exception as e:
            print(f"Error setting image for SAM: {str(e)}")
            return False
    
    def predict_everything(self, points_per_side=32, min_mask_region_area=100):
        """
        Generate masks for everything in the image.
        
        Args:
            points_per_side (int): Number of points to sample along each side of the image.
            min_mask_region_area (int): Minimum area of a mask region to keep.
            
        Returns:
            tuple: (masks, scores, logits) where masks are the segmentation masks,
                  scores are the confidence scores, and logits are the raw model outputs.
        """
        try:
            # Load model if not loaded
            if self.model is None:
                if not self.load_model():
                    return None, None, None
            
            # Generate masks
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=True,
                mask_input=None,
                points_per_side=points_per_side,
                min_mask_region_area=min_mask_region_area
            )
            
            return masks, scores, logits
        except Exception as e:
            print(f"Error predicting everything with SAM: {str(e)}")
            return None, None, None
    
    def predict_points(self, points, point_labels, multimask_output=True):
        """
        Generate masks based on input points.
        
        Args:
            points (numpy.ndarray): Array of point coordinates, shape (N, 2).
            point_labels (numpy.ndarray): Array of point labels (1 for foreground, 0 for background), shape (N,).
            multimask_output (bool): Whether to return multiple masks per input.
            
        Returns:
            tuple: (masks, scores, logits) where masks are the segmentation masks,
                  scores are the confidence scores, and logits are the raw model outputs.
        """
        try:
            # Load model if not loaded
            if self.predictor is None:
                if not self.load_model():
                    return None, None, None
            
            # Generate masks
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=point_labels,
                multimask_output=multimask_output
            )
            
            return masks, scores, logits
        except Exception as e:
            print(f"Error predicting with points using SAM: {str(e)}")
            return None, None, None
    
    def predict_box(self, box, multimask_output=True):
        """
        Generate masks based on a bounding box.
        
        Args:
            box (numpy.ndarray): Bounding box in XYXY format, shape (4,).
            multimask_output (bool): Whether to return multiple masks per input.
            
        Returns:
            tuple: (masks, scores, logits) where masks are the segmentation masks,
                  scores are the confidence scores, and logits are the raw model outputs.
        """
        try:
            # Load model if not loaded
            if self.predictor is None:
                if not self.load_model():
                    return None, None, None
            
            # Generate masks
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],  # Add batch dimension
                multimask_output=multimask_output
            )
            
            return masks, scores, logits
        except Exception as e:
            print(f"Error predicting with box using SAM: {str(e)}")
            return None, None, None
    
    def unload_model(self):
        """Unload the model to free up memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
        
        # Force CUDA memory cleanup if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"SAM model {self.model_name} unloaded")
