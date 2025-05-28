
import os
import torch
import numpy as np
from PIL import Image
import sys

# Add the parent directory to the path to import model_downloader
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from model_downloader import download_dino_model

class GroundingDINODetector:
    def __init__(self, model_name="groundingdino_swinb", device=None):
        """
        Initialize the GroundingDINO detector.
        
        Args:
            model_name (str): Name of the DINO model to use.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.processor = None
    
    def load_model(self):
        """Load the GroundingDINO model."""
        try:
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict
            
            # Download the model if it doesn't exist
            model_path = download_dino_model(self.model_name)
            
            # Load configuration
            config_file = os.path.join(parent_dir, "models", "grounding-dino", "GroundingDINO_SwinB_cfg.py")
            if not os.path.exists(config_file):
                # Create a default config file if it doesn't exist
                self._create_default_config(config_file)
            
            args = SLConfig.fromfile(config_file)
            args.device = self.device
            
            # Build model
            model = build_model(args)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
            model.eval()
            model = model.to(self.device)
            
            self.model = model
            
            print(f"GroundingDINO model {self.model_name} loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"Error loading GroundingDINO model: {str(e)}")
            return False
    
    def _create_default_config(self, config_path):
        """Create a default configuration file for GroundingDINO."""
        config_content = """
from groundingdino.util.slconfig import SLConfig

def get_config():
    config = SLConfig()
    config.MODEL = SLConfig()
    config.MODEL.DEVICE = "cuda"
    
    # Backbone
    config.MODEL.BACKBONE = SLConfig()
    config.MODEL.BACKBONE.TYPE = "swin"
    config.MODEL.BACKBONE.NAME = "swinB"
    config.MODEL.BACKBONE.PRETRAINED = ""
    config.MODEL.BACKBONE.SWIN = SLConfig()
    config.MODEL.BACKBONE.SWIN.PRETRAIN_IMG_SIZE = 384
    config.MODEL.BACKBONE.SWIN.PATCH_SIZE = 4
    config.MODEL.BACKBONE.SWIN.IN_CHANS = 3
    config.MODEL.BACKBONE.SWIN.EMBED_DIM = 128
    config.MODEL.BACKBONE.SWIN.DEPTHS = [2, 2, 18, 2]
    config.MODEL.BACKBONE.SWIN.NUM_HEADS = [4, 8, 16, 32]
    config.MODEL.BACKBONE.SWIN.WINDOW_SIZE = 12
    config.MODEL.BACKBONE.SWIN.MLP_RATIO = 4.0
    config.MODEL.BACKBONE.SWIN.QKV_BIAS = True
    config.MODEL.BACKBONE.SWIN.QK_SCALE = None
    config.MODEL.BACKBONE.SWIN.DROP_RATE = 0.0
    config.MODEL.BACKBONE.SWIN.ATTN_DROP_RATE = 0.0
    config.MODEL.BACKBONE.SWIN.DROP_PATH_RATE = 0.3
    config.MODEL.BACKBONE.SWIN.APE = False
    config.MODEL.BACKBONE.SWIN.PATCH_NORM = True
    config.MODEL.BACKBONE.SWIN.USE_CHECKPOINT = False
    config.MODEL.BACKBONE.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    
    # Transformer
    config.MODEL.TRANSFORMER = SLConfig()
    config.MODEL.TRANSFORMER.ENCODER = SLConfig()
    config.MODEL.TRANSFORMER.ENCODER.EMBED_DIM = 256
    config.MODEL.TRANSFORMER.ENCODER.NUM_HEADS = 8
    config.MODEL.TRANSFORMER.ENCODER.ATTN_DROPOUT = 0.1
    config.MODEL.TRANSFORMER.ENCODER.DIM_FEEDFORWARD = 2048
    config.MODEL.TRANSFORMER.ENCODER.DROPOUT = 0.1
    config.MODEL.TRANSFORMER.ENCODER.ACTIVATION = "relu"
    config.MODEL.TRANSFORMER.ENCODER.NUM_LAYERS = 6
    config.MODEL.TRANSFORMER.ENCODER.NORMALIZE_BEFORE = False
    
    # Text Encoder
    config.MODEL.TEXT_ENCODER = SLConfig()
    config.MODEL.TEXT_ENCODER.WIDTH = 768
    config.MODEL.TEXT_ENCODER.CONTEXT_LENGTH = 77
    config.MODEL.TEXT_ENCODER.NUM_LAYERS = 12
    config.MODEL.TEXT_ENCODER.VOCAB_SIZE = 49408
    config.MODEL.TEXT_ENCODER.PROJ_DIM = 256
    config.MODEL.TEXT_ENCODER.N_CTX = 16
    
    # Decoder
    config.MODEL.DECODER = SLConfig()
    config.MODEL.DECODER.HIDDEN_DIM = 256
    config.MODEL.DECODER.NHEADS = 8
    config.MODEL.DECODER.DIM_FEEDFORWARD = 2048
    config.MODEL.DECODER.DROPOUT = 0.1
    config.MODEL.DECODER.ACTIVATION = "relu"
    config.MODEL.DECODER.NUM_LAYERS = 6
    config.MODEL.DECODER.PRE_NORM = False
    
    # Matcher
    config.MODEL.MATCHER = SLConfig()
    config.MODEL.MATCHER.COST_CLASS = 2.0
    config.MODEL.MATCHER.COST_BBOX = 5.0
    config.MODEL.MATCHER.COST_GIOU = 2.0
    config.MODEL.MATCHER.COST_MASK = 1.0
    config.MODEL.MATCHER.COST_DICE = 1.0
    
    # Loss Weights
    config.MODEL.LOSS = SLConfig()
    config.MODEL.LOSS.MASKS = 1.0
    config.MODEL.LOSS.BOXES = 1.0
    config.MODEL.LOSS.CLASS = 2.0
    
    return config

config = get_config()
"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            f.write(config_content)
        
        print(f"Created default GroundingDINO config at {config_path}")
    
    def detect(self, image, text_prompt, box_threshold=0.35, text_threshold=0.25):
        """
        Detect objects in an image based on a text prompt.
        
        Args:
            image (PIL.Image or numpy.ndarray): Input image.
            text_prompt (str): Text prompt for detection.
            box_threshold (float): Box confidence threshold.
            text_threshold (float): Text confidence threshold.
            
        Returns:
            tuple: (boxes, phrases, scores) where boxes are the bounding boxes,
                  phrases are the detected object labels, and scores are the confidence scores.
        """
        try:
            from groundingdino.util.inference import load_image, predict
            
            # Load model if not loaded
            if self.model is None:
                if not self.load_model():
                    return None, None, None
            
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
            
            # Prepare image for model
            image_tensor = load_image(image_np).to(self.device)
            
            # Run inference
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            
            # Convert boxes to numpy for easier handling
            boxes_np = boxes.cpu().numpy()
            scores_np = logits.cpu().numpy()
            
            return boxes_np, phrases, scores_np
        except Exception as e:
            print(f"Error during GroundingDINO detection: {str(e)}")
            return None, None, None
    
    def unload_model(self):
        """Unload the model to free up memory."""
        if self.model is not None:
            del self.model
            self.model = None
            
            # Force CUDA memory cleanup if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"GroundingDINO model {self.model_name} unloaded")
