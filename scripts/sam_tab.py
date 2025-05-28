
import os
import gradio as gr
from modules import script_callbacks, shared
import sys

# Add the parent directory to the path to import model_downloader
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from model_downloader import get_downloaded_models, download_sam_model, download_dino_model, SAM_MODELS, DINO_MODELS
from scripts.sam import SAMSegmenter
from scripts.dino import GroundingDINODetector

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
import tempfile
import time
import json

# Global variables
sam_model = None
dino_model = None
current_image = None
current_masks = None
current_mask_index = 0
points = []
point_labels = []
box_points = []

def on_ui_tabs():
    # Check if we should force CPU mode
    force_cpu = False
    for cmd_opt in sys.argv:
        if cmd_opt == "--forge-anything-cpu-only":
            force_cpu = True
            print("Forge Anything: Forcing CPU mode")
    
    # Check if we should download all models
    download_all = False
    for cmd_opt in sys.argv:
        if cmd_opt == "--forge-anything-download-all":
            download_all = True
            print("Forge Anything: Downloading all models")
    
    if download_all:
        from model_downloader import download_all_models
        download_all_models()
    
    # Get available models
    downloaded_models = get_downloaded_models()
    
    # Create lists for dropdown menus
    sam_model_choices = [model["name"] for model in downloaded_models["sam"]]
    dino_model_choices = [model["name"] for model in downloaded_models["dino"]]
    
    # Add not-yet-downloaded models
    for model_name in SAM_MODELS:
        if model_name not in sam_model_choices:
            sam_model_choices.append(model_name)
    
    for model_name in DINO_MODELS:
        if model_name not in dino_model_choices:
            dino_model_choices.append(model_name)
    
    # Default to mobile_sam if available
    default_sam_model = "mobile_sam" if "mobile_sam" in sam_model_choices else sam_model_choices[0] if sam_model_choices else None
    default_dino_model = "groundingdino_swinb" if "groundingdino_swinb" in dino_model_choices else dino_model_choices[0] if dino_model_choices else None
    
    with gr.Blocks(analytics_enabled=False, css=get_css()) as segment_anything_tab:
        with gr.Row():
            gr.HTML("<h1>Segment Anything</h1>")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                with gr.Group(elem_id="model_selection_group"):
                    gr.HTML("<h2>Model Selection</h2>")
                    
                    with gr.Row():
                        sam_model_dropdown = gr.Dropdown(
                            choices=sam_model_choices,
                            value=default_sam_model,
                            label="SAM Model",
                            elem_id="sam_model_dropdown"
                        )
                        sam_download_button = gr.Button("Download", elem_id="sam_download_button")
                    
                    with gr.Row():
                        dino_model_dropdown = gr.Dropdown(
                            choices=dino_model_choices,
                            value=default_dino_model,
                            label="DINO Model (for text prompts)",
                            elem_id="dino_model_dropdown"
                        )
                        dino_download_button = gr.Button("Download", elem_id="dino_download_button")
                    
                    use_cpu_checkbox = gr.Checkbox(
                        value=force_cpu,
                        label="Use CPU (slower but uses less VRAM)",
                        elem_id="use_cpu_checkbox"
                    )
                
                # Segmentation options
                with gr.Group(elem_id="segmentation_options_group"):
                    gr.HTML("<h2>Segmentation Options</h2>")
                    
                    segmentation_mode = gr.Radio(
                        choices=["Everything", "Text Prompt", "Points", "Box"],
                        value="Everything",
                        label="Segmentation Mode",
                        elem_id="segmentation_mode"
                    )
                    
                    with gr.Group(visible=False) as text_prompt_group:
                        text_prompt = gr.Textbox(
                            label="Text Prompt",
                            placeholder="Enter objects to detect (e.g., 'cat, dog, person')",
                            elem_id="text_prompt"
                        )
                        text_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.25,
                            step=0.01,
                            label="Text Confidence Threshold",
                            elem_id="text_threshold"
                        )
                        box_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.01,
                            label="Box Confidence Threshold",
                            elem_id="box_threshold"
                        )
                    
                    with gr.Group(visible=False) as points_group:
                        point_label = gr.Radio(
                            choices=["Foreground", "Background"],
                            value="Foreground",
                            label="Point Label",
                            elem_id="point_label"
                        )
                        clear_points_button = gr.Button("Clear Points", elem_id="clear_points_button")
                    
                    with gr.Group(visible=False) as box_group:
                        box_instructions = gr.HTML("<p>Click and drag to draw a box around an object.</p>")
                        clear_box_button = gr.Button("Clear Box", elem_id="clear_box_button")
                    
                    with gr.Group() as common_options:
                        multimask_output = gr.Checkbox(
                            value=True,
                            label="Generate Multiple Masks",
                            elem_id="multimask_output"
                        )
                        
                        points_per_side = gr.Slider(
                            minimum=8,
                            maximum=64,
                            value=32,
                            step=4,
                            label="Points Per Side (for 'Everything' mode)",
                            elem_id="points_per_side"
                        )
                        
                        min_mask_region_area = gr.Slider(
                            minimum=0,
                            maximum=1000,
                            value=100,
                            step=10,
                            label="Minimum Mask Region Area",
                            elem_id="min_mask_region_area"
                        )
                
                # Action buttons
                with gr.Group(elem_id="action_buttons_group"):
                    generate_button = gr.Button("Generate Mask", variant="primary", elem_id="generate_button")
                    
                    with gr.Row():
                        prev_mask_button = gr.Button("Previous Mask", elem_id="prev_mask_button")
                        next_mask_button = gr.Button("Next Mask", elem_id="next_mask_button")
                    
                    export_button = gr.Button("Export Mask", elem_id="export_button")
                    
                    mask_info = gr.HTML("<p>No mask generated yet.</p>", elem_id="mask_info")
            
            with gr.Column(scale=1):
                # Image upload and display
                with gr.Group(elem_id="image_group"):
                    input_image = gr.Image(
                        type="pil",
                        label="Input Image",
                        elem_id="input_image",
                        tool="editor",
                        height=512
                    )
                    
                    output_image = gr.Image(
                        type="pil",
                        label="Segmentation Result",
                        elem_id="output_image",
                        height=512
                    )
                    
                    mask_image = gr.Image(
                        type="pil",
                        label="Mask Only",
                        elem_id="mask_image",
                        height=512
                    )
        
        # Event handlers
        def download_sam(model_name):
            if not model_name:
                return "Please select a model first."
            
            try:
                path = download_sam_model(model_name)
                return f"Downloaded {model_name} to {path}"
            except Exception as e:
                return f"Error downloading {model_name}: {str(e)}"
        
        def download_dino(model_name):
            if not model_name:
                return "Please select a model first."
            
            try:
                path = download_dino_model(model_name)
                return f"Downloaded {model_name} to {path}"
            except Exception as e:
                return f"Error downloading {model_name}: {str(e)}"
        
        def update_segmentation_mode(mode):
            return {
                text_prompt_group: mode == "Text Prompt",
                points_group: mode == "Points",
                box_group: mode == "Box"
            }
        
        def clear_points():
            global points, point_labels
            points = []
            point_labels = []
            return "Points cleared."
        
        def clear_box():
            global box_points
            box_points = []
            return "Box cleared."
        
        def on_image_change(image):
            global current_image, current_masks, current_mask_index, points, point_labels, box_points
            current_image = image
            current_masks = None
            current_mask_index = 0
            points = []
            point_labels = []
            box_points = []
            return None, None, "Image loaded. Ready for segmentation."
        
        def on_image_click(evt: gr.SelectData, image, mode, label):
            global points, point_labels, box_points
            
            if mode == "Points":
                # Add point
                x, y = evt.index
                points.append([x, y])
                point_labels.append(1 if label == "Foreground" else 0)
                
                # Draw points on image
                img = image.copy()
                draw = ImageDraw.Draw(img)
                
                for i, (px, py) in enumerate(points):
                    color = "green" if point_labels[i] == 1 else "red"
                    draw.ellipse([(px-5, py-5), (px+5, py+5)], fill=color)
                
                return img, f"Added {label} point at ({x}, {y}). Total points: {len(points)}"
            
            elif mode == "Box":
                # Add box point
                x, y = evt.index
                
                if len(box_points) < 2:
                    box_points.append([x, y])
                    
                    # Draw box on image
                    img = image.copy()
                    draw = ImageDraw.Draw(img)
                    
                    if len(box_points) == 1:
                        # Draw first point
                        px, py = box_points[0]
                        draw.ellipse([(px-5, py-5), (px+5, py+5)], fill="blue")
                        return img, f"Added first box point at ({px}, {py}). Click again to complete the box."
                    
                    elif len(box_points) == 2:
                        # Draw box
                        x1, y1 = box_points[0]
                        x2, y2 = box_points[1]
                        draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=2)
                        return img, f"Box created from ({x1}, {y1}) to ({x2}, {y2})."
                
                return image, "Box already created. Click 'Clear Box' to start over."
            
            return image, ""
        
        def generate_mask(sam_model_name, dino_model_name, use_cpu, mode, text_prompt_value, 
                          text_threshold_value, box_threshold_value, multimask, points_per_side_value, 
                          min_area_value, image):
            global current_image, current_masks, current_mask_index, sam_model, dino_model, points, point_labels, box_points
            
            if image is None:
                return None, None, "Please upload an image first."
            
            current_image = image
            device = "cpu" if use_cpu else "cuda" if torch.cuda.is_available() else "cpu"
            
            try:
                # Initialize SAM model
                if sam_model is None or sam_model.model_name != sam_model_name or sam_model.device != device:
                    if sam_model is not None:
                        sam_model.unload_model()
                    
                    sam_model = SAMSegmenter(model_name=sam_model_name, device=device)
                    if not sam_model.load_model():
                        return None, None, f"Failed to load SAM model {sam_model_name}. Make sure it's downloaded."
                
                # Set image
                if not sam_model.set_image(image):
                    return None, None, "Failed to set image for SAM model."
                
                # Generate masks based on mode
                if mode == "Everything":
                    masks, scores, _ = sam_model.predict_everything(
                        points_per_side=points_per_side_value,
                        min_mask_region_area=min_area_value
                    )
                    
                    if masks is None:
                        return None, None, "Failed to generate masks in 'Everything' mode."
                    
                    current_masks = masks
                    current_mask_index = 0
                    
                    # Create visualization
                    result_image, mask_only = create_mask_overlay(image, masks[current_mask_index])
                    
                    return result_image, mask_only, f"Generated {len(masks)} masks. Showing mask 1/{len(masks)} (score: {scores[current_mask_index]:.3f})"
                
                elif mode == "Text Prompt":
                    if not text_prompt_value:
                        return None, None, "Please enter a text prompt."
                    
                    # Initialize DINO model
                    if dino_model is None or dino_model.model_name != dino_model_name or dino_model.device != device:
                        if dino_model is not None:
                            dino_model.unload_model()
                        
                        dino_model = GroundingDINODetector(model_name=dino_model_name, device=device)
                        if not dino_model.load_model():
                            return None, None, f"Failed to load DINO model {dino_model_name}. Make sure it's downloaded."
                    
                    # Detect objects with DINO
                    boxes, phrases, scores = dino_model.detect(
                        image=image,
                        text_prompt=text_prompt_value,
                        box_threshold=box_threshold_value,
                        text_threshold=text_threshold_value
                    )
                    
                    if boxes is None or len(boxes) == 0:
                        return None, None, f"No objects detected with text prompt: '{text_prompt_value}'"
                    
                    # Convert to image size
                    h, w = np.array(image).shape[:2]
                    boxes_scaled = boxes.copy()
                    boxes_scaled[:, 0] *= w
                    boxes_scaled[:, 1] *= h
                    boxes_scaled[:, 2] *= w
                    boxes_scaled[:, 3] *= h
                    boxes_scaled = boxes_scaled.astype(int)
                    
                    # Generate masks for each box
                    all_masks = []
                    all_scores = []
                    
                    for i, box in enumerate(boxes_scaled):
                        masks, scores, _ = sam_model.predict_box(box, multimask_output=multimask)
                        
                        if masks is not None and len(masks) > 0:
                            for j in range(len(masks)):
                                all_masks.append(masks[j])
                                all_scores.append(scores[j])
                    
                    if not all_masks:
                        return None, None, "Failed to generate masks from detected objects."
                    
                    current_masks = np.array(all_masks)
                    current_mask_index = 0
                    
                    # Create visualization
                    result_image, mask_only = create_mask_overlay(image, current_masks[current_mask_index])
                    
                    return result_image, mask_only, f"Generated {len(current_masks)} masks from {len(boxes)} objects. Showing mask 1/{len(current_masks)} (score: {all_scores[current_mask_index]:.3f})"
                
                elif mode == "Points":
                    if not points:
                        return None, None, "Please add points by clicking on the image."
                    
                    # Convert points to numpy arrays
                    point_coords = np.array(points)
                    point_label_array = np.array(point_labels)
                    
                    # Generate masks
                    masks, scores, _ = sam_model.predict_points(
                        points=point_coords,
                        point_labels=point_label_array,
                        multimask_output=multimask
                    )
                    
                    if masks is None:
                        return None, None, "Failed to generate masks from points."
                    
                    current_masks = masks
                    current_mask_index = 0
                    
                    # Create visualization
                    result_image, mask_only = create_mask_overlay(image, masks[current_mask_index])
                    
                    return result_image, mask_only, f"Generated {len(masks)} masks from points. Showing mask 1/{len(masks)} (score: {scores[current_mask_index]:.3f})"
                
                elif mode == "Box":
                    if len(box_points) != 2:
                        return None, None, "Please draw a box by clicking two points on the image."
                    
                    # Convert box to XYXY format
                    x1, y1 = box_points[0]
                    x2, y2 = box_points[1]
                    box = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                    
                    # Generate masks
                    masks, scores, _ = sam_model.predict_box(
                        box=box,
                        multimask_output=multimask
                    )
                    
                    if masks is None:
                        return None, None, "Failed to generate masks from box."
                    
                    current_masks = masks
                    current_mask_index = 0
                    
                    # Create visualization
                    result_image, mask_only = create_mask_overlay(image, masks[current_mask_index])
                    
                    return result_image, mask_only, f"Generated {len(masks)} masks from box. Showing mask 1/{len(masks)} (score: {scores[current_mask_index]:.3f})"
                
                return None, None, f"Unknown segmentation mode: {mode}"
            
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, None, f"Error generating mask: {str(e)}"
        
        def prev_mask():
            global current_masks, current_mask_index, current_image
            
            if current_masks is None or current_image is None:
                return None, None, "No masks generated yet."
            
            if len(current_masks) <= 1:
                return None, None, "Only one mask available."
            
            current_mask_index = (current_mask_index - 1) % len(current_masks)
            result_image, mask_only = create_mask_overlay(current_image, current_masks[current_mask_index])
            
            return result_image, mask_only, f"Showing mask {current_mask_index + 1}/{len(current_masks)}"
        
        def next_mask():
            global current_masks, current_mask_index, current_image
            
            if current_masks is None or current_image is None:
                return None, None, "No masks generated yet."
            
            if len(current_masks) <= 1:
                return None, None, "Only one mask available."
            
            current_mask_index = (current_mask_index + 1) % len(current_masks)
            result_image, mask_only = create_mask_overlay(current_image, current_masks[current_mask_index])
            
            return result_image, mask_only, f"Showing mask {current_mask_index + 1}/{len(current_masks)}"
        
        def export_mask():
            global current_masks, current_mask_index
            
            if current_masks is None:
                return "No mask to export."
            
            try:
                # Create a temporary directory
                temp_dir = tempfile.mkdtemp()
                
                # Save the mask as a PNG
                mask = current_masks[current_mask_index]
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                timestamp = int(time.time())
                mask_path = os.path.join(temp_dir, f"mask_{timestamp}.png")
                mask_image.save(mask_path)
                
                # Copy to output directory
                output_dir = os.path.join(shared.opts.outdir_samples or "outputs", "forge-anything")
                os.makedirs(output_dir, exist_ok=True)
                
                output_path = os.path.join(output_dir, f"mask_{timestamp}.png")
                import shutil
                shutil.copy2(mask_path, output_path)
                
                return f"Mask exported to {output_path}"
            except Exception as e:
                return f"Error exporting mask: {str(e)}"
        
        # Connect event handlers
        sam_download_button.click(
            fn=download_sam,
            inputs=[sam_model_dropdown],
            outputs=[mask_info]
        )
        
        dino_download_button.click(
            fn=download_dino,
            inputs=[dino_model_dropdown],
            outputs=[mask_info]
        )
        
        segmentation_mode.change(
            fn=update_segmentation_mode,
            inputs=[segmentation_mode],
            outputs=[text_prompt_group, points_group, box_group]
        )
        
        clear_points_button.click(
            fn=clear_points,
            inputs=[],
            outputs=[mask_info]
        )
        
        clear_box_button.click(
            fn=clear_box,
            inputs=[],
            outputs=[mask_info]
        )
        
        input_image.change(
            fn=on_image_change,
            inputs=[input_image],
            outputs=[output_image, mask_image, mask_info]
        )
        
        input_image.select(
            fn=on_image_click,
            inputs=[input_image, segmentation_mode, point_label],
            outputs=[input_image, mask_info]
        )
        
        generate_button.click(
            fn=generate_mask,
            inputs=[
                sam_model_dropdown, dino_model_dropdown, use_cpu_checkbox,
                segmentation_mode, text_prompt, text_threshold, box_threshold,
                multimask_output, points_per_side, min_mask_region_area,
                input_image
            ],
            outputs=[output_image, mask_image, mask_info]
        )
        
        prev_mask_button.click(
            fn=prev_mask,
            inputs=[],
            outputs=[output_image, mask_image, mask_info]
        )
        
        next_mask_button.click(
            fn=next_mask,
            inputs=[],
            outputs=[output_image, mask_image, mask_info]
        )
        
        export_button.click(
            fn=export_mask,
            inputs=[],
            outputs=[mask_info]
        )
    
    return [(segment_anything_tab, "Segment Anything", "segment_anything_tab")]

def create_mask_overlay(image, mask):
    """Create a visualization of the mask overlaid on the image."""
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Create a colored mask
    colored_mask = np.zeros_like(image_np)
    colored_mask[mask > 0] = [0, 255, 0]  # Green mask
    
    # Create mask overlay
    alpha = 0.5
    overlay = cv2.addWeighted(image_np, 1, colored_mask, alpha, 0)
    
    # Create mask-only image (white on black)
    mask_only = np.zeros_like(image_np)
    mask_only[mask > 0] = [255, 255, 255]  # White mask
    
    # Convert back to PIL Images
    overlay_pil = Image.fromarray(overlay)
    mask_only_pil = Image.fromarray(mask_only)
    
    return overlay_pil, mask_only_pil

def get_css():
    """Get the CSS for the UI."""
    css_file = os.path.join(parent_dir, "style.css")
    
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            return f.read()
    
    return ""

script_callbacks.on_ui_tabs(on_ui_tabs)
