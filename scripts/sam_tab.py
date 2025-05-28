import os
import gradio as gr
from modules import script_callbacks, shared
from scripts.sam import get_sam_model_ids, get_sam_model_list, get_sam_model_dir, get_sam_model, get_sam_mask, get_sam_mask_by_text, get_sam_mask_by_points, get_sam_mask_by_box, get_sam_mask_by_everything
from scripts.dino import get_dino_model_list, get_dino_model_dir, get_dino_model, get_dino_mask

# Define the SAM Tab class
class SAMTab:
    def __init__(self):
        self.sam_model = None
        self.dino_model = None
        self.sam_model_id = None
        self.dino_model_id = None
        self.input_image = None
        self.current_mask = None
        self.points = []
        self.point_labels = []
        self.box = None
        self.text_prompt = ""
        self.use_dino = False
        self.use_everything = False
        self.use_points = False
        self.use_box = False
        self.use_text = False

    def refresh_sam_models(self, sam_model_dropdown):
        model_list = get_sam_model_list()
        return gr.Dropdown.update(choices=model_list, value=model_list[0] if len(model_list) > 0 else None)

    def refresh_dino_models(self, dino_model_dropdown):
        model_list = get_dino_model_list()
        return gr.Dropdown.update(choices=model_list, value=model_list[0] if len(model_list) > 0 else None)

    def load_sam_model(self, model_name, use_cpu):
        if model_name is None or model_name == "":
            return "No SAM model selected"
        
        try:
            self.sam_model_id = model_name
            self.sam_model = get_sam_model(model_name, use_cpu)
            return f"Successfully loaded SAM model: {model_name}"
        except Exception as e:
            return f"Error loading SAM model: {str(e)}"

    def load_dino_model(self, model_name):
        if model_name is None or model_name == "":
            return "No DINO model selected"
        
        try:
            self.dino_model_id = model_name
            self.dino_model = get_dino_model(model_name)
            return f"Successfully loaded DINO model: {model_name}"
        except Exception as e:
            return f"Error loading DINO model: {str(e)}"

    def process_image(self, image, sam_model_name, use_cpu, dino_model_name, use_dino, use_everything, use_points, use_box, use_text, text_prompt, points, point_labels, box):
        if image is None:
            return None, "No image provided"
        
        self.input_image = image
        
        # Load SAM model if not loaded or if model changed
        if self.sam_model is None or self.sam_model_id != sam_model_name:
            load_result = self.load_sam_model(sam_model_name, use_cpu)
            if "Error" in load_result:
                return None, load_result
        
        # Load DINO model if using text prompts and not loaded or if model changed
        if use_dino and (self.dino_model is None or self.dino_model_id != dino_model_name):
            load_result = self.load_dino_model(dino_model_name)
            if "Error" in load_result:
                return None, load_result
        
        try:
            # Process based on selected method
            if use_everything:
                mask, mask_image = get_sam_mask_by_everything(self.sam_model, image)
                self.current_mask = mask
                return mask_image, "Generated mask using 'everything' mode"
            
            elif use_text and use_dino:
                if text_prompt.strip() == "":
                    return None, "Text prompt is empty"
                mask, mask_image = get_sam_mask_by_text(self.sam_model, self.dino_model, image, text_prompt)
                self.current_mask = mask
                return mask_image, f"Generated mask using text prompt: {text_prompt}"
            
            elif use_points and points and point_labels:
                mask, mask_image = get_sam_mask_by_points(self.sam_model, image, points, point_labels)
                self.current_mask = mask
                return mask_image, "Generated mask using points"
            
            elif use_box and box:
                mask, mask_image = get_sam_mask_by_box(self.sam_model, image, box)
                self.current_mask = mask
                return mask_image, "Generated mask using bounding box"
            
            else:
                return None, "Please select a segmentation method and provide required inputs"
                
        except Exception as e:
            return None, f"Error generating mask: {str(e)}"

    def export_mask(self, output_path):
        if self.current_mask is None:
            return "No mask to export"
        
        try:
            if not output_path.endswith('.png'):
                output_path += '.png'
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.current_mask.save(output_path)
            return f"Mask exported to {output_path}"
        except Exception as e:
            return f"Error exporting mask: {str(e)}"

# Create the UI tabs
def on_ui_tabs():
    sam_tab = SAMTab()
    
    with gr.Blocks(analytics_enabled=False) as segment_anything_interface:
        with gr.Row():
            gr.HTML(
                """
                <div style="text-align: center; max-width: 650px; margin: 0 auto">
                    <h1 style="font-weight: 900; font-size: 2.5rem; margin-bottom: 0.5rem">
                        Segment Anything
                    </h1>
                    <p style="margin-bottom: 0.7rem">
                        A powerful image segmentation tool powered by Meta's Segment Anything Model (SAM)
                    </p>
                </div>
                """
            )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                with gr.Box():
                    with gr.Group():
                        gr.HTML("<h2>Model Selection</h2>")
                        
                        with gr.Row():
                            sam_model_dropdown = gr.Dropdown(
                                label="SAM Model", 
                                choices=get_sam_model_list(), 
                                value=get_sam_model_list()[0] if len(get_sam_model_list()) > 0 else None
                            )
                            sam_refresh_button = gr.Button("🔄", elem_id="refresh_sam_models", variant="secondary", size="sm")
                        
                        use_cpu = gr.Checkbox(label="Use CPU for SAM", value=False)
                        
                        with gr.Row():
                            dino_model_dropdown = gr.Dropdown(
                                label="DINO Model (for text prompts)", 
                                choices=get_dino_model_list(), 
                                value=get_dino_model_list()[0] if len(get_dino_model_list()) > 0 else None
                            )
                            dino_refresh_button = gr.Button("🔄", elem_id="refresh_dino_models", variant="secondary", size="sm")
                
                # Segmentation methods
                with gr.Box():
                    with gr.Group():
                        gr.HTML("<h2>Segmentation Method</h2>")
                        
                        with gr.Row():
                            use_everything = gr.Checkbox(label="Everything Mode", value=True)
                            use_text = gr.Checkbox(label="Text Prompt", value=False)
                        
                        with gr.Row():
                            use_points = gr.Checkbox(label="Points", value=False)
                            use_box = gr.Checkbox(label="Bounding Box", value=False)
                        
                        text_prompt = gr.Textbox(label="Text Prompt", placeholder="Enter objects to segment (e.g., 'cat, dog')", visible=True)
                
                # Actions
                with gr.Box():
                    with gr.Group():
                        gr.HTML("<h2>Actions</h2>")
                        
                        process_button = gr.Button("Generate Mask", variant="primary")
                        status_text = gr.Textbox(label="Status", interactive=False)
                        
                        export_path = gr.Textbox(label="Export Path", placeholder="/path/to/save/mask.png")
                        export_button = gr.Button("Export Mask")
                        export_status = gr.Textbox(label="Export Status", interactive=False)
            
            with gr.Column(scale=1):
                # Image input and output
                input_image = gr.Image(label="Input Image", type="pil")
                output_image = gr.Image(label="Segmentation Mask", type="pil")
        
        # Event handlers
        sam_refresh_button.click(
            fn=sam_tab.refresh_sam_models,
            inputs=[sam_model_dropdown],
            outputs=[sam_model_dropdown]
        )
        
        dino_refresh_button.click(
            fn=sam_tab.refresh_dino_models,
            inputs=[dino_model_dropdown],
            outputs=[dino_model_dropdown]
        )
        
        process_button.click(
            fn=sam_tab.process_image,
            inputs=[
                input_image,
                sam_model_dropdown,
                use_cpu,
                dino_model_dropdown,
                use_text,
                use_everything,
                use_points,
                use_box,
                use_text,
                text_prompt,
                input_image,  # Placeholder for points
                input_image,  # Placeholder for point_labels
                input_image,  # Placeholder for box
            ],
            outputs=[output_image, status_text]
        )
        
        export_button.click(
            fn=sam_tab.export_mask,
            inputs=[export_path],
            outputs=[export_status]
        )
        
        # Checkbox interactions to ensure only one method is selected at a time
        def update_checkboxes(everything, text, points, box, selected):
            if selected == "everything" and everything:
                return gr.Checkbox.update(value=False), gr.Checkbox.update(value=False), gr.Checkbox.update(value=False), gr.Textbox.update(visible=False)
            elif selected == "text" and text:
                return gr.Checkbox.update(value=False), gr.Checkbox.update(value=False), gr.Checkbox.update(value=False), gr.Textbox.update(visible=True)
            elif selected == "points" and points:
                return gr.Checkbox.update(value=False), gr.Checkbox.update(value=False), gr.Checkbox.update(value=False), gr.Textbox.update(visible=False)
            elif selected == "box" and box:
                return gr.Checkbox.update(value=False), gr.Checkbox.update(value=False), gr.Checkbox.update(value=False), gr.Textbox.update(visible=False)
            return gr.Checkbox.update(), gr.Checkbox.update(), gr.Checkbox.update(), gr.Textbox.update()
        
        use_everything.change(
            fn=lambda x: update_checkboxes(x, False, False, False, "everything"),
            inputs=[use_everything],
            outputs=[use_text, use_points, use_box, text_prompt]
        )
        
        use_text.change(
            fn=lambda x: update_checkboxes(False, x, False, False, "text"),
            inputs=[use_text],
            outputs=[use_everything, use_points, use_box, text_prompt]
        )
        
        use_points.change(
            fn=lambda x: update_checkboxes(False, False, x, False, "points"),
            inputs=[use_points],
            outputs=[use_everything, use_text, use_box, text_prompt]
        )
        
        use_box.change(
            fn=lambda x: update_checkboxes(False, False, False, x, "box"),
            inputs=[use_box],
            outputs=[use_everything, use_text, use_points, text_prompt]
        )
    
    return [(segment_anything_interface, "Segment Anything", "segment_anything_tab")]

# Register the tab with the WebUI
script_callbacks.on_ui_tabs(on_ui_tabs)