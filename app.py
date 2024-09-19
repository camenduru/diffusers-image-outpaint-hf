import gradio as gr
import spaces
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download

from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

from PIL import Image, ImageDraw
import numpy as np

MODELS = {
    "RealVisXL V5.0 Lightning": "SG161222/RealVisXL_V5.0_Lightning",
}

config_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="config_promax.json",
)

config = ControlNetModel_Union.load_config(config_file)
controlnet_model = ControlNetModel_Union.from_config(config)
model_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="diffusion_pytorch_model_promax.safetensors",
)
state_dict = load_state_dict(model_file)
model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
    controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
)
model.to(device="cuda", dtype=torch.float16)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
).to("cuda")

pipe = StableDiffusionXLFillPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0_Lightning",
    torch_dtype=torch.float16,
    vae=vae,
    controlnet=model,
    variant="fp16",
).to("cuda")

pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

prompt = "high quality"
(
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,
) = pipe.encode_prompt(prompt, "cuda", True)



@spaces.GPU
def infer(image, model_selection, ratio_choice, overlap_width):

    source = image
    
    if ratio_choice == "16:9":
        target_ratio = (16, 9)  # Set the new target ratio to 16:9
        target_width = 1280  # Adjust target width based on desired resolution
        overlap = overlap_width
        #fade_width = 24
        max_height = 720  # Adjust max height instead of width
        
        # Resize the image if it's taller than max_height
        if source.height > max_height:
            scale_factor = max_height / source.height
            new_height = max_height
            new_width = int(source.width * scale_factor)
            source = source.resize((new_width, new_height), Image.LANCZOS)
        
        # Calculate the required width for the 16:9 ratio
        target_width = (source.height * target_ratio[0]) // target_ratio[1]
        
        # Calculate margins (now left and right)
        margin_x = (target_width - source.width) // 2
        
        # Calculate new output size
        output_size = (target_width, source.height)
        
        # Create a white background
        background = Image.new('RGB', output_size, (255, 255, 255))
        
        # Calculate position to paste the original image
        position = (margin_x, 0)
        
        # Paste the original image onto the white background
        background.paste(source, position)
        
        # Create the mask
        mask = Image.new('L', output_size, 255)  # Start with all white
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([
            (margin_x + overlap, overlap),
            (margin_x + source.width - overlap, source.height - overlap)
        ], fill=0)
        
        # Prepare the image for ControlNet
        cnet_image = background.copy()
        cnet_image.paste(0, (0, 0), mask)
    
        for image in pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=cnet_image,
        ):
            yield cnet_image, image
    
        image = image.convert("RGBA")
        cnet_image.paste(image, (0, 0), mask)
    
        yield background, cnet_image

    elif ratio_choice == "9:16":
        
        target_ratio=(9, 16)
        target_height=1280
        overlap=overlap_width
        #fade_width=24
        max_width = 720
        # Resize the image if it's wider than max_width
        if source.width > max_width:
            scale_factor = max_width / source.width
            new_width = max_width
            new_height = int(source.height * scale_factor)
            source = source.resize((new_width, new_height), Image.LANCZOS)
        
        # Calculate the required height for 9:16 ratio
        target_height = (source.width * target_ratio[1]) // target_ratio[0]
        
        # Calculate margins (only top and bottom)
        margin_y = (target_height - source.height) // 2
        
        # Calculate new output size
        output_size = (source.width, target_height)
        
        # Create a white background
        background = Image.new('RGB', output_size, (255, 255, 255))
        
        # Calculate position to paste the original image
        position = (0, margin_y)
        
        # Paste the original image onto the white background
        background.paste(source, position)
        
        # Create the mask
        mask = Image.new('L', output_size, 255)  # Start with all white
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([
            (overlap, margin_y + overlap),
            (source.width - overlap, margin_y + source.height - overlap)
        ], fill=0)
        
        # Prepare the image for ControlNet
        cnet_image = background.copy()
        cnet_image.paste(0, (0, 0), mask)
    
        for image in pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=cnet_image,
        ):
            yield cnet_image, image
    
        image = image.convert("RGBA")
        cnet_image.paste(image, (0, 0), mask)
    
        yield background, cnet_image

    elif ratio_choice == "1:1":
        target_ratio = (1, 1)
        target_size = (1024, 1024)
        overlap = overlap_width

        if source.width > target_size[0] or source.height > target_size[1]:
            scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
            new_width = int(source.width * scale_factor)
            new_height = int(source.height * scale_factor)
            source = source.resize((new_width, new_height), Image.LANCZOS)
        
        margin_x = (target_size[0] - source.width) // 2
        margin_y = (target_size[1] - source.height) // 2

        background = Image.new('RGB', target_size, (255, 255, 255))
        background.paste(source, (margin_x, margin_y))

        mask = Image.new('L', target_size, 255)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([
            (margin_x + overlap, margin_y + overlap),
            (margin_x + source.width - overlap, margin_y + source.height - overlap)
        ], fill=0)

        cnet_image = background.copy()
        cnet_image.paste(0, (0, 0), mask)

        for image in pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=cnet_image,
        ):
            yield cnet_image, image

        image = image.convert("RGBA")
        cnet_image.paste(image, (0, 0), mask)

        yield background, cnet_image
        

def clear_result():
    return gr.update(value=None)


css = """
.gradio-container {
    width: 1024px !important;
}
"""


title = """<h1 align="center">Diffusers Image Outpaint</h1>
<div align="center">Drop an image you would like to extend, pick your expected ratio and hit Generate.</div>
"""

with gr.Blocks(css=css) as demo:
    with gr.Column():
        
        gr.HTML(title)

        with gr.Row():
            
            with gr.Column():
                
                input_image = gr.Image(
                    type="pil",
                    label="Input Image",
                    sources=["upload"],
                )
     
                with gr.Row():
                    ratio = gr.Radio(
                        label="Expected ratio", 
                        choices=["1:1", "9:16", "16:9"],
                        value = "9:16"
                    )
                    model_selection = gr.Dropdown(
                        choices=list(MODELS.keys()),
                        value="RealVisXL V5.0 Lightning",
                        label="Model",
                    )

                overlap_width = gr.Slider(
                    label="Mask overlap width",
                    minimum = 1,
                    maximum = 50,
                    value = 42,
                    step = 1
                )
    
                run_button = gr.Button("Generate")

                gr.Examples(
                    examples = [
                        ["./examples/example_1.webp", "RealVisXL V5.0 Lightning", "9:16"],
                        ["./examples/example_2.jpg", "RealVisXL V5.0 Lightning", "16:9"],
                        ["./examples/example_3.jpg", "RealVisXL V5.0 Lightning", "1:1"]
                    ],
                    inputs = [input_image, model_selection, ratio]
                )
            
            with gr.Column():
                result = ImageSlider(
                    interactive=False,
                    label="Generated Image",
                )

    run_button.click(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=infer,
        inputs=[input_image, model_selection, ratio, overlap_width],
        outputs=result,
    )


demo.launch(share=False)