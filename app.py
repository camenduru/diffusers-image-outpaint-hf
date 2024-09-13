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



"""
def fill_image(image, model_selection):

    margin = 256
    overlap = 24
    # Open the original image
    source = image  # Changed from image["background"] to match new input format
    
    # Calculate new output size
    output_size = (source.width + 2*margin, source.height + 2*margin)
    
    # Create a white background
    background = Image.new('RGB', output_size, (255, 255, 255))
    
    # Calculate position to paste the original image
    position = (margin, margin)
    
    # Paste the original image onto the white background
    background.paste(source, position)
    
    # Create the mask
    mask = Image.new('L', output_size, 255)  # Start with all white
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle([
        (position[0] + overlap, position[1] + overlap),
        (position[0] + source.width - overlap, position[1] + source.height - overlap)
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
        yield image, cnet_image

    image = image.convert("RGBA")
    cnet_image.paste(image, (0, 0), mask)

    yield background, cnet_image
"""

@spaces.GPU
def fill_image(image, model_selection):
    source = image
    target_ratio=(9, 16)
    target_height=1280
    overlap=48
    fade_width=24
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
        yield image, cnet_image

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


title = """<h1 align="center">Diffusers Image Fill</h1>
<div align="center">Draw the mask over the subject you want to erase or change.</div>
"""

with gr.Blocks(css=css) as demo:
    gr.HTML(title)

    run_button = gr.Button("Generate")

    with gr.Row():
        input_image = gr.Image(
            type="pil",
            label="Input Image",
            sources=["upload"],
        )

        result = ImageSlider(
            interactive=False,
            label="Generated Image",
        )

    model_selection = gr.Dropdown(
        choices=list(MODELS.keys()),
        value="RealVisXL V5.0 Lightning",
        label="Model",
    )

    run_button.click(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=fill_image,
        inputs=[input_image, model_selection],
        outputs=result,
    )


demo.launch(share=False)
