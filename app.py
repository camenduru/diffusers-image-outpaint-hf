import gradio as gr
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


def infer(image, model_selection, width, height, overlap_width, num_inference_steps, prompt_input=None):
    source = image
    target_size = (width, height)
    target_ratio = (width, height)  # Calculate aspect ratio from width and height
    overlap = overlap_width

    # Upscale if source is smaller than target in both dimensions
    if source.width < target_size[0] and source.height < target_size[1]:
        scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
        new_width = int(source.width * scale_factor)
        new_height = int(source.height * scale_factor)
        source = source.resize((new_width, new_height), Image.LANCZOS)

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

    final_prompt = "high quality"
    if prompt_input.strip() != "":
        final_prompt += ", " + prompt_input

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(final_prompt, "cuda", True)

    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
        num_inference_steps=num_inference_steps
    ):
        yield cnet_image, image

    image = image.convert("RGBA")
    cnet_image.paste(image, (0, 0), mask)

    yield background, cnet_image

def preload_presets(target_ratio):
    if target_ratio == "9:16":
        changed_width = 720
        changed_height = 1280
        return changed_width, changed_height, gr.update(open=False)
    elif target_ratio == "16:9":
        changed_width = 1280
        changed_height = 720
        return changed_width, changed_height, gr.update(open=False)
    elif target_ratio == "Custom":
        return 720, 1280, gr.update(open=True)

def clear_result():
    return gr.update(value=None)


css = """
.gradio-container {
    width: 1200px !important;
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
                    height = 300
                )
                
                prompt_input = gr.Textbox(label="Prompt (Optional)")
                
                with gr.Row():
                    target_ratio = gr.Radio(
                        label = "Expected Ratio",
                        choices = ["9:16", "16:9", "Custom"],
                        value = "9:16",
                        scale = 2
                    )
                    
                    run_button = gr.Button("Generate", scale=1)

                with gr.Accordion(label="Advanced settings", open=False) as settings_panel:
                    with gr.Column(): 
                        with gr.Row():
                            width_slider = gr.Slider(
                                label="Width",
                                minimum=720,
                                maximum=1440,
                                step=8,
                                value=720,  # Set a default value
                            )
                            height_slider = gr.Slider(
                                label="Height",
                                minimum=720,
                                maximum=1440,
                                step=8,
                                value=1280,  # Set a default value
                            )
                        with gr.Row():
                            model_selection = gr.Dropdown(
                                choices=list(MODELS.keys()),
                                value="RealVisXL V5.0 Lightning",
                                label="Model",
                            )
                            num_inference_steps = gr.Slider(label="Steps", minimum=4, maximum=12, step=1, value=8 )

                        overlap_width = gr.Slider(
                            label="Mask overlap width",
                            minimum=1,
                            maximum=50,
                            value=42,
                            step=1
                        )

                gr.Examples(
                    examples=[
                        ["./examples/example_1.webp", "RealVisXL V5.0 Lightning", 1280, 720],  
                        ["./examples/example_2.jpg", "RealVisXL V5.0 Lightning", 720, 1280],  
                        ["./examples/example_3.jpg", "RealVisXL V5.0 Lightning", 1024, 1024],  
                    ],
                    inputs=[input_image, model_selection, width_slider, height_slider],
                )

            with gr.Column():
                result = ImageSlider(
                    interactive=False,
                    label="Generated Image",
                )

    target_ratio.change(
        fn = preload_presets,
        inputs = [target_ratio],
        outputs = [width_slider, height_slider, settings_panel],
        queue = False
    )
    run_button.click(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=infer,
        inputs=[input_image, model_selection, width_slider, height_slider, overlap_width, num_inference_steps, prompt_input],
        outputs=result,
    )

    prompt_input.submit(
        fn=clear_result,
        inputs=None,
        outputs=result,
    ).then(
        fn=infer,
        inputs=[input_image, model_selection, width_slider, height_slider, overlap_width, num_inference_steps, prompt_input],
        outputs=result,
    )

demo.queue(max_size=12).launch(share=True, show_error=True, show_api=True, inline=False)