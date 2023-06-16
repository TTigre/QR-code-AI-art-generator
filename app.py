import torch
import gradio as gr
from PIL import Image
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
)
from diffusers.utils import load_image
from PIL import Image

controlnet = ControlNetModel.from_pretrained(
    "DionTimmer/controlnet_qrcode-control_v1p_sd15", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
)

pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()


def resize_for_condition_image(input_image: Image.Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


def inference(
    init_image: Image.Image,
    qrcode_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 10.0,
    controlnet_conditioning_scale: float = 2.0,
    strength: float = 0.8,
    seed: int = -1,
    num_inference_steps: int = 30,
):
    init_image = resize_for_condition_image(init_image, 768)
    qrcode_image = resize_for_condition_image(qrcode_image, 768)

    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        control_image=qrcode_image,  # type: ignore
        width=768,
        height=768,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        strength=float(strength),
        num_inference_steps=num_inference_steps,
    )
    return out.images[0]


with gr.Blocks() as blocks:
    gr.Markdown(
        """# AI QR Code Generator
                
                model: https://huggingface.co/DionTimmer/controlnet_qrcode-control_v1p_sd15
                """
    )

    with gr.Row():
        with gr.Column():
            init_image = gr.Image(label="Init Image", type="pil")
            qr_code_image = gr.Image(label="QR Code Image", type="pil")
            prompt = gr.Textbox(label="Prompt")
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="ugly, disfigured, low quality, blurry, nsfw",
            )
            with gr.Accordion(label="Params"):
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=50.0,
                    step=0.1,
                    value=10.0,
                    label="Guidance Scale",
                )
                controlnet_conditioning_scale = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    step=0.1,
                    value=2.0,
                    label="Controlnet Conditioning Scale",
                )
                strength = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.1, value=0.8, label="Strength"
                )
                seed = gr.Slider(
                    minimum=-1,
                    maximum=9999999999,
                    step=1,
                    value=2313123,
                    label="Seed",
                    randomize=True,
                )
            run_btn = gr.Button("Run")
        with gr.Column():
            result_image = gr.Image(label="Result Image")
    run_btn.click(
        inference,
        inputs=[
            init_image,
            qr_code_image,
            prompt,
            negative_prompt,
            guidance_scale,
            controlnet_conditioning_scale,
            strength,
            seed,
        ],
        outputs=[result_image],
    )

    gr.Examples(
        examples=[
            [
                "./examples/init.jpeg",
                "./examples/qrcode.png",
                "crisp QR code prominently displayed on a billboard amidst the bustling skyline of New York City, with iconic landmarks subtly featured in the background.",
                "ugly, disfigured, low quality, blurry, nsfw",
                10.0,
                2.0,
                0.8,
                2313123,
            ]
        ],
        fn=inference,
        inputs=[
            init_image,
            qr_code_image,
            prompt,
            negative_prompt,
            guidance_scale,
            controlnet_conditioning_scale,
            strength,
            seed,
        ],
        outputs=[result_image],
    )

blocks.queue()
blocks.launch()
