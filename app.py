import torch
import gradio as gr
from PIL import Image
import qrcode
from pathlib import Path

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
)

from PIL import Image

qrcode_generator = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=0,
)

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


sd_pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
)
sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
sd_pipe = sd_pipe.to("cuda")


sd_pipe.enable_xformers_memory_efficient_attention()
sd_pipe.enable_model_cpu_offload()


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
    qr_code_content: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 10.0,
    controlnet_conditioning_scale: float = 2.0,
    strength: float = 0.8,
    seed: int = -1,
    num_inference_steps: int = 30,
):
    print(init_image, qrcode_image, qr_code_content, prompt, negative_prompt)
    if prompt is None or prompt == "":
        raise gr.Error("Prompt is required")

    if qrcode_image is None and qr_code_content == "":
        raise gr.Error("QR Code Image or QR Code Content is required")

    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()

    if init_image is None:
        print("Generating random image from prompt using Stable Diffusion")
        # generate image from prompt
        out = sd_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=25,
            num_images_per_prompt=1,
        )  # type: ignore

        init_image = out.images[0]
    else:
        print("Using provided init image")
        init_image = resize_for_condition_image(init_image, 768)

    if qr_code_content != "":
        print("Generating QR Code from content")
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_code_content)
        qr.make(fit=True)

        qrcode_image = qr.make_image(fill_color="black", back_color="white")
        qrcode_image = resize_for_condition_image(qrcode_image, 768)
    else:
        print("Using QR Code Image")
        qrcode_image = resize_for_condition_image(qrcode_image, 768)

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        control_image=qrcode_image,  # type: ignore
        width=768,  # type: ignore
        height=768,  # type: ignore
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),  # type: ignore
        generator=generator,
        strength=float(strength),
        num_inference_steps=num_inference_steps,
    )
    return out.images[0]  # type: ignore


with gr.Blocks() as blocks:
    gr.Markdown(
        """
# AI QR Code Generator

model: https://huggingface.co/DionTimmer/controlnet_qrcode-control_v1p_sd15

<a href="https://huggingface.co/spaces/huggingface-projects/AI-QR-code-generator?duplicate=true" style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
<img style="margin-bottom: 0em;display: inline;margin-top: -.25em;" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a> for no queue on your own hardware.</p>
                """
    )

    with gr.Row():
        with gr.Column():
            qr_code_content = gr.Textbox(
                label="QR Code Content",
                info="QR Code Content or URL",
                value="",
            )
            prompt = gr.Textbox(
                label="Prompt",
                info="Prompt is required. If init image is not provided, then it will be generated from prompt using Stable Diffusion 2.1",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="ugly, disfigured, low quality, blurry, nsfw",
            )
            init_image = gr.Image(label="Init Image (Optional)", type="pil")

            qr_code_image = gr.Image(
                label="QR Code Image (Optional)",
                type="pil",
            )

            with gr.Accordion(label="Params"):
                gr.Markdown(
                    "**Note: The QR Code Image functionality is highly dependent on the params below.**"
                )
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=50.0,
                    step=0.01,
                    value=10.0,
                    label="Guidance Scale",
                )
                controlnet_conditioning_scale = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    step=0.01,
                    value=2.0,
                    label="Controlnet Conditioning Scale",
                )
                strength = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01, value=0.8, label="Strength"
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
            qr_code_content,
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
                "",
                "crisp QR code prominently displayed on a billboard amidst the bustling skyline of New York City, with iconic landmarks subtly featured in the background.",
                "ugly, disfigured, low quality, blurry, nsfw",
                10.0,
                2.0,
                0.8,
                2313123,
            ],
            [
                "./examples/init.jpeg",
                None,
                "https://huggingface.co",
                "crisp QR code prominently displayed on a billboard amidst the bustling skyline of New York City, with iconic landmarks subtly featured in the background.",
                "ugly, disfigured, low quality, blurry, nsfw",
                10.0,
                2.0,
                0.8,
                2313123,
            ],
            [
                None,
                None,
                "https://huggingface.co/spaces/huggingface-projects/AI-QR-code-generator",
                "beautiful sunset in San Francisco with Golden Gate bridge in the background",
                "ugly, disfigured, low quality, blurry, nsfw",
                10.0,
                2.7,
                0.8,
                7878952477,
            ],
            [
                None,
                None,
                "https://huggingface.co",
                "A flying cat over a jungle",
                "ugly, disfigured, low quality, blurry, nsfw",
                10.0,
                2.7,
                0.8,
                23123124123,
            ],
        ],
        fn=inference,
        inputs=[
            init_image,
            qr_code_image,
            qr_code_content,
            prompt,
            negative_prompt,
            guidance_scale,
            controlnet_conditioning_scale,
            strength,
            seed,
        ],
        outputs=[result_image],
        cache_examples=True,
    )

blocks.queue()
blocks.launch()
