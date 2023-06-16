import torch
import gradio as gr
from PIL import Image
import qrcode
from pathlib import Path
from multiprocessing import cpu_count
import requests
import io
from PIL import Image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
)

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
HF_TOKEN = os.environ.get("HF_TOKEN")

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

qrcode_generator = qrcode.QRCode(
    version=1,
    error_correction=qrcode.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)

controlnet = ControlNetModel.from_pretrained(
    "DionTimmer/controlnet_qrcode-control_v1p_sd15", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()


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


SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "DPM++ Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True),
    "Heun": lambda config: HeunDiscreteScheduler.from_config(config),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
    "DDIM": lambda config: DDIMScheduler.from_config(config),
    "DEIS": lambda config: DEISMultistepScheduler.from_config(config),
}


def inference(
    qr_code_content: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 10.0,
    controlnet_conditioning_scale: float = 2.0,
    strength: float = 0.8,
    seed: int = -1,
    init_image: Image.Image | None = None,
    qrcode_image: Image.Image | None = None,
    use_qr_code_as_init_image = True,
    sampler = "DPM++ Karras SDE",
):
    if prompt is None or prompt == "":
        raise gr.Error("Prompt is required")

    if qrcode_image is None and qr_code_content == "":
        raise gr.Error("QR Code Image or QR Code Content is required")

    pipe.scheduler = SAMPLER_MAP[sampler](pipe.scheduler.config)

    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()

    if qr_code_content != "" or qrcode_image.size == (1, 1):
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

    # hack due to gradio examples
    if use_qr_code_as_init_image:
        init_image = qrcode_image
    elif init_image is None or init_image.size == (1, 1):
        print("Generating random image from prompt using Stable Diffusion")
        # generate image from prompt
        image_bytes = query({"inputs": prompt})
        init_image = Image.open(io.BytesIO(image_bytes))
    else:
        print("Using provided init image")
        init_image = resize_for_condition_image(init_image, 768)

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=qrcode_image,
        control_image=qrcode_image,  # type: ignore
        width=768,  # type: ignore
        height=768,  # type: ignore
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),  # type: ignore
        generator=generator,
        strength=float(strength),
        num_inference_steps=40,
    )
    return out.images[0]  # type: ignore


with gr.Blocks() as blocks:
    gr.Markdown(
        """
# QR Code AI Art Generator

## 💡 How to generate beautiful QR codes

There are two modes to generate beautiful QR codes:

1. **Blend-in mode**. Use the QR code image as the initial image **and** the control image. 
When using the QR code as both the init and control image, you can get QR Codes that blend in **very** naturally with your provided prompt.
The strength parameter defines how much noise is added to your QR code and the noisy QR code is then guided towards both your prompt and the QR code image via Controlnet.
Make sure to leave the radio *Use QR code as init image* checked and use a high strength value (between 0.8 and 0.95) and choose a lower conditioning scale (between 0.7 and 1.3).
This mode arguably achieves the asthetically most appealing images, but also requires more tuning of the controlnet conditioning scale and the strength value. If the generated image 
looks way to much like the original QR code, make sure to gently increase the *strength* value and reduce the *conditioning* scale. Also check out the examples below.

2. **Condition-only mode**. Use the QR code image **only** as the control image and denoise from a provided initial image.
When providing an initial image or letting SD 2.1 generate the initial image, you have much more freedom to decide how the generated QR code can look like depending on your provided image.
This mode allows you to stongly steer the generated QR code into a style, landscape, motive that you provided before-hand. This mode tends to generate QR codes that 
are less *"blend-in"* with the QR code itself. Make sure to choose high controlnet conditioning scales between 2.0 and 3.0 and lower strength values between 0.5 and 0.7. Also check examples below.

model: https://huggingface.co/DionTimmer/controlnet_qrcode-control_v1p_sd15

<a href="https://huggingface.co/spaces/huggingface-projects/QR-code-AI-art-generator?duplicate=true" style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
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
            with gr.Accordion(label="QR Code Image (Optional)", open=False):
                qr_code_image = gr.Image(
                    label="QR Code Image (Optional). Leave blank to automatically generate QR code",
                    type="pil",
                )

            prompt = gr.Textbox(
                label="Prompt",
                info="Prompt that guides the generation towards",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="ugly, disfigured, low quality, blurry, nsfw",
            )
            use_qr_code_as_init_image = gr.Checkbox(label="Use QR code as init image", value=True, interactive=True, info="Whether init image should be QR code. Unclick to pass init image or generate init image with Stable Diffusion 2.1")

            with gr.Accordion(label="Init Images (Optional)", open=False, visible=False) as init_image_acc:
                init_image = gr.Image(label="Init Image (Optional). Leave blank to generate image with SD 2.1", type="pil")

            def change_view(qr_code_as_image: bool):
                if not qr_code_as_image:
                    return {init_image_acc: gr.update(visible=True)}
                else:
                    return {init_image_acc: gr.update(visible=False)}

            use_qr_code_as_init_image.change(change_view, inputs=[use_qr_code_as_init_image], outputs=[init_image_acc])

            with gr.Accordion(
                label="Params: The generated QR Code functionality is largely influenced by the parameters detailed below",
                open=True,
            ):
                controlnet_conditioning_scale = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    step=0.01,
                    value=1.1,
                    label="Controlnet Conditioning Scale",
                )
                strength = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01, value=0.9, label="Strength"
                )
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=50.0,
                    step=0.25,
                    value=7.5,
                    label="Guidance Scale",
                )
                sampler = gr.Dropdown(choices=list(SAMPLER_MAP.keys()), value="DPM++ Karras SDE")
                seed = gr.Slider(
                    minimum=-1,
                    maximum=9999999999,
                    step=1,
                    value=2313123,
                    label="Seed",
                    randomize=True,
                )
            with gr.Row():
                run_btn = gr.Button("Run")
        with gr.Column():
            result_image = gr.Image(label="Result Image")
    run_btn.click(
        inference,
        inputs=[
            qr_code_content,
            prompt,
            negative_prompt,
            guidance_scale,
            controlnet_conditioning_scale,
            strength,
            seed,
            init_image,
            qr_code_image,
            use_qr_code_as_init_image,
            sampler,
        ],
        outputs=[result_image],
    )

    # gr.Examples(
    #     examples=[
    #         [
    #             "https://huggingface.co/",
    #             "A sky view of a colorful lakes and rivers flowing through the desert",
    #             "ugly, disfigured, low quality, blurry, nsfw",
    #             7.5,
    #             1.3,
    #             0.9,
    #             5392011833,
    #             None,
    #             None,
    #             True,
    #             "DPM++ Karras SDE",
    #         ],
    #         [
    #             "https://huggingface.co/spaces/huggingface-projects/QR-code-AI-art-generator",
    #             "billboard amidst the bustling skyline of New York City, with iconic landmarks subtly featured in the background.",
    #             "ugly, disfigured, low quality, blurry, nsfw",
    #             13.37,
    #             2.81,
    #             0.68,
    #             2313123,
    #             "./examples/hack.png",
    #             "./examples/hack.png",
    #             False,
    #             "DDIM",
    #         ],
    #         [
    #             "https://huggingface.co/spaces/huggingface-projects/QR-code-AI-art-generator",
    #             "beautiful sunset in San Francisco with Golden Gate bridge in the background",
    #             "ugly, disfigured, low quality, blurry, nsfw",
    #             11.01,
    #             2.61,
    #             0.66,
    #             1423585430,
    #             "./examples/hack.png",
    #             "./examples/hack.png",
    #             False,
    #             "DDIM",
    #         ],
    #         [
    #             "https://huggingface.co",
    #             "A flying cat over a jungle",
    #             "ugly, disfigured, low quality, blurry, nsfw",
    #             13,
    #             2.81,
    #             0.66,
    #             2702246671,
    #             "./examples/hack.png",
    #             "./examples/hack.png",
    #             False,
    #             "DDIM",
    #         ],
    #         [
    #             "",
    #             "crisp QR code prominently displayed on a billboard amidst the bustling skyline of New York City, with iconic landmarks subtly featured in the background.",
    #             "ugly, disfigured, low quality, blurry, nsfw",
    #             10.0,
    #             2.0,
    #             0.8,
    #             2313123,
    #             "./examples/init.jpeg",
    #             "./examples/qrcode.png",
    #             False,
    #             "DDIM",
    #         ],
    #     ],
    #     fn=inference,
    #     inputs=[
    #         qr_code_content,
    #         prompt,
    #         negative_prompt,
    #         guidance_scale,
    #         controlnet_conditioning_scale,
    #         strength,
    #         seed,
    #         init_image,
    #         qr_code_image,
    #         use_qr_code_as_init_image,
    #         sampler,
    #     ],
    #     outputs=[result_image],
    #     cache_examples=True,
    #  )

blocks.queue(concurrency_count=1, max_size=20)
blocks.launch(share=True)
