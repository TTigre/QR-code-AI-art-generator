import torch
import gradio as gr
from PIL import Image
import qrcode
from pathlib import Path
from multiprocessing import cpu_count
import requests
import io
import os
from PIL import Image
import random
from qreader import QReader
import cv2
from numpy import asarray

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

# Create a QRCode generator
qrcode_generator = qrcode.QRCode(
    version=1,  # Set the version of the QRCode
    error_correction=qrcode.ERROR_CORRECT_H,  # Set the error correction level
    box_size=10,  # Set the size of each box in the QRCode
    border=4,  # Set the size of the border around the QRCode
)

# Load a pre-trained ControlNet model for image-to-image conversion
controlnet = ControlNetModel.from_pretrained(
    "DionTimmer/controlnet_qrcode-control_v1p_sd15",  # Specify the name of the pre-trained model
    torch_dtype=torch.float16,  # Set the torch dtype to float16 for memory efficiency
)

# Create a StableDiffusionControlNetImg2ImgPipeline
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # Specify the name of the pre-trained pipeline
    controlnet=controlnet,  # Pass the loaded ControlNet model to the pipeline
    safety_checker=None,  # Set the safety checker to None
    torch_dtype=torch.float16,  # Set the torch dtype to float16 for memory efficiency
).to("cuda")  # Move the pipeline to the CUDA device

# Enable memory-efficient attention in the pipeline
pipe.enable_xformers_memory_efficient_attention()



def resize_for_condition_image(input_image: Image.Image, resolution: int) -> Image.Image:
    # Convert the input image to RGB format
    input_image = input_image.convert("RGB")

    # Get the width and height of the input image
    W, H = input_image.size

    # Calculate the scaling factor based on the resolution and the minimum dimension of the image
    k = float(resolution) / min(H, W)

    # Scale the width and height of the image based on the scaling factor
    H *= k
    W *= k

    # Round the scaled width and height to the nearest multiple of 64
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64

    # Resize the image with the new width and height using Lanczos resampling method
    img = input_image.resize((W, H), resample=Image.LANCZOS)

    # Return the resized image
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
    # Check if prompt is provided
    if prompt is None or prompt == "":
        raise gr.Error("Prompt is required")

    # Check if either QR Code Image or QR Code Content is provided
    if qrcode_image is None and qr_code_content == "":
        raise gr.Error("QR Code Image or QR Code Content is required")

    # Set the scheduler based on the sampler type
    pipe.scheduler = SAMPLER_MAP[sampler](pipe.scheduler.config)

    # Set the generator seed if provided
    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()

    # Generate QR Code Image from content if provided, else use the provided QR Code Image
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

    # Set the init_image as the modified QR Code Image
    init_image = qrcode_image

    # Perform inference with the provided parameters
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=qrcode_image,
        control_image=qrcode_image,
        width=768,
        height=768,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        strength=float(strength),
        num_inference_steps=40,
    )
    
    # Return the resulting image
    return out.images[0]

global qreader
qreader = QReader()

def try_inference(
    qr_code_content: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 10.0,
    controlnet_conditioning_scale: float = 2.0,
    strength: float = 0.8,
    seed: int = -1,
    init_image: Image.Image | None = None,
    qrcode_image: Image.Image | None = None,
    use_qr_code_as_init_image: bool = True,
    sampler: str = "DPM++ Karras SDE"
    ) -> Image.Image | None :

    # Perform inference using the provided parameters
    image = inference(
        qr_code_content,
        prompt,
        negative_prompt,
        guidance_scale,
        controlnet_conditioning_scale,
        strength,
        seed,
        init_image,
        qrcode_image,
        use_qr_code_as_init_image,
        sampler
    )

    # Convert image to RGB format
    image = cv2.cvtColor(asarray(image), cv2.COLOR_BGR2RGB)
    
    # Decode QR code from the image
    decoded_text = qreader.decode(image=image)
    
    # Check if the decoded text matches the provided QR code content
    if decoded_text is not None and decoded_text == qr_code_content:
        return image
    else:
        return None

def valid_QR_code_image(
    steps: int,
    qr_code_content: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 10.0,
    controlnet_conditioning_scale: float = 2.5,
    strength: float = 0.8,
    seed: int = -1,
    init_image: Image.Image | None = None,
    qrcode_image: Image.Image | None = None,
    use_qr_code_as_init_image = True,
    sampler = "DPM++ Karras SDE",
):
    # Set the initial range for conditioning scale
    conditioning_max = 2.0
    conditioning_min = 0.6
    conditioning_current = 1.3
    last_working = 2.0
    working_image = None

    # Iterate for the specified number of steps
    for i in range(steps):
        # Try generating an image using inference function
        image = try_inference(
            qr_code_content, prompt, negative_prompt, guidance_scale, conditioning_current,
            strength, seed, init_image, qrcode_image, use_qr_code_as_init_image, sampler
        )
        if image is not None:
            # If an image is successfully generated, update the range of conditioning scale
            conditioning_max = conditioning_current
            last_working = conditioning_current
            conditioning_current = (conditioning_max + conditioning_min) / 2.0
            working_image = image
            print("Match Found!")
        else:
            # If no image is generated, update the range of conditioning scale
            conditioning_min = conditioning_current
            conditioning_current = (conditioning_max + conditioning_min) / 2.0
            print("No Match!")

    if working_image is None:
        # If no working image is found, use a higher conditioning scale
        conditioning_current = 2.0
        working_image = inference(
            qr_code_content, prompt, negative_prompt, guidance_scale, conditioning_current,
            strength, seed, init_image, qrcode_image, use_qr_code_as_init_image, sampler
        )

    # Generate more QR-like images
    final_images: List[Image.Image] = []
    final_images.append(working_image)
    print("Generating more QR like images")

    for i in range(4):
        print("Generating image " + str(i+2))
        conditioning_current += 0.2
        final_images.append(inference(
            qr_code_content, prompt, negative_prompt, guidance_scale, conditioning_current,
            strength, seed, init_image, qrcode_image, use_qr_code_as_init_image, sampler
        ))

    # Return the generated images
    return (
        final_images[0], final_images[1], final_images[2], final_images[3], final_images[4]
    )

def valid_QR_code_image_GUI(
    steps: int,  # Number of steps for the QR code validation process
    qr_code_content: str,  # Content of the QR code to validate
    prompt: str,  # Prompt message for positive validation
    negative_prompt: str,  # Prompt message for negative validation
    guidance_scale: float = 10.0,  # Scale factor for guidance loss
    strength: float = 0.8,  # Strength of the validation process
    seed: int = -1,  # Seed for random number generation
    init_image: Image.Image | None = None,  # Initial image for the validation process
    qrcode_image: Image.Image | None = None,  # QR code image for the validation process
    use_qr_code_as_init_image = True,  # Flag to determine whether to use the QR code as initial image
    sampler = "DPM++ Karras SDE",  # Sampler to use for generating images
):
    # Set the conditioning scale for the controlnet
    controlnet_conditioning_scale: float = 1.3
    
    # Call the valid_QR_code_image function with the provided arguments
    return valid_QR_code_image(
        steps,
        qr_code_content,
        prompt,
        negative_prompt,
        guidance_scale,
        controlnet_conditioning_scale,
        strength,
        seed,
        init_image,
        qrcode_image,
        use_qr_code_as_init_image,
        sampler
    )


# Create a block of code
with gr.Blocks() as blocks:
    gr.Markdown(
        """
# QR Code AI Art Generator

## 💡 Cómo generar códigos QR hermosos

Utilizamos la imagen del código QR como la imagen inicial **y** la imagen de control, lo que te permite generar QR codes que se integren de forma **muy natural** con tu indicación proporcionada.
El parámetro de fuerza define cuánto ruido se agrega a tu código QR, y luego se guía al código QR ruidoso tanto hacia tu indicación como hacia la imagen del código QR mediante Controlnet.
Si necesitas cambiar la fuerza utiliza un valor de fuerza alto entre 0.8 y 0.95.


Se recomienda prompts con formas no demasiado específicas.
Usualmente personas o retratos no funcionan muy bien.
Los prompts de birdview suelen funcionar bien.
Los prompts en inglés suelen proporcionar mejores resultados.


Algunos ejemplos exitosos son:
- Colonial town birdview
- Sunny tropical island
- Cyberpunk city
- Galaxies with explosions

                """
    )

    # Create a row
    with gr.Row():
        # Create a column
        with gr.Column():
            # Create a textbox for QR code content
            qr_code_content = gr.Textbox(
                label="Contenido del código QR",
                info="Contenido o URL del código QR",
                value="",
            )

            # Create an accordion for QR code image (optional)
            with gr.Accordion(label="Imagen del código QR (Opcional)", open=False):
                qr_code_image = gr.Image(
                    label="Imagen del código QR (Opcional). Dejar en blanco para generar automáticamente",
                    type="pil",
                )

            # Create a textbox for prompt
            prompt = gr.Textbox(
                label="Prompt",
                info="Prompt que guía la generación",
            )

            # Create a slider for number of steps
            steps = gr.Slider(
                minimum=1, maximum=20, step=1, value=5, label="Pasos", info="Cantidad de imágenes generadas en búsqueda de la más artística",
            )

            # Create a checkbox for using QR code as initial image
            use_qr_code_as_init_image = gr.Checkbox(label="Usar código QR como imagen inicial", value=True, interactive=False, info="Si la imagen inicial debe ser un código QR. Desmarca para omitir la imagen inicial o genera la imagen inicial con Stable Diffusion")

            # Create an accordion for initial images (optional)
            with gr.Accordion(label="Imagenes iniciales (Opcional)", open=False, visible=False) as init_image_acc:
                init_image = gr.Image(label="Imagen inicial (Opcional). Dejar en blanco para generar la imagen con SD", type="pil")

            # Create an accordion for hyperparameters
            with gr.Accordion(
                label="Hiperparámetros: La funcionalidad del código QR generado está influenciada por los parámetros detallados a continuación.",
                open=False,
            ):
                # Create a textbox for negative prompt
                negative_prompt = gr.Textbox(
                    label="Prompt negativo",
                    value="ugly, disfigured, low quality, blurry, nsfw",
                    info="Prompt que se evita generar",
                )

                # Create a slider for strength
                strength = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01, value=0.9, label="Fuerza",
                    info="Cambiar su valor para decidir cuanto se ajusta la imagen generada al prompt",
                )

                # Create a slider for guidance scale
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=50.0,
                    step=0.25,
                    value=7.5,
                    label="Escala de guía",
                )

                # Create a dropdown for sampler
                sampler = gr.Dropdown(choices=list(SAMPLER_MAP.keys()), value="DPM++ Karras SDE", label="Sampler")

                # Create a slider for seed
                seed = gr.Slider(
                    minimum=-1,
                    maximum=9999999999,
                    step=1,
                    value=2313123,
                    label="Seed",
                    randomize=True,
                )

            # Create a row
            with gr.Row():
                # Create a button for running the code
                run_btn = gr.Button("Correr")

        # Create a column for result images
        with gr.Column():
            result_image1 = gr.Image(label="Imagen resultante 1 (La más artística)")
            result_image2 = gr.Image(label="Imagen resultante 2")
            result_image3 = gr.Image(label="Imagen resultante 3")
            result_image4 = gr.Image(label="Imagen resultante 4")
            result_image5 = gr.Image(label="Imagen resultante 5 (La más escaneable)")
    
    # When the run button is clicked, execute the function valid_QR_code_image_GUI
    run_btn.click(
        valid_QR_code_image_GUI,
        inputs=[
            steps,
            qr_code_content,
            prompt,
            negative_prompt,
            guidance_scale,
            strength,
            seed,
            init_image,
            qr_code_image,
            use_qr_code_as_init_image,
            sampler,
        ],
        outputs=[result_image1,result_image2,result_image3,result_image4,result_image5,],
    )
# Queue the code blocks with a concurrency count of 1 and a max size of 20
blocks.queue(concurrency_count=1, max_size=20)

# Launch the blocks with sharing enabled and debugging enabled
blocks.launch(share=True, debug=True)
