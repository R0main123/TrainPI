import torch
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline

def generation(positive_prompt="", negative_prompt=""):
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16
    ).to("mps")

    init_image = Image.open("../TrainPI/data/content/helicopter.jpg").convert("RGB")
    mask_image = Image.open("../TrainPI/src/segmentation/helicopter_mask.png").convert("RGB")

    result = pipe(
        prompt=positive_prompt,
        image=init_image,
        mask_image=mask_image,
        guidance_scale=7.5,
        num_inference_steps=50,
        negative_prompt=negative_prompt
    ).images[0]

    result.save("generation.png")
