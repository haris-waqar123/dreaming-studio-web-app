import os
import time
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

router = APIRouter()

templates = Jinja2Templates(directory="templates")

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors"
local_ckpt_path = "dreamshaper-Lightening/dream_shaper_lightning_4step_unet.safetensors"

# Ensure the local directory exists.
os.makedirs(os.path.dirname(local_ckpt_path), exist_ok=True)

print(torch.cuda.is_available())

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)

# Check if the model checkpoint is already saved locally, if not download and save it.
if not os.path.isfile(local_ckpt_path):
    ckpt_data = load_file(hf_hub_download(repo, ckpt), device="cuda")
    torch.save(ckpt_data, local_ckpt_path)  # Save the model locally
else:
    ckpt_data = torch.load(local_ckpt_path)  # Load the model from the local file system

unet.load_state_dict(ckpt_data)

pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

@router.post("/generate-image", response_class=HTMLResponse)
async def handle_image_generation(request: Request, prompt: str = Form(...)):
    timestamp = int(time.time())
    image_filename = f"generated_image_{timestamp}.png"
    image_path = f"static/gen_images/{image_filename}"

    image = pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
    
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    
    image.save(image_path)

    return templates.TemplateResponse("index.html", {"request": request, "section": "generate", "values": {"prompt": prompt}, "image_path_1": image_path})
