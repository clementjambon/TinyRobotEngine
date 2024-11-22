# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
import os
from transformers import AutoModelForVision2Seq, AutoProcessor

from PIL import Image
import requests

import torch
import time

device = "mps"
OUTPATH = "llm/embeds/openvla-7b"

with torch.no_grad():

    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # Grab image input & format prompt
    image: Image.Image = Image.open(
        requests.get(
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIjNjirrnsk7_v3b7NmJ83TgLPw4ZCuwoLnKe5v3244fKd9ILpssEwnrEj_kZUJo5Bm2o&usqp=CAU",
            stream=True,
        ).raw
    )
    prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

    # Preprocess
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    pixels_values = inputs["pixel_values"]
    # Extract patch features
    patch_features = vla.vision_backbone(pixels_values)
    # Project
    projected_patch_embeddings = vla.projector(patch_features)
    os.makedirs(OUTPATH, exist_ok=True)
    with open(os.path.join(f"{OUTPATH}", "embeds.bin"), "wb") as f:
        f.write(projected_patch_embeddings.cpu().float().numpy().tobytes())
