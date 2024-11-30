# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
import os
from transformers import AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction

from PIL import Image
import requests

import torch
import time

with torch.no_grad():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = OpenVLAForActionPrediction.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    vla.processor = processor

    # Typical bridge_orig input
    INSTRUCTION = "pick up the blue fork and place it on the left of the pot"

    # Grab image input & format prompt
    image: Image.Image = Image.open(
        requests.get(
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIjNjirrnsk7_v3b7NmJ83TgLPw4ZCuwoLnKe5v3244fKd9ILpssEwnrEj_kZUJo5Bm2o&usqp=CAU",
            stream=True,
        ).raw
    )
    prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut:"

    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
