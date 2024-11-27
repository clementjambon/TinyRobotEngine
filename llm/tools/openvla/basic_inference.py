# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor

from PIL import Image
import requests

import torch
import time

device = "mps"

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

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
start = time.time()
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Your prediction code here
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
#action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
end = time.time()
length = end - start
print("time of prediction:", length)
print("predicted action:", action)