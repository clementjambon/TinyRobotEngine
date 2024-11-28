# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
import os
import argparse
from transformers import AutoModelForVision2Seq, AutoProcessor

from PIL import Image
import requests

import torch
import timm

device = "mps"


def save_weights(weights: torch.Tensor, output: str, name: str):
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, f"{name}.bin"), "wb") as f:
        f.write(weights.cpu().float().numpy().tobytes())


def save_featurizer(output, featurizer: timm.models.vision_transformer.VisionTransformer, pixel_values):
    save_weights(pixel_values, output, "pixel_values")
    patch_embeds = featurizer.patch_embed(pixel_values)
    print("patch_embeds", patch_embeds.shape)
    embed_dim = patch_embeds.shape[-1]
    save_weights(patch_embeds.reshape(16, 16, embed_dim).permute((2, 0, 1)), output, "patch_embeds")
    embeddings = featurizer._pos_embed(patch_embeds)
    print("embeddings", embeddings.shape)
    save_weights(embeddings, output, "embeddings")
    patch_features = featurizer(pixel_values)
    save_weights(patch_features, output, "patch_features")


@torch.no_grad()
def main(args):
    os.makedirs(args.output, exist_ok=True)

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
    pixel_values = inputs["pixel_values"]

    save_featurizer(os.path.join(args.output, "featurizer"), vla.vision_backbone.featurizer, pixel_values[:, :3])
    save_featurizer(
        os.path.join(args.output, "fused_featurizer"), vla.vision_backbone.fused_featurizer, pixel_values[:, 3:]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="assets/openvla/tests/model/")
    args = parser.parse_args()

    main(args)
