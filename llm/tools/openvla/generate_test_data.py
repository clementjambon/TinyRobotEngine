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
        f.write(weights.contiguous().cpu().float().numpy().tobytes())


def save_featurizer(output, featurizer: timm.models.vision_transformer.VisionTransformer, pixel_values):
    save_weights(pixel_values, output, "pixel_values")
    patch_embeds = featurizer.patch_embed(pixel_values)
    save_weights(patch_embeds, output, "patch_embeds")
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

    # with open(os.path.join(args.output, "pixel_values_featurizer.bin"), "wb") as f:
    #     f.write(pixel_values[:, :3].contiguous().cpu().float().numpy().tobytes())
    # with open(os.path.join(args.output, "pixel_values_fused_featurizer.bin"), "wb") as f:
    #     f.write(pixel_values[:, 3:].contiguous().cpu().float().numpy().tobytes())
    # # Extract patch features
    # patch_embeds = vla.vision_backbone.featurizer.patch_embeds(pixel_values[:, :3])
    # patch_features_featurizer = vla.vision_backbone.featurizer(pixel_values[:, :3])
    # print(patch_features_featurizer.shape)
    # with open(os.path.join(args.output, "patch_features_featurizer.bin"), "wb") as f:
    #     f.write(patch_features_featurizer.contiguous().cpu().float().numpy().tobytes())
    # patch_features_fused_featurizer = vla.vision_backbone.fused_featurizer(pixel_values[:, 3:])
    # with open(os.path.join(args.output, "patch_features_fused_featurizer.bin"), "wb") as f:
    #     f.write(patch_features_fused_featurizer.contiguous().cpu().float().numpy().tobytes())
    # # Project
    # projected_patch_embeddings = vla.projector(
    #     torch.cat([patch_features_featurizer, patch_features_fused_featurizer], dim=2)
    # )
    # with open(os.path.join(args.output, "projected_patch_embeddings.bin"), "wb") as f:
    #     f.write(projected_patch_embeddings.contiguous().cpu().float().numpy().tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="assets/openvla/tests/model/")
    args = parser.parse_args()

    main(args)
