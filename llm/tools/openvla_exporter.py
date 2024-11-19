"""Implementation of exporting VILA PyTorch model to TinyChatEngine format.

Usage:
   python vila_exporter.py <path of hugging face model checkpoint> <output dir>

Example commandline:
   python tools/vila_exporter.py --model models/vila-7b --output models/VILA_7B
"""

import argparse
import math
import os
import struct
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


@torch.no_grad()
def _export_model(model, prefix):

    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "lm_head.bin"), "wb") as f:
        f.write(model.language_model.lm_head._parameters["weight"].cpu().float().numpy().tobytes())
    _export_llama_model(model.language_model.model, os.path.join(f"{outpath}", "decoder"))

    # Export to Clip's folder "models/CLIP_ViT_Large"
    # _export_mm_projector(model.model.mm_projector, f"models/CLIP_ViT_Large/mm_projector")
    # for idx, mm_projector in enumerate(model.model.mm_projector):
    #     if idx == 0 or idx == 2:
    #         # _export_mm_projector(mm_projector, os.path.join(outpath, f"mm_projector_{idx}"))
    #         # Export to Clip's folder "models/CLIP_ViT_Large"
    #         _export_mm_projector(mm_projector, f"models/CLIP_ViT_Large/mm_projector_{idx}")


def _export_mm_projector(mm_projector, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(mm_projector.weight.cpu().float().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "bias.bin"), "wb") as f:
        f.write(mm_projector.bias.cpu().float().numpy().tobytes())


def _export_embed_tokens(embed_tokens, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(embed_tokens.weight.cpu().float().numpy().tobytes())


def _export_llama_model(model, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)

    _export_embed_tokens(model.embed_tokens, os.path.join(outpath, "embed_tokens"))
    _export_LlamaRMSNorm(model.norm, os.path.join(outpath, "norm"))
    for idx, layer in enumerate(model.layers):
        _export_llama_layer(layer, os.path.join(outpath, f"layer{idx}"))


def _export_LlamaRMSNorm(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op.weight.cpu().float().numpy().tobytes())


def _export_llama_layer(layer, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    _export_attention_params(layer.self_attn, os.path.join(outpath, "self_attn"))
    _export_LlamaRMSNorm(layer.input_layernorm, os.path.join(outpath, "input_layernorm"))
    _export_LlamaRMSNorm(
        layer.post_attention_layernorm,
        os.path.join(outpath, "post_attention_layernorm"),
    )
    _export_linearfp(layer.mlp.gate_proj, os.path.join(outpath, "gate_proj"))
    _export_linearfp(layer.mlp.down_proj, os.path.join(outpath, "down_proj"))
    _export_linearfp(layer.mlp.up_proj, os.path.join(outpath, "up_proj"))


def _export_linearfp(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op._parameters["weight"].cpu().float().numpy().tobytes())


def _export_rotaryEmbedding(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "cos_cached.bin"), "wb") as f:
        f.write(op.cos_cached.cpu().float().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "sin_cached.bin"), "wb") as f:
        f.write(op.sin_cached.cpu().float().numpy().tobytes())


def _export_BMM_F32T(alpha, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "alpha.bin"), "wb") as f:
        f.write(struct.pack("f", alpha))


def _export_attention_params(attn, prefix: str):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    _export_linearfp(attn.k_proj, os.path.join(outpath, "k_proj"))
    _export_linearfp(attn.v_proj, os.path.join(outpath, "v_proj"))
    _export_linearfp(attn.q_proj, os.path.join(outpath, "q_proj"))
    _export_linearfp(attn.o_proj, os.path.join(outpath, "o_proj"))
    qk_bmm_alpha = 1 / math.sqrt(attn.head_dim)
    _export_BMM_F32T(qk_bmm_alpha, os.path.join(outpath, "qk_bmm"))
    _export_rotaryEmbedding(attn.rotary_emb, os.path.join(outpath, "rotary_emb"))


def main():
    """Export an OpenVLA model to TinyChatEngine format."""
    parser = argparse.ArgumentParser(description="export OpenVLA pytorch model to TinyChatEngine format.")
    parser.add_argument("--hf_path", type=str, help="Path to huggingface model hub", default="openvla/openvla-7b")
    parser.add_argument(
        "--output", type=str, help="Output directory of the exported model", default="models/openvla-7b"
    )

    args = parser.parse_args()

    model = AutoModelForVision2Seq.from_pretrained(
        args.hf_path,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print("Start exporting VILA model...")
    _export_model(model, args.output)
    print("Finished exporting VILA model.")


if __name__ == "__main__":
    main()
