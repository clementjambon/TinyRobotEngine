# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
import os
from argparse import ArgumentParser
from typing import List, Any

import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer

from PIL import Image
import requests

import torch
import time

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def save_tensor(weights: torch.Tensor, output: str, name: str, is_int: bool = False):
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, f"{name}.bin"), "wb") as f:
        if is_int:
            f.write(weights.cpu().int().numpy().tobytes())
        else:
            f.write(weights.cpu().float().numpy().tobytes())


def save_info(list: List[int], output: str, name: str):
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, f"{name}.bin"), "wb") as f:
        for x in list:
            f.write(x.to_bytes(4, byteorder="little"))


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained(args.hf_name, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.hf_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    vocab_size = processor.tokenizer.vocab_size

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in args.hf_name else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        args.data_root_dir,
        args.dataset_name,
        batch_transform,
        resize_resolution=(224, 224),  # 224 is hard coded, originally tuple(vla.module.config.image_sizes)
        shuffle_buffer_size=100_000,
        image_aug=False,
    )

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=args.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Evaluate!
    with tqdm.tqdm(total=args.n_max if args.n_max > 0 else len(vla_dataset), leave=False) as progress:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * args.batch_size >= args.n_max:
                break

            pixels_values = batch["pixel_values"].to(torch.bfloat16).to(device)
            attention_mask = batch["attention_mask"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"]

            action_gt = labels[:, -8:-1]
            save_tensor(action_gt.cpu(), args.output, f"{batch_idx:04d}_action_gt", is_int=True)

            # Extract patch features
            patch_features = vla.vision_backbone(pixels_values)

            # Project
            projected_patch_embeddings = vla.projector(patch_features)
            save_tensor(projected_patch_embeddings.cpu(), args.output, f"{batch_idx:04d}_projected_patch_embeddings")

            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # Get Input Embeddings (from Language Model Embeddings)
            save_tensor(input_ids, args.output, f"{batch_idx:04d}_input_ids", is_int=True)
            save_info([input_ids.shape[-1]], args.output, f"{batch_idx:04d}_info")
            input_embeddings = vla.get_input_embeddings()(input_ids)

            # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )
            save_tensor(multimodal_embeddings.cpu(), args.output, f"{batch_idx:04d}_multimodal_embeddings")

            if args.export_logits:
                multimodal_attention_mask = None
                if attention_mask is not None:
                    multimodal_attention_mask = torch.cat(
                        [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                    )

                # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
                #   => We'll ignore the per-token outputs for each of the patch embeddings as well!
                multimodal_labels = None
                if labels is not None:
                    projected_patch_labels = torch.full(
                        (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                        IGNORE_INDEX,
                        dtype=labels.dtype,
                        device=labels.device,
                    )
                    multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)
                    multimodal_labels = torch.cat(
                        [labels[multimodal_indices, :1], projected_patch_labels, labels[multimodal_indices, 1:]], dim=1
                    )

                llm_output = vla.language_model(
                    input_ids=None,
                    attention_mask=multimodal_attention_mask,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=multimodal_embeddings,
                    labels=multimodal_labels,
                    use_cache=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None,
                )

                save_tensor(llm_output.logits[..., :vocab_size].cpu(), args.output, f"{batch_idx:04d}_logits")

            progress.update()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf_name", type=str, default="openvla/openvla-7b")
    parser.add_argument("--output", type=str, default="embeds/OpenVLA_7B")
    parser.add_argument("--data_root_dir", type=str, default="./datasets")
    parser.add_argument("--dataset_name", type=str, default="bridge_orig")
    parser.add_argument("--n_max", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--export_logits", action="store_true")

    args = parser.parse_args()
    main(args)
