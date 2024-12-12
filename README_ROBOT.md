# TinyRobotEngine 🤖: On-Device VLA Model Inference

## CLARIFICATIONS

Before getting any further, please note that this repo was designed for **executing OpenVLA on a Mac (ARM)**.

## Walkthrough

### Setup

Please start by following the initial installation instructions of TinyChatEngine.

Compile the executable using
```shell
make robot -j
```

In your conda environment, install OpenVLA. This is necessary to produce embeddings and run tests.
```shell
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .
```

### Demo

You can directly download prequantized INT4 weights (with AWQ) [here](https://drive.google.com/file/d/1P_L6UzYWV5IuX1QSr3KVEXcSQ96nPYOL/view?usp=sharing) and copy them in `llm/INT4/models/OpenVLA_7B_fake_awq/`. You can also download the projected vision embeddings [here](https://drive.google.com/file/d/1QzYbbwMJjj-VI4rxx7aH5TJJ9uCGkDNY/view?usp=sharing) and copy them in `llm/embeds/OpenVLA_7B`. Please follow [Pre-computing image embeddings](#precomputing-image-embeddings) if you want to do this manully.

With this, you can try TinyRobotEngine using:
```shell
./robot OpenVLA_7B_fake_awq INT4 8 embeds/OpenVLA_7B/0000_projected_patch_embeddings.bin

# More generally:
# ./robot OpenVLA_7B_PATH INT4 NUM_THREADS EMBED_PATH
```

Feel free to use other embeddings, check [Precomputing Image Embeddings](#precomputing-image-embeddings) for that. You can also quantize you own weights as described in [Converting weights to binaries](#converting-weights-to-binaries) and [Quantizing to INT4](#quantizing-to-int4-for-arm).

### Profile inference

You can test and profile (LLM) inference using
```shell
make test_OpenVLAGenerate -j
./test_OpenVLAGenerate
make profile_OpenVLAGenerate -j
./profile_OpenVLAGenerate
```

NB: To do this, please make sure you precomputed **ALL** image embeddings from [our subset of the OpenVLA dataset](https://drive.google.com/file/d/1SVoF6u_8pmx5sPWcj4bXETbRlflmFlbZ/view?usp=drive_link) or downloaded them from [here](https://drive.google.com/file/d/1idtrAgZ99IvgVVKQ4S9QPcWRmMJYRZ0U/view?usp=sharing) and copied them following the instructions in [Demo](#demo).

These scripts will compute the **average (token-wise) accuracy** and the **average time it takes to infer the 7 action tokens** on your machine.

NB: You may want to increase the number of open files that can be open with (for example)
```shell
ulimit -n 1000000
```

## Advanced Walkthrough

### Converting weights to binaries

Once you're done, download the weights of the OpenVLA model and convert them to binaries using
```shell
cd llm 
python tools/openvla_exporter.py
```

By default, this will download the [openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b) model from HuggingFace and store the corresponding weights to `models/OpenVLA_7B`.

If you want to load a "fake" quantized network from our [openvla-awq](https://github.com/seanxzhan/llm-awq) pipeline, make sure to copy the corresponding weights `models/openvla-7b-pseudo.pt` and run
```shell
python tools/openvla_exporter.py --lm_path models/openvla-7b-pseudo.pt --output models/OpenVLA_7B_fake_awq
```
The corresponding weights will be saved in `models/OpenVLA_7B_fake_awq`.

NB: we currently export the weights of the vision backbone but inference isn't fully working yet.

### Quantizing to INT4 (for ARM)

You can then quantize the corresponding weights to INT4 using the SIMD-aware weight packing described in [Section 4.2](https://arxiv.org/pdf/2306.00978) of the AWQ paper with
```shell
# For the base model:
python tools/model_quantizer.py --model_path models/OpenVLA_7B
# For the AWQ-scaled model:
python tools/model_quantizer.py --model_path models/OpenVLA_7B_fake_awq
```
The resulting weights will be stored in `INT4/models/OpenVLA_7B` (resp. `INT4/models/OpenVLA_7B_fake_awq`).

### Precomputing image embeddings

We currently do NOT support the vision backbone (although an unsafe/unverified implementation of it exists in this repo). As a consequence, you need to precompute embeddings. We provide a script to do so. Please download an OpenVLA dataset (e.g, [this one](https://drive.google.com/file/d/1SVoF6u_8pmx5sPWcj4bXETbRlflmFlbZ/view?usp=drive_link)) and unzip it into `./datasets`. Then, run
```shell
python tools/openvla/export_image_tokens_dataset.py (--hf_path {HF_PATH} --n_max {NB_DATAPOINTS} --dataset_name {DATASET_NAME} --output {OUTPUT_PATH})
```

By default, this will export embeddings as `embeds/OpenVLA_7B/0000_projected_patch_embeddings.bin` (where `0000` corresponds to a data point). 
NB: we also export other embeddings for unit tests (use `--export_logits` to get the corresponding logits).

If you want to export the full dataset, use `--n_max -1`.

### Interactive mode

With this, you can try run inference using with the interactive mode
```shell
./robot (OpenVLA_7B INT4 NUM_THREADS EMBED_PATH)
```
where `EMBED_PATH` is the path of the embeddings you precomputed before.

## Findings about generation

Below, we describe how we analyzed OpenVLA's data processing in order to match generation within TinyRobotEngine.

* When called with `"In: What action should the robot take to pick up the blue fork and place it on the left of the pot?\nOut:"` (through `basic_inference.py`), the model will call `self.generate` with `'<s> In: What action should the robot take to pick up the blue fork and place it on the left of the pot?\nOut: '` (exactly. Notice the empty token that was added at the end. More on this further down.).
* The forward function of `PrismaticForConditionalGeneration` is then called with `input_ids=tensor([[    1,   512, 29901,  1724,  3158,   881,   278, 19964,  2125,   304,
          5839,   701,   278,  7254, 27350,   322,  2058,   372,   373,   278,
          2175,   310,   278,  3104, 29973,    13,  3744, 29901, 29871]])` (29 tokens) which yield `input_embeddings` with shape `torch.Size([1, 29, 4096])`. `projected_patch_embeddings` has shape `torch.Size([1, 256, 4096])`.
* All of them are concatenated to produce a `multimodal_embeddings` of shape `torch.Size([1, 285, 4096])` (where `29 + 256=285`) but the start token `<s>` is placed **before the image embedding** and the `multimodal_attention_mask` has shape `torch.Size([1, 284])`.
* For generation, we have `max_new_tokens=7` (easy), `greedy_search=GenerationMode.GREEDY_SEARCH` = no sampling!
* The special empty token 29871 has to be inserted after `"Out: "`. See [this](https://github.com/openvla/openvla/blob/0214a0c7c09942fb8e0ec3c3948c00e4e8949911/prismatic/extern/hf/modeling_prismatic.py#L510): `The special empty token ('') does not already appear after the colon (':') token in the prompt (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time`.
