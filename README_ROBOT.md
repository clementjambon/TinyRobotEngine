# TinyRobotEngine 🤖: On-Device VLA Model Inference

## CLARIFICATIONS

Before getting any further, please not that this repo was designed for **executing OpenVLA on a Mac**.

## Walkthrough

### Setup

Please start by following the initial installation instructions of TinyChatEngine. Make sure that `nlohmann-json` is properly installed.

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

NB: we currently export the weights of the vision backbone but inference isn't fully operational yet (TODO(Clement)).

### Quantizing to INT4 (for ARM)

You can then quantize the corresponding weights to INT4 using the SIMD-aware weight packing described in [Section 4.2](https://arxiv.org/pdf/2306.00978) of the AWQ paper with
```shell
# For the base model:
python tools/model_quantizer.py --model_path models/OpenVLA_7B
# For the AWQ-scaled model:
python tools/model_quantizer.py --model_path models/OpenVLA_7B_fake_awq
```
The resulting weights will be stored in `INT4/models/OpenVLA_7B` (resp. `INT4/models/OpenVLA_7B_fake_awq`).

### Pre-computing image embeddings

We currently do NOT support the vision backbone (although an unsafe/unverified implementation of it exists in this repo). As a consequence, you need to precompute embeddings. We provide a script to do so. Please download an OpenVLA dataset (e.g, [this one](https://drive.google.com/file/d/1SVoF6u_8pmx5sPWcj4bXETbRlflmFlbZ/view?usp=drive_link)) and unzip it into `./datasets`. Then, run
```shell
python tools/openvla/export_image_tokens_dataset.py (--hf_path {HF_PATH} --n_max {NB_DATAPOINTS} --dataset_name {DATASET_NAME} --output {OUTPUT_PATH})
```

By default, this will export embeddings as `embeds/OpenVLA_7B/0000_projected_patch_embeddings.bin` (where `0000` corresponds to a data point). 
NB: we also export other embeddings for unit tests.

### Inference

With this, you can then run inference using
```shell
./robot (OpenVLA_7B INT4 NUM_THREADS EMBED_PATH)
```

## Profile inference

You can profile inference using
```shell
make profile_OpenVLAGenerate -j
./profile_OpenVLAGenerate
```