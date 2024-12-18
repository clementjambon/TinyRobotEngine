#ifndef MODEL_H
#define MODEL_H
#include <cstring>

struct model_config {
    int batch;
    int num_heads;
    int num_kv_heads;
    int num_layers;
    int max_sqlen;
    int embed_dim;
    int hidden_dim;
    int vocsize;
    int padding_idx;
    float rms_norm_eps;  // RMSNorm epsilon (only for LLaMA models)
    // Below are for Clip models
    int image_size;
    int patch_size;
    int projection_dim;
    int mmproj_dim;
    // Below is for ViT with registers
    int num_registers;
    bool class_token;
    bool layer_scale;

    model_config() : model_config(1, 32, 32, 2048, 4096, 11008, 32000, 1, 1e-6, 0, 0, 0, 0) {}
    model_config(int batch, int num_heads, int num_layers, int max_sqlen, int embed_dim, int hidden_dim, int vocsize,
                 int padding_idx, float rms_norm_eps)
        : batch(batch),
          num_heads(num_heads),
          num_layers(num_layers),
          max_sqlen(max_sqlen),
          embed_dim(embed_dim),
          hidden_dim(hidden_dim),
          vocsize(vocsize),
          padding_idx(padding_idx),
          rms_norm_eps(rms_norm_eps) {}
    // GQA/MQA models
    model_config(int batch, int num_heads, int num_kv_heads, int num_layers, int max_sqlen, int embed_dim,
                 int hidden_dim, int vocsize, int padding_idx, float rms_norm_eps)
        : batch(batch),
          num_heads(num_heads),
          num_kv_heads(num_kv_heads),
          num_layers(num_layers),
          max_sqlen(max_sqlen),
          embed_dim(embed_dim),
          hidden_dim(hidden_dim),
          vocsize(vocsize),
          padding_idx(padding_idx),
          rms_norm_eps(rms_norm_eps) {}
    // Clip models
    model_config(int batch, int num_heads, int num_layers, int max_sqlen, int embed_dim, int hidden_dim, int vocsize,
                 int padding_idx, float rms_norm_eps, int image_size, int patch_size, int projection_dim,
                 int mmproj_dim)
        : batch(batch),
          num_heads(num_heads),
          num_layers(num_layers),
          max_sqlen(max_sqlen),
          embed_dim(embed_dim),
          hidden_dim(hidden_dim),
          vocsize(vocsize),
          padding_idx(padding_idx),
          rms_norm_eps(rms_norm_eps),
          image_size(image_size),
          patch_size(patch_size),
          projection_dim(projection_dim),
          mmproj_dim(mmproj_dim),
          class_token(true),
          layer_scale(false) {}
    // ViT with registers
    model_config(int batch, int num_heads, int num_layers, int max_sqlen, int embed_dim, int hidden_dim, int vocsize,
                 int padding_idx, float rms_norm_eps, int image_size, int patch_size, int projection_dim,
                 int mmproj_dim, int num_registers, bool class_token, bool layer_scale)
        : batch(batch),
          num_heads(num_heads),
          num_layers(num_layers),
          max_sqlen(max_sqlen),
          embed_dim(embed_dim),
          hidden_dim(hidden_dim),
          vocsize(vocsize),
          padding_idx(padding_idx),
          rms_norm_eps(rms_norm_eps),
          image_size(image_size),
          patch_size(patch_size),
          projection_dim(projection_dim),
          mmproj_dim(mmproj_dim),
          num_registers(num_registers),
          class_token(class_token),
          layer_scale(layer_scale) {}
};

struct vit_model_config {
    int image_size = 224;
    int patch_size = 14;
    int num_patches = (image_size / patch_size) * (image_size / patch_size);
    // + registers (= 4 for dino-v2) + class embeding (=0 for dino-v2)
    int num_positions = num_patches + 4;
    int projection_dim = 1024;
    // int mmproj_dim = 4096;
    // float image_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    // float image_std[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    // IMAGENET_STATS
    float image_mean[3] = {0.485f, 0.456f, 0.406f};
    float image_std[3] = {0.229f, 0.224f, 0.225f};
};

enum {
    OPT_125M,
    OPT_1_3B,
    OPT_6_7B,
    LLaMA_7B,
    LLaMA_13B,
    CodeLLaMA_7B,
    CodeLLaMA_13B,
    StarCoder_15_5B,
    LLaVA_7B,
    LLaVA_13B,
    VILA_2_7B,
    VILA_7B,
    VILA_13B,
    Clip_ViT_Large,
    Mistral_7B,
    LLaMA_3_8B,
    VILA1_5_8B,
    OpenVLA_7B,
    DINO_v2,
    SIGLIP,
};
enum { FP32, QINT8, INT4 };

const struct model_config opt_6_7B(1, 32, 32, 2048, 4096, 16384, 50272, 1, 0);
const struct model_config opt_1_3B(1, 32, 24, 2048, 2048, 8192, 50272, 1, 0);
const struct model_config opt_125m(1, 12, 12, 2048, 768, 3072, 50272, 1, 0);
const struct model_config llama_7B(1, 32, 32, 32, 2048, 4096, 11008, 32000, 1, 1e-6);
const struct model_config llama_13B(1, 40, 40, 40, 2048, 5120, 13824, 32000, 1, 1e-6);
const struct model_config codellama_7B(1, 32, 32, 32, 2048, 4096, 11008, 32016, 1, 1e-5);
const struct model_config codellama_13B(1, 40, 40, 40, 2048, 5120, 13824, 32016, 1, 1e-5);
const struct model_config starcoder_15_5B(1, 48, 40, 2048, 6144, 24576, 49152, 1, 0);
const struct model_config llava_7B(1, 32, 32, 32, 2048, 4096, 11008, 32000, 1, 1e-5);
const struct model_config llava_13B(1, 40, 40, 40, 2048, 5120, 13824, 32000, 1, 1e-5);
const struct model_config vila_2_7B(1, 20, 20, 32, 2048, 2560, 6912, 32000, 1, 1e-5);
const struct model_config vila_7B(1, 32, 32, 32, 2048, 4096, 11008, 32000, 1, 1e-5);
const struct model_config vila_13B(1, 40, 40, 40, 2048, 5120, 13824, 32000, 1, 1e-5);
const struct model_config clip_vit_large(1, 16, 23, 2048, 1024, 4096, 0, 1, 0, 336, 14, 768,
                                         4096);  // llava's and vila's clip model uses only 23 layers out of 24
const struct model_config mistral_7B(1, 32, 8, 32, 2048, 4096, 14336, 32000, 1, 1e-5);
const struct model_config llama_3_8B(1, 32, 8, 32, 2048, 4096, 14336, 128256, 1, 1e-5);
// Copied from LLAVA (TODO(clem): check that)
const struct model_config openvla_7B(1, 32, 32, 32, 2048, 4096, 11008, 32000, 1, 1e-5);
// Vision towers following
// https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
// NOTE: hidden_dim = 4 * embed_dim, voc_size = 0,
// TODO: no_embed tokens, no pre_norm, don't forget additional bias
// Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
const struct model_config vit_large_patch14_reg4_dinov2(1, 16, 24, 2048, 1024, 4096, 0, 1, 0, 224, 14, 1024, 4096, 4,
                                                        true, true);
const struct model_config vit_so400m_patch14_siglip_224(1, 16, 26, 2048, 1152, 4304, 0, 1, 0, 224, 14, 1024, 4096, 0,
                                                        false, false);

static struct model_config get_opt_model_config(int choise) {
    struct model_config ret;
    switch (choise) {
        case OPT_125M:
            ret = opt_125m;
            break;
        case OPT_1_3B:
            ret = opt_1_3B;
            break;
        case OPT_6_7B:
            ret = opt_6_7B;
            break;
        case LLaMA_7B:
            ret = llama_7B;
            break;
        case LLaMA_13B:
            ret = llama_13B;
            break;
        case CodeLLaMA_7B:
            ret = codellama_7B;
            break;
        case CodeLLaMA_13B:
            ret = codellama_13B;
            break;
        case StarCoder_15_5B:
            ret = starcoder_15_5B;
            break;
        case LLaVA_7B:
            ret = llava_7B;
            break;
        case LLaVA_13B:
            ret = llava_13B;
            break;
        case VILA_2_7B:
            ret = vila_2_7B;
            break;
        case VILA_7B:
            ret = vila_7B;
            break;
        case VILA_13B:
            ret = vila_13B;
            break;
        case Clip_ViT_Large:
            ret = clip_vit_large;
            break;
        case Mistral_7B:
            ret = mistral_7B;
            break;
        case LLaMA_3_8B:
            ret = llama_3_8B;
            break;
        case VILA1_5_8B:
            ret = vila_7B;
            break;
        case OpenVLA_7B:
            ret = openvla_7B;
            break;
        case DINO_v2:
            ret = vit_large_patch14_reg4_dinov2;
            break;
        case SIGLIP:
            ret = vit_so400m_patch14_siglip_224;
            break;
        default:
            throw("Unsupported model choice.");
            break;
    }
    return ret;
}

#endif
