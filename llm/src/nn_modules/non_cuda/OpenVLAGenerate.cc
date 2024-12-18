#include <signal.h>

#include <sstream>
#include <string>
#include <thread>

#include "Generate.h"
#include "LLaMATokenizer.h"
#include "OpenVLAActionDetokenizer.h"
#include "clip_utils.h"
#include "common.h"
#include "interface.h"
#include "utils.h"

// TODO: that's only for dino for now!
struct openvla_image_embed {
    float *embed;
    int n_image_pos;
};

// Image embedding
// Load pre-computed embeddings
static struct openvla_image_embed *load_image_embed(std::string embed_filename, const int embed_dim);

// Run inference on vision backbones
static struct openvla_image_embed *load_image(std::string image, const vit_model_config *vit_config,
                                              void *vit_model_ptr);
struct openvla_image_embed *vit_image_embed_make_with_filename(const vit_model_config *vit_config, void *vit_model_ptr,
                                                               const char *image_path);
struct openvla_image_embed *vit_image_embed_make_with_bytes(const vit_model_config *vit_config, void *vit_model_ptr,
                                                            const unsigned char *image_bytes, int image_bytes_length);
static bool vit_image_embed_make_with_clip_img(const vit_model_config *vit_config, void *clip_model_ptr,
                                               const clip_image_u8 *img, float **image_embd_out, int *n_img_pos_out);
static bool encode_image_with_vit(const vit_model_config *vit_config, void *vit_model_ptr, const clip_image_u8 *img,
                                  float *image_embd, int *n_img_pos);
bool vit_image_preprocess(const vit_model_config *vit_config, void *vit_model_ptr, const clip_image_u8 *img,
                          clip_image_f32 *res, const bool pad2square);

// Clip value between a and b
static float clip(const float value, const float lower, const float upper);

void print_openvla_output(struct OpenVLAGenerate_Output output) {
    for (int i = 0; i < output.generated_ids.size(); ++i) {
        std::cout << "[" << i << "] " << output.generated_ids.at(i) << " ---> " << output.generated_actions.at(i)
                  << std::endl;
    };
}

struct OpenVLAGenerate_Output OpenVLAGenerate(struct OpenVLAGenerate_Input &input) {
    if (input.verbose) {
        if (input.input_ids.length() > 0) {
        } else {
        }
        std::cout << "Received text: " << input.text << std::endl;
    }

    std::vector<int> generated_ids;
    std::vector<float> generated_actions;

    // Create the Action Detokenizer
    const action_stats *act_stats = new action_stats();

    // =================
    // Tokenize (or not)
    // =================
    const int max_token = 2048;
    std::vector<int> input_ids(max_token);
    if (input.input_ids.length() > 0) {
        const int n = input.input_ids.length();
        for (int i = 0; i < input.input_ids.length(); ++i) {
            input_ids.at(i) = input.input_ids(0, 0, i);
        }
        input_ids.resize(n);
    } else {
        llama_vocab vocab = llama_init_vocab(input.voc_path.c_str());
        const int n = llama_tokenize(vocab, input.text.c_str(), input_ids.data(), input_ids.size(), true);
        input_ids.resize(n);
    }

    // cf.
    // https://github.com/openvla/openvla/blob/0214a0c7c09942fb8e0ec3c3948c00e4e8949911/prismatic/extern/hf/modeling_prismatic.py#L510
    assert(input_ids.at(input_ids.size() - 1) == 29871);
    if (input.verbose) {
        std::cout << "Input IDs(" << input_ids.size() << "): ";
        for (int i = 0; i < input_ids.size(); ++i) {
            std::cout << input_ids.at(i) << "; ";
        }
        std::cout << std::endl;
    }

    // =================
    // Iterate
    // =================

    int action_idx = 0;
    int break_cnt = 2;
    bool has_past_kv = false;
    std::vector<Matrix3D<float>> past_keys, past_values;
    int n_remain = input.generation_config.n_predict;

    while (n_remain != 0 && break_cnt) {
        std::vector<float> logits(input.generation_config.n_vocab);

        // This is the number of logits we wish to sample from
        // NB: the first time, we have to pass everyone!
        int sqlen = 1;

        Int4LlamaForCausalLM *model = static_cast<Int4LlamaForCausalLM *>(input.llama_model_ptr);
        struct Int4LlamaForCausalLM_output model_output;
        struct Int4LlamaForCausalLM_input model_input;
        if (has_past_kv) {
            Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
            model_input = Int4LlamaForCausalLM_input(input_ids_mat, past_keys, past_values, true);
        } else {
            // Load and preprocess image or embedding
            auto start = std::chrono::high_resolution_clock::now();
            auto image_embed = load_image_embed(input.img_embed_path, input.model_config.embed_dim);
            // auto image_embed = load_image(img_path, &featurizer_config, featurizer_model_ptr);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            if (input.verbose) {
                std::cout << "Image loading time: " << elapsed.count() << " s\n";
            }
            // TODO(clem): de-hardcode this!
            const int n_image_tokens = 256;
            sqlen = input_ids.size() + n_image_tokens;
            int first_sqlen = input_ids.size();
            Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, first_sqlen);
            Matrix3D<float> image_embed_mat(image_embed->embed, 1, n_image_tokens, 4096);
            model_input = {input_ids_mat, image_embed_mat, true};
        }
        // WARNING: this will also count prefilling!
        STATS_START("Inference latency");
        model_output = model->forward(input.llama_param_path, model_input);
        STATS_END("Inference latency");
        past_keys = model_output.past_keys;
        past_values = model_output.past_values;
        // memcpy model_ouput.logits[-1] to logits
        memcpy(logits.data(), &model_output.logits.m_data[(sqlen - 1) * input.generation_config.n_vocab],
               input.generation_config.n_vocab * sizeof(float));

        // Make sure the KV cache is set after this (always!
        has_past_kv = true;

        // ================
        // Sample (GREEDY)
        // ================
        const int n_vocab = input.generation_config.n_vocab;

        std::vector<OPT_token_data> candidates;
        candidates.reserve(n_vocab);
        for (int token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(OPT_token_data{token_id, logits[token_id], 0.0f});
        }

        OPT_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

        // OpenVLA uses GREEDY_SEARCH
        // = choose most-likely token at each step of sampling
        int id = sample_token_greedy(&candidates_p);

        // ================
        // Process samples
        // ================

        if (id == 2) {
            std::cout << "EOSSSS" << std::endl;
            break_cnt--;
            continue;
        }  // eos
        else if (id == 1)
            continue;
        break_cnt = 2;

        generated_ids.push_back(id);
        // The new input id is just the new token
        input_ids = std::vector<int>{id};

        float action = token_id_to_action(id, action_idx, act_stats);
        generated_actions.push_back(action);

        --n_remain;
        ++action_idx;
    }

    OpenVLAGenerate_Output output{generated_ids, generated_actions};
    return output;
}

// Load precomputed embeddings
static struct openvla_image_embed *load_image_embed(std::string embed_filename, const int embed_dim) {
    const int n_data = 1;
    // TODO: check this!
    const int n_patches = 256;
    // Load image embeddings
    Matrix3D<float> embweight(new float[n_data * embed_dim * n_patches], n_data, n_patches, embed_dim);
    embweight.load(embed_filename.c_str());

    // Create abstract embeddings
    // TODO: load only one!
    auto result = (openvla_image_embed *)malloc(sizeof(openvla_image_embed));
    result->embed = embweight.m_data;
    result->n_image_pos = 0;
    return result;
}

/*
The codes below for image preprocessing are adapted from llama.cpp:
https://github.com/ggerganov/llama.cpp
*/
static struct openvla_image_embed *load_image(std::string image, const vit_model_config *vit_config,
                                              void *vit_model_ptr) {
    // load and preprocess the image
    openvla_image_embed *embed = NULL;
    embed = vit_image_embed_make_with_filename(vit_config, vit_model_ptr, image.c_str());
    if (!embed) {
        fprintf(stderr, "%s: is %s really an image file?\n", __func__, image.c_str());
        return NULL;
    }

    return embed;
}

struct openvla_image_embed *vit_image_embed_make_with_filename(const vit_model_config *vit_config, void *vit_model_ptr,
                                                               const char *image_path) {
    unsigned char *image_bytes;
    long image_bytes_length;
    auto loaded = load_file_to_bytes(image_path, &image_bytes, &image_bytes_length);
    if (!loaded) {
        fprintf(stderr, "%s: failed to load %s\n", __func__, image_path);
        return NULL;
    }

    auto embed = vit_image_embed_make_with_bytes(vit_config, vit_model_ptr, image_bytes, image_bytes_length);
    free(image_bytes);

    return embed;
}

struct openvla_image_embed *vit_image_embed_make_with_bytes(const vit_model_config *vit_config, void *vit_model_ptr,
                                                            const unsigned char *image_bytes, int image_bytes_length) {
    clip_image_u8 *img = make_clip_image_u8();
    if (!clip_image_load_from_bytes(image_bytes, image_bytes_length, img)) {
        clip_image_u8_free(img);
        fprintf(stderr, "%s: can't load image from bytes, is it a valid image?", __func__);
        return NULL;
    }

    float *image_embed = NULL;
    int n_image_pos = 0;
    bool image_embed_result =
        vit_image_embed_make_with_clip_img(vit_config, vit_model_ptr, img, &image_embed, &n_image_pos);
    if (!image_embed_result) {
        clip_image_u8_free(img);
        fprintf(stderr, "%s: coulnd't embed the image\n", __func__);
        return NULL;
    }

    clip_image_u8_free(img);
    auto result = (openvla_image_embed *)malloc(sizeof(openvla_image_embed));
    result->embed = image_embed;
    result->n_image_pos = n_image_pos;
    return result;
}

size_t vit_embd_nbytes(const vit_model_config *vit_config) {
    return vit_config->num_patches * vit_config->projection_dim * sizeof(float);
}

static bool vit_image_embed_make_with_clip_img(const vit_model_config *vit_config, void *vit_model_ptr,
                                               const clip_image_u8 *img, float **image_embd_out, int *n_img_pos_out) {
    float *image_embd = (float *)malloc(vit_embd_nbytes(vit_config));
    if (!image_embd) {
        fprintf(stderr, "Unable to allocate memory for image embeddings\n");
        free(image_embd);
        return false;
    }

    int n_img_pos;
    if (!encode_image_with_vit(vit_config, vit_model_ptr, img, image_embd, &n_img_pos)) {
        fprintf(stderr, "%s: cannot encode image, aborting\n", __func__);
        free(image_embd);
        return false;
    }
    *image_embd_out = image_embd;
    *n_img_pos_out = n_img_pos;

    return true;
}

static bool encode_image_with_vit(const vit_model_config *vit_config, void *vit_model_ptr, const clip_image_u8 *img,
                                  float *image_embd, int *n_img_pos) {
    clip_image_f32 *img_res = make_clip_image_f32();
    if (!vit_image_preprocess(vit_config, vit_model_ptr, img, img_res, /*pad2square =*/true)) {
        fprintf(stderr, "%s: unable to preprocess image\n", __func__);
        clip_image_f32_free(img_res);
        return false;
    }

    Fp32Dinov2VisionTransformer *vit_model = static_cast<Fp32Dinov2VisionTransformer *>(vit_model_ptr);
    struct Fp32Dinov2VisionTransformer_input model_input;
    struct Fp32Dinov2VisionTransformer_output model_output;
    Matrix3D<float> input_image(img_res->data, 3, img_res->nx, img_res->ny);
    model_input = {input_image};
    model_output = vit_model->forward(model_input);
    memcpy(image_embd, model_output.last_hidden_state.m_data, vit_embd_nbytes(vit_config));

    clip_image_f32_free(img_res);

    return true;
}

// normalize: x = (x - mean) / std
// TODO: implement bicubic interpolation instead of linear.
bool vit_image_preprocess(const vit_model_config *vit_config, void *vit_model_ptr, const clip_image_u8 *img,
                          clip_image_f32 *res, const bool pad2square) {
    // the logic below is to pad the shorter side to the longer side with a background color: rgb(122, 116, 104)
    // see
    // https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/conversation.py#L113-L156

    clip_image_u8 *temp = make_clip_image_u8();  // we will keep the input image data here temporarily
    if (pad2square && img->nx != img->ny) {
        int longer_side = std::max(img->nx, img->ny);
        temp->nx = longer_side;
        temp->ny = longer_side;
        temp->size = 3 * longer_side * longer_side;
        temp->data = new uint8_t[temp->size]();
        uint8_t bc[3] = {122, 116, 104};  // bakground color in RGB from LLaVA

        // fill with background color
        for (size_t i = 0; i < temp->size; i++) {
            temp->data[i] = bc[i % 3];
        }

        // copy from the input image
        for (int y = 0; y < img->ny; y++) {
            for (int x = 0; x < img->nx; x++) {
                const int i = 3 * (y * img->nx + x);
                const int j = 3 * (y * temp->nx + x);
                temp->data[j] = img->data[i];
                temp->data[j + 1] = img->data[i + 1];
                temp->data[j + 2] = img->data[i + 2];
            }
        }
    } else {
        temp->nx = img->nx;
        temp->ny = img->ny;
        temp->size = img->size;
        temp->data = new uint8_t[temp->size]();
        memcpy(&temp->data[0], &img->data[0], temp->size);  // copy
    }

    const int nx = temp->nx;
    const int ny = temp->ny;

    const int nx2 = vit_config->image_size;
    const int ny2 = vit_config->image_size;

    res->nx = nx2;
    res->ny = ny2;
    res->size = 3 * nx2 * ny2;
    res->data = new float[res->size]();

    const float scale = std::max(nx, ny) / (float)vit_config->image_size;

    const int nx3 = int(nx / scale + 0.5f);
    const int ny3 = int(ny / scale + 0.5f);

    const auto &m3 = vit_config->image_mean;  // {0.48145466f, 0.4578275f, 0.40821073f};
    const auto &s3 = vit_config->image_std;   // {0.26862954f, 0.26130258f, 0.27577711f};

    for (int y = 0; y < ny3; y++) {
        for (int x = 0; x < nx3; x++) {
            for (int c = 0; c < 3; c++) {
                // linear interpolation
                const float sx = (x + 0.5f) * scale - 0.5f;
                const float sy = (y + 0.5f) * scale - 0.5f;

                const int x0 = std::max(0, (int)std::floor(sx));
                const int y0 = std::max(0, (int)std::floor(sy));

                const int x1 = std::min(x0 + 1, nx - 1);
                const int y1 = std::min(y0 + 1, ny - 1);

                const float dx = sx - x0;
                const float dy = sy - y0;

                const int j00 = 3 * (y0 * nx + x0) + c;
                const int j01 = 3 * (y0 * nx + x1) + c;
                const int j10 = 3 * (y1 * nx + x0) + c;
                const int j11 = 3 * (y1 * nx + x1) + c;

                const float v00 = temp->data[j00];
                const float v01 = temp->data[j01];
                const float v10 = temp->data[j10];
                const float v11 = temp->data[j11];

                const float v0 = v00 * (1.0f - dx) + v01 * dx;
                const float v1 = v10 * (1.0f - dx) + v11 * dx;

                const float v = v0 * (1.0f - dy) + v1 * dy;

                const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);

                const int i = 3 * (y * nx3 + x) + c;

                res->data[i] = ((float(v2) / 255.0f) - m3[c]) / s3[c];
            }
        }
    }
    clip_image_u8_free(temp);

    return true;
}
