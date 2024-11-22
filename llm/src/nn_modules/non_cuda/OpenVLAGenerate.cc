#include <signal.h>

#include <sstream>
#include <string>
#include <thread>

#include "Generate.h"
#include "LLaMATokenizer.h"
#include "common.h"
#include "interface.h"
#include "utils.h"

#define STB_IMAGE_IMPLEMENTATION
// #include "stb_image.h"

// struct clip_model_config {
//     int image_size = 336;
//     int patch_size = 14;
//     int num_patches = (image_size / patch_size) * (image_size / patch_size);
//     int num_positions = num_patches + 1;
//     int projection_dim = 768;
//     int mmproj_dim = 4096;
//     // float image_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
//     // float image_std[3] = {0.26862954f, 0.26130258f, 0.27577711f};
//     float image_mean[3] = {0.48145466f, 0.48145466f, 0.48145466f};
//     float image_std[3] = {0.26862954f, 0.26862954f, 0.26862954f};
// };

// struct llava_image_embed {
//     float *embed;
//     int n_image_pos;
// };

// struct clip_image_u8 {
//     int nx;
//     int ny;
//     uint8_t *data = NULL;
//     size_t size;
// };

// struct clip_image_f32 {
//     int nx;
//     int ny;
//     float *data = NULL;
//     size_t size;
// };

// clip_image_u8 *make_clip_image_u8() { return new clip_image_u8(); }
// clip_image_f32 *make_clip_image_f32() { return new clip_image_f32(); }
// void clip_image_u8_free(clip_image_u8 *img) {
//     if (img->data) {
//         delete[] img->data;
//     }
//     delete img;
// }
// void clip_image_f32_free(clip_image_f32 *img) {
//     if (img->data) {
//         delete[] img->data;
//     }
//     delete img;
// }

// static struct llava_image_embed *load_image(std::string image, void *clip_model_ptr, bool is_vila);
// struct llava_image_embed *llava_image_embed_make_with_filename(clip_model_config *clip_config, void *clip_model_ptr,
//                                                                const char *image_path, bool is_vila);
// static bool load_file_to_bytes(const char *path, unsigned char **bytesOut, long *sizeOut);
// struct llava_image_embed *llava_image_embed_make_with_bytes(clip_model_config *clip_config, void *clip_model_ptr,
//                                                             const unsigned char *image_bytes, int image_bytes_length,
//                                                             bool is_vila);
// bool clip_image_load_from_bytes(const unsigned char *bytes, size_t bytes_length, struct clip_image_u8 *img);
// static bool llava_image_embed_make_with_clip_img(clip_model_config *clip_config, void *clip_model_ptr,
//                                                  const clip_image_u8 *img, float **image_embd_out, int
//                                                  *n_img_pos_out, bool is_vila);
// static bool encode_image_with_clip(clip_model_config *clip_config, void *clip_model_ptr, const clip_image_u8 *img,
//                                    float *image_embd, int *n_img_pos, bool is_vila);
// bool clip_image_preprocess(clip_model_config *clip_config, void *clip_model_ptr, const clip_image_u8 *img,
//                            clip_image_f32 *res, const bool pad2square);

// // Function to speak in the background
// static void sayInBackground(const std::string &text) {
//     std::string command = "./application/sts_utils/speak \"" + text + "\"";
//     int result = std::system(command.c_str());
//     (void)result;
// }
struct openvla_image_embed {
    float *embed;
    int n_image_pos;
};

static struct openvla_image_embed *load_image_embed(std::string embed_filename, const int embed_dim);

std::string OpenVLAGenerate(std::string llama_param_path, void *llama_model_ptr, int model_type, std::string text,
                            std::string img_path, const struct opt_params generation_config,
                            const struct model_config model_config, std::string voc_path, bool interactive,
                            bool voicechat, bool is_vila) {
    std::vector<int> last_n_tokens(generation_config.n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    std::vector<int> embd;
    std::vector<int> generate_ids;

    // Tokenize first-part text
    const int max_token = 2048;
    std::vector<int> input_ids(max_token);
    llama_vocab vocab = llama_init_vocab(voc_path.c_str());
    const int n = llama_tokenize(vocab, text.c_str(), input_ids.data(), input_ids.size(), true);
    input_ids.resize(n);

    int n_consumed = 0;
    while ((int)input_ids.size() > n_consumed) {
        embd.push_back(input_ids[n_consumed]);
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(input_ids[n_consumed]);
        ++n_consumed;

        if ((int)embd.size() >= generation_config.n_batch) {
            break;
        }
    }

    bool previous_two_hash = false;
    int break_cnt = 2;
    bool new_prompt = true;
    static bool first_prompt = true;
    static bool has_past_kv = false;
    static std::vector<Matrix3D<float>> past_keys, past_values;
    int n_remain = generation_config.n_predict;
    std::string output;
    while (n_remain != 0 && break_cnt) {
        std::vector<float> logits(generation_config.n_vocab);

        int sqlen = 1;
        if (new_prompt) {
            sqlen = input_ids.size();
        }
        if (model_type == LLaVA_INT4 || model_type == VILA_INT4) {
            Int4LlamaForCausalLM *model = static_cast<Int4LlamaForCausalLM *>(llama_model_ptr);
            struct Int4LlamaForCausalLM_output model_output;
            struct Int4LlamaForCausalLM_input model_input;
            if (has_past_kv) {
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                model_input = {input_ids_mat, past_keys, past_values};
            } else {
                // Load and preprocess image
                // auto start = std::chrono::high_resolution_clock::now();
                auto image_embed = load_image_embed(img_path, model_config.embed_dim);
                // auto end = std::chrono::high_resolution_clock::now();
                // std::chrono::duration<double> elapsed = end - start;
                // std::cout << "Image loading time: " << elapsed.count() << " s\n";
                // TODO(clem): de-hardcode this!
                const int n_image_tokens = 256;
                sqlen = input_ids.size() + n_image_tokens;
                int first_sqlen = input_ids.size();
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, first_sqlen);
                Matrix3D<float> image_embed_mat(image_embed->embed, 1, n_image_tokens, 4096);
                model_input = {input_ids_mat, image_embed_mat};
            }
            if (!new_prompt) STATS_START("Inference latency");
            // auto start = std::chrono::high_resolution_clock::now();
            model_output = model->forward(llama_param_path, model_input);
            // auto end = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> elapsed = end - start;
            // static bool flag = true;
            // if (flag) {
            //     std::cout << "Inference time: " << elapsed.count() << " s\n";
            //     flag = false;
            // }
            if (!new_prompt) STATS_END("Inference latency");
            past_keys = model_output.past_keys;
            past_values = model_output.past_values;
            // memcpy model_ouput.logits[-1] to logits
            memcpy(logits.data(), &model_output.logits.m_data[(sqlen - 1) * generation_config.n_vocab],
                   generation_config.n_vocab * sizeof(float));
        } else if (model_type == LLaVA_FP32 || model_type == VILA_FP32) {
            Fp32LlamaForCausalLM *model = static_cast<Fp32LlamaForCausalLM *>(llama_model_ptr);
            struct Fp32LlamaForCausalLM_output model_output;
            struct Fp32LlamaForCausalLM_input model_input;
            if (has_past_kv) {
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, sqlen);
                model_input = {input_ids_mat, past_keys, past_values};
            } else {
                // auto image_embed = load_image(img_path, clip_model_ptr, is_vila);
                // TODO(clem): de-hardcode this!
                const int n_image_tokens = 256;
                auto image_embed = load_image_embed(img_path, model_config.embed_dim);
                sqlen = input_ids.size() + n_image_tokens;
                int first_sqlen = input_ids.size();
                Matrix3D<int> input_ids_mat(input_ids.data(), 1, 1, first_sqlen);
                Matrix3D<float> image_embed_mat(image_embed->embed, 1, n_image_tokens, 4096);
                model_input = {input_ids_mat, image_embed_mat};
            }
            if (!new_prompt) STATS_START("Inference latency");
            model_output = model->forward(model_input);
            if (!new_prompt) STATS_END("Inference latency");
            past_keys = model_output.past_keys;
            past_values = model_output.past_values;
            // memcpy model_ouput.logits[-1] to logits
            memcpy(logits.data(), &model_output.logits.m_data[(sqlen - 1) * generation_config.n_vocab],
                   generation_config.n_vocab * sizeof(float));
        }
        has_past_kv = true;

        if (first_prompt) {
            break;
        }

        // Generate
        const int n_ctx = generation_config.n_ctx;
        const float temp = generation_config.temp;
        const int32_t top_k = generation_config.top_k <= 0 ? generation_config.n_vocab : generation_config.top_k;
        const float top_p = generation_config.top_p;
        const float tfs_z = generation_config.tfs_z;
        const float typical_p = generation_config.typical_p;
        const int32_t repeat_last_n = generation_config.repeat_last_n < 0 ? n_ctx : generation_config.repeat_last_n;
        const float repeat_penalty = generation_config.repeat_penalty;
        const float alpha_presence = generation_config.presence_penalty;
        const float alpha_frequency = generation_config.frequency_penalty;
        const int mirostat = generation_config.mirostat;
        const float mirostat_tau = generation_config.mirostat_tau;
        const float mirostat_eta = generation_config.mirostat_eta;
        const int n_vocab = generation_config.n_vocab;

        std::vector<OPT_token_data> candidates;
        candidates.reserve(n_vocab);
        for (int token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(OPT_token_data{token_id, logits[token_id], 0.0f});
        }

        OPT_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

        // Apply penalties
        auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
        sample_repetition_penalty(&candidates_p, last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                  last_n_repeat, repeat_penalty);
        sample_frequency_and_presence_penalties(&candidates_p,
                                                last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                last_n_repeat, alpha_frequency, alpha_presence);

        int id = 0;
        if (temp <= 0) {
            id = sample_token_greedy(&candidates_p);
        } else {
            if (mirostat == 1) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                const int mirostat_m = 100;
                sample_temperature(&candidates_p, temp);
                id =
                    sample_token_mirostat(n_vocab, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
            } else if (mirostat == 2) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                sample_temperature(&candidates_p, temp);
                id = sample_token_mirostat_v2(&candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
            } else {
                // Temperature sampling
                sample_top_k(&candidates_p, top_k, 1);
                sample_tail_free(&candidates_p, tfs_z, 1);
                sample_typical(&candidates_p, typical_p, 1);
                sample_top_p(&candidates_p, top_p, 1);
                sample_temperature(&candidates_p, temp);
                id = sample_token(&candidates_p);
            }
        }

        if (id == 2) {
            break_cnt--;
            continue;
        }  // eos
        else if (id == 1)
            continue;
        break_cnt = 2;

        bool skip = false;
        if (id == 2277 && !previous_two_hash) {
            previous_two_hash = true;
            skip = true;
        } else if (previous_two_hash && id == 29937) {  // token = #
            break_cnt = 0;
            skip = true;
        } else {
            if (previous_two_hash) std::cout << "##" << std::endl;
            previous_two_hash = false;
        }

        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
        embd.push_back(id);
        generate_ids.push_back(id);
        input_ids = std::vector<int>{id};

        if (interactive && !skip) {
            output += llama_id_to_token(vocab, id);
            std::cout << llama_id_to_token(vocab, id) << std::flush;
        }

        new_prompt = false;
        --n_remain;
    }

    if (interactive && !first_prompt) {
        std::cout << std::endl;
    }
    first_prompt = false;

    // Set prompt color
    set_print_yellow();
    Profiler::getInstance().report_internal();
    Profiler::getInstance().reset();
    // Reset color
    set_print_reset();

    return output;
}

/*
The codes below for image preprocessing are adapted from llama.cpp:
https://github.com/ggerganov/llama.cpp
*/

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
