#include <iostream>

#include "Generate.h"
#include "LLaMATokenizer.h"
#include "interface.h"
#include "utils_memalloc.h"

int NUM_THREAD = 8;

std::string format_number(std::string placeholder, int value) {
    char buff[256];
    snprintf(buff, sizeof(buff), placeholder.c_str(), value);
    std::string buffAsStdStr = buff;
    return buffAsStdStr;
}

int main() {
    MemoryAllocator mem_buf;
    std::string voc_path = "models/llama_vocab.bin";

    // ===============
    // Load model(s)
    // ===============

    // Load model
    std::string llama_param_path = "INT4/models/OpenVLA_7B";
    const struct model_config llama_config = get_opt_model_config(OpenVLA_7B);
    Int4LlamaForCausalLM llama_model = Int4LlamaForCausalLM(llama_param_path, llama_config);

    // Generation config
    struct opt_params generation_config;
    generation_config.n_vocab = 32000;
    // NB: action dimension!!!!
    // That's 7 joints in the original implementation
    generation_config.n_predict = 7;

    // Not used
    struct vit_model_config featurizer_config;

    // ===============
    // Load data
    // ===============
    int sample_idx = 0;

    std::string info_path = format_number("embeds/OpenVLA_7B/%04d_info.bin", sample_idx);
    std::string input_ids_path = format_number("embeds/OpenVLA_7B/%04d_input_ids.bin", sample_idx);
    std::string img_embed_path = format_number("embeds/OpenVLA_7B/%04d_projected_patch_embeddings.bin", sample_idx);
    std::string action_path = format_number("embeds/OpenVLA_7B/%04d_action_gt.bin", sample_idx);

    // Load Input ids
    Matrix3D<int> gt_info(mem_buf.get_intbuffer(1), 1, 1, 1);
    gt_info.load(info_path.c_str());
    int n_ids = gt_info(0, 0, 0);
    Matrix3D<int> gt_input_ids(mem_buf.get_intbuffer(n_ids), 1, 1, n_ids);
    gt_input_ids.load(input_ids_path.c_str());
    std::cout << "Input IDs: ";
    for (int i = 0; i < gt_input_ids.length(); ++i) {
        std::cout << gt_input_ids(0, 0, i) << "; ";
    }
    std::cout << std::endl;
    // DEBUG: Detokenize
    std::string detokenized;
    llama_vocab vocab = llama_init_vocab(voc_path.c_str());
    for (int i = 0; i < gt_input_ids.length(); ++i) {
        detokenized += llama_id_to_token(vocab, gt_input_ids(0, 0, i));
    }
    std::cout << "Detokenized: " << detokenized << std::endl;

    // Load GT action ids
    Matrix3D<int> gt_action(mem_buf.get_intbuffer(7), 1, 1, 7);
    gt_action.load(action_path.c_str());
    std::cout << "Action IDs: ";
    for (int i = 0; i < gt_action.length(); ++i) {
        std::cout << gt_action(0, 0, i) << std::endl;
    }
    std::cout << std::endl;

    // ===============
    // Inference
    // ===============

    for (int i = 0; i < 10; ++i) {
        OpenVLAGenerate_Input input(llama_config, llama_param_path, &llama_model, voc_path,
                                    "In: What action should the robot take to pick up the blue fork and place it on "
                                    "the left of the pot?\nOut: ",
                                    img_embed_path, generation_config);
        OpenVLAGenerate_Output output = OpenVLAGenerate(input);

        print_openvla_output(output);
    }

    // Set prompt color
    set_print_yellow();
    Profiler::getInstance().report_internal();
    Profiler::getInstance().reset();
    // Reset make
    set_print_reset();
}
