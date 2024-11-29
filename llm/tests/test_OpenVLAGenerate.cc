#include <iostream>

#include "Generate.h"

int NUM_THREAD = 8;

int main() {
    std::string llama_param_path = "INT4/models/OpenVLA_7B";
    std::string img_path = "embeds/OpenVLA_7B/0000_projected_patch_embeddings.bin";
    struct model_config llama_config = get_opt_model_config(OpenVLA_7B);

    // Load model
    Int4LlamaForCausalLM llama_model = Int4LlamaForCausalLM(llama_param_path, llama_config);

    // Generation config
    struct opt_params generation_config;
    generation_config.top_k = 0;
    generation_config.temp = 1.0f;
    generation_config.n_vocab = 32000;
    // NB: action dimension!!!!
    // That's 7 joints in the original implementation
    generation_config.n_predict = 7;

    // Not used
    struct vit_model_config featurizer_config;

    std::vector<float> output = OpenVLAGenerate(llama_param_path, &llama_model, featurizer_config, NULL, LLaVA_INT4,
                                                "What action should the robot take to move the blocks?", img_path,
                                                generation_config, llama_config, "models/llama_vocab.bin", false);

    std::cout << "generated:" << output.size() << std::endl;
    for (auto it = output.begin(); it != output.end(); ++it) {
        std::cout << *it << std::endl;
    };
}
