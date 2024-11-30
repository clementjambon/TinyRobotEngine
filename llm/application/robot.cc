#include <cstring>
#include <iostream>
#include <map>
#include <string>

#include "Generate.h"
#include "interface.h"

std::map<std::string, int> model_config = {
    {"OpenVLA_7B", OpenVLA_7B}, {"OpenVLA_7B_fake_awq", OpenVLA_7B}, {"DINO_v2", DINO_v2}, {"SIGLIP", SIGLIP}};

std::map<std::string, std::string> model_path = {{"OpenVLA_7B", "models/OpenVLA_7B"},
                                                 {"OpenVLA_7B_fake_awq", "models/OpenVLA_7B_fake_awq"}};

std::map<std::string, int> data_format_list = {
    {"FP32", FP32}, {"INT8", QINT8}, {"INT4", INT4}, {"int4", INT4}, {"fp32", FP32},
};

bool convertToBool(const char* str) {
    if (strcmp(str, "true") == 0 || strcmp(str, "1") == 0) {
        return true;
    } else if (strcmp(str, "false") == 0 || strcmp(str, "0") == 0) {
        return false;
    } else {
        std::cerr << "Error: Invalid boolean value: " << str << std::endl;
        exit(EXIT_FAILURE);
    }
}

int NUM_THREAD = 8;

int main(int argc, char* argv[]) {
    std::string target_model = "OpenVLA_7B";
    std::string target_data_format = "INT4";
    bool instruct = true;
    std::string img_path = "embeds/OpenVLA_7B/0000_projected_patch_embeddings.bin";
    Profiler::getInstance().for_demo = true;

    // Set prompt color
    set_print_yellow();
    std::cout << "TinyRobotEngine: https://github.com/clementjambon/TinyRobotEngine/" << std::endl;

    if (argc >= 3 && argc <= 5) {
        auto target_str = argv[1];
        target_model = argv[1];

        // Number of threads
        if (argc >= 4) {
            NUM_THREAD = atoi(argv[3]);
        }
        // Load image
        if (argc == 5) {
            img_path = argv[4];
        }

        if (model_config.count(target_model) == 0) {
            std::cerr << "Model config:" << target_str << " unsupported" << std::endl;
            std::cerr << "Please select one of the following:";
            for (const auto& k : model_config) {
                std::cerr << k.first << ", ";
            }
            std::cerr << std::endl;
            throw("Unsupported model\n");
        }
        std::cout << "Using model: " << argv[1] << std::endl;

        auto data_format_input = argv[2];
        if (data_format_list.count(data_format_input) == 0) {
            std::cerr << "Data format:" << data_format_input << " unsupported" << std::endl;
            std::cerr << "Please select one of the following: ";
            for (const auto& k : data_format_list) {
                std::cerr << k.first << ", ";
            }
            std::cerr << std::endl;
            throw("Unsupported data format\n");
        }
        target_data_format = argv[2];
        std::cout << "Using data format: " << argv[2] << std::endl;

        std::cout << "Using img: " + img_path << std::endl;
    } else if (argc == 2) {
        auto target_str = argv[1];
        target_model = argv[1];
        if (model_config.count(target_model) == 0) {
            std::cerr << "Model config:" << target_str << " unsupported" << std::endl;
            std::cerr << "Please select one of the following: ";
            for (const auto& k : model_config) {
                std::cerr << k.first << ", ";
            }
            std::cerr << std::endl;
            throw("Unsupported model\n");
        }
        std::cout << "Using model: " << argv[1] << std::endl;

        auto data_format_input = "INT4";
    }
    // DEFAULT
    else {
        target_model = "OpenVLA_7B";
        target_data_format = "INT4";
        std::cout << "Using model: " + target_model << std::endl;
        std::cout << "Using data format: " + target_data_format << std::endl;
        std::cout << "Using img: " + img_path << std::endl;
    }

    // Only OpenVLA_7B is supported now
    if (true) {
        int format_id = data_format_list[target_data_format];

        // Load model
        // TODO: load vision model!
        std::cout << "Loading model... " << std::flush;
        // std::string clip_m_path = model_path["Clip_ViT_Large"];
        std::string llama_m_path = model_path[target_model];

        // int clip_model_id = model_config["Clip_ViT_Large"];
        int llama_model_id = model_config[target_model];

#ifdef MODEL_PREFIX
        llama_m_path = MODEL_PREFIX + llama_m_path;
#endif

        // GREEDY_SEARCH!
        // Generation config
        struct opt_params generation_config;
        generation_config.n_vocab = 32000;
        // NB: action dimension!!!!
        // That's 7 joints in the original implementation
        generation_config.n_predict = 7;

        int prompt_iter = 0;

        if (format_id == INT4) {
            // const struct vit_model_config featurizer_config = vit_model_config();
            // Fp32Dinov2VisionTransformer featurizer_model = Fp32Dinov2VisionTransformer(
            //     llama_m_path + "/vision_backbone/featurizer", get_opt_model_config(model_config["DINO_v2"]));
            // Fp32Dinov2VisionTransformer fused_featurizer_model = Fp32Dinov2VisionTransformer(
            //     llama_m_path + "/vision_backbone/fused_featurizer", get_opt_model_config(model_config["SIGLIP"]));

            llama_m_path = "INT4/" + llama_m_path;
            const struct model_config llama_config = get_opt_model_config(OpenVLA_7B);
            Int4LlamaForCausalLM llama_model = Int4LlamaForCausalLM(llama_m_path, llama_config);

            // Get input from the user
            while (true) {
                std::string input;
                std::string input_prefix = "In: What action should the robot take to ";
                // Set prompt color
                set_print_yellow();
                std::cout << std::endl;
                // std::cout << "USER: ";
                // set user input color
                set_print_red();
                std::cout << input_prefix;
                std::getline(std::cin, input);
                // Don't forget to append this
                input += "\nOut: ";
                // reset color
                set_print_reset();

                OpenVLAGenerate_Input generation_input(llama_config, llama_m_path, &llama_model,
                                                       "models/llama_vocab.bin", input, img_path, generation_config,
                                                       false);

                OpenVLAGenerate_Output output = OpenVLAGenerate(generation_input);

                print_openvla_output(output);

                // Set prompt color
                set_print_yellow();
                Profiler::getInstance().report_internal();
                Profiler::getInstance().reset();
                // Reset make
                set_print_reset();
            }
        } else {
            std::cout << std::endl;
            std::cerr << "At this time, we only support INT4 for OpenVLA_7B." << std::endl;
        }
    }
};
