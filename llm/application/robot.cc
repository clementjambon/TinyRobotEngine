#include <cstring>
#include <iostream>
#include <map>
#include <string>

#include "Generate.h"
#include "interface.h"

std::map<std::string, int> model_config = {
    {"OpenVLA_7B", OpenVLA_7B},
};

std::map<std::string, std::string> model_path = {{"OpenVLA_7B", "models/openvla-7b"}};

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

int NUM_THREAD = 5;

int main(int argc, char* argv[]) {
    std::string target_model = "OpenVLA_7B";
    std::string target_data_format = "INT4";
    bool instruct = true;
    std::string img_path = "images/monalisa.jpg";
    Profiler::getInstance().for_demo = true;

    // Set prompt color
    set_print_yellow();
    std::cout << "TinyChatEngine by MIT HAN Lab: https://github.com/mit-han-lab/TinyChatEngine" << std::endl;

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
        if (target_data_format == "INT4" || target_data_format == "int4")
            std::cout << "Using AWQ for 4bit quantization: https://github.com/mit-han-lab/llm-awq" << std::endl;
        else
            std::cout << "Using data format: " << argv[2] << std::endl;
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
        target_data_format = "INT8";
        std::cout << "Using model: " + target_model << std::endl;
        std::cout << "Using data format: " + target_data_format << std::endl;
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

        struct opt_params generation_config;
        generation_config.n_predict = 512;
        generation_config.repeat_penalty = 1.1f;
        generation_config.temp = 0.2f;
        generation_config.n_vocab = 32000;

        int prompt_iter = 0;

        if (format_id == FP32) {
            // Fp32CLIPVisionTransformer clip_model =
            //     Fp32CLIPVisionTransformer(clip_m_path, get_opt_model_config(clip_model_id), false);
            Fp32LlamaForCausalLM llama_model = Fp32LlamaForCausalLM(llama_m_path, get_opt_model_config(llama_model_id));

            // Get input from the user
            while (true) {
                std::string input;
                if (prompt_iter == 1) {
                    // Set prompt color
                    set_print_yellow();
                    std::cout << "Finished!" << std::endl << std::endl;
                    // reset color
                    set_print_reset();
                }
                if (prompt_iter > 0) {
                    if (true) {
                        // Set prompt color
                        set_print_yellow();
                        std::cout << "USER: ";
                        // set user input color
                        set_print_red();
                        std::getline(std::cin, input);
                        // reset color
                        set_print_reset();
                    }
                    if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.") break;
                    std::cout << "ASSISTANT: ";
                }

                if (prompt_iter == 0) {
                    input = "This is a chat between a user and an assistant.\n\n### USER: ";
                    prompt_iter += 1;
                } else if (prompt_iter == 1) {
                    input = "\n" + input + "\n### ASSISTANT:";
                    prompt_iter += 1;
                } else {
                    input = "### USER: " + input + "\n### ASSISTANT: \n";
                }

                // LLaVAGenerate(llama_m_path, &llama_model, clip_m_path, &clip_model, LLaVA_FP32, input, img_path,
                //               generation_config, "models/llama_vocab.bin", true, false, false);
            }
        } else if (format_id == INT4) {
            // Fp32CLIPVisionTransformer clip_model =
            //     Fp32CLIPVisionTransformer(clip_m_path, get_opt_model_config(clip_model_id), false);
            llama_m_path = "INT4/" + llama_m_path;
            Int4LlamaForCausalLM llama_model = Int4LlamaForCausalLM(llama_m_path, get_opt_model_config(llama_model_id));

            // Get input from the user
            while (true) {
                if (prompt_iter == 1) {
                    // Set prompt color
                    set_print_yellow();
                    std::cout << "Finished!" << std::endl << std::endl;
                    // reset color
                    set_print_reset();
                }
                std::string input;
                if (prompt_iter > 0) {
                    if (true) {
                        // Set prompt color
                        set_print_yellow();
                        std::cout << "USER: ";
                        // set user input color
                        set_print_red();
                        std::getline(std::cin, input);
                        // reset color
                        set_print_reset();
                    }
                    if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.") break;
                    std::cout << "ASSISTANT: ";
                }

                if (prompt_iter == 0) {
                    input = "This is a chat between a user and an assistant.\n\n### USER: ";
                    prompt_iter += 1;
                } else if (prompt_iter == 1) {
                    input = "\n" + input + "\n### ASSISTANT:";
                    prompt_iter += 1;
                } else {
                    input = "### USER: " + input + "\n### ASSISTANT: \n";
                }

                // LLaVAGenerate(llama_m_path, &llama_model, clip_m_path, &clip_model, LLaVA_INT4, input, img_path,
                //               generation_config, "models/llama_vocab.bin", true, use_voicechat, false);
            }
        } else {
            std::cout << std::endl;
            std::cerr << "At this time, we only support FP32 and INT4 for LLaVA_7B." << std::endl;
        }
    }
};
