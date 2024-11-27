#include "Fp32Dinov2Encoder.h"

#include <cstring>
#include <iostream>

#include "utils.h"

Fp32Dinov2Encoder::Fp32Dinov2Encoder(std::string param_path, const struct model_config config) {
    // Load all the encoder layers
    for (int layer_idx = 0; layer_idx < config.num_layers; layer_idx++) {
        DEBUG_INS(std::cout << "Start loading layer:" << layer_idx << "..." << std::endl;)

        std::string path = param_path + "/layer" + std::to_string(layer_idx);
        Fp32Dinov2EncoderLayer layer = Fp32Dinov2EncoderLayer(path, config, layer_idx);

        this->layers.push_back(layer);
    }
};

// Fp32Dinov2Encoder
struct Fp32Dinov2Encoder_output Fp32Dinov2Encoder::forward(const struct Fp32Dinov2Encoder_input &input) {
    PROFILE_START(profile_name);
    int sqlen = input.hidden_states.m_dim_y;

    // Go through each layer
    Matrix3D<float> hidden_states = input.hidden_states;
    std::vector<Matrix3D<float>> past_keys, past_values;
    for (int i = 0; i < this->layers.size(); i++) {
        if (!input.has_past_keys_values) {
            struct Fp32Dinov2EncoderLayer_input l_i = {hidden_states, input.attention_mask};
            struct Fp32Dinov2EncoderLayer_output l_o = this->layers[i].forward(l_i);
            hidden_states = l_o.hidden_states;
            past_keys.push_back(l_o.past_key_value.first);
            past_values.push_back(l_o.past_key_value.second);
        } else {
            struct Fp32Dinov2EncoderLayer_input l_i = {hidden_states, input.attention_mask, input.past_keys[i],
                                                       input.past_values[i]};
            struct Fp32Dinov2EncoderLayer_output l_o = this->layers[i].forward(l_i);
            hidden_states = l_o.hidden_states;
            past_keys.push_back(l_o.past_key_value.first);
            past_values.push_back(l_o.past_key_value.second);
        }
    }

    struct Fp32Dinov2Encoder_output output = {hidden_states, past_keys, past_values};
    PROFILE_END(profile_name);
    return output;
}
