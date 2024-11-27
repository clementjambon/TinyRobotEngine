#include "Fp32Dinov2VisionTransformer.h"

#include <cstring>
#include <iostream>

#include "utils.h"

Fp32Dinov2VisionTransformer::Fp32Dinov2VisionTransformer(std::string param_path, const struct model_config config) {
    this->num_patches = config.image_size / config.patch_size;
    this->num_patch_tokens = num_patches * num_patches;
    this->num_tokens_extended = num_patch_tokens + int(config.class_token) + config.num_registers;
    this->voc_size = config.vocsize;
    this->embed_dim = config.embed_dim;
    this->hidden_dim = config.hidden_dim;
    this->num_heads = config.num_heads;
    this->padding_idx = config.padding_idx;
    this->class_token = config.class_token;
    this->num_registers = config.num_registers;
    this->layer_scale = config.layer_scale;

    allocate_aligned_memory(patch_embeds_buf, num_patch_tokens * config.embed_dim * sizeof(float));  // TODO
    allocate_aligned_memory(class_embeds_buf, int(this->class_token) * config.embed_dim * sizeof(float));
    allocate_aligned_memory(reg_embeds_buf, this->num_registers * config.embed_dim * sizeof(float));
    allocate_aligned_memory(pos_embeds_buf, num_tokens_extended * config.embed_dim * sizeof(float));
    allocate_aligned_memory(last_hidden_states_buf, num_patch_tokens * config.embed_dim * sizeof(float));
    allocate_aligned_memory(hidden_states_buf, num_tokens_extended * config.embed_dim * sizeof(float));
    allocate_aligned_memory(embeddings_buf, num_tokens_extended * config.embed_dim * sizeof(float));

    this->encoder = Fp32Dinov2Encoder(param_path + "/encoder", config);

    int max_sqlen = config.max_sqlen;

    // Class Embedding
    if (this->class_token) {
        read_to_array((param_path + "/embeddings/class_embedding/weight.bin").c_str(), class_embeds_buf,
                      config.embed_dim);
    }
    // Register Embedding
    if (this->num_registers > 0) {
        read_to_array((param_path + "/embeddings/register_embedding/weight.bin").c_str(), reg_embeds_buf,
                      this->num_registers * config.embed_dim);
    }
    // Patch Embedding
    struct Conv2D_params embed_patch;
    float *patch_weight_buf;
    float *patch_bias_buf;
    allocate_aligned_memory(patch_weight_buf, 14 * 14 * 3 * config.embed_dim * sizeof(float));
    allocate_aligned_memory(patch_bias_buf, config.embed_dim * sizeof(float));
    Matrix4D<float> patch_weight(patch_weight_buf, 3, 14, 14, config.embed_dim);  // TODO
    Matrix3D<float> patch_bias(patch_bias_buf, 1, 1, config.embed_dim);
    embed_patch.weight = patch_weight;
    embed_patch.bias = patch_bias;
    embed_patch.stride_width = 14;
    embed_patch.stride_height = 14;
    this->embed_patch = Conv2D(embed_patch);
    this->embed_patch.has_bias = true;
    load_Conv2D(this->embed_patch, param_path + "/embeddings/patch_embedding");
    // Position Embedding
    float *posweight_buf;
    allocate_aligned_memory(posweight_buf, config.embed_dim * num_tokens_extended * sizeof(float));
    Matrix3D<float> posweight(posweight_buf, 1, num_tokens_extended, config.embed_dim);
    this->embed_positions = Embedding(config.embed_dim, num_tokens_extended, padding_idx, posweight);
    load_Embedding_params(this->embed_positions, param_path + "/embeddings/position_embedding");
};

// Fp32Dinov2VisionTransformer:
struct Fp32Dinov2VisionTransformer_output Fp32Dinov2VisionTransformer::forward(
    const struct Fp32Dinov2VisionTransformer_input &input) {
    PROFILE_START(profile_name);
    int sqlen = input.input_image.m_dim_z, batch_size = input.input_image.m_dim_x, past_key_values_length = 0;

    if (input.has_past_keys_values) {
        past_key_values_length = input.past_keys[0].m_dim_y;
    }

    // Attention mask: NULL
    Matrix3D<float> causal_attention_mask;

    // Input image
    Matrix3D<float> input_image(input.input_image.m_data, input.input_image.m_dim_x, input.input_image.m_dim_y,
                                input.input_image.m_dim_z);

    // Patch embeddings
    Matrix3D<float> patch_embeds(patch_embeds_buf, this->embed_dim, this->num_patches, this->num_patches);  // TODO
    this->embed_patch.forward(input_image, patch_embeds);
    // Class embeddings + register embeddings
    Matrix3D<float> embeddings(embeddings_buf, 1, this->num_tokens_extended, this->embed_dim);
    int offset = 0;
    if (this->class_token) {
        Matrix3D<float> class_embeds(class_embeds_buf, 1, 1, this->embed_dim);
        // Concate class embeddings with patch embeddings into embeddings
        memcpy(embeddings.m_data, class_embeds.m_data, class_embeds.length() * sizeof(float));
        offset += class_embeds.length();
    }
    if (this->num_registers > 0) {
        Matrix3D<float> reg_embeds(reg_embeds_buf, 1, this->num_registers, this->embed_dim);
        // Concate register embeddings embeddings with patch embeddings into embeddings
        memcpy(embeddings.m_data + offset, reg_embeds.m_data, reg_embeds.length() * sizeof(float));
        offset += reg_embeds.length();
    }
    memcpy(embeddings.m_data + offset, patch_embeds.m_data, patch_embeds.length() * sizeof(float));
    // Position embeddings
    int position_ids_buf[this->num_tokens_extended];
    Matrix3D<int> position_ids(position_ids_buf, 1, 1, this->num_tokens_extended);
    for (int i = 0; i < this->num_tokens_extended; i++) position_ids.m_data[i] = i + past_key_values_length;
    Matrix3D<float> pos_embeds(pos_embeds_buf, 1, this->num_tokens_extended, this->embed_dim);
    this->embed_positions.forward(position_ids, pos_embeds);

    assert(embeddings.m_dim_x == pos_embeds.m_dim_x);
    assert(embeddings.m_dim_y == pos_embeds.m_dim_y);
    assert(embeddings.m_dim_z == pos_embeds.m_dim_z);
    for (int i = 0; i < embeddings.length(); i++) {
        embeddings.m_data[i] = embeddings.m_data[i] + pos_embeds.m_data[i];
    }

    std::cout << "Managed to embed!" << std::endl;

    // CLIP Encoder
    struct Fp32Dinov2Encoder_output encoder_output;
    if (input.has_past_keys_values) {
        struct Fp32Dinov2Encoder_input encoder_input = {embeddings, causal_attention_mask, input.past_keys,
                                                        input.past_values};
        encoder_output = this->encoder.forward(encoder_input);
    } else {
        struct Fp32Dinov2Encoder_input encoder_input = {embeddings, causal_attention_mask};
        encoder_output = this->encoder.forward(encoder_input);
    }

    // Last hidden states
    Matrix3D<float> last_hidden_states(last_hidden_states_buf, 1, this->num_patch_tokens, this->embed_dim);
    // Copy encoder_output.last_hidden_state[class_token+register_tokens:] to last_hidden_states
    // TODO: check that!
    memcpy(last_hidden_states.m_data, encoder_output.last_hidden_state.m_data + offset,
           last_hidden_states.length() * sizeof(float));

    struct Fp32Dinov2VisionTransformer_output output;
    output = {last_hidden_states, encoder_output.past_keys, encoder_output.past_values, embeddings, patch_embeds};

    PROFILE_END(profile_name);
    return output;
}
