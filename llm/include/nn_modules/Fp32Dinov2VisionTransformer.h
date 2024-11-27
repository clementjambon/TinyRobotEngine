#include <cstdlib>
#include <string>
#include <vector>

#include "Fp32Dinov2Encoder.h"
#include "common.h"
#include "operators.h"

struct Fp32Dinov2VisionTransformer_output {
    Matrix3D<float> last_hidden_state;
    std::vector<Matrix3D<float>> past_keys, past_values;
    Matrix3D<float> embeddings, patch_embeds;
};
struct Fp32Dinov2VisionTransformer_input {
    Matrix3D<float> input_image;
    std::vector<Matrix3D<float>> past_keys, past_values;
    bool has_past_keys_values;

    Fp32Dinov2VisionTransformer_input() {}
    Fp32Dinov2VisionTransformer_input(Matrix3D<float> input_image_) : input_image(input_image_) {
        has_past_keys_values = false;
    }
    Fp32Dinov2VisionTransformer_input(Matrix3D<float> input_image_, std::vector<Matrix3D<float>> past_keys_,
                                      std::vector<Matrix3D<float>> past_values_)
        : input_image(input_image_), past_keys(past_keys_), past_values(past_values_) {
        has_past_keys_values = true;
    }
};

class Fp32Dinov2VisionTransformer {
   public:
    Fp32Dinov2VisionTransformer(std::string param_path, const struct model_config config);
    Fp32Dinov2VisionTransformer() {};
    struct Fp32Dinov2VisionTransformer_output forward(const struct Fp32Dinov2VisionTransformer_input& input);
    Embedding embed_positions;
    Conv2D embed_patch;
    int voc_size, embed_dim, padding_idx, hidden_dim, num_heads, image_size, patch_size, num_patches, projection_dim,
        mmproj_dim, num_patch_tokens, num_tokens_extended, num_registers, layer_scale;
    bool class_token;
    std::vector<Fp32Dinov2EncoderLayer> layers;
    std::string profile_name = "Fp32Dinov2VisionTransformer";

   private:
    Fp32Dinov2Encoder encoder;
    float* patch_embeds_buf;
    float* class_embeds_buf;
    float* reg_embeds_buf;
    float* pos_embeds_buf;
    float* last_hidden_states_buf;
    float* hidden_states_buf;
    float* embeddings_buf;
};
