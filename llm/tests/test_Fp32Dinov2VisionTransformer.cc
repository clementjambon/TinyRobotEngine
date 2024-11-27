#include <chrono>
#include <cstring>

#include "Fp32Dinov2VisionTransformer.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"

int NUM_THREAD = 8;

void test_Fp32Dinov2VisionTransformer() {
    struct model_config config = get_opt_model_config(DINO_v2);
    MemoryAllocator mem_buf;

    // Load model
    Fp32Dinov2VisionTransformer featurizer_model =
        Fp32Dinov2VisionTransformer("models/OpenVLA_7B/vision_backbone/featurizer", config);

    // Load processed image
    int num_pixels = 3 * config.image_size * config.image_size;
    Matrix3D<float> pixel_values(mem_buf.get_fpbuffer(num_pixels), 3, config.image_size, config.image_size);
    pixel_values.load("assets/openvla/tests/model/pixel_values_featurizer.bin");
    struct Fp32Dinov2VisionTransformer_input model_input = {pixel_values};
    struct Fp32Dinov2VisionTransformer_output model_output;
    model_output = featurizer_model.forward(model_input);

    // Load ground truth features
    int num_patches = config.image_size / config.patch_size;
    int num_image_tokens = num_patches * num_patches;
    Matrix3D<float> output_gt(mem_buf.get_fpbuffer(num_image_tokens * config.embed_dim), 1, num_image_tokens,
                              config.embed_dim);
    output_gt.load("assets/openvla/tests/model/patch_features_featurizer.bin");

    bool success = check_two_equal(output_gt.m_data, model_output.last_hidden_state.m_data, output_gt.length(), 1e-8);

    if (!success)
        std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    else
        std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;

    // Fp32LlamaForCausalLM model = Fp32LlamaForCausalLM("models/LLaMA_7B", config);

    // struct Fp32LlamaForCausalLM_output output_1st = model.forward(input_1st);

    // Matrix3D<float> logits(mem_buf.get_fpbuffer(b * sqlen * voc_size), b, sqlen, voc_size);
    // logits.load("assets/llama/tests/model/1st_logits.bin");
    // // print_first_k_elelment("O", output_1st.logits.m_data, 20);
    // // print_first_k_elelment("G", logits.m_data, 20);
    // bool success = check_two_equal(output_1st.logits.m_data, logits.m_data, logits.length(), 1e-8);

    // Matrix3D<float> temp_key_value(mem_buf.get_fpbuffer(b * sqlen * embed_dim), num_heads, sqlen,
    //                                embed_dim / num_heads);
    // Profiler::getInstance().report();
    // Profiler::getInstance().reset();

    // // generating phase: 2nd run
    // Matrix3D<int> input_ids_2nd(mem_buf.get_intbuffer(sqlen), b, 1, 1);
    // input_ids_2nd.load("assets/llama/tests/model/2nd_input_ids.bin");
    // struct Fp32LlamaForCausalLM_input input_2nd = {input_ids_2nd, output_1st.past_keys, output_1st.past_values};

    // struct Fp32LlamaForCausalLM_output output_2nd = model.forward(input_2nd);

    // logits = Matrix3D<float>(mem_buf.get_fpbuffer(b * 1 * voc_size), b, 1, voc_size);
    // logits.load("assets/llama/tests/model/2nd_logits.bin");
    // // print_first_k_elelment("O", output_2nd.logits.m_data, 20);
    // // print_first_k_elelment("G", logits.m_data, 20);
    // success &= check_two_equal(output_2nd.logits.m_data, logits.m_data, logits.length(), 1e-8);

    // Profiler::getInstance().report();
    // if (!success)
    //     std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    // else
    //     std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() { test_Fp32Dinov2VisionTransformer(); }
