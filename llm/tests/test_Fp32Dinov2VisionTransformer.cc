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

    // Load ground truth features
    int num_patches = config.image_size / config.patch_size;
    int num_image_tokens = num_patches * num_patches;
    int num_extended_tokens = num_image_tokens + int(config.class_token) + config.num_registers;

    Matrix3D<float> gt_patch_embeds(mem_buf.get_fpbuffer(num_image_tokens * config.embed_dim), config.embed_dim,
                                    num_patches, num_patches);
    gt_patch_embeds.load("assets/openvla/tests/model/featurizer/patch_embeds.bin");
    std::cout << "gt_patch_embeds" << std::endl;
    gt_patch_embeds.print_dims();

    Matrix3D<float> gt_embeddings(mem_buf.get_fpbuffer(num_extended_tokens * config.embed_dim), 1, num_extended_tokens,
                                  config.embed_dim);
    gt_embeddings.load("assets/openvla/tests/model/featurizer/embeddings.bin");
    std::cout << "gt_embeddings" << std::endl;
    gt_embeddings.print_dims();

    Matrix3D<float> gt_output(mem_buf.get_fpbuffer(num_image_tokens * config.embed_dim), 1, num_image_tokens,
                              config.embed_dim);
    gt_output.load("assets/openvla/tests/model/featurizer/patch_features.bin");
    std::cout << "gt_output" << std::endl;
    gt_output.print_dims();

    // Load processed image
    int num_pixels = 3 * config.image_size * config.image_size;
    Matrix3D<float> pixel_values(mem_buf.get_fpbuffer(num_pixels), 3, config.image_size, config.image_size);
    pixel_values.load("assets/openvla/tests/model/featurizer/pixel_values.bin");
    // WARNING: for now, we cheat!
    struct Fp32Dinov2VisionTransformer_input model_input(pixel_values);
    struct Fp32Dinov2VisionTransformer_output model_output;
    model_output = featurizer_model.forward(model_input);

    bool success = check_two_equal(gt_patch_embeds, model_output.patch_embeds, 1e-8);
    // bool success = check_two_equal(gt_embeddings, model_output.embeddings, 1e-8);

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
