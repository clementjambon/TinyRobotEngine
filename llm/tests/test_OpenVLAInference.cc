#include <chrono>
#include <cstring>

#include "Int4llamaForCausalLM.h"
#include "operators.h"
#include "utils.h"
#include "utils_memalloc.h"

int NUM_THREAD = 8;

void test_OpenVLAInference() {
    struct model_config config = get_opt_model_config(OpenVLA_7B);
    MemoryAllocator mem_buf;

    // ==========
    // MODEL
    // ==========
    Int4LlamaForCausalLM llama_model = Int4LlamaForCausalLM("INT4/models/OpenVLA_7B", config);

    // ==========
    // GT DATA
    // ==========
    // Info
    Matrix3D<int> gt_info(mem_buf.get_intbuffer(1), 1, 1, 1);
    gt_info.load("embeds/OpenVLA_7B/0000_info.bin");
    int n_ids = gt_info(0, 0, 0);
    std::cout << n_ids << std::endl;
    // Input ids
    Matrix3D<int> gt_input_ids(mem_buf.get_intbuffer(n_ids), 1, 1, n_ids);
    gt_input_ids.load("embeds/OpenVLA_7B/0000_input_ids.bin");
    // Patch embeds
    Matrix3D<float> gt_patch_embeds(mem_buf.get_fpbuffer(256 * config.embed_dim), 1, 256, config.embed_dim);
    gt_patch_embeds.load("embeds/OpenVLA_7B/0000_projected_patch_embeddings.bin");
    // Logits
    Matrix3D<float> gt_logits(mem_buf.get_fpbuffer((256 + n_ids) * config.vocsize), 1, 256 + n_ids, config.vocsize);
    gt_logits.load("embeds/OpenVLA_7B/0000_logits.bin");

    // ==========
    // INFERENCE
    // ==========
    struct Int4LlamaForCausalLM_output model_output;
    struct Int4LlamaForCausalLM_input model_input;

    const int n_image_tokens = 256;
    int sqlen = gt_input_ids.length() + n_image_tokens;
    int first_sqlen = gt_input_ids.length();
    model_input = {gt_input_ids, gt_patch_embeds};

    model_output = llama_model.forward("INT4/models/OpenVLA_7B", model_input);

    model_output.logits.print_dims();
    gt_logits.print_dims();
    assert(model_output.logits.same_dims(gt_logits));
    for (int z = 0; z < gt_logits.m_dim_z; ++z) {
        std::cout << model_output.logits(0, gt_logits.m_dim_y - 1, z) << "; " << gt_logits(0, gt_logits.m_dim_y - 1, z)
                  << std::endl;
    }

    // if (!success)
    //     std::cout << "-------- Test of " << __func__ << ": Fail! -------- " << std::endl;
    // else
    //     std::cout << "-------- Test of " << __func__ << ": Passed! -------- " << std::endl;
}

int main() { test_OpenVLAInference(); }
