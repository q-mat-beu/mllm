#include "attention.h"
#include "backend.h"
#include <stdexcept>
#include <cmath>
#include <memory>
#ifdef ENABLE_PROFILING
#include <chrono> // Added for timing
#include <iostream> // Added for printing timing
#endif

MultiHeadAttention::MultiHeadAttention(Backend* backend, int embed_dim, int num_heads)
    : backend(backend),
      embed_dim(embed_dim),
      num_heads(num_heads),
      head_dim(embed_dim / num_heads)
{
    if (embed_dim % num_heads != 0) {
        throw std::runtime_error("Embedding dimension must be divisible by the number of heads");
    }

    proj_q = std::make_unique<Linear>(backend, embed_dim, embed_dim);
    proj_k = std::make_unique<Linear>(backend, embed_dim, embed_dim);
    proj_v = std::make_unique<Linear>(backend, embed_dim, embed_dim);
    proj_o = std::make_unique<Linear>(backend, embed_dim, embed_dim);
}

MultiHeadAttention::~MultiHeadAttention() {
}

void MultiHeadAttention::load_weights(
    const std::vector<float>& w_q_data, const std::vector<float>& b_q_data,
    const std::vector<float>& w_k_data, const std::vector<float>& b_k_data,
    const std::vector<float>& w_v_data, const std::vector<float>& b_v_data,
    const std::vector<float>& w_o_data, const std::vector<float>& b_o_data
) {
    proj_q->load_weights(w_q_data, b_q_data);
    proj_k->load_weights(w_k_data, b_k_data);
    proj_v->load_weights(w_v_data, b_v_data);
    proj_o->load_weights(w_o_data, b_o_data);
}

std::unique_ptr<Tensor> MultiHeadAttention::forward(const Tensor* input, bool apply_mask) {
#ifdef ENABLE_PROFILING
    auto start_attn_forward = std::chrono::high_resolution_clock::now();
#endif

    auto input_shape = input->get_shape();
    if (input_shape.size() != 3) { // Expect 3D
        throw std::runtime_error("MultiHeadAttention::forward expects a 3D input tensor (batch, seq, dim)");
    }
    int batch_size = input_shape[0];
    int seq_len = input_shape[1];

#ifdef ENABLE_PROFILING
    auto start_proj = std::chrono::high_resolution_clock::now();
#endif
    auto q = proj_q->forward(input);
    auto k = proj_k->forward(input);
    auto v = proj_v->forward(input);
#ifdef ENABLE_PROFILING
    auto end_proj = std::chrono::high_resolution_clock::now();
    std::cout << "    Attention Projections took: " << std::chrono::duration<double>(end_proj - start_proj).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_reshape = std::chrono::high_resolution_clock::now();
#endif
    q->reshape({batch_size, seq_len, num_heads, head_dim});
    k->reshape({batch_size, seq_len, num_heads, head_dim});
    v->reshape({batch_size, seq_len, num_heads, head_dim});
#ifdef ENABLE_PROFILING
    auto end_reshape = std::chrono::high_resolution_clock::now();
    std::cout << "    Attention Reshapes took: " << std::chrono::duration<double>(end_reshape - start_reshape).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_transpose1 = std::chrono::high_resolution_clock::now();
#endif
    auto q_transposed = backend->create_tensor({batch_size, num_heads, seq_len, head_dim}, MLLM_FLOAT32);
    q_transposed->allocate();
    backend->transpose(q.get(), q_transposed.get(), 1, 2);

    auto k_transposed = backend->create_tensor({batch_size, num_heads, seq_len, head_dim}, MLLM_FLOAT32);
    k_transposed->allocate();
    backend->transpose(k.get(), k_transposed.get(), 1, 2);

    auto v_transposed = backend->create_tensor({batch_size, num_heads, seq_len, head_dim}, MLLM_FLOAT32);
    v_transposed->allocate();
    backend->transpose(v.get(), v_transposed.get(), 1, 2);
#ifdef ENABLE_PROFILING
    auto end_transpose1 = std::chrono::high_resolution_clock::now();
    std::cout << "    Attention Transpose 1 took: " << std::chrono::duration<double>(end_transpose1 - start_transpose1).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_k_T = std::chrono::high_resolution_clock::now();
#endif
    auto k_T = backend->create_tensor({batch_size, num_heads, head_dim, seq_len}, MLLM_FLOAT32);
    k_T->allocate();
    backend->transpose(k_transposed.get(), k_T.get(), 2, 3);
#ifdef ENABLE_PROFILING
    auto end_k_T = std::chrono::high_resolution_clock::now();
    std::cout << "    Attention K_T transpose took: " << std::chrono::duration<double>(end_k_T - start_k_T).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_matmul1 = std::chrono::high_resolution_clock::now();
#endif
    auto scores = backend->create_tensor({batch_size, num_heads, seq_len, seq_len}, MLLM_FLOAT32);
    scores->allocate();
    backend->matrix_multiply(q_transposed.get(), k_T.get(), scores.get());
#ifdef ENABLE_PROFILING
    auto end_matmul1 = std::chrono::high_resolution_clock::now();
    std::cout << "    Attention Matmul 1 (scores) took: " << std::chrono::duration<double>(end_matmul1 - start_matmul1).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_scale = std::chrono::high_resolution_clock::now();
#endif
    float scale_factor = 1.0f / sqrtf(head_dim);
    backend->scale(scores.get(), scores.get(), scale_factor);
#ifdef ENABLE_PROFILING
    auto end_scale = std::chrono::high_resolution_clock::now();
    std::cout << "    Attention Scale took: " << std::chrono::duration<double>(end_scale - start_scale).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_mask = std::chrono::high_resolution_clock::now();
#endif
    if (apply_mask) {
        backend->apply_causal_mask(scores.get());
    }
#ifdef ENABLE_PROFILING
    auto end_mask = std::chrono::high_resolution_clock::now();
    std::cout << "    Attention Mask took: " << std::chrono::duration<double>(end_mask - start_mask).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_softmax = std::chrono::high_resolution_clock::now();
#endif
    backend->softmax_rowwise(scores.get(), scores.get());
#ifdef ENABLE_PROFILING
    auto end_softmax = std::chrono::high_resolution_clock::now();
    std::cout << "    Attention Softmax took: " << std::chrono::duration<double>(end_softmax - start_softmax).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_matmul2 = std::chrono::high_resolution_clock::now();
#endif
    auto context = backend->create_tensor({batch_size, num_heads, seq_len, head_dim}, MLLM_FLOAT32);
    context->allocate();
    backend->matrix_multiply(scores.get(), v_transposed.get(), context.get());
#ifdef ENABLE_PROFILING
    auto end_matmul2 = std::chrono::high_resolution_clock::now();
    std::cout << "    Attention Matmul 2 (context) took: " << std::chrono::duration<double>(end_matmul2 - start_matmul2).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_transpose2 = std::chrono::high_resolution_clock::now();
#endif
    auto context_transposed = backend->create_tensor({batch_size, seq_len, num_heads, head_dim}, MLLM_FLOAT32);
    context_transposed->allocate();
    backend->transpose(context.get(), context_transposed.get(), 1, 2);
#ifdef ENABLE_PROFILING
    auto end_transpose2 = std::chrono::high_resolution_clock::now();
    std::cout << "    Attention Transpose 2 took: " << std::chrono::duration<double>(end_transpose2 - start_transpose2).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_reshape2 = std::chrono::high_resolution_clock::now();
#endif
    context_transposed->reshape({batch_size, seq_len, embed_dim});
#ifdef ENABLE_PROFILING
    auto end_reshape2 = std::chrono::high_resolution_clock::now();
    std::cout << "    Attention Reshape 2 took: " << std::chrono::duration<double>(end_reshape2 - start_reshape2).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto start_proj_o = std::chrono::high_resolution_clock::now();
#endif
    auto final_output = proj_o->forward(context_transposed.get());
#ifdef ENABLE_PROFILING
    auto end_proj_o = std::chrono::high_resolution_clock::now();
    std::cout << "    Attention Output Projection took: " << std::chrono::duration<double>(end_proj_o - start_proj_o).count() << " s" << std::endl;
#endif

#ifdef ENABLE_PROFILING
    auto end_attn_forward = std::chrono::high_resolution_clock::now();
    std::cout << "  MultiHeadAttention::forward total took: " << std::chrono::duration<double>(end_attn_forward - start_attn_forward).count() << " s" << std::endl;
#endif

    return final_output;
}