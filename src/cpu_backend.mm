#include "cpu_backend.h"
#include "cpu_tensor.h"
#include <Accelerate/Accelerate.h>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <omp.h> // Added for OpenMP

std::unique_ptr<Tensor> CpuBackend::create_tensor(const std::vector<int>& shape, DataType dtype) {
    return std::make_unique<CpuTensor>(shape, dtype);
}

void CpuBackend::matrix_multiply(const Tensor* inA, const Tensor* inB, Tensor* outC) {
    auto shapeA = inA->get_shape();
    auto shapeB = inB->get_shape();
    auto shapeC = outC->get_shape();
    int rank = shapeA.size();

    if (rank < 2) {
        throw std::runtime_error("matrix_multiply on CPU backend requires at least 2D tensors.");
    }

    int M = shapeA[rank - 2];
    int K = shapeA[rank - 1];
    int N = shapeB[rank - 1];

    if (shapeA[rank - 1] != shapeB[rank - 2] || shapeC[rank - 2] != M || shapeC[rank - 1] != N) {
        throw std::runtime_error("Matrix dimensions are not compatible for multiplication.");
    }

    const float* A_data = (const float*)inA->get_data();
    const float* B_data = (const float*)inB->get_data();
    float* C_data = (float*)outC->get_data();

    int batch_size = 1;
    for (int i = 0; i < rank - 2; ++i) {
        batch_size *= shapeA[i];
    }

    size_t strideA = M * K;
    size_t strideB = K * N;
    size_t strideC = M * N;

    for (int b = 0; b < batch_size; ++b) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    1.0f, A_data + b * strideA, K,
                    B_data + b * strideB, N,
                    0.0f, C_data + b * strideC, N);
    }
}

void CpuBackend::add(const Tensor* inA, const Tensor* inB, Tensor* outC) {
    if (inA->get_size() != inB->get_size() || inA->get_size() != outC->get_size()) {
        throw std::runtime_error("Tensors must have the same size for addition.");
    }

    const float* A = (const float*)inA->get_data();
    const float* B = (const float*)inB->get_data();
    float* C = (float*)outC->get_data();

    for (size_t i = 0; i < inA->get_size(); ++i) {
        C[i] = A[i] + B[i];
    }
}

void CpuBackend::broadcast_add(const Tensor* inA, const Tensor* inB_bias, Tensor* outC) {
    const float* A = (const float*)inA->get_data();
    const float* bias = (const float*)inB_bias->get_data();
    float* C = (float*)outC->get_data();

    auto shapeA = inA->get_shape();
    size_t last_dim = shapeA.back();
    size_t num_rows = inA->get_size() / last_dim;

    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < last_dim; ++j) {
            C[i * last_dim + j] = A[i * last_dim + j] + bias[j];
        }
    }
}

void CpuBackend::softmax_rowwise(const Tensor* in, Tensor* out) {
    const float* in_data = (const float*)in->get_data();
    float* out_data = (float*)out->get_data();

    auto shape = in->get_shape();
    size_t row_size = shape.back();
    size_t num_rows = in->get_size() / row_size;

    for (size_t i = 0; i < num_rows; ++i) {
        const float* row_in = in_data + i * row_size;
        float* row_out = out_data + i * row_size;

        float max_val = row_in[0];
        for (size_t j = 1; j < row_size; ++j) {
            if (row_in[j] > max_val) {
                max_val = row_in[j];
            }
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < row_size; ++j) {
            row_out[j] = std::exp(row_in[j] - max_val);
            sum_exp += row_out[j];
        }

        for (size_t j = 0; j < row_size; ++j) {
            row_out[j] /= sum_exp;
        }
    }
}

void CpuBackend::scale(const Tensor* in, Tensor* out, float scale_factor) {
    float* out_data = (float*)out->get_data();
    // If in and out are different tensors, copy in to out first
    if (in != out) {
        const float* in_data = (const float*)in->get_data();
        memcpy(out_data, in_data, in->get_byte_size());
    }
    cblas_sscal(in->get_size(), scale_factor, out_data, 1);
}

void CpuBackend::transpose(const Tensor* in, Tensor* out, int dim1, int dim2) {
    auto in_shape = in->get_shape();
    auto out_shape = out->get_shape();
    const float* in_data = (const float*)in->get_data();
    float* out_data = (float*)out->get_data();

    std::vector<int> in_strides(in_shape.size());
    in_strides.back() = 1;
    for (int i = in_shape.size() - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
    }

    std::vector<int> out_strides(out_shape.size());
    out_strides.back() = 1;
    for (int i = out_shape.size() - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    std::vector<int> perm(in_shape.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[dim1], perm[dim2]);

    #pragma omp parallel for
    for(size_t i = 0; i < in->get_size(); ++i) {
        size_t in_idx = i;
        size_t out_idx = 0;
        std::vector<int> in_coords(in_shape.size());
        
        size_t temp_idx = in_idx;
        for(size_t j = 0; j < in_shape.size(); ++j) {
            in_coords[j] = temp_idx / in_strides[j];
            temp_idx %= in_strides[j];
        }

        std::vector<int> out_coords(out_shape.size());
        for(size_t j = 0; j < perm.size(); ++j) {
            out_coords[j] = in_coords[perm[j]];
        }
        
        for(size_t j = 0; j < out_shape.size(); ++j) {
            out_idx += out_coords[j] * out_strides[j];
        }
        out_data[out_idx] = in_data[in_idx];
    }
}

void CpuBackend::layernorm(const Tensor* in, Tensor* out, const Tensor* gamma, const Tensor* beta, float epsilon) {
    const float* in_data = (const float*)in->get_data();
    float* out_data = (float*)out->get_data();
    const float* gamma_data = (const float*)gamma->get_data();
    const float* beta_data = (const float*)beta->get_data();

    auto shape = in->get_shape();
    size_t normalized_shape = shape.back();
    size_t num_rows = in->get_size() / normalized_shape;

    for (size_t i = 0; i < num_rows; ++i) {
        const float* row_in = in_data + i * normalized_shape;
        float* row_out = out_data + i * normalized_shape;

        float mean = 0.0f;
        for (size_t j = 0; j < normalized_shape; ++j) {
            mean += row_in[j];
        }
        mean /= normalized_shape;

        float variance = 0.0f;
        for (size_t j = 0; j < normalized_shape; ++j) {
            variance += (row_in[j] - mean) * (row_in[j] - mean);
        }
        variance /= normalized_shape;

        float inv_std = 1.0f / std::sqrt(variance + epsilon);

        for (size_t j = 0; j < normalized_shape; ++j) {
            row_out[j] = (row_in[j] - mean) * inv_std * gamma_data[j] + beta_data[j];
        }
    }
}

void CpuBackend::lookup(const Tensor* weights, const Tensor* indices, Tensor* out) {
    const float* weights_data = (const float*)weights->get_data();
    const int* indices_data = (const int*)indices->get_data();
    float* out_data = (float*)out->get_data();

    auto out_shape = out->get_shape();
    size_t embedding_dim = out_shape.back();
    size_t num_indices = indices->get_size();

    for (size_t i = 0; i < num_indices; ++i) {
        int token_index = indices_data[i];
        const float* weight_row = weights_data + token_index * embedding_dim;
        float* out_row = out_data + i * embedding_dim;
        memcpy(out_row, weight_row, embedding_dim * sizeof(float));
    }
}

void CpuBackend::gelu(const Tensor* in, Tensor* out) {
    const float* in_data = (const float*)in->get_data();
    float* out_data = (float*)out->get_data();

    for (size_t i = 0; i < in->get_size(); ++i) {
        float x = in_data[i];
        float sigmoid_arg = 1.702f * x;
        float sigmoid_val = 1.0f / (1.0f + expf(-sigmoid_arg));
        out_data[i] = x * sigmoid_val;
    }
}

void CpuBackend::transpose2d(const Tensor* in, Tensor* out) {
    auto in_shape = in->get_shape();
    if (in_shape.size() != 2) {
        throw std::runtime_error("Transpose2d currently only supports 2D tensors.");
    }
    int rows = in_shape[0];
    int cols = in_shape[1];

    const float* in_data = (const float*)in->get_data();
    float* out_data = (float*)out->get_data();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out_data[j * rows + i] = in_data[i * cols + j];
        }
    }
}

void CpuBackend::apply_causal_mask(Tensor* scores) {
    auto shape = scores->get_shape();
    if (shape.size() != 4) {
        throw std::runtime_error("apply_causal_mask expects a 4D tensor (batch, heads, seq, seq)");
    }

    float* data = (float*)scores->get_data();
    int seq_len = shape[2];
    int batch_size = shape[0] * shape[1];
    int matrix_size = seq_len * seq_len;

    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                if (j > i) {
                    data[b * matrix_size + i * seq_len + j] = -INFINITY;
                }
            }
        }
    }
}