#pragma once

#include <memory>
#include <vector>

#include "tensor.h"

class Backend {
public:
    virtual ~Backend() = default;

    virtual std::unique_ptr<Tensor> create_tensor(const std::vector<int>& shape, DataType dtype) = 0;

    virtual void matrix_multiply(const Tensor* inA, const Tensor* inB, Tensor* outC) = 0;
    virtual void add(const Tensor* inA, const Tensor* inB, Tensor* outC) = 0;
    virtual void broadcast_add(const Tensor* inA, const Tensor* inB_bias, Tensor* outC) = 0;
    virtual void softmax_rowwise(const Tensor* in, Tensor* out) = 0;
    virtual void scale(const Tensor* in, Tensor* out, float scale_factor) = 0;
    virtual void transpose(const Tensor* in, Tensor* out, int dim1, int dim2) = 0;
    virtual void layernorm(const Tensor* in, Tensor* out, const Tensor* gamma, const Tensor* beta, float epsilon) = 0;
    virtual void lookup(const Tensor* weights, const Tensor* indices, Tensor* out) = 0;
    virtual void gelu(const Tensor* in, Tensor* out) = 0;
    virtual void transpose2d(const Tensor* in, Tensor* out) = 0;
    virtual void apply_causal_mask(Tensor* scores) = 0;
};