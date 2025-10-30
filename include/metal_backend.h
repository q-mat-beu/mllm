#pragma once

#include "backend.h"
#include "metal_types.h"
#import <Metal/Metal.h>

class MetalBackend : public Backend {
public:
    MetalBackend();
    ~MetalBackend() override;

    std::unique_ptr<Tensor> create_tensor(const std::vector<int>& shape, DataType dtype) override;

    id<MTLDevice> get_device() const;
    id<MTLCommandQueue> get_command_queue() const;
    id<MTLLibrary> get_library() const;

    void matrix_multiply(const Tensor* inA, const Tensor* inB, Tensor* outC) override;
    void add(const Tensor* inA, const Tensor* inB, Tensor* outC) override;
    void broadcast_add(const Tensor* inA, const Tensor* inB_bias, Tensor* outC) override;
    void softmax_rowwise(const Tensor* in, Tensor* out) override;
    void scale(const Tensor* in, Tensor* out, float scale_factor) override;
    void transpose(const Tensor* in, Tensor* out, int dim1, int dim2) override;
    void layernorm(const Tensor* in, Tensor* out, const Tensor* gamma, const Tensor* beta, float epsilon) override;
    void lookup(const Tensor* weights, const Tensor* indices, Tensor* out) override;
    void gelu(const Tensor* in, Tensor* out) override;
    void transpose2d(const Tensor* in, Tensor* out) override;
    void apply_causal_mask(Tensor* scores) override;

private:
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    id<MTLLibrary> library;
};