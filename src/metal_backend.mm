#include "metal_backend.h"
#include "metal_tensor.h"
#include "metal_types.h"
#include <stdexcept>
#include <vector>
#include <numeric>
#include <cmath>

std::unique_ptr<Tensor> MetalBackend::create_tensor(const std::vector<int>& shape, DataType dtype) {
    return std::make_unique<MetalTensor>(shape, dtype, this);
}

MetalBackend::MetalBackend() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("Metal is not supported on this device");
    }
    command_queue = [device newCommandQueue];

    NSError* error = nil;
    NSString* library_path_str = [NSString stringWithFormat:@"%@/default.metallib", [[NSBundle mainBundle] bundlePath]];
    NSURL* library_url = [NSURL fileURLWithPath:library_path_str];
    library = [device newLibraryWithURL:library_url error:&error];
    if (!library) {
        throw std::runtime_error("Failed to create Metal library");
    }
}

MetalBackend::~MetalBackend() {
    [library release];
    [command_queue release];
    [device release];
}

id<MTLDevice> MetalBackend::get_device() const {
    return device;
}

id<MTLCommandQueue> MetalBackend::get_command_queue() const {
    return command_queue;
}

id<MTLLibrary> MetalBackend::get_library() const {
    return library;
}

void MetalBackend::matrix_multiply(const Tensor* inA, const Tensor* inB, Tensor* outC) {
    const MetalTensor* mt_inA = dynamic_cast<const MetalTensor*>(inA);
    const MetalTensor* mt_inB = dynamic_cast<const MetalTensor*>(inB);
    MetalTensor* mt_outC = dynamic_cast<MetalTensor*>(outC);

    auto shapeA = mt_inA->get_shape();
    int rank = shapeA.size();

    if (rank == 2) {
        NSError* error = nil;
        id<MTLFunction> kernel_func = [library newFunctionWithName:@"matmul"];
        if (!kernel_func) {
            throw std::runtime_error("Failed to create Metal kernel function");
        }

        id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:kernel_func error:&error];
        if (!pipeline_state) {
            throw std::runtime_error("Failed to create Metal pipeline state");
        }

        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

        [command_encoder setComputePipelineState:pipeline_state];
        [command_encoder setBuffer:mt_inA->get_buffer() offset:0 atIndex:0];
        [command_encoder setBuffer:mt_inB->get_buffer() offset:0 atIndex:1];
        [command_encoder setBuffer:mt_outC->get_buffer() offset:0 atIndex:2];

        MatMulParams params = {
            .widthA = (uint32_t)mt_inA->get_shape()[1],
            .heightA = (uint32_t)mt_inA->get_shape()[0],
            .widthB = (uint32_t)mt_inB->get_shape()[1],
            .heightB = (uint32_t)mt_inB->get_shape()[0],
            .widthC = (uint32_t)mt_outC->get_shape()[1],
            .heightC = (uint32_t)mt_outC->get_shape()[0],
        };
        [command_encoder setBytes:&params length:sizeof(params) atIndex:3];

        MTLSize grid_size = MTLSizeMake(params.widthC, params.heightC, 1);
        
        NSUInteger threadgroup_width = [pipeline_state threadExecutionWidth];
        NSUInteger threadgroup_height = [pipeline_state maxTotalThreadsPerThreadgroup] / threadgroup_width;
        MTLSize threadgroup_size = MTLSizeMake(threadgroup_width, threadgroup_height, 1);

        [command_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [command_encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        [kernel_func release];
        [pipeline_state release];
    } else if (rank > 2) {
        auto shapeB = mt_inB->get_shape();
        auto shapeC = mt_outC->get_shape();

        int batch_size = 1;
        for (int i = 0; i < rank - 2; ++i) {
            batch_size *= shapeA[i];
        }

        int M = shapeA[rank - 2];
        int K = shapeA[rank - 1];
        int N = shapeB[rank - 1];

        NSError* error = nil;
        id<MTLFunction> kernel_func = [library newFunctionWithName:@"matmul_batched"];
        if (!kernel_func) {
            throw std::runtime_error("Failed to create Metal kernel function for matmul_batched");
        }

        id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:kernel_func error:&error];
        if (!pipeline_state) {
            throw std::runtime_error("Failed to create Metal pipeline state for matmul_batched");
        }

        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

        [command_encoder setComputePipelineState:pipeline_state];
        [command_encoder setBuffer:mt_inA->get_buffer() offset:0 atIndex:0];
        [command_encoder setBuffer:mt_inB->get_buffer() offset:0 atIndex:1];
        [command_encoder setBuffer:mt_outC->get_buffer() offset:0 atIndex:2];

        BatchedMatMulParams params;
        params.heightA = M;
        params.widthA = K;
        params.heightB = K;
        params.widthB = N;
        params.heightC = M;
        params.widthC = N;
        params.batch_size = batch_size;
        params.strideA = M * K;
        params.strideB = K * N;
        params.strideC = M * N;

        [command_encoder setBytes:&params length:sizeof(params) atIndex:3];

        MTLSize grid_size = MTLSizeMake(params.widthC, params.heightC, params.batch_size);

        NSUInteger threadgroup_width = 8;
        NSUInteger threadgroup_height = 8;
        NSUInteger threadgroup_depth = 1;
        MTLSize threadgroup_size = MTLSizeMake(threadgroup_width, threadgroup_height, threadgroup_depth);

        [command_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [command_encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        [kernel_func release];
        [pipeline_state release];
    }
}

void MetalBackend::add(const Tensor* inA, const Tensor* inB, Tensor* outC) {
    const MetalTensor* mt_inA = dynamic_cast<const MetalTensor*>(inA);
    const MetalTensor* mt_inB = dynamic_cast<const MetalTensor*>(inB);
    MetalTensor* mt_outC = dynamic_cast<MetalTensor*>(outC);

    NSError* error = nil;
    id<MTLFunction> kernel_func = [library newFunctionWithName:@"add"];
    if (!kernel_func) {
        throw std::runtime_error("Failed to create Metal kernel function");
    }

    id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:kernel_func error:&error];
    if (!pipeline_state) {
        throw std::runtime_error("Failed to create Metal pipeline state");
    }

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

    [command_encoder setComputePipelineState:pipeline_state];
    [command_encoder setBuffer:mt_inA->get_buffer() offset:0 atIndex:0];
    [command_encoder setBuffer:mt_inB->get_buffer() offset:0 atIndex:1];
    [command_encoder setBuffer:mt_outC->get_buffer() offset:0 atIndex:2];

    MTLSize grid_size = MTLSizeMake(mt_outC->get_size(), 1, 1);
    
    NSUInteger threadgroup_size = [pipeline_state maxTotalThreadsPerThreadgroup];
    MTLSize threadgroup = MTLSizeMake(threadgroup_size, 1, 1);

    [command_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup];
    [command_encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    [kernel_func release];
    [pipeline_state release];
}

void MetalBackend::broadcast_add(const Tensor* inA, const Tensor* inB_bias, Tensor* outC) {
    const MetalTensor* mt_inA = dynamic_cast<const MetalTensor*>(inA);
    const MetalTensor* mt_inB_bias = dynamic_cast<const MetalTensor*>(inB_bias);
    MetalTensor* mt_outC = dynamic_cast<MetalTensor*>(outC);

    NSError* error = nil;
    id<MTLFunction> kernel_func = [library newFunctionWithName:@"broadcast_add"];
    if (!kernel_func) {
        throw std::runtime_error("Failed to create Metal kernel function");
    }

    id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:kernel_func error:&error];
    if (!pipeline_state) {
        throw std::runtime_error("Failed to create Metal pipeline state");
    }

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

    [command_encoder setComputePipelineState:pipeline_state];
    [command_encoder setBuffer:mt_inA->get_buffer() offset:0 atIndex:0];
    [command_encoder setBuffer:mt_inB_bias->get_buffer() offset:0 atIndex:1];
    [command_encoder setBuffer:mt_outC->get_buffer() offset:0 atIndex:2];

    uint32_t width = (uint32_t)mt_inA->get_shape().back(); // Assuming bias is added to the last dimension
    [command_encoder setBytes:&width length:sizeof(width) atIndex:3];

    MTLSize grid_size = MTLSizeMake(mt_outC->get_size(), 1, 1);
    
    NSUInteger threadgroup_size = [pipeline_state maxTotalThreadsPerThreadgroup];
    MTLSize threadgroup = MTLSizeMake(threadgroup_size, 1, 1);

    [command_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup];
    [command_encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    [kernel_func release];
    [pipeline_state release];
}

void MetalBackend::softmax_rowwise(const Tensor* in, Tensor* out) {
    const MetalTensor* mt_in = dynamic_cast<const MetalTensor*>(in);
    MetalTensor* mt_out = dynamic_cast<MetalTensor*>(out);

    NSError* error = nil;
    id<MTLFunction> kernel_func = [library newFunctionWithName:@"softmax_rowwise"];
    if (!kernel_func) {
        throw std::runtime_error("Failed to create Metal kernel function for softmax_rowwise");
    }

    id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:kernel_func error:&error];
    if (!pipeline_state) {
        throw std::runtime_error("Failed to create Metal pipeline state for softmax_rowwise");
    }

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

    [command_encoder setComputePipelineState:pipeline_state];
    [command_encoder setBuffer:mt_in->get_buffer() offset:0 atIndex:0];
    [command_encoder setBuffer:mt_out->get_buffer() offset:0 atIndex:1];

    uint32_t row_size = mt_in->get_shape().back();
    [command_encoder setBytes:&row_size length:sizeof(row_size) atIndex:2];

    size_t num_rows = mt_in->get_size() / row_size;
    MTLSize grid_size = MTLSizeMake(num_rows, 1, 1);

    NSUInteger threadgroup_size_val = [pipeline_state maxTotalThreadsPerThreadgroup];
    if (threadgroup_size_val > num_rows) {
        threadgroup_size_val = num_rows;
    }
    MTLSize threadgroup_size = MTLSizeMake(threadgroup_size_val, 1, 1);

    [command_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [command_encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    [kernel_func release];
    [pipeline_state release];
}

void MetalBackend::scale(const Tensor* in, Tensor* out, float scale_factor) {
    const MetalTensor* mt_in = dynamic_cast<const MetalTensor*>(in);
    MetalTensor* mt_out = dynamic_cast<MetalTensor*>(out);

    NSError* error = nil;
    id<MTLFunction> kernel_func = [library newFunctionWithName:@"scale"];
    if (!kernel_func) {
        throw std::runtime_error("Failed to create Metal kernel function for scale");
    }

    id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:kernel_func error:&error];
    if (!pipeline_state) {
        throw std::runtime_error("Failed to create Metal pipeline state for scale");
    }

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

    [command_encoder setComputePipelineState:pipeline_state];
    [command_encoder setBuffer:mt_in->get_buffer() offset:0 atIndex:0];
    [command_encoder setBuffer:mt_out->get_buffer() offset:0 atIndex:1];
    [command_encoder setBytes:&scale_factor length:sizeof(scale_factor) atIndex:2];

    MTLSize grid_size = MTLSizeMake(mt_in->get_size(), 1, 1);
    
    NSUInteger threadgroup_size_val = [pipeline_state maxTotalThreadsPerThreadgroup];
    MTLSize threadgroup_size = MTLSizeMake(threadgroup_size_val, 1, 1);

    [command_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [command_encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    [kernel_func release];
    [pipeline_state release];
}

void MetalBackend::transpose(const Tensor* in, Tensor* out, int dim1, int dim2) {
    const MetalTensor* mt_in = dynamic_cast<const MetalTensor*>(in);
    MetalTensor* mt_out = dynamic_cast<MetalTensor*>(out);

    auto in_shape = mt_in->get_shape();
    int rank = in_shape.size();

    if (rank == 3) {
        NSError* error = nil;
        id<MTLFunction> kernel_func = [library newFunctionWithName:@"transpose"];
        if (!kernel_func) {
            throw std::runtime_error("Failed to create Metal kernel function for transpose");
        }

        id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:kernel_func error:&error];
        if (!pipeline_state) {
            throw std::runtime_error("Failed to create Metal pipeline state for transpose");
        }

        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

        [command_encoder setComputePipelineState:pipeline_state];
        [command_encoder setBuffer:mt_in->get_buffer() offset:0 atIndex:0];
        [command_encoder setBuffer:mt_out->get_buffer() offset:0 atIndex:1];

        TransposeParams params;
        params.dims[0] = in_shape[0];
        params.dims[1] = in_shape[1];
        params.dims[2] = in_shape[2];

        params.perm[0] = 0;
        params.perm[1] = 1;
        params.perm[2] = 2;
        
        params.perm[dim1] = dim2;
        params.perm[dim2] = dim1;

        [command_encoder setBytes:&params length:sizeof(params) atIndex:2];

        auto out_shape = mt_out->get_shape();
        MTLSize grid_size = MTLSizeMake(out_shape[0], out_shape[1], out_shape[2]);
        
        NSUInteger threadgroup_width = 8;
        NSUInteger threadgroup_height = 8;
        NSUInteger threadgroup_depth = [pipeline_state maxTotalThreadsPerThreadgroup] / (threadgroup_width * threadgroup_height);
        if (threadgroup_depth == 0) threadgroup_depth = 1;

        MTLSize threadgroup_size = MTLSizeMake(threadgroup_width, threadgroup_height, threadgroup_depth);

        [command_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [command_encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        [kernel_func release];
        [pipeline_state release];
    } else if (rank == 4) {
        NSError* error = nil;
        id<MTLFunction> kernel_func = [library newFunctionWithName:@"transpose4d"];
        if (!kernel_func) {
            throw std::runtime_error("Failed to create Metal kernel function for transpose4d");
        }

        id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:kernel_func error:&error];
        if (!pipeline_state) {
            throw std::runtime_error("Failed to create Metal pipeline state for transpose4d");
        }

        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

        [command_encoder setComputePipelineState:pipeline_state];
        [command_encoder setBuffer:mt_in->get_buffer() offset:0 atIndex:0];
        [command_encoder setBuffer:mt_out->get_buffer() offset:0 atIndex:1];

        Transpose4DParams params;
        for(int i=0; i<4; ++i) params.dims[i] = in_shape[i];
        for(int i=0; i<4; ++i) params.perm[i] = i;
        
        params.perm[dim1] = dim2;
        params.perm[dim2] = dim1;

        [command_encoder setBytes:&params length:sizeof(params) atIndex:2];

        MTLSize grid_size = MTLSizeMake(mt_out->get_size(), 1, 1);
        
        NSUInteger threadgroup_size_val = [pipeline_state maxTotalThreadsPerThreadgroup];
        MTLSize threadgroup_size = MTLSizeMake(threadgroup_size_val, 1, 1);

        [command_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [command_encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        [kernel_func release];
        [pipeline_state release];
    } else {
        throw std::runtime_error("Transpose currently only supports 3D and 4D tensors.");
    }
}

void MetalBackend::layernorm(const Tensor* in, Tensor* out, const Tensor* gamma, const Tensor* beta, float epsilon) {
    const MetalTensor* mt_in = dynamic_cast<const MetalTensor*>(in);
    MetalTensor* mt_out = dynamic_cast<MetalTensor*>(out);
    const MetalTensor* mt_gamma = dynamic_cast<const MetalTensor*>(gamma);
    const MetalTensor* mt_beta = dynamic_cast<const MetalTensor*>(beta);

    NSError* error = nil;
    id<MTLFunction> kernel_func = [library newFunctionWithName:@"layernorm"];
    if (!kernel_func) {
        throw std::runtime_error("Failed to create Metal kernel function for layernorm");
    }

    id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:kernel_func error:&error];
    if (!pipeline_state) {
        throw std::runtime_error("Failed to create Metal pipeline state for layernorm");
    }

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

    [command_encoder setComputePipelineState:pipeline_state];
    [command_encoder setBuffer:mt_in->get_buffer() offset:0 atIndex:0];
    [command_encoder setBuffer:mt_out->get_buffer() offset:0 atIndex:1];
    [command_encoder setBuffer:mt_gamma->get_buffer() offset:0 atIndex:2];
    [command_encoder setBuffer:mt_beta->get_buffer() offset:0 atIndex:3];

    uint32_t normalized_shape = mt_in->get_shape().back();
    LayerNormParams params = {
        .normalized_shape = normalized_shape,
        .epsilon = epsilon
    };
    [command_encoder setBytes:&params length:sizeof(params) atIndex:4];

    // The grid size is the number of rows to normalize
    size_t num_rows = mt_in->get_size() / normalized_shape;
    MTLSize grid_size = MTLSizeMake(num_rows, 1, 1);

    NSUInteger threadgroup_size_val = [pipeline_state maxTotalThreadsPerThreadgroup];
    if (threadgroup_size_val > num_rows) {
        threadgroup_size_val = num_rows;
    }
    MTLSize threadgroup_size = MTLSizeMake(threadgroup_size_val, 1, 1);

    [command_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [command_encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    [kernel_func release];
    [pipeline_state release];
}

void MetalBackend::lookup(const Tensor* weights, const Tensor* indices, Tensor* out) {
    const MetalTensor* mt_weights = dynamic_cast<const MetalTensor*>(weights);
    const MetalTensor* mt_indices = dynamic_cast<const MetalTensor*>(indices);
    MetalTensor* mt_out = dynamic_cast<MetalTensor*>(out);

    NSError* error = nil;
    id<MTLFunction> kernel_func = [library newFunctionWithName:@"lookup"];
    if (!kernel_func) {
        throw std::runtime_error("Failed to create Metal kernel function for lookup");
    }

    id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:kernel_func error:&error];
    if (!pipeline_state) {
        throw std::runtime_error("Failed to create Metal pipeline state for lookup");
    }

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

    [command_encoder setComputePipelineState:pipeline_state];
    [command_encoder setBuffer:mt_weights->get_buffer() offset:0 atIndex:0];
    [command_encoder setBuffer:mt_indices->get_buffer() offset:0 atIndex:1];
    [command_encoder setBuffer:mt_out->get_buffer() offset:0 atIndex:2];

    // The last dimension of the output is the embedding dimension
    uint32_t embedding_dim = mt_out->get_shape().back();
    LookupParams params = {
        .embedding_dim = embedding_dim
    };
    [command_encoder setBytes:&params length:sizeof(params) atIndex:3];

    // The grid size is the number of indices to look up
    size_t num_indices = mt_indices->get_size();
    MTLSize grid_size = MTLSizeMake(num_indices, 1, 1);

    NSUInteger threadgroup_size_val = [pipeline_state maxTotalThreadsPerThreadgroup];
    if (threadgroup_size_val > num_indices) {
        threadgroup_size_val = num_indices;
    }
    MTLSize threadgroup_size = MTLSizeMake(threadgroup_size_val, 1, 1);

    [command_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [command_encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    [kernel_func release];
    [pipeline_state release];
}

void MetalBackend::gelu(const Tensor* in, Tensor* out) {
    const MetalTensor* mt_in = dynamic_cast<const MetalTensor*>(in);
    MetalTensor* mt_out = dynamic_cast<MetalTensor*>(out);

    NSError* error = nil;
    id<MTLFunction> kernel_func = [library newFunctionWithName:@"gelu"];
    if (!kernel_func) {
        throw std::runtime_error("Failed to create Metal kernel function for gelu");
    }

    id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:kernel_func error:&error];
    if (!pipeline_state) {
        throw std::runtime_error("Failed to create Metal pipeline state for gelu");
    }

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

    [command_encoder setComputePipelineState:pipeline_state];
    [command_encoder setBuffer:mt_in->get_buffer() offset:0 atIndex:0];
    [command_encoder setBuffer:mt_out->get_buffer() offset:0 atIndex:1];

    MTLSize grid_size = MTLSizeMake(mt_in->get_size(), 1, 1);

    NSUInteger threadgroup_size_val = [pipeline_state maxTotalThreadsPerThreadgroup];
    if (threadgroup_size_val > mt_in->get_size()) {
        threadgroup_size_val = mt_in->get_size();
    }
    MTLSize threadgroup_size = MTLSizeMake(threadgroup_size_val, 1, 1);

    [command_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [command_encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    [kernel_func release];
    [pipeline_state release];
}

void MetalBackend::transpose2d(const Tensor* in, Tensor* out) {
    const MetalTensor* mt_in = dynamic_cast<const MetalTensor*>(in);
    MetalTensor* mt_out = dynamic_cast<MetalTensor*>(out);

    if (mt_in->get_shape().size() != 2) {
        throw std::runtime_error("Transpose2d currently only supports 2D tensors.");
    }

    NSError* error = nil;
    id<MTLFunction> kernel_func = [library newFunctionWithName:@"transpose2d"];
    if (!kernel_func) {
        throw std::runtime_error("Failed to create Metal kernel function for transpose2d");
    }

    id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:kernel_func error:&error];
    if (!pipeline_state) {
        throw std::runtime_error("Failed to create Metal pipeline state for transpose2d");
    }

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

    [command_encoder setComputePipelineState:pipeline_state];
    [command_encoder setBuffer:mt_in->get_buffer() offset:0 atIndex:0];
    [command_encoder setBuffer:mt_out->get_buffer() offset:0 atIndex:1];

    auto in_shape = mt_in->get_shape();
    Transpose2DParams params = {
        .width = (uint32_t)in_shape[1],
        .height = (uint32_t)in_shape[0]
    };
    [command_encoder setBytes:&params length:sizeof(params) atIndex:2];

    MTLSize grid_size = MTLSizeMake(in_shape[1], in_shape[0], 1);
    
    NSUInteger threadgroup_width = [pipeline_state threadExecutionWidth];
    NSUInteger threadgroup_height = [pipeline_state maxTotalThreadsPerThreadgroup] / threadgroup_width;
    MTLSize threadgroup_size = MTLSizeMake(threadgroup_width, threadgroup_height, 1);

    [command_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [command_encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    [kernel_func release];
    [pipeline_state release];
}

void MetalBackend::apply_causal_mask(Tensor* scores) {
    MetalTensor* mt_scores = dynamic_cast<MetalTensor*>(scores);

    auto shape = mt_scores->get_shape();
    if (shape.size() != 4) {
        throw std::runtime_error("apply_causal_mask expects a 4D tensor (batch, heads, seq, seq)");
    }

    uint32_t seq_len = shape[2];
    uint32_t batch_size = shape[0] * shape[1];

    NSError* error = nil;
    id<MTLFunction> kernel_func = [library newFunctionWithName:@"apply_causal_mask"];
    if (!kernel_func) {
        throw std::runtime_error("Failed to create Metal kernel function for apply_causal_mask");
    }

    id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:kernel_func error:&error];
    if (!pipeline_state) {
        throw std::runtime_error("Failed to create Metal pipeline state for apply_causal_mask");
    }

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

    [command_encoder setComputePipelineState:pipeline_state];
    [command_encoder setBuffer:mt_scores->get_buffer() offset:0 atIndex:0];
    [command_encoder setBytes:&seq_len length:sizeof(seq_len) atIndex:1];

    MTLSize grid_size = MTLSizeMake(seq_len, seq_len, batch_size);

    NSUInteger threadgroup_width = 8;
    NSUInteger threadgroup_height = 8;
    NSUInteger threadgroup_depth = 1;
    MTLSize threadgroup_size = MTLSizeMake(threadgroup_width, threadgroup_height, threadgroup_depth);

    [command_encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [command_encoder endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    [kernel_func release];
    [pipeline_state release];
}
