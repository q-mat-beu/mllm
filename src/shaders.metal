#include <metal_stdlib>
#include "metal_types.h"
using namespace metal;

kernel void matmul(const device float* inA [[buffer(0)]],
                   const device float* inB [[buffer(1)]],
                   device float* outC [[buffer(2)]],
                   constant MatMulParams& params [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.widthC || gid.y >= params.heightC) {
        return;
    }

    float sum = 0.0f;
    for (uint i = 0; i < params.widthA; ++i) {
        sum += inA[gid.y * params.widthA + i] * inB[i * params.widthB + gid.x];
    }
    outC[gid.y * params.widthC + gid.x] = sum;
}

kernel void matmul_batched(const device float* inA [[buffer(0)]],
                           const device float* inB [[buffer(1)]],
                           device float* outC [[buffer(2)]],
                           constant BatchedMatMulParams& params [[buffer(3)]],
                           uint3 gid [[thread_position_in_grid]]) {
    // gid.x -> output column
    // gid.y -> output row
    // gid.z -> batch index

    if (gid.x >= params.widthC || gid.y >= params.heightC || gid.z >= params.batch_size) {
        return;
    }

    uint batch_offset_A = gid.z * params.strideA;
    uint batch_offset_B = gid.z * params.strideB;
    uint batch_offset_C = gid.z * params.strideC;

    float sum = 0.0f;
    for (uint i = 0; i < params.widthA; ++i) {
        uint indexA = batch_offset_A + gid.y * params.widthA + i;
        uint indexB = batch_offset_B + i * params.widthB + gid.x;
        sum += inA[indexA] * inB[indexB];
    }

    uint indexC = batch_offset_C + gid.y * params.widthC + gid.x;
    outC[indexC] = sum;
}

kernel void add(const device float* inA [[buffer(0)]],
                const device float* inB [[buffer(1)]],
                device float* outC [[buffer(2)]],
                uint gid [[thread_position_in_grid]]) {
    outC[gid] = inA[gid] + inB[gid];
}

kernel void broadcast_add(const device float* inA [[buffer(0)]],
                         const device float* inB_bias [[buffer(1)]],
                         device float* outC [[buffer(2)]],
                         constant uint& width [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    uint col = gid % width;
    outC[gid] = inA[gid] + inB_bias[col];
}

kernel void softmax_rowwise(const device float* in [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant uint& row_size [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) { // 1D grid over rows

    uint row_start_idx = gid * row_size;

    // 1. Find max value in the row
    float max_val = -FLT_MAX;
    for (uint i = 0; i < row_size; ++i) {
        max_val = fmax(max_val, in[row_start_idx + i]);
    }

    // 2. Calculate sum of exps
    float sum_exp = 0.0f;
    for (uint i = 0; i < row_size; ++i) {
        sum_exp += exp(in[row_start_idx + i] - max_val);
    }

    // 3. Calculate softmax
    for (uint i = 0; i < row_size; ++i) {
        uint idx = row_start_idx + i;
        if (sum_exp == 0.0f) { // Avoid division by zero
            out[idx] = 0.0f;
        } else {
            out[idx] = exp(in[idx] - max_val) / sum_exp;
        }
    }
}

kernel void scale(const device float* in [[buffer(0)]],
                  device float* out [[buffer(1)]],
                  constant float& scale_factor [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    out[gid] = in[gid] * scale_factor;
}

kernel void transpose(const device float* in [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant TransposeParams& params [[buffer(2)]],
                      uint3 gid [[thread_position_in_grid]]) {

    uint d1 = params.dims[1];
    uint d2 = params.dims[2];

    uint p0 = params.perm[0];
    uint p1 = params.perm[1];
    uint p2 = params.perm[2];

    uint out_d0 = params.dims[p0];
    uint out_d1 = params.dims[p1];
    uint out_d2 = params.dims[p2];

    if (gid.x >= out_d0 || gid.y >= out_d1 || gid.z >= out_d2) {
        return;
    }
    
    uint in_coords[3];
    in_coords[p0] = gid.x;
    in_coords[p1] = gid.y;
    in_coords[p2] = gid.z;

    uint in_idx = in_coords[0] * d1 * d2 + in_coords[1] * d2 + in_coords[2];
    uint out_idx = gid.x * out_d1 * out_d2 + gid.y * out_d2 + gid.z;

    out[out_idx] = in[in_idx];
}

kernel void transpose4d(const device float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant Transpose4DParams& params [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) { // 1D grid

    uint out_idx = gid;

    // Output strides (for decoding out_idx)
    uint out_strides[3];
    out_strides[2] = params.dims[params.perm[3]];
    out_strides[1] = out_strides[2] * params.dims[params.perm[2]];
    out_strides[0] = out_strides[1] * params.dims[params.perm[1]];

    // Calculate output coordinates
    uint out_coords[4];
    uint temp_idx = out_idx;
    out_coords[0] = temp_idx / out_strides[0];
    temp_idx %= out_strides[0];
    out_coords[1] = temp_idx / out_strides[1];
    temp_idx %= out_strides[1];
    out_coords[2] = temp_idx / out_strides[2];
    out_coords[3] = temp_idx % out_strides[2];

    // Get input coordinates by un-permuting output coordinates
    uint in_coords[4];
    in_coords[params.perm[0]] = out_coords[0];
    in_coords[params.perm[1]] = out_coords[1];
    in_coords[params.perm[2]] = out_coords[2];
    in_coords[params.perm[3]] = out_coords[3];

    // Input strides (for encoding in_idx)
    uint in_strides[3];
    in_strides[2] = params.dims[3];
    in_strides[1] = in_strides[2] * params.dims[2];
    in_strides[0] = in_strides[1] * params.dims[1];

    // Calculate input index
    uint in_idx = in_coords[0] * in_strides[0] +
                  in_coords[1] * in_strides[1] +
                  in_coords[2] * in_strides[2] +
                  in_coords[3];

    if (out_idx < params.dims[0] * params.dims[1] * params.dims[2] * params.dims[3]) {
        out[out_idx] = in[in_idx];
    }
}

kernel void apply_causal_mask(device float* scores [[buffer(0)]],
                              constant uint& seq_len [[buffer(1)]],
                              uint3 gid [[thread_position_in_grid]]) {
    // gid.x -> col
    // gid.y -> row
    // gid.z -> batch item

    if (gid.x > gid.y) {
        uint index = gid.z * seq_len * seq_len + gid.y * seq_len + gid.x;
        scores[index] = -1.0e9f; // A large negative number
    }
}

kernel void layernorm(const device float* in [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      const device float* gamma [[buffer(2)]],
                      const device float* beta [[buffer(3)]],
                      constant LayerNormParams& params [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) { // 1D grid over rows

    uint feature_dim = params.normalized_shape;
    uint row_start_idx = gid * feature_dim;

    // 1. Calculate mean
    float sum = 0.0f;
    for (uint i = 0; i < feature_dim; ++i) {
        sum += in[row_start_idx + i];
    }
    float mean = sum / feature_dim;

    // 2. Calculate variance
    float sum_sq = 0.0f;
    for (uint i = 0; i < feature_dim; ++i) {
        float val = in[row_start_idx + i] - mean;
        sum_sq += val * val;
    }
    float variance = sum_sq / feature_dim;
    float inv_stddev = rsqrt(variance + params.epsilon);

    // 3. Normalize
    for (uint i = 0; i < feature_dim; ++i) {
        uint idx = row_start_idx + i;
        out[idx] = (in[idx] - mean) * inv_stddev * gamma[i] + beta[i];
    }
}

kernel void lookup(const device float* weights [[buffer(0)]],
                   const device int* indices [[buffer(1)]], // Assuming integer indices
                   device float* out [[buffer(2)]],
                   constant LookupParams& params [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {

    uint embedding_dim = params.embedding_dim;
    
    // 1. Get the index for this thread
    int token_index = indices[gid];

    // 2. Calculate start of source and destination rows
    uint src_offset = token_index * embedding_dim;
    uint dest_offset = gid * embedding_dim;

    // 3. Copy the embedding vector
    for (uint i = 0; i < embedding_dim; ++i) {
        out[dest_offset + i] = weights[src_offset + i];
    }
}

kernel void gelu(const device float* in [[buffer(0)]],
                 device float* out [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
    
    float x = in[gid];
    float sigmoid_arg = 1.702f * x;
    float sigmoid_val = 1.0f / (1.0f + exp(-sigmoid_arg));
    out[gid] = x * sigmoid_val;
}

kernel void transpose2d(const device float* in [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant Transpose2DParams& params [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    out[gid.x * params.height + gid.y] = in[gid.y * params.width + gid.x];
}