#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define NS_RETURNS_INNER_POINTER
#define NS_ASSUME_NONNULL_BEGIN
#define NS_ASSUME_NONNULL_END
#define API_AVAILABLE(...)
#define API_UNAVAILABLE(...)
#define NS_SWIFT_UNAVAILABLE(...)
#define NS_REFINED_FOR_SWIFT
#define NS_DESIGNATED_INITIALIZER
#else
#include <cstdint>
#endif

// Define the MatMulParams struct for both Metal and C++
struct MatMulParams {
    uint32_t widthA;
    uint32_t heightA;
    uint32_t widthB;
    uint32_t heightB;
    uint32_t widthC;
    uint32_t heightC;
};

struct BatchedMatMulParams {
    uint32_t widthA, heightA;
    uint32_t widthB, heightB;
    uint32_t widthC, heightC;
    uint32_t batch_size;
    uint32_t strideA, strideB, strideC;
};

struct TransposeParams {
    uint32_t dims[3];
    uint32_t perm[3];
};

struct Transpose4DParams {
    uint32_t dims[4];
    uint32_t perm[4];
};

struct LayerNormParams {
    uint32_t normalized_shape;
    float epsilon;
};

struct LookupParams {
    uint32_t embedding_dim;
};

struct Transpose2DParams {
    uint32_t width;
    uint32_t height;
};