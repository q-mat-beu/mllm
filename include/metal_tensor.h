#pragma once

#include "tensor.h"
#include <vector>
#import <Metal/Metal.h>

class MetalBackend;

class MetalTensor : public Tensor {
public:
    MetalTensor(const std::vector<int>& shape, DataType dtype, MetalBackend* backend);
    ~MetalTensor() override;

    void allocate() override;
    void reshape(const std::vector<int>& new_shape) override;

    void copy_from_float(const std::vector<float>& data) override;
    void copy_to_float(std::vector<float>& data) const override;
    void copy_from_int(const std::vector<int>& data) override;
    void copy_to_int(std::vector<int>& data) const override;

    const std::vector<int>& get_shape() const override { return shape; }
    DataType get_dtype() const override { return dtype; }
    size_t get_size() const override;
    size_t get_byte_size() const override;

    void* get_data() override;
    const void* get_data() const override;

    id<MTLBuffer> get_buffer() const { return buffer; }

private:
    std::vector<int> shape;
    DataType dtype;
    id<MTLBuffer> buffer;
    MetalBackend* backend;
};
