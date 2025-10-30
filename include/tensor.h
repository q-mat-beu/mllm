#pragma once

#include <vector>
#include <string>
#include <memory>

enum DataType {
    MLLM_FLOAT32,
    MLLM_INT32
};

class Tensor {
public:
    virtual ~Tensor() = default;

    virtual void allocate() = 0;
    virtual void reshape(const std::vector<int>& new_shape) = 0;

    virtual void copy_from_float(const std::vector<float>& data) = 0;
    virtual void copy_to_float(std::vector<float>& data) const = 0;
    virtual void copy_from_int(const std::vector<int>& data) = 0;
    virtual void copy_to_int(std::vector<int>& data) const = 0;

    virtual const std::vector<int>& get_shape() const = 0;
    virtual DataType get_dtype() const = 0;
    virtual size_t get_size() const = 0;
    virtual size_t get_byte_size() const = 0;

    virtual void* get_data() = 0;
    virtual const void* get_data() const = 0;
};
