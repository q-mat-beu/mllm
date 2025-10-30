#include "cpu_tensor.h"
#include <numeric>
#include <stdexcept>

CpuTensor::CpuTensor(const std::vector<int>& shape, DataType dtype)
    : shape(shape), dtype(dtype), data(nullptr) {}

CpuTensor::~CpuTensor() {
    if (data) {
        free(data);
    }
}

void CpuTensor::allocate() {
    if (!data) {
        data = malloc(get_byte_size());
    }
}

void CpuTensor::reshape(const std::vector<int>& new_shape) {
    size_t new_size = 1;
    for (int dim : new_shape) {
        new_size *= dim;
    }
    if (new_size != get_size()) {
        throw std::runtime_error("Cannot reshape tensor to a different number of elements.");
    }
    this->shape = new_shape;
}

void CpuTensor::copy_from_float(const std::vector<float>& data) {
    if (this->dtype != MLLM_FLOAT32) {
        throw std::runtime_error("Cannot copy float data to a non-float tensor.");
    }
    if (data.size() != get_size()) {
        throw std::runtime_error("Size of data to copy does not match tensor size.");
    }
    memcpy(this->data, data.data(), get_byte_size());
}

void CpuTensor::copy_to_float(std::vector<float>& data) const {
    if (this->dtype != MLLM_FLOAT32) {
        throw std::runtime_error("Cannot copy non-float tensor data to a float vector.");
    }
    data.resize(get_size());
    memcpy(data.data(), this->data, get_byte_size());
}

void CpuTensor::copy_from_int(const std::vector<int>& data) {
    if (this->dtype != MLLM_INT32) {
        throw std::runtime_error("Cannot copy int data to a non-int tensor.");
    }
    if (data.size() != get_size()) {
        throw std::runtime_error("Size of data to copy does not match tensor size.");
    }
    memcpy(this->data, data.data(), get_byte_size());
}

void CpuTensor::copy_to_int(std::vector<int>& data) const {
    if (this->dtype != MLLM_INT32) {
        throw std::runtime_error("Cannot copy non-int tensor data to an int vector.");
    }
    data.resize(get_size());
    memcpy(data.data(), this->data, get_byte_size());
}

size_t CpuTensor::get_size() const {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

size_t CpuTensor::get_byte_size() const {
    size_t element_size = 0;
    switch (dtype) {
        case MLLM_FLOAT32:
            element_size = sizeof(float);
            break;
        case MLLM_INT32:
            element_size = sizeof(int);
            break;
        default:
            throw std::runtime_error("Unsupported data type in get_byte_size.");
    }
    return get_size() * element_size;
}
