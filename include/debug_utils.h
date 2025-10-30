#pragma once

#include "tensor.h"
#include <string>
#include <vector> // For std::vector in print_tensor

// Helper function for debugging
void print_tensor_stats(const std::string& name, const Tensor* tensor);
void print_tensor_data(const std::string& name, const Tensor* tensor);
