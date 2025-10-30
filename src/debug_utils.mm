#include "debug_utils.h"
#include "tensor.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

// Helper function for debugging
// void print_tensor_stats(const std::string& name, Tensor* tensor) {
//     std::vector<float> data;
//     tensor->copy_to_float(data);
//     if (data.empty()) {
//         std::cout << name << " is empty." << std::endl;
//         return;
//     }

//     float min_val = data[0];
//     float max_val = data[0];
//     double sum_val = 0.0;
//     bool has_nan = false;

//     for (float val : data) {
//         if (std::isnan(val)) {
//             has_nan = true;
//         }
//         min_val = std::min(min_val, val);
//         max_val = std::max(max_val, val);
//         sum_val += val;
//     }
//     double mean_val = sum_val / data.size();

//     std::cout << "--- " << name << " Stats ---" << std::endl;
//     std::cout << "  Shape: [";
//     for (int dim : tensor->get_shape()) {
//         std::cout << dim << " ";
//     }
//     std::cout << "]" << std::endl;
//     std::cout << "  Min: " << min_val << ", Max: " << max_val << ", Mean: " << mean_val;
//     if (has_nan) {
//         std::cout << ", CONTAINS NAN!";
//     }
//     std::cout << std::endl;
// }
