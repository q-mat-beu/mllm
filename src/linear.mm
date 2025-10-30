#include "linear.h"
#include "backend.h"
#include <memory>
#include <stdexcept>

Linear::Linear(Backend* backend, int in_features, int out_features)
    : backend(backend), in_features(in_features), out_features(out_features) {
    
    weight = backend->create_tensor({in_features, out_features}, MLLM_FLOAT32);
    bias = backend->create_tensor({out_features}, MLLM_FLOAT32);

    weight->allocate();
    bias->allocate();
}

Linear::~Linear() {
    // No need to delete, std::unique_ptr will handle it
}

void Linear::load_weights(const std::vector<float>& weight_data, const std::vector<float>& bias_data) {
    if (weight_data.size() != weight->get_size()) {
        throw std::runtime_error("Incorrect size for weight data provided to Linear layer.");
    }
    if (bias_data.size() != bias->get_size()) {
        throw std::runtime_error("Incorrect size for bias data provided to Linear layer.");
    }
    weight->copy_from_float(weight_data);
    bias->copy_from_float(bias_data);
}

std::unique_ptr<Tensor> Linear::forward(const Tensor* input) {
    auto original_shape = input->get_shape();
    if (original_shape.back() != in_features) {
        throw std::runtime_error("Linear layer input features mismatch.");
    }

    int total_rows = 1;
    for (size_t i = 0; i < original_shape.size() - 1; ++i) {
        total_rows *= original_shape[i];
    }
    
    auto reshaped_input = backend->create_tensor({total_rows, in_features}, input->get_dtype());
    reshaped_input->allocate();
    
    std::vector<float> input_data;
    input->copy_to_float(input_data);
    reshaped_input->copy_from_float(input_data);

    std::vector<int> proj_shape_2d = {total_rows, out_features};
    auto output_proj = backend->create_tensor(proj_shape_2d, MLLM_FLOAT32);
    output_proj->allocate();

    backend->matrix_multiply(reshaped_input.get(), weight.get(), output_proj.get());

    std::vector<int> final_output_shape = original_shape;
    final_output_shape.back() = out_features;
    output_proj->reshape(final_output_shape);

    auto output = backend->create_tensor(final_output_shape, MLLM_FLOAT32);
    output->allocate();
    backend->broadcast_add(output_proj.get(), bias.get(), output.get());

    return output;
}