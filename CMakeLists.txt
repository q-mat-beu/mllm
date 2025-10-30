cmake_minimum_required(VERSION 3.20)
project(mllm LANGUAGES CXX OBJCXX)

option(ENABLE_PROFILING "Enable performance profiling output" OFF)

if (ENABLE_PROFILING)
    add_compile_definitions(ENABLE_PROFILING)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD 17)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib -lomp")

include(FetchContent)

# nlohmann/json for JSON parsing
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
)

# Google Test for testing
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

FetchContent_MakeAvailable(json googletest)

# Add include directory
include_directories(include)

# Main executable
add_executable(mllm src/main.cpp src/tokenizer.cpp src/metal_tensor.mm src/cpu_tensor.cpp src/metal_backend.mm src/cpu_backend.mm src/attention.mm src/linear.mm src/layernorm.mm src/embedding.mm src/transformer_block.mm src/gpt2_model.mm src/inference.mm src/gpt2_inference.mm src/debug_utils.mm)
target_link_libraries(mllm PRIVATE nlohmann_json::nlohmann_json)

target_link_libraries(mllm PRIVATE "-framework Metal" "-framework Foundation" "-framework Accelerate")



# Testing
enable_testing()
add_executable(mllm_tests tests/test_main.cpp tests/test_tokenizer.cpp src/tokenizer.cpp src/metal_tensor.mm src/cpu_tensor.cpp src/metal_backend.mm src/cpu_backend.mm tests/test_metal_backend.mm tests/test_cpu_backend.mm src/attention.mm tests/test_attention.mm src/linear.mm tests/test_linear.mm src/layernorm.mm tests/test_layernorm.mm src/embedding.mm tests/test_embedding.mm tests/test_activations.mm src/transformer_block.mm tests/test_transformer_block.mm src/gpt2_model.mm tests/test_gpt2_model.mm src/inference.mm src/gpt2_inference.mm tests/test_causal_mask.mm src/debug_utils.mm tests/test_gpt2_model_performance.mm)
target_link_libraries(mllm_tests PRIVATE GTest::gtest_main nlohmann_json::nlohmann_json)

target_link_libraries(mllm_tests PRIVATE "-framework Metal" "-framework Foundation" "-framework Accelerate")



include_directories(include)

# Compile Metal shaders
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/default.metallib
    COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders.metal -o ${CMAKE_BINARY_DIR}/default.air -I ${CMAKE_CURRENT_SOURCE_DIR}/include
    COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/default.air -o ${CMAKE_BINARY_DIR}/default.metallib
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders.metal
)

add_custom_target(Shaders DEPENDS ${CMAKE_BINARY_DIR}/default.metallib)

add_dependencies(mllm Shaders)
add_dependencies(mllm_tests Shaders)

add_custom_command(TARGET mllm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_BINARY_DIR}/default.metallib
            $<TARGET_FILE_DIR:mllm>/default.metallib)

add_custom_command(TARGET mllm_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_BINARY_DIR}/default.metallib
            $<TARGET_FILE_DIR:mllm_tests>/default.metallib)

# Copy resources directory to build directory
add_custom_command(TARGET mllm_tests POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_CURRENT_SOURCE_DIR}/resources
                   $<TARGET_FILE_DIR:mllm_tests>/resources)

include(GoogleTest)
gtest_discover_tests(mllm_tests)
