# MLLM: Metal/CPU Large Language Model Inference

## Project Description

MLLM (Metal Large Language Model) is a C++/Objective-C++ project designed for efficient inference of Large Language Models, specifically GPT-2, leveraging Apple's Metal framework for GPU acceleration on macOS and providing a CPU-based fallback using the Accelerate framework for optimized linear algebra operations.

This project aims to provide a clear, performant, and extensible foundation for experimenting with LLM inference on Apple hardware, with a focus on demonstrating both GPU and CPU execution paths.

## Features

*   **GPT-2 Model Inference:** Implements the forward pass for a GPT-2 small model.
*   **Metal Backend:** Utilizes Apple's Metal framework for high-performance GPU-accelerated tensor operations.
*   **CPU Backend:** Provides a CPU-based backend leveraging Apple's Accelerate framework (BLAS/LAPACK) for optimized linear algebra.
*   **Backend Agnostic Design:** A flexible architecture allowing easy switching between Metal and CPU backends.
*   **Modular Layer Implementation:** Clear and separate implementations for core transformer layers (Linear, LayerNorm, MultiHeadAttention, Embedding, TransformerBlock).
*   **Tokenizer:** Basic BPE tokenizer implementation compatible with GPT-2.
*   **Profiling:** Built-in profiling capabilities to measure performance of individual operations, toggleable via a compiler flag.

## Prerequisites

Before you begin, ensure you have the following installed:

*   **macOS:** This project is developed for macOS and heavily relies on Apple's frameworks.
*   **Xcode Command Line Tools:**
    ```bash
    xcode-select --install
    ```
*   **CMake:** Version 3.20 or higher.
    ```bash
    brew install cmake
    ```
*   **Homebrew (Optional, but Recommended for `libomp`):**
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
*   **libomp (for CPU backend optimization):**
    ```bash
    brew install libomp
    ```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/mllm.git
    cd mllm
    ```

2.  **Download GPT-2 Weights and BPE Resources:**
    This project uses the GPT-2 small model weights and BPE tokenizer files. You will need to download `vocab.json`, `merges.txt`, and `pytorch_model.bin` (or a converted `gpt2_weights.json`) and place them in the `resources/` directory.

    *   **Option 1: Manual Download (Recommended for `pytorch_model.bin` conversion)**
        You can download these files from Hugging Face's `gpt2` model repository:
        -   `vocab.json`: [https://huggingface.co/gpt2/resolve/main/vocab.json](https://huggingface.co/gpt2/resolve/main/vocab.json)
        -   `merges.txt`: [https://huggingface.co/gpt2/resolve/main/merges.txt](https://huggingface.co/gpt2/resolve/main/merges.txt)
        -   `pytorch_model.bin`: [https://huggingface.co/gpt2/resolve/main/pytorch_model.bin](https://huggingface.co/gpt2/resolve/main/pytorch_model.bin)

        Place `vocab.json` and `merges.txt` directly into the `resources/` directory.

        To convert `pytorch_model.bin` into the `gpt2_weights.json` format expected by this project, use the provided Python script:
        ```bash
        pip install torch transformers
        python resources/convert_gpt2_weights.py --pytorch_model_path resources/pytorch_model.bin --output_path resources/gpt2_weights.json
        ```

3.  **Compile the Project:**
    ```bash
    mkdir build
    cd build
    cmake ..
    cmake --build .
    ```

    **Note for OpenMP:** If you encounter issues with OpenMP not being found, you might need to manually specify its location during CMake configuration. After installing `libomp` via Homebrew, you can try:
    ```bash
    cmake -DCMAKE_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
          -DCMAKE_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
          -DCMAKE_OBJCXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
          -DCMAKE_EXE_LINKER_FLAGS="-L/opt/homebrew/opt/libomp/lib -lomp" ..
    cmake --build .
    ```

## Usage

Run the `mllm` executable from the `build` directory.

```bash
./mllm --prompt "Hello, my name is" --model ../resources/gpt2_weights.json --max_tokens 50 --temperature 0.7 --backend metal
```

**Command-line Arguments:**

*   `--prompt <text>`: The input prompt for the model.
*   `--model <path>`: Path to the GPT-2 weights JSON file (e.g., `../resources/gpt2_weights.json`).
*   `--max_tokens <N>`: Maximum number of tokens to generate (default: 50).
*   `--temperature <T>`: Sampling temperature (default: 1.0).
*   `--top_k <K>`: Top-K sampling (default: 40).
*   `--top_p <P>`: Top-P (nucleus) sampling (default: 0.9).
*   `--no_top_k`: Disable Top-K sampling.
*   `--no_top_p`: Disable Top-P sampling.
*   `--vocab_path <path>`: Path to `vocab.json` (default: `resources/vocab.json`).
*   `--merges_path <path>`: Path to `merges.txt` (default: `resources/merges.txt`).
*   `--backend <metal|cpu>`: Select the backend for inference (`metal` or `cpu`, default: `metal`).

## Testing

To run the unit tests:

```bash
cd build
ctest
```

## Profiling

To enable detailed performance profiling output during compilation, configure CMake with the `ENABLE_PROFILING` option:

```bash
cd build
cmake -DENABLE_PROFILING=ON ..
cmake --build .
```

Then, run the performance tests or the main executable. The profiling output will be printed to `stdout`.

```bash
./mllm_tests --gtest_filter=GPT2ModelPerformanceTest.ForwardPassCpu
# or
./mllm --prompt "Hello" --model ../resources/gpt2_weights.json --backend cpu
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
