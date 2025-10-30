import json
import os
import argparse

try:
    import torch
    import numpy as np
except ImportError:
    print("Error: torch and numpy are required. Please install them with 'pip install torch numpy'")
    exit(1)

def convert_gpt2_weights(pytorch_bin_path, output_json_path, embed_dim, num_layers):
    """
    Converts PyTorch GPT-2 model weights from a .bin file to a custom JSON format.

    Args:
        pytorch_bin_path (str): Path to the PyTorch model's state_dict (.bin file).
        output_json_path (str): Path where the converted JSON weights will be saved.
        embed_dim (int): The embedding dimension of the GPT-2 model (e.g., 768 for small).
        num_layers (int): The number of transformer layers in the GPT-2 model (e.g., 12 for small).
    """
    if not os.path.exists(pytorch_bin_path):
        print(f"Error: PyTorch model file not found at {pytorch_bin_path}")
        return

    print(f"Loading PyTorch state_dict from {pytorch_bin_path}...")
    try:
        state_dict = torch.load(pytorch_bin_path, map_location='cpu')
        print("PyTorch state_dict loaded successfully.")
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        return

    our_weights = {}

    def get_tensor(key):
        if key not in state_dict:
            raise KeyError(f"Weight '{key}' not found in the model file.")
        return state_dict[key].float()

    try:
        # Token Embeddings (wte)
        print("Converting token embeddings (wte.weight)...")
        our_weights['wte.weight'] = get_tensor('wte.weight').flatten().tolist()

        # Position Embeddings (wpe)
        print("Converting position embeddings (wpe.weight)...")
        our_weights['wpe.weight'] = get_tensor('wpe.weight').flatten().tolist()

        # Transformer Blocks (h.0 to h.num_layers-1)
        for i in range(num_layers):
            print(f"Converting weights for transformer block h.{i}...")
            prefix = f"h.{i}."
            our_prefix = f"h.{i}."

            # LayerNorm 1 (ln_1)
            our_weights[our_prefix + 'ln_1.weight'] = get_tensor(prefix + 'ln_1.weight').flatten().tolist()
            our_weights[our_prefix + 'ln_1.bias'] = get_tensor(prefix + 'ln_1.bias').flatten().tolist()

            # Attention (attn)
            c_attn_weight = get_tensor(prefix + 'attn.c_attn.weight').numpy()
            c_attn_bias = get_tensor(prefix + 'attn.c_attn.bias').numpy()

            # Split Q, K, V weights and biases
            w_q_data = c_attn_weight[:, :embed_dim].flatten().tolist()
            w_k_data = c_attn_weight[:, embed_dim:2*embed_dim].flatten().tolist()
            w_v_data = c_attn_weight[:, 2*embed_dim:].flatten().tolist()

            b_q_data = c_attn_bias[:embed_dim].flatten().tolist()
            b_k_data = c_attn_bias[embed_dim:2*embed_dim].flatten().tolist()
            b_v_data = c_attn_bias[2*embed_dim:].flatten().tolist()
            
            our_weights[our_prefix + 'attn.w_q_data'] = w_q_data
            our_weights[our_prefix + 'attn.b_q_data'] = b_q_data
            our_weights[our_prefix + 'attn.w_k_data'] = w_k_data
            our_weights[our_prefix + 'attn.b_k_data'] = b_k_data
            our_weights[our_prefix + 'attn.w_v_data'] = w_v_data
            our_weights[our_prefix + 'attn.b_v_data'] = b_v_data

            # c_proj.weight (output projection)
            our_weights[our_prefix + 'attn.c_proj.weight'] = get_tensor(prefix + 'attn.c_proj.weight').flatten().tolist()
            our_weights[our_prefix + 'attn.c_proj.bias'] = get_tensor(prefix + 'attn.c_proj.bias').flatten().tolist()

            # LayerNorm 2 (ln_2)
            our_weights[our_prefix + 'ln_2.weight'] = get_tensor(prefix + 'ln_2.weight').flatten().tolist()
            our_weights[our_prefix + 'ln_2.bias'] = get_tensor(prefix + 'ln_2.bias').flatten().tolist()

            # MLP (Feed-Forward Network)
            our_weights[our_prefix + 'mlp.c_fc.weight'] = get_tensor(prefix + 'mlp.c_fc.weight').flatten().tolist()
            our_weights[our_prefix + 'mlp.c_fc.bias'] = get_tensor(prefix + 'mlp.c_fc.bias').flatten().tolist()
            our_weights[our_prefix + 'mlp.c_proj.weight'] = get_tensor(prefix + 'mlp.c_proj.weight').flatten().tolist()
            our_weights[our_prefix + 'mlp.c_proj.bias'] = get_tensor(prefix + 'mlp.c_proj.bias').flatten().tolist()

        # Final LayerNorm (ln_f)
        print("Converting final LayerNorm (ln_f)...")
        our_weights['ln_f.weight'] = get_tensor('ln_f.weight').flatten().tolist()
        our_weights['ln_f.bias'] = get_tensor('ln_f.bias').flatten().tolist()

        # Language Model Head (lm_head)
        print("Converting language model head (lm_head)...")
        if 'lm_head.weight' in state_dict:
            # lm_head is a standard nn.Linear layer, so its weight is (vocab_size, embed_dim).
            # My C++ Linear layer expects (embed_dim, vocab_size), so I need to transpose it.
            our_weights['lm_head.weight'] = get_tensor('lm_head.weight').transpose(0, 1).flatten().tolist()
        else:
            # wte.weight is (vocab_size, embed_dim). It's used as the weight for the lm_head (tied weights).
            # It also needs to be transposed before use in the final linear layer.
            print("Warning: 'lm_head.weight' not found. Using token embeddings (wte.weight) instead (tied weights).")
            our_weights['lm_head.weight'] = get_tensor('wte.weight').transpose(0, 1).flatten().tolist()
        
        # lm_head.bias is not always present
        if 'lm_head.bias' in state_dict:
            our_weights['lm_head.bias'] = get_tensor('lm_head.bias').flatten().tolist()
        else:
            print("Info: 'lm_head.bias' not found. This is common if the model doesn't use a bias for the LM head.")
            our_weights['lm_head.bias'] = []


    except KeyError as e:
        print(f"Error during weight conversion: {e}")
        return

    print(f"Saving converted weights to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(our_weights, f) # Removed indent for smaller file size
    print("Conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GPT-2 PyTorch weights to custom JSON format.")
    parser.add_argument("--model_path", type=str, default="pytorch_model.bin",
                        help="Path to the PyTorch model.bin file.")
    parser.add_argument("--output_path", type=str, default="gpt2_weights.json",
                        help="Path to save the output JSON file.")
    parser.add_argument("--embed_dim", type=int, default=768,
                        help="Embedding dimension of the model (e.g., 768 for GPT-2 small).")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of transformer layers in the model (e.g., 12 for GPT-2 small).")

    args = parser.parse_args()

    convert_gpt2_weights(args.model_path, args.output_path, args.embed_dim, args.num_layers)
    
    print("\nTo use these weights, you will also need:")
    print("- vocab.json (from Hugging Face GPT-2 model card)")
    print("- merges.txt (from Hugging Face GPT-2 model card)")
    print("Place these in your 'resources' directory or specify their paths via CLI arguments in the main C++ application.")
