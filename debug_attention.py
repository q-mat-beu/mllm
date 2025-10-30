
import torch
import torch.nn.functional as F
import numpy as np

# Parameters from the C++ test
embed_dim = 4
num_heads = 2
seq_len = 3
head_dim = embed_dim // num_heads

# Input data from the C++ test
input_data = torch.tensor([
    [1.0, 0.0, 0.5, 0.0],
    [0.0, 1.0, 0.0, 0.5],
    [0.5, 0.0, 1.0, 0.0]
], dtype=torch.float32)

# Weights (Identity)
_w_q = torch.eye(embed_dim, dtype=torch.float32)
_w_k = torch.eye(embed_dim, dtype=torch.float32)
_w_v = torch.eye(embed_dim, dtype=torch.float32)
_w_o = torch.eye(embed_dim, dtype=torch.float32)

# Biases (Zeros)
_b_q = torch.zeros(embed_dim, dtype=torch.float32)
_b_k = torch.zeros(embed_dim, dtype=torch.float32)
_b_v = torch.zeros(embed_dim, dtype=torch.float32)
_b_o = torch.zeros(embed_dim, dtype=torch.float32)

# 1. Project to Q, K, V
# Since weights are identity and biases are zero, Q, K, V are the same as the input
q = input_data
k = input_data
v = input_data

# 2. Reshape and transpose for multi-head attention
# (seq, dim) -> (seq, heads, head_dim) -> (heads, seq, head_dim)
q = q.view(seq_len, num_heads, head_dim).transpose(0, 1)
k = k.view(seq_len, num_heads, head_dim).transpose(0, 1)
v = v.view(seq_len, num_heads, head_dim).transpose(0, 1)

# 3. Scaled Dot-Product Attention
# (heads, seq, head_dim) @ (heads, head_dim, seq) -> (heads, seq, seq)
scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
attn_weights = F.softmax(scores, dim=-1)

# (heads, seq, seq) @ (heads, seq, head_dim) -> (heads, seq, head_dim)
context = torch.matmul(attn_weights, v)

# 4. Transpose and reshape back
# (heads, seq, head_dim) -> (seq, heads, head_dim) -> (seq, dim)
context = context.transpose(0, 1).contiguous().view(seq_len, embed_dim)

# 5. Final output projection
# Since weights are identity and biases are zero, output is the same as context
final_output = context

# Print the golden values
print("Golden values for the attention test:")
print(final_output.flatten().tolist())
