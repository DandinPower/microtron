import torch
import torch.nn as nn

from dataclasses import dataclass
from torch.nn import functional as F

@dataclass
class GptOssConfig:
    vocab_size: int = 201088
    hidden_size: int = 1440
    intermediate_size: int = 1440
    num_hidden_layers: int = 12
    num_attention_heads: int = 64
    head_dim: int = 64
    num_local_experts: int = 8
    num_key_value_heads: int = 8
    attention_dropout: float = 0.0
    experts_per_token: int = 4
    sliding_window: int = 128

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, hidden_states):
        return F.rms_norm(hidden_states, normalized_shape=self.weight.shape, weight=self.weight, eps=self.eps)

class GptOss(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass