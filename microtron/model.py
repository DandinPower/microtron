import torch
import torch.nn as nn

from dataclasses import dataclass
from torch.nn import functional as F

@dataclass
class GptOssConfig:
    vocab_size: int = 201088
    hidden_size: int = 1024
    intermediate_size: int = 1024
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

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.sinks = nn.Parameter(torch.ones(self.num_attention_heads))
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias = True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = True)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias = True)


class Router(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.topk = config.experts_per_token
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.weight = nn.Parameter(torch.ones(self.num_experts, self.hidden_size))
        self.bias = nn.Parameter(torch.ones(self.num_experts))


class Experts(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.gate_up_proj = nn.Parameter(torch.ones(self.num_experts, self.hidden_size, 2 * self.intermediate_size))
        self.gate_up_proj_bias = nn.Parameter(torch.ones(self.num_experts, 2 * self.intermediate_size))
        self.down_proj = nn.Parameter(torch.ones(self.num_experts, self.hidden_size, self.intermediate_size))
        self.down_proj_bias = nn.Parameter(torch.ones(self.num_experts, self.intermediate_size))


class MLP(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.router = Router(config)
        self.experts = Experts(config)


class Block(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config: GptOssConfig = config
        self.self_attn = CausalSelfAttention(config)
        self.mlp = MLP(config)  
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)


class GptOss(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config: GptOssConfig = config
        self.model = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size),
            layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        ))
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)