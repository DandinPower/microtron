import torch
from transformers import GptOssForCausalLM

from .model import GptOss, GptOssConfig

def main():
    # for the native bf16! thanks to unsloth
    model_hf = GptOssForCausalLM.from_pretrained("unsloth/gpt-oss-20b-BF16", dtype=torch.bfloat16)  # 21b
    state_dict_hf = model_hf.state_dict()

    for k, v in state_dict_hf.items():
        print(k, v.shape)
    total_params = sum(p.numel() for p in model_hf.parameters())
    print(total_params)

    base_config = GptOssConfig()
    base_config.hidden_size = 2880
    base_config.intermediate_size = 2880
    base_config.num_local_experts = 32
    base_config.num_hidden_layers = 24

    model_mine = GptOss(base_config)
    state_dict_mine = model_mine.state_dict()

    for k, v in state_dict_mine.items():
        print(k, v.shape)

    total_params = sum(p.numel() for p in model_mine.parameters())
    print(total_params)