import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class MHA(nn.Module):
    def __init__(self,config: GPTConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        self.scale = self.head_dim ** -0.5
        self.c_attn = nn.Linear(self.n_embd, self.n_embd * 3)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, N, n_embd = x.shape
        qkv = self.c_attn(x)
        q,k,v = qkv.split(n_embd, dim=-1)
        q = q.view(B,N, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B,N, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B,N, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        logits = q @ k.transpose(-2, -1) * self.scale
        logits = logits.masked_fill(self.mask[:, :, :N, :N] == 0, -float('inf'))
        weights = F.softmax(logits, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, N, self.n_embd)
        out = self.c_proj(out)
        return out
#





class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x



class GPT_Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MHA(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config


        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional embedding
            h = nn.ModuleList([GPT_Block(config) for _ in range(config.n_layer)]), # the transformer
            ln_f = nn.LayerNorm(config.n_embd), # layernorm before
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    def forward(self, x):
        word_emb = self.transformer.wte(x)
        pos_enc = self.transformer.wpe(torch.arange(x.shape[1], device=x.device))
        #print("word_emb", word_emb.shape)
        #print("pos_enc", pos_enc.shape)
        x = word_emb + pos_enc
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
#
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = cls(config)
        model_dict = model.state_dict()
        gpt_head_model = GPT2LMHeadModel.from_pretrained(model_type)
        params_dict = gpt_head_model.state_dict()
        # copy shared weights
        #for key in params_dict.keys():
        #    if key in model_dict:
        #        model_dict[key] = params_dict[key]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k in params_dict.keys():
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert params_dict[k].shape[::-1] == model_dict[k].shape
                with torch.no_grad():
                    model_dict[k].copy_(params_dict[k].t())
            else:
                # vanilla copy over the other parameters
                assert params_dict[k].shape == model_dict[k].shape
                with torch.no_grad():
                    model_dict[k].copy_(params_dict[k])
#
        return model
