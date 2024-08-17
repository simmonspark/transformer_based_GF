import math
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.embd_dim % config.attention_head_n == 0
        self.att_l = nn.Linear(config.embd_dim, config.embd_dim * 3, bias=False)
        self.proj = nn.Linear(config.embd_dim, config.embd_dim, bias=False)
        self.att_drop = nn.Dropout(0.1)
        self.res_drop = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(config.embd_dim, bias=None)

    def forward(self, x, attention_mask=None):
        res = x

        B, T, C = x.size()

        q, k, v = self.att_l(x).split(self.config.embd_dim, dim=2)

        q = q.view(B, T, self.config.attention_head_n, C // self.config.attention_head_n).transpose(1, 2)
        k = k.view(B, T, self.config.attention_head_n, C // self.config.attention_head_n).transpose(1, 2)
        v = v.view(B, T, self.config.attention_head_n, C // self.config.attention_head_n).transpose(1, 2)

        att: torch.Tensor = (q @ k.transpose(-1,-2)) / math.sqrt(C // self.config.attention_head_n)

        if attention_mask is not None:
            # Expand attention_mask to match the shape of att
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_length)

            # Ensure attention_mask is in the correct dtype
            attention_mask = attention_mask.to(torch.bool)

            # Apply the mask
            # Note: ~attention_mask inverts the mask, so only positions with False in attention_mask get -inf
            att = att.masked_fill(~attention_mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.att_drop(att)
        att = att @ v
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        att = att + res
        att = self.res_drop(att)
        att = self.norm(att)

        return att
