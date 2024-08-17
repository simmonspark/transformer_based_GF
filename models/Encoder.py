import torch.nn as nn
import copy
from models.attention import MultiHeadAttention
from models.embedding import PosAndWordEmbedding
import torch.nn.functional as F
import torch


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embd_dim, 2 * config.embd_dim, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(2 * config.embd_dim, config.embd_dim, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head = MLP(config)
        self.drop = nn.Dropout(0.1)
        self.att_l = MultiHeadAttention(config)
        self.norm = nn.LayerNorm(config.embd_dim, bias = False)

    def forward(self, x, attention_mask=None):
        x = self.att_l(x, attention_mask)
        res = x
        x = self.head(x)
        x = res + x
        x = self.norm(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ebd = PosAndWordEmbedding(config)
        self.block_list = nn.ModuleList([TransformerBlock(config) for _ in range(config.encoder_layer_n)])

    def forward(self, x, attention_mask = None):
        x = self.ebd(x)
        for block in self.block_list:
            x = block(x, attention_mask)
        return x

