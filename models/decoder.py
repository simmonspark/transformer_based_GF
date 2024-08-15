import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from embedding import PosAndWordEmbedding
from Encoder import TransformerBlock

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.att = MultiHeadAttention(config)
        self.norm = nn.LayerNorm(config.embd_dim)
