import torch.nn as nn
import torch.nn.functional as F
from models.attention import MultiHeadAttention
from models.embedding import PosAndWordEmbedding
from models.Encoder import MLP


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.att = nn.MultiheadAttention(embed_dim=config.embd_dim, num_heads=config.attention_head_n)
        self.transformerblock = nn.ModuleDict(dict(
            mhe=nn.MultiheadAttention(embed_dim=config.embd_dim, num_heads=config.attention_head_n),
            mlp=MLP(config),
            ln=nn.LayerNorm(config.embd_dim),
            do=nn.Dropout(0.1),
        ))

    def forward(self, key, query, x, attention_mask=None):
        residual = x
        decoder_attention,_ = self.att(x, x, x)

        decoder_attention = self.transformerblock.do(decoder_attention + residual)

        '''if key.size(1) > query.size(1):
            key = key[:, :query.size(1), :]'''

        mha_output, _ = self.transformerblock.mhe(query, key, value=decoder_attention)
        mha_output = self.transformerblock.do(mha_output + decoder_attention)
        mha_output = self.transformerblock.ln(mha_output)

        res = mha_output
        decoder_attention_output = self.transformerblock.mlp(mha_output)
        decoder_attention_output = self.transformerblock.do(decoder_attention_output + res)
        decoder_attention_output = self.transformerblock.ln(decoder_attention_output)

        return decoder_attention_output


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.embd = nn.Embedding(config.vocab_size, config.embd_dim)
        self.posembd = PosAndWordEmbedding(config)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.decoder_layer_n)])
        self.drop = nn.Dropout(0.1)

    def forward(self, x, encoder_out, mask):
        x = self.drop(self.posembd(x))
        for block in self.blocks:
            x = block(encoder_out, x, x, mask)
        return x
