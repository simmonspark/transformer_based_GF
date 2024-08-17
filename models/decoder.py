import torch.nn as nn
import torch.nn.functional as F
from models.attention import MultiHeadAttention
from models.embedding import PosAndWordEmbedding
from models.Encoder import MLP


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.att = nn.MultiheadAttention(768, 4)
        self.transformerblock = nn.ModuleDict(dict(
            mhe=nn.MultiheadAttention(embed_dim=config.embd_dim, num_heads=config.attention_head_n),
            mlp=MLP(config),
            ln1=nn.LayerNorm(config.embd_dim, eps=1e-6),
            ln2=nn.LayerNorm(config.embd_dim, eps=1e-6),
            do=nn.Dropout(0.1),
        ))

    def forward(self, key, query, x, trg_mask, key_padding_mask):
        residual = x
        assert trg_mask is not None

        # Self-Attention
        x = x.transpose(0, 1)
        decoder_attention, _ = self.att(x, x, x)
        decoder_attention = decoder_attention.transpose(0, 1)

        # Apply Dropout and Add & Norm
        decoder_attention = self.transformerblock.do(decoder_attention + residual)
        decoder_attention = self.transformerblock.ln1(decoder_attention)  # First LayerNorm

        # Cross-Attention with Key Padding Mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.transpose(0, 1)
        mha_output, _ = self.transformerblock.mhe(query, key, value=decoder_attention)

        # Apply Dropout and Add & Norm
        mha_output = self.transformerblock.do(mha_output + decoder_attention)
        mha_output = self.transformerblock.ln2(mha_output)  # Second LayerNorm

        # MLP and Final Add & Norm
        res = mha_output
        decoder_attention_output = self.transformerblock.mlp(mha_output)
        decoder_attention_output = self.transformerblock.do(decoder_attention_output + res)
        decoder_attention_output = self.transformerblock.ln2(decoder_attention_output)  # Reuse the same LayerNorm

        return decoder_attention_output


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.embd = nn.Embedding(config.vocab_size, config.embd_dim)
        self.posembd = PosAndWordEmbedding(config)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.decoder_layer_n)])
        self.drop = nn.Dropout(0.1)

    def forward(self, x, encoder_out, trg_mask, satt_mask):
        x = self.drop(self.posembd(x))
        for block in self.blocks:
            x = block(encoder_out, x, x, trg_mask, satt_mask)
        return x
