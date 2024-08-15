import torch
import torch.nn as nn


class PosAndWordEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embd_layer = nn.Embedding(self.config.vocab_size, self.config.embd_dim)
        self.pos_embd = nn.Embedding(self.config.block_size, self.config.embd_dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, mode = 'encoder'):
        b, t = x.size()
        x = self.embd_layer(x)


        device = x.device

        pos_raw_ids = torch.arange(0, t, dtype=torch.long, device=device)

        pos = self.pos_embd(pos_raw_ids)

        x = self.drop(x + pos)
        return x
