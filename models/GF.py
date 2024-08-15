import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):

    def __init__(self, config):
        super(self).__init__()

        self.encoder = Encoder(config)

        self.decoder = Decoder(config)

        self.fc_out = nn.Linear(config.embd_dim, config.vocab_size)

    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask

    def forward(self, source, target):
        trg_mask = self.make_trg_mask(target)
        enc_out = self.encoder(source)
        outputs = self.decoder(target, enc_out, trg_mask)
        output = F.softmax(self.fc_out(outputs), dim=-1)
        return output

    def generate_tokens(self, source, max_tokens, device):
        self.eval()
        with torch.no_grad():

            target = torch.zeros(source.size(0), 1).long().to(device)

            for _ in range(max_tokens):
                output = self(source, target)
                next_token_probs = output[:, -1, :]
                next_token = torch.argmax(next_token_probs, dim=-1).unsqueeze(-1)

                target = torch.cat([target, next_token], dim=1)

                if next_token.item() == 2:
                    break

        return target
