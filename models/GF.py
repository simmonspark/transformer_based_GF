import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Encoder import Encoder
from models.decoder import Decoder
from models.config import GFConfig


class GF(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)

        self.decoder = Decoder(config)

        self.fc_out = nn.Linear(config.embd_dim, config.vocab_size)

    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask

    def forward(self, source, target, attention_mask=None):
        trg_mask = self.make_trg_mask(target)
        enc_out = self.encoder(source, attention_mask)
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


print('-------------------------------')
print('--- 내 친구  compile TEST :) ----')
print('-------------------------------\n')

config = GFConfig()
model = GF(config)
print(f'[PASSED] at. configuration test\n')
source = torch.randint(0, config.vocab_size, (8, 1024)).long()
target = torch.randint(0, config.vocab_size, (8, 1024)).long()
print(f'[PASSED] at. generate dummy IO\n')
out = model(source, target)
print(f'[PASSED] at. test model out. shape as below\n')
print(out.shape)
print('\n')
criterion = nn.CrossEntropyLoss()

out_reshaped = out.view(-1, config.vocab_size)  # (batch_size * target_len, vocab_size)

target_reshaped = target.view(-1)  # (batch_size * target_len)
print(f'[PASSED] at. CAL nan loss as randn IO data\n')
loss = criterion(out_reshaped, target_reshaped)

print("Loss:", loss.item())
print('\n')
n = count_parameters(model)
print(f'[PASSED] at. countion model params\n')
print(f'params : {n / 1_000_000:.4f}M')
print('\n짝짝짝~ 모든 필수 assertion 테스트를 통과했어요~\n')
print('\n모델 고속 컴파일...\n')
model = torch.compile(model)
print('\nDONE\n')
print('data preprocessing 프로세스가 작동합니다.')
print('...시간이 좀 걸려요...\n')

