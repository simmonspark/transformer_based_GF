import os
import torch
from models.config import GFConfig
from tokenizers import Tokenizer
from models.GF import GF
from utils import PrepareData
from dataset import GFDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from transformers import  T5ForConditionalGeneration

IsComplie = False
mode = 'scratch'  # scratch, resume, fine, fine_resume
lr = 1e-6
betas = (0.9, 0.95)
epoch = 100
batch_size = 8

criterion = nn.CrossEntropyLoss()
cfg = GFConfig()
abs_path = '/media/sien/DATA/DATA/dataset/GFData/Training'
spesific_path = 'use_ful'
data_dir = os.path.join(abs_path, spesific_path)
sub_categories = ['미용', '건강', '연애/결혼', '일상대화']
spetial_tokens = ['[START],[UNK],[EOS]']

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(9434)

print('======================================')
print('       시언이의 챗봇 train process      ')
print('======================================\n')
print("Initializing a new model from scratch")
print("defaulting to vocab_size of GPT-2 to 53000 (53004 rounded up for efficiency)\n")
print('---- Config as follow ----\n')
print(cfg)

scaler = torch.cuda.amp.GradScaler(enabled=True)

print('cuda amp GradScaler at 16bit cal [ON]')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(f'TOKENIZERS_PARALLELISM [OFF]')

tokenizer_path = "korean_tokenizer.json"

tokenizer = Tokenizer.from_file(tokenizer_path)

model = GF(cfg)

if mode == 'resume':
    checkpoint = torch.load('out/checkpoint.pt')
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.to('cuda')
elif mode == 'scratch':
    print('\ntrain from scratch')
    model = model.to('cuda')
elif mode == 'fine':
    test_model = T5ForConditionalGeneration(T5ForConditionalGeneration.from_pretrained("KETI-AIR/ke-t5-small").config)
    model = test_model.to('cuda')
elif mode == 'fine_resume':
    test_model = T5ForConditionalGeneration(T5ForConditionalGeneration.from_pretrained("KETI-AIR/ke-t5-small").config)
    model = test_model.to('cuda')
    checkpoint = torch.load('out/checkpoint.pt')
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.to('cuda')

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas)

train_data, val_data = PrepareData()

train_ds = GFDataset(train_data)
val_ds = GFDataset(val_data)

train_loader = DataLoader(train_ds, batch_size=batch_size, pin_memory=True, num_workers=3)
val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=True, num_workers=3)

if compile:
    print("compiling the model... (시간이 좀 걸려요..)")
    unoptimized_model = model
    model = torch.compile(model)
    model = model.to('cuda')


@torch.no_grad()
def cal_loss():
    print('\nvalidation start\n')
    model.eval()
    losses = []
    for D in tqdm(val_loader):
        x, y, att_mask = D
        x = x.to('cuda')
        y = y.to(torch.long).to('cuda')
        att_mask = att_mask.to('cuda')
        with torch.cuda.amp.autocast(enabled=True):
            if mode.startswith('fine'):
                pred = model(x, y, att_mask)
                loss = pred.loss
            else :
                pred = model(x, y, att_mask)
                pred = pred.view(-1, cfg.vocab_size)
                y = y.view(-1)
                loss = criterion(pred, y)
        losses.append(loss)
    model.train()
    print(f'val loss : {sum(losses) / len(losses)}')
    return sum(losses) / len(losses)




for iter in range(epoch):

    g_loss = []
    for D in tqdm(train_loader):
        x, y, att_mask = D
        x = x.to('cuda')
        y = y.to(torch.long).to('cuda')
        att_mask = att_mask.to('cuda')
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            pred = model(x, y, att_mask)
            if mode.startswith('fine'):
                loss = pred.loss
            else :
                pred = pred.view(-1, cfg.vocab_size)
                y = y.view(-1)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()


        g_loss.append(loss.item())
    print(f"iter {iter}: loss {sum(g_loss) / len(g_loss):.4f}")
    val_loss = cal_loss()
    print(f"Validation loss: {val_loss:.4f}")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }
    torch.save(checkpoint, os.path.join('out', 'checkpoint.pt'))
    print('saved_checkpoint!\n')
