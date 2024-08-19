import torch
from transformers import T5ForConditionalGeneration
from dataset import GFDataset
from utils import *
from torch.utils.data import DataLoader
import os
from models.GF import GF
from models.config import GFConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"
cfg = GFConfig()
model = GF(cfg)
checkpoint = torch.load('out/checkpoint.pt')
state_dict = checkpoint['model_state_dict']
new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model = model.to('cuda')

tokenizer_path = "korean_tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)

abs_path = '/media/sien/DATA/DATA/dataset/GFData/Training'
spesific_path = 'use_ful'
data_dir = os.path.join(abs_path, spesific_path)

train_data, val_data = PrepareData()

train_ds = GFDataset(train_data)
val_ds = GFDataset(val_data)

batch_size = 8


train_loader = DataLoader(train_ds, batch_size=batch_size, pin_memory=True, num_workers=3)
val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=True, num_workers=3, shuffle=True)
with torch.no_grad():
    source, target, attention_mask = next(iter(val_loader))
    input_token = source
    output_logits = model(source=source.to('cuda'),target= target.to('cuda'),attention_mask= attention_mask.to('cuda'))

    predicted_sequence = torch.argmax(output_logits, dim=-1)

    for _ in range(batch_size):
        decoded_source = tokenizer.decode(input_token[_].tolist(), skip_special_tokens=True)
        decoded_pred = tokenizer.decode(predicted_sequence[_].tolist(), skip_special_tokens=True)
        decoded_string = tokenizer.decode(target[_].tolist(), skip_special_tokens=True)
        print(f'입력은 다음과 같습니다. --> {decoded_source}')
        print(f'대답은 다음과 같습니다. --> {decoded_pred}\n')
        #print(f'정답은 다음과 같습니다. --> {decoded_string}')



