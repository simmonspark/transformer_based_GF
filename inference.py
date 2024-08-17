import torch
from transformers import T5ForConditionalGeneration
from dataset import GFDataset
from utils import *
from torch.utils.data import DataLoader
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

test_model = T5ForConditionalGeneration(T5ForConditionalGeneration.from_pretrained("KETI-AIR/ke-t5-small").config)
checkpoint = torch.load('out/checkpoint.pt')
state_dict = checkpoint['model_state_dict']
new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
test_model.load_state_dict(new_state_dict)
model = test_model.to('cuda')

tokenizer_path = "korean_tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)

abs_path = '/media/sien/DATA/DATA/dataset/GFData/Training'
spesific_path = 'use_ful'
data_dir = os.path.join(abs_path, spesific_path)

train_data, val_data = PrepareData()

train_ds = GFDataset(train_data)
val_ds = GFDataset(val_data)

train_loader = DataLoader(train_ds, batch_size=4, pin_memory=True, num_workers=3,shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4, pin_memory=True, num_workers=3)


def test_model():
    results = []
    x, y, mask = next(iter(train_loader))
    x = x.to('cuda')
    y = y.to(torch.long).to('cuda')
    att_mask = mask.to('cuda')

    with torch.cuda.amp.autocast(enabled=True):
        outputs = model.generate(input_ids=x)

    for _ in range(4):
        input_token = tokenizer.decode(x[_].tolist(), skip_special_tokens=True)
        pred_text = tokenizer.decode(outputs[_].tolist(), skip_special_tokens=True)
        target_text = tokenizer.decode(y[_].tolist(), skip_special_tokens=True)
        results.append((input_token, pred_text, target_text))

    return results


test_results = test_model()
for i, (input_token, pred_text, target) in enumerate(test_results):
    print(f"Example {i + 1}")
    print(f"입력은 다음과 같습니다.--->: {input_token}")
    print(f"응답은 자음과 같습니다.---> : {pred_text}\n")
    print(f"정답은 다음과 같습니다.---> : {target}\n")
