from torch.utils.data import Dataset
import torch
from tokenizers import Tokenizer

tokenizer_path = "korean_tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)
max_length = 1024


def add_padding(ids, max_length=max_length, pad_id=0):
    if len(ids) < max_length:
        return ids + [pad_id] * (max_length - len(ids))
    return ids[:max_length]


class GFDataset(Dataset):
    def __init__(self, data_dic):
        super().__init__()
        self.data_dic = data_dic

    def __getitem__(self, item):
        input_data, label = self.data_dic['input_data'][item], self.data_dic['label'][item]

        input_data = tokenizer.encode(input_data)

        att_mask = add_padding(input_data.attention_mask)

        input_data = add_padding(input_data.ids)

        label = tokenizer.encode(label)

        label = add_padding(label.ids)

        return torch.Tensor(input_data).type(torch.int), torch.Tensor(label).type(torch.int), torch.Tensor(att_mask).type(torch.int)

    def __len__(self):
        return len(self.data_dic['input_data'])
