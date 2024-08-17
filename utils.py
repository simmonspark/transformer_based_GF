import os
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset
from tqdm import tqdm
from tokenizers import Tokenizer
from dataset import GFDataset
from torch.utils.data import DataLoader

tokenizer_path = "korean_tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)
max_length = 1024


def add_padding(ids, max_length=max_length, pad_id=0):
    if len(ids) < max_length:
        return ids + [pad_id] * (max_length - len(ids))
    return ids[:max_length]


'''
20만개의 데이터 중 5만개를 sampling한다.
이 중에서 sub-sampling-category로는 미용, 건강, 연애/결혼 총 15000개의 데이터를 사용한다.  
'''

abs_path = '/media/sien/DATA/DATA/dataset/GFData/Training'
spesific_path = 'use_ful'
data_dir = os.path.join(abs_path, spesific_path)
sub_categories = ['미용', '건강', '연애/결혼', '일상대화']
spetial_tokens = ['[START],[UNK],[EOS]']


def PrepareData(data_path=data_dir):
    full_path = []
    for dir, _, path in tqdm(os.walk(data_dir), desc='Collection data path'):
        for name in path:
            F = os.path.join(dir, name)
            full_path.append(F)
    print('\n[DONE] collection data dir\n')
    input_data = []
    error_files = []
    for path in tqdm(full_path, desc='Processing raw Json as input'):

        try:
            with open(path, 'r', encoding='utf-8') as file:
                raw_json = json.load(file)
                for M in raw_json['info']:
                    if M['category'] in sub_categories:
                        tmp = []
                        for dialogue in M['annotations']['lines']:
                            # tmp.append(dialogue['norm_text'])
                            text = '[START]' + dialogue['norm_text'] + '[EOS]'
                            tmp.append(text)
                            label = tmp

                        input_data.append({'input_data': tmp[:-1], 'label': label[1:]})
        except json.JSONDecodeError as e:
            print(f'\n[ERROR] JSONDecodeError 발생: {path}\n')
            error_files.append(path)
        except Exception as e:
            print(f'\n[ERROR] 파일 처리 중 오류 발생: {path} - {e}\n')
    test_data = input_data[0]
    for i in range(len(test_data['input_data'])):
        print(test_data['input_data'][i])
        print('\n')
        print(test_data['label'][i])
        print('\n')
    print('QA test 출력입니다.\n')
    dataset = Dataset.from_list(input_data)
    flatten_input = [item for sublist in dataset['input_data'] for item in sublist]
    flatten_label = [item for sublist in dataset['label'] for item in sublist]
    dataset_train = dict(input_data=flatten_input[0:90000], label=flatten_label[0:90000])
    dataset_val = dict(input_data=flatten_input[90000:], label=flatten_label[90000:])

    return dataset_train, dataset_val


def PrepareToknizingData(data_path=data_dir):
    full_path = []
    for dir, _, path in tqdm(os.walk(data_dir), desc='Collection data path'):
        for name in path:
            F = os.path.join(dir, name)
            full_path.append(F)
    print('\n[DONE] collection data dir\n')
    input_data = []
    error_files = []
    for path in tqdm(full_path, desc='Processing raw Json as input'):

        try:
            with open(path, 'r', encoding='utf-8') as file:
                raw_json = json.load(file)
                for M in raw_json['info']:
                    if M['category'] in sub_categories:
                        tmp = []
                        att_tmp = []
                        for dialogue in M['annotations']['lines']:
                            # tmp.append(dialogue['norm_text'])
                            text = '[START]' + dialogue['norm_text'] + '[EOS]'
                            text = tokenizer.encode(text)
                            att_mask = text.attention_mask
                            att_mask = add_padding(att_mask)
                            att_tmp.append(att_mask)
                            text = add_padding(text.ids)
                            tmp.append(text)
                            label = tmp

                        input_data.append({'input_data': tmp[:-1], 'attention_mask': att_tmp[:-1], 'label': label[1:]})
        except json.JSONDecodeError as e:
            print(f'\n[ERROR] JSONDecodeError 발생: {path}\n')
            error_files.append(path)

    test_data = input_data[0]
    for i in range(len(test_data['input_data'])):
        print(test_data['input_data'][i])
        print('\n')
        print(test_data['label'][i])
        print('\n')
    print('QA test 출력입니다.\n')
    dataset = Dataset.from_list(input_data)
    return dataset


if __name__ == "__main__":
    train_data, val_data = PrepareData()

    print()
    # t_data = PrepareToknizingData()
    dataset = GFDataset(train_data)
    x, y, a = next(iter(dataset))
    print()
    # flatten_input = [item for sublist in data['input_data'] for item in sublist]
    # flatten_label = [item for sublist in data['label'] for item in sublist]
    test_loader = DataLoader(dataset, batch_size=8, pin_memory=True, pin_memory_device='cuda')
    while True:
        next(iter(test_loader))
        print('test call [PASS]')
