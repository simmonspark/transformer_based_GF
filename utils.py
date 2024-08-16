import os
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset
from tqdm import tqdm

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
    return Dataset.from_list(input_data)


if __name__ == "__main__":
    data = PrepareData()
