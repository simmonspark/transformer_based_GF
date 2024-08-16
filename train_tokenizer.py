from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.processors import BertProcessing
from utils import PrepareData

dataset = PrepareData()
dataset = [[str(D) for D in dataset]]

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
trainer = trainers.WordPieceTrainer(
    vocab_size=53000,  # 얼마나 많은 단어를 기억할거냐
    min_frequency=1,  # 얼마나 자주 등장해야 vocab에 추가할건지
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"]
)

tokenizer.train_from_iterator(dataset, trainer)

encoded = tokenizer.encode("[START]개강 전에 머리 자르러 언제 가지?[EOS]")
print(encoded.ids)
decoded_string = tokenizer.decode(encoded.ids, skip_special_tokens=True)
print(f'Decoded string -> {decoded_string}')

tokenizer.save("korean_tokenizer.json")

# print(tokenizer.get_vocab())
