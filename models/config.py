from dataclasses import dataclass
@dataclass
class GFConfig:
    '''
    : define
     -> 모델에 관련한 파라미터를 정의합니다.
    '''
    block_size : int = 1024
    vocab_size : int = 50304
    encoder_layer_n : int = 6
    attention_head_n : int = 4
    embd_dim : int = 256
    bias : bool = False


