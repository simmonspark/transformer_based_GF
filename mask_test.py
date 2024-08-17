import torch
import torch.nn as nn

# 설정 값들
batch_size = 8
seq_len = 1024
embedding_dim = 768
num_heads = 4

# 임의의 입력 데이터
x = torch.randn(batch_size, seq_len, embedding_dim)  # shape: [8, 1024, 768]

# MultiheadAttention 모듈 초기화
multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)

# Attention mask 생성 (예시로, 자기 자신을 마스킹하는 것)
# 일반적으로 attn_mask는 [seq_len, seq_len] 형식이거나 [batch_size, num_heads, seq_len, seq_len]일 수 있습니다.
# 여기서는 간단히 [seq_len, seq_len]으로 시작합니다.
attn_mask = torch.ones(seq_len, seq_len)  # 예시로 모든 곳에 1을 할당한 mask

# attn_mask는 [batch_size * num_heads, seq_len, seq_len] 형태로 변환되어야 합니다.
# 하지만 여기서는 [seq_len, seq_len]이므로 바로 사용 가능합니다.
# 실전에서는 [batch_size, num_heads, seq_len, seq_len] 형태로 생성한 후 아래와 같이 변환해야 합니다.

# 만약 batch_size와 num_heads가 있는 마스크를 변환해야 한다면:
attn_mask = attn_mask.unsqueeze(0).repeat(batch_size * num_heads, 1, 1)  # shape: [32, 1024, 1024]

# Input을 multi-head attention에 맞게 변환
x = x.transpose(0, 1)  # shape: [1024, 8, 768]

# MultiheadAttention 호출 (트랜스포즈한 입력과 변환된 마스크를 전달)
output, attn_output_weights = multihead_attn(x, x, x, attn_mask=attn_mask)

# 결과를 다시 [batch_size, seq_len, embedding_dim] 형태로 변환
output = output.transpose(0, 1)  # shape: [8, 1024, 768]

print(output.shape)  # [8, 1024, 768]이어야 함
