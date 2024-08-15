import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

class WordEmbedding(nn.Module):
    def __init__(self,config):
        super().__init__()

