# Modelo baseline LSTM simple
import torch
import torch.nn as nn

class BaselineLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()