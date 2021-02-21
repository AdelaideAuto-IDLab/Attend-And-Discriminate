import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


__all__ = ['SelfAttention', 'TemporalAttention']


def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    """
    Create and initialize a `nn.Conv1d` layer with spectral normalization.
    """
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    # return spectral_norm(conv)
    return conv


class SelfAttention(nn.Module):
    """
    # self-attention implementation from https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
    Self attention layer for nd
    """
    def __init__(self, n_channels: int, div):
        super(SelfAttention, self).__init__()

        if n_channels > 1:
            self.query = conv1d(n_channels, n_channels//div)
            self.key = conv1d(n_channels, n_channels//div)
        else:
            self.query = conv1d(n_channels, n_channels)
            self.key = conv1d(n_channels, n_channels)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class TemporalAttention(nn.Module):
    """
    Temporal attention module
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, x):
        out = self.fc(x).squeeze(2)
        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 0)
        return context
