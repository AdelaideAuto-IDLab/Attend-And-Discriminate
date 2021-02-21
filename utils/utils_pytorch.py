import torch.nn as nn
from utils.utils import paint

__all__ = ["get_info_params", "get_info_layers", "init_weights_orthogonal"]


def get_info_params(model):
    """
    Display a summary of trainable/frozen network parameter counts
    :param model:
    :return:
    """
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    print(paint(f"[-] {num_trainable}/{num_total} trainable parameters", "blue"))


def get_info_layers(model):
    """
    Display network layer information
    :param model:
    :return:
    """
    print("Layer Name \t\t Parameter Size")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t\t", model.state_dict()[param_tensor].size())


def init_weights_orthogonal(m):
    """
    Orthogonal initialization of layer parameters
    :param m:
    :return:
    """
    if type(m) == nn.LSTM or type(m) == nn.GRU:
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    elif type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)
