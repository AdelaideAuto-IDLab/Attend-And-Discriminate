import os

import torch
import torch.nn as nn

from utils.utils import paint, makedir
from utils.utils_pytorch import (
    get_info_params,
    get_info_layers,
    init_weights_orthogonal,
)
from utils.utils_attention import SelfAttention, TemporalAttention
from settings import get_args

__all__ = ["create"]


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        activation,
        sa_div,
    ):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(1, filter_num, (filter_size, 1))
        self.conv2 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv3 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv4 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            filter_num * input_dim,
            hidden_dim,
            enc_num_layers,
            bidirectional=enc_is_bidirectional,
            dropout=dropout_rnn,
        )

        self.ta = TemporalAttention(hidden_dim)
        self.sa = SelfAttention(filter_num, sa_div)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        # apply self-attention on each temporal dimension (along sensor and feature dimensions)
        refined = torch.cat(
            [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1,
        )
        x = refined.permute(3, 0, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)
        outputs, h = self.rnn(x)

        # apply temporal attention on GRU outputs
        out = self.ta(outputs)
        return out


class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_class):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, z):
        return self.fc(z)


class AttendDiscriminate(nn.Module):
    def __init__(
        self,
        model,
        dataset,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        dropout_cls,
        activation,
        sa_div,
        num_class,
        train_mode,
        experiment,
    ):
        super(AttendDiscriminate, self).__init__()

        self.experiment = f"train_{experiment}" if train_mode else experiment
        self.model = model
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        print(paint(f"[STEP 3] Creating {self.model} HAR model ..."))

        self.fe = FeatureExtractor(
            input_dim,
            hidden_dim,
            filter_num,
            filter_size,
            enc_num_layers,
            enc_is_bidirectional,
            dropout,
            dropout_rnn,
            activation,
            sa_div,
        )

        self.dropout = nn.Dropout(dropout_cls)
        self.classifier = Classifier(hidden_dim, num_class)
        self.register_buffer(
            "centers", (torch.randn(num_class, self.hidden_dim).cuda())
        )

        # do not create log directories if we are only testing the models module
        if experiment != "test_models":
            if train_mode:
                makedir(self.path_checkpoints)
                makedir(self.path_logs)
            makedir(self.path_visuals)

    def forward(self, x):
        feature = self.fe(x)
        z = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature)
        )
        out = self.dropout(feature)
        logits = self.classifier(out)
        return z, logits

    @property
    def path_checkpoints(self):
        return f"./models/{self.dataset}/{self.experiment}/checkpoints/"

    @property
    def path_logs(self):
        return f"./models/{self.dataset}/{self.experiment}/logs/"

    @property
    def path_visuals(self):
        return f"./models/{self.dataset}/{self.experiment}/visuals/"


__factory = {
    "AttendDiscriminate": AttendDiscriminate,
}


def create(model, config):
    if model not in __factory.keys():
        raise KeyError(f"[!] Unknown HAR model: {model}")
    return __factory[model](**config)


def main():

    # get experiment arguments
    args, _, config_model = get_args()
    args.experiment = "test_models"
    config_model["experiment"] = "test_models"

    # [STEP 1] create synthetic HAR batch
    data_synthetic = torch.randn((args.batch_size, args.window, args.input_dim)).cuda()

    # [STEP 2] create HAR models
    if torch.cuda.is_available():
        model = create(args.model, config_model).cuda()
        torch.backends.cudnn.benchmark = True
        get_info_params(model)
        get_info_layers(model)
        model.apply(init_weights_orthogonal)

    model.eval()
    with torch.no_grad():
        print(paint("[*] Performing a forward pass with a synthetic batch..."))
        z, logits = model(data_synthetic)
        print(f"\t input: {data_synthetic.shape} {data_synthetic.dtype}")
        print(f"\t z: {z.shape} {z.dtype}")
        print(f"\t logits: {logits.shape} {logits.dtype}")


if __name__ == "__main__":
    main()
