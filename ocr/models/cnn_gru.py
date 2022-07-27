import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, vocab_size):
        super(CRNN, self).__init__()
        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(1792, 256)
        self.gru1 = nn.GRU(input_size=256, hidden_size=256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, vocab_size)

    def forward(self, x):
        x = self.resnet(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.dropout(self.fc1(x))
        output, _ = self.gru1(x)
        x = self.fc2(output)
        x = x.permute(1, 0, 2)

        return x