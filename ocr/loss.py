import torch
import torch.nn as nn


def calculate_loss(inputs, texts, label_converter, device):
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    inputs = inputs.log_softmax(2)
    input_size, batch_size, _ = inputs.size()
    input_size = torch.full(size=(batch_size,), fill_value=input_size, dtype=torch.int32)
    #print(texts)
    encoded_texts, text_lens = label_converter.encode(texts)
    loss = criterion(inputs, encoded_texts.to(device), input_size.to(device), text_lens.to(device))
    return loss
