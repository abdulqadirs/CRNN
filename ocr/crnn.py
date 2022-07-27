import torch
from tqdm import tqdm
import os
from loss import calculate_loss
from datasets import load_metric
cer_metric = load_metric("cer")

def training(model, training_loader, validation_loader, optimizer, label_converter, epochs, start_epoch,  validate_every, output_dir):
    device = 'cpu'
    running_loss = 0
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        for i, data in tqdm(enumerate(training_loader)):
            images = data[0]
            labels = data[1]
            output = model(images)
            loss = calculate_loss(output, labels, label_converter, device)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(training_loader)
        print('Epoch {} loss: {}'.format(epoch, epoch_loss))
        
        if epoch % validate_every == 0:
            avg_cer = validation(net, validation_loader)

def validation(model, validation_loader, label_converter):
    device = 'cpu'
    model.eval()
    eval_cer = []
    for i, data in tqdm(enumerate(validation_loader)):
        images = data[0]
        labels = data[1]
        output = model(images)
        encoded_text = outputs.squeeze().argmax(1)
        decoded_text = label_converter.decode(encoded_text)

        cer = cer_metric.compute(predictions=[decoded_text], references=[labels[0]])
        eval_cer.append(cer)
    avg_cer = sum(eval_cer) / len(validation_loader)
    print('Validation finished.')
    print('Average Character Error Rate (CER): '.formate(avg_cer))
    return avg_cer
