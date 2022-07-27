import os
import torch
from pathlib import Path

from data import get_train_loader, get_test_loader
from models.cnn_gru import CRNN
from optimizer import sgd_optimizer
from crnn import training
from label_converter import LabelConverter

def main():
    images_dir = Path('../dataset/upti-1/ligature_undegraded/')
    labels_dir = Path('../dataset/upti-1/groundtruth')
    output_dir = Path('../output/')
    training_loader, validation_loader = get_train_loader(images_dir, labels_dir)
    testing_loader = get_test_loader(images_dir, labels_dir)
   
    chars = ['م', 'ی', 'ں', ' ', 'د', 'س', 'گ', 'ر', 'ا', 'و', 'ن', 'ے', 'ک', 'ق', 'ت',
         'پ', 'ھ', 'ٹ', 'ڈ', 'ہ', 'ب', '(', 'ع', ')',
         'غ', 'ل', 'ً', 'ط', 'ز', 'خ', 'ش', '‘', 'ف', '۔', 'ئ', 'ح', 'چ', 'ڑ', 'ؤ', 
         'ج', 'ث', 'ذ', 'ظ', 'آ', '،', '.', 'ص', 'ء', 'ض', 'ٰ', "'", '”', '“', 'ٴ', 
         ':',  '-', ',', 'أ', '؟', 
         '"', 'ژ', '/',  'ة', '’', 'ه', '…',
          'ِ', 'ُ', 'ّ', 'ْ', '+', 'ﷺ', 'َ', '%'
         , '#', '؛', '!', 'ٓ', 
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '$']
    label_converter = LabelConverter(chars)
    vocab_size = label_converter.get_vocab_size()

    crnn = CRNN(vocab_size)
    optimizer = sgd_optimizer(crnn)
    epochs = 50
    start_epoch = 1
    validate_every = 5

    training(crnn, training_loader, validation_loader, optimizer, label_converter, epochs, start_epoch, validate_every, output_dir)


if __name__ == "__main__":
    main()
