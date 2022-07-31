import os
import torch
from pathlib import Path
from config import Config
from utils.read_config import reading_config
from utils.parse_arguments import parse_training_arguments
from data import get_train_loader, get_test_loader
from models.cnn_gru import CRNN
from optimizer import sgd_optimizer
from crnn import training
from label_converter import LabelConverter

def main():
    
    args, _ = parse_training_arguments()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    if args.checkpoints is not None:
        checkpoints_file = Path(args.checkpoints)
    
    if args.training:
        config_file = Path('../config.ini')
        reading_config(config_file)
        training_loader, validation_loader = get_train_loader(images_dir, labels_dir)
   
        chars = ['م', 'ی', 'ں', ' ', 'د', 'س', 'گ', 'ر', 'ا', 'و', 'ن', 'ے', 'ک', 'ق', 'ت',
            'پ', 'ھ', 'ٹ', 'ڈ', 'ہ', 'ب', '(', 'ع', ')',
            'غ', 'ل', 'ً', 'ط', 'ز', 'خ', 'ش', '‘', 'ف', '۔', 'ئ', 'ح', 'چ', 'ڑ', 'ؤ', 
            'ج', 'ث', 'ذ', 'ظ', 'آ', '،', '.', 'ص', 'ء', 'ض', 'ٰ', "'", '”', '“', 'ٴ', 
            ':',  '-', ',', 'أ', '؟', 
            '"', 'ژ', '/',  'ة', '’', 'ه', '…',
            'ِ', 'ُ', 'ّ', 'ْ', '+', 'ﷺ', 'َ', '%
            ',  '#', '؛', '!', 'ٓ', 
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '$']
        label_converter = LabelConverter(chars)
        vocab_size = label_converter.get_vocab_size()
        
        crnn = CRNN(vocab_size).to(Config.get("device"))
        learning_rate = Config.get('learning_rate')
        momentum = Config.get('momentum')
        optimizer = sgd_optimizer(crnn, learning_rate, momentum)
        epochs = Config.get('epochs')
        start _epoch = 1
        validate_every = Config.get('validate_every')
        
        if args.checkpoints is not None:
            checkpoint_file = Path(args.checkpoints)
            checkpoint = torch.load(checkpoint_file, map_location=Config.get('device'))
            start_epoch = checkpoint['epoch'] + 1
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            training(crnn, training_loader, validation_loader, optimizer, label_converter, epochs, start_epoch, validate_every, output_dir)
        else:
            start_epoch = 1
            training(crnn, training_loader, validation_loader, optimizer, label_converter, epochs, start_epoch, validate_every, output_dir)

if __name__ == "__main__":
    main()
