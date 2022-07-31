import configparser
import logging
import torch
import config

def reading_config(file_path):
    """
    Reads the config settings from config file and makes them accessible to the project using config.py.
    Args:
        file_path (Path): The path of config file.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    config = configparser.ConfigParser()
    try:
        config.read(file_path)
        print('Reading the config file from: %s' % file_path)
    except FileNotFoundError:
        print("Config file doesn't exist.")

    #GPUs
    Config.set("disable_cuda", config.getboolean("GPU", "disable_cuda", fallback=False))
    if not Config.get("disable_cuda") and torch.cuda.is_available():
        Config.set("device", "cpu")
        print('GPU is available')
    else:
        Config.set("device", "cpu")
        print('Only CPU is available')


    #data
    Config.set("image_channels", config.getint("data", "image_channels", fallback=3))
    Config.set("image_width", config.getint("data", "image_width", fallback=1500))
    Config.set("image_height", config.getint("data", "image_height", fallback=100))
    Cofig.set("flip_image", config.getbollean("data", "flip_image", fallback=True))

    #model
    Config.set("gru_hidden_size", config.getint("model", "gru_hidden_size", facllback=256))
    Config.set("dropout", config.getfloat("model", "dropout", fallback=0.2))
    
    #Training
    Config.set("training_batch_size", config.getint("training", "batch_size", fallback=8))
    Config.set("epochs", config.getint("training", "epochs", fallback=50))
    Config.set("learning_rate", config.getfloat("training", "learning_rate", fallback=0.0001))
    Config.set("momentum", config.getfloat("training", "momentum", fallback=0.9))
    Config.set("weight_decay", config.getfloat("training", "weight_decay", fallback=0.001))

    #validation
    Config.set("validation_batch_size", config.getint("validation", "batch_size", fallback=1))
    Config.set("validate_every", config.getint("validation", "validate_every", fallback=1))

    #testing
    Config.set("testing_batch_size", config.getint("testing", "batch_size", fallback=1))

