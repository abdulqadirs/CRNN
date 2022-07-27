import configparser
import logging

config_path = '/home/aqadir/Documents/BEYOND-DATA/CRNN/config.ini'
config = configparser.ConfigParser()
config.read(config_path)
disable_gpu = config.getboolean('GPU', 'disable_cuda', fallback=False)
print(disable_gpu)