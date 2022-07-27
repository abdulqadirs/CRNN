import os
import numpy as np
from pathlib import Path
import random
from PIL import Image
import PIL
from torch.utils.data import Dataset, DataLoader
import string
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

class UPTIDataset(Dataset):
    def __init__(self, image_files, images_dir, labels_dir):
        self.image_files = image_files
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transforms.Compose([transforms.Resize((100, 1500)), transforms.ToTensor()])
    
    def __getitem__(self, index):
        img_file = self.image_files[index]
        img_path = Path(self.images_dir / img_file)
        label_file = img_file.split('.')[0] + '.gt.txt'
        label_path = Path(self.labels_dir / label_file)
        #reading text
        f = open(label_path, 'r')
        text = f.read().strip()
        f.close()
        text = text.replace(' ', '$') #replacing white space the '$'
        text = text.replace('\ue002', '')
        text = text.replace('\ue000', '')
        #reading image
        img = Image.open(img_path).convert('RGB')
        #flippling the image because urdu start from right to left
        img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        img = self.transform(img)
        
        return img, text

    def __len__(self):
        return len(self.image_files)

def get_train_loader(images_dir, labels_dir):
    image_files = os.listdir(images_dir)
    upti_dataset = UPTIDataset(image_files, images_dir, labels_dir)
    dataset_size = len(upti_dataset)
    indices = list(range(dataset_size))
    training_split = int(0.8 * dataset_size)

    np.random.seed(96)
    np.random.shuffle(indices)

    train_indices = indices[:training_split]
    valid_indices = indices[training_split:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_batch_size = 4
    valid_batch_size = 1

    training_loader = DataLoader(upti_dataset,
                        num_workers = 0,
                        batch_size = train_batch_size,
                        sampler = train_sampler)

    validation_loader = DataLoader(upti_dataset,
                        num_workers = 0,
                        batch_size = valid_batch_size,
                        sampler = valid_sampler)

    return training_loader, validation_loader

def get_test_loader(images_dir, labels_dir):
    image_files = os.listdir(images_dir)
    upti_dataset = UPTIDataset(image_files, images_dir, labels_dir)
    dataset_size = len(upti_dataset)
    test_batch_size = 1
    testing_loader = DataLoader(upti_dataset, num_workers = 0, batch_size = test_batch_size)

    return testing_loader

images_dir = Path('dataset/upti-1/ligature_undegraded/')
labels_dir = Path('dataset/upti-1/groundtruth')
training_loader, validation_loader = get_train_loader(images_dir, labels_dir)
testing_loader = get_test_loader(images_dir, labels_dir)
print(len(testing_loader))
for data in testing_loader:
    images = data[0]
    labels = data[1]
    print(images.shape)
    print(labels[0])
    print(len(labels))
    break