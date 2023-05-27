import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(labels)
        self.transform = transform
        self.target_transform = target_transform
        self.label2id = {'angel': 0, 'apple': 1, 'arm': 2, 'banana': 3, 'baseball': 4, 'basketball': 5, 'bear': 6, 'beard': 7, 'bird': 8, 'book': 9, 'bowtie': 10, 'bread': 11, 'butterfly': 12, 'cake': 13, 'campfire': 14, 'carrot': 15, 'cat': 16, 'cloud': 17, 'coffee_cup': 18, 'crown': 19, 'diamond': 20, 'dog': 21, 'donut': 22, 'eye': 23, 'face': 24, 'flower': 25, 'garden': 26, 'hand': 27, 'headphones': 28, 'house_plant': 29, 'ice_cream': 30, 'leaf': 31, 'light_bulb': 32, 'lightning': 33, 'ocean': 34, 'palm_tree': 35, 'pizza': 36, 'rabbit': 37, 'smiley_face': 38, 'snowflake': 39, 'snowman': 40, 'star': 41, 'strawberry': 42, 'sun': 43, 'teddy-bear': 44}
        self.num_classes = 45

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{self.img_labels.iloc[idx, 0]}.jpg')
        # image = read_image(img_path)
        image = Image.open(img_path, 'r')
        label = self.label2id[self.img_labels.iloc[idx, 1]]
        one_hot = np.zeros(self.num_classes)
        one_hot[label] = 1
        
        if self.transform:
            image = self.transform(image)

        return image, one_hot