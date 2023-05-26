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
        self.label2id = {'angel': 0, 'apple': 1, 'arm': 2, 'banana': 3, 'baseball': 4, 'basketball': 5, 'bear': 6, 'beard': 7, 'bird': 8, 'birthday_cake': 9, 'book': 10, 'bowtie': 11, 'bread': 12, 'broccoli': 13, 'butterfly': 14, 'cake': 15, 'campfire': 16, 'candle': 17, 'carrot': 18, 'cat': 19, 'circle': 20, 'cloud': 21, 'coffee_cup': 22, 'cookie': 23, 'crown': 24, 'diamond': 25, 'dog': 26, 'dolphin': 27, 'donut': 28, 'duck': 29, 'ear': 30, 'envelope': 31, 'eye': 32, 'face': 33, 'fish': 34, 'flower': 35, 'foot': 36, 'frog': 37, 'garden': 38, 'hamburger': 39, 'hand': 40, 'headphones': 41, 'hourglass': 42, 'house': 43, 'house_plant': 44, 'ice_cream': 45, 'key': 46, 'laptop': 47, 'leaf': 48, 'light_bulb': 49, 'lightning': 50, 'lipstick': 51, 'lollipop': 52, 'megaphone': 53, 'microphone': 54, 'moon': 55, 'mountain': 56, 'mouse': 57, 'moustache': 58, 'mouth': 59, 'mushroom': 60, 'ocean': 61, 'octopus': 62, 'palm_tree': 63, 'panda': 64, 'pencil': 65, 'pizza': 66, 'rabbit': 67, 'rain': 68, 'rainbow': 69, 'shark': 70, 'sheep': 71, 'skull': 72, 'smiley_face': 73, 'snail': 74, 'snowflake': 75, 'snowman': 76, 'square': 77, 'squirrel': 78, 'star': 79, 'steak': 80, 'stop_sign': 81, 'strawberry': 82, 'sun': 83, 'teapot': 84, 'teddy-bear': 85, 'tree': 86, 'triangle': 87, 'watermelon': 88, 'whale': 89}
        self.num_classes = 90

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{self.img_labels.iloc[idx, 0]}.jpg')
        # image = read_image(img_path)
        image = Image.open(img_path, 'r')
        label = self.label2id[self.img_labels.iloc[idx, 1]]
        one_hot = np.zeros(90)
        one_hot[label] = 1
        
        if self.transform:
            image = self.transform(image)

        return image, one_hot