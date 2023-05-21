import torch
from tqdm import tqdm
from torchvision import datasets, models, transforms
import os
import argparse


def arg():
    parser = argparse.ArgumentParser(description='classification')

    parser.add_argument('--lr', default=0.0001, type=int)




def main(args):

    data_dir = 'data/hymenoptera_data'
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    
    dataset_train = datasets.ImageFolder(os.path.join(data_dir, 'train/images'), data_transforms['train'])
    dataset_val = datasets.ImageFolder(os.path.join(data_dir, 'val/images'), data_transforms['val'])

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=4, shuffle=True, num_workers=4)

    

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, )
    # lr_schedluer = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_)




    for sample, target in tqdm(dataset_train):
        
    
        model.train()


if __name__ == '__main__':
    args = arg()
    main(args)