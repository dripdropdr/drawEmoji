import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import datasets, models, transforms
import os
import argparse
import wandb
import datetime
import time
from dataset import CustomImageDataset

def arg():
    parser = argparse.ArgumentParser(description='classification')

    parser.add_argument('--lr', default=0.001, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--save_interval', default=5)

    parser.add_argument('--data', default='dataset')
    parser.add_argument('--output', default='output/emoji')

    return parser


def main(args):

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3)
    model.fc = nn.Linear(model.fc.in_features, 90)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    lr_schedluer = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, verbose=True)

    # dataset/loader
    input_size = 64
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dataset_train = CustomImageDataset(os.path.join(args.data, 'train/images'), os.path.join(args.data, 'train/train.csv'), data_transforms['train'])
    dataset_val = CustomImageDataset(os.path.join(args.data, 'val/images'), os.path.join(args.data, 'val/test.csv'), data_transforms['val'])

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # wandb
    config = vars(args)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(project='drawemoji', name=f'experiments_{now}', config = config)

    # train
    for epoch in range(args.epochs):

        train_one_epoch(model, criterion, optimizer, dataloader_train, device, epoch)
        lr_schedluer.step()
        evaluate(model, criterion, optimizer, dataloader_val, device, epoch)

        if (epoch % args.save_interval == 0):
            torch.save(model.state_dict(), f'{args.output}/chekcpoint{epoch:04}.pth')


def train_one_epoch(model, criterion, optimizer, dataloader_train, device, epoch):

    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for sample, target in tqdm(dataloader_train, desc=f'Epoch[{epoch}]:'):
        sample = sample.to(device)
        target = target.to(device)

        output = model(sample)
        loss = criterion(output, target)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        running_loss = loss.item()
        running_acc = (torch.argmax(output, axis=1) == torch.argmax(target, axis=1)).float().mean().item()
        epoch_loss += running_loss
        epoch_acc += running_acc
        wandb.log({'train_loss':running_loss, 'train_acc':running_acc, 'lr':optimizer.param_groups[0]['lr']})

    epoch_loss /= len(dataloader_train)
    epoch_acc /= len(dataloader_train)
    wandb.log({'epoch_loss': epoch_loss, 'epoch_acc' : epoch_acc})
        
def evaluate(model, criterion, optimizer, dataloader_val, device, epoch):
    model.eval()
    running_loss = 0
    running_acc = 0
    for sample, target in tqdm(dataloader_val, desc=f'Epoch[{epoch}]:'):

        sample = sample.to(device)
        target = target.to(device)

        output = model(sample)
        loss = criterion(output, target)

        running_loss += loss.item() / sample.size(0)
        running_acc += torch.sum(output == target) / sample.size(0)
        wandb.log({'eval_loss':running_loss, 'eval_acc':running_acc, 'lr':optimizer.param_groups[0]['lr']})


if __name__ == '__main__':
    parser = arg()
    args = parser.parse_args()
    main(args)