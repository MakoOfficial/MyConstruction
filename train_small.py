import csv
import os
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch import Tensor

from torch.utils.data import Dataset
import torchvision.datasets as datasets

import torch.nn.functional as F

import random

from torch.optim.lr_scheduler import StepLR


import warnings

import torchvision.transforms as transforms
from func import print

warnings.filterwarnings("ignore")

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False


def train_fn(net, train_loader, loss_fn, epoch, optimizer):
    '''
    checkpoint is a dict
    '''
    global total_size
    global training_loss

    for batch_idx, data in enumerate(train_loader):
        image = data[0]
        image = image.type(torch.FloatTensor).cuda()

        # batch_size = len(data[1])
        # label = F.one_hot(data[1]-1, num_classes=230).float().cuda()
        # label = (data[1] - 1).type(torch.LongTensor).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        featureOri, featureEcf, decoder_output = net(image)

        loss = (1 - flags["alpha"]) * loss_fn(featureOri, decoder_output) + \
               flags["alpha"] * loss_fn(featureEcf, decoder_output)
        loss.backward()
        # backward,update parameter
        optimizer.step()
        batch_loss = loss.item()

        training_loss += batch_loss
        total_size += 1
    return training_loss / total_size


import time
from models.model import Reconstruct


def map_fn(flags):
    mymodel = Reconstruct().cuda()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),  # 3 is bicubic
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_set = datasets.ImageFolder(data_dir, transform=transform_train)
    print(train_set)

    print(train_set.__len__())
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=flags['batch_size'],
        shuffle=True,
        num_workers=flags['num_workers'],
        drop_last=True,
        pin_memory=True
    )

    loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss(reduction='sum')
    # loss_fn = nn.CrossEntropyLoss(reduction='sum')
    lr = flags['lr']

    wd = 1e-4

    optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))
    #   optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = wd)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

    ## Trains
    for epoch in range(flags['num_epochs']):
        global training_loss
        training_loss = torch.tensor([0], dtype=torch.float32)
        global total_size
        total_size = torch.tensor([0], dtype=torch.float32)

        start_time = time.time()
        train_fn(mymodel, train_loader, loss_fn, epoch, optimizer)

        train_loss = training_loss / total_size
        print(
            f'training loss is {train_loss}, time : {time.time() - start_time}, lr:{optimizer.param_groups[0]["lr"]}')
        scheduler.step()

    torch.save(mymodel.state_dict(), '/'.join([save_path, f'{model_name}.bin']))


if __name__ == "__main__":
    save_path = '../../autodl-tmp/Reconstruction400'
    os.makedirs(save_path, exist_ok=True)
    model_name = f'Reconstruction400'

    flags = {}
    flags['lr'] = 5e-4
    flags['batch_size'] = 8
    flags['num_workers'] = 8
    flags['num_epochs'] = 160
    flags['seed'] = 1
    flags['alpha'] = 0.8

    data_dir = '../../autodl-tmp/ori/'
    print(f'{save_path} start')
    map_fn(flags)
