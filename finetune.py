import csv
import os
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch import Tensor

from torch.utils.data import Dataset

import torch.nn.functional as F

import random

from torch.optim.lr_scheduler import StepLR

from albumentations.augmentations.transforms import Lambda, RandomBrightnessContrast
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate, HorizontalFlip
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.crops.transforms import RandomResizedCrop
from albumentations import Compose

import warnings

import torchvision.transforms as transforms
from func import print

warnings.filterwarnings("ignore")

seed = 1  # seed必须是int，可以自行设置
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed)  # 多卡模式下，让所有显卡生成的随机数一致？这个待验证
np.random.seed(seed)  # numpy产生的随机数一致
random.seed(seed)

# CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
# 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
torch.backends.cudnn.deterministic = True

# 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
# 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
torch.backends.cudnn.benchmark = False

norm_mean = [0.143]  # 0.458971
norm_std = [0.144]  # 0.225609

RandomErasing = transforms.RandomErasing(scale=(0.02, 0.08), ratio=(0.5, 2), p=0.8)


def randomErase(image, **kwargs):
    return RandomErasing(image)


def sample_normalize(image, **kwargs):
    image = image / 255
    channel = image.shape[2]
    mean, std = image.reshape((-1, channel)).mean(axis=0), image.reshape((-1, channel)).std(axis=0)
    return (image - mean) / (std + 1e-3)


transform_train = Compose([
    # RandomBrightnessContrast(p = 0.8),
    # Resize(height=512, width=512),
    RandomResizedCrop(512, 512, (0.5, 1.0), p=0.5),
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0.0,
                     p=0.8),
    # HorizontalFlip(p = 0.5),

    # ShiftScaleRotate(shift_limit = 0.2, scale_limit = 0.2, rotate_limit=20, p = 0.8),
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.8, contrast_limit=(-0.3, 0.2)),
    Lambda(image=sample_normalize),
    ToTensorV2(),
    Lambda(image=randomErase)

])

transform_val = Compose([
    Lambda(image=sample_normalize),
    ToTensorV2(),
])

transform_test = Compose([
    Lambda(image=sample_normalize),
    ToTensorV2(),
])


class BAATrainDataset(Dataset):
    def __init__(self, df, file_path):
        def preprocess_df(df):
            # nomalize boneage distribution
            # df['zscore'] = df['boneage'].map(lambda x: (x - boneage_mean) / boneage_div)
            # change the type of gender, change bool variable to float32
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path

    def __getitem__(self, index):
        row = self.df.iloc[index]
        num = int(row['id'])
        # return (transform_train(image=read_image(f"{self.file_path}/{num}.png"))['image'],
        #         Tensor([row['male']])), row['zscore']
        return (transform_train(image=cv2.imread(f"{self.file_path}/{num}.png", cv2.IMREAD_COLOR))['image'],
                # Tensor([row['male']])), Tensor([row['boneage']]).to(torch.int64)
                Tensor([row['male']])), row['boneage']

    def __len__(self):
        return len(self.df)


class BAAValDataset(Dataset):
    def __init__(self, df, file_path):
        def preprocess_df(df):
            # change the type of gender, change bool variable to float32
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return (transform_val(image=cv2.imread(f"{self.file_path}/{int(row['id'])}.png", cv2.IMREAD_COLOR))['image'],
                Tensor([row['male']])), row['boneage']

    def __len__(self):
        return len(self.df)


def create_data_loader(train_df, val_df, train_root, val_root):
    return BAATrainDataset(train_df, train_root), BAAValDataset(val_df, val_root)


def L1_penalty(net, alpha):
    l1_penalty = torch.nn.L1Loss(size_average=False)
    loss = 0
    for param in net.MLP.parameters():
        loss += torch.sum(torch.abs(param))
    for param2 in net.classifier.parameters():
        loss += torch.sum(torch.abs(param2))

    return alpha * loss

def train_fn(net, train_loader, loss_fn, epoch, optimizer):
    '''
    checkpoint is a dict
    '''
    global total_size
    global training_loss

    net.train()
    for batch_idx, data in enumerate(train_loader):
        image, gender = data[0]
        image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()

        batch_size = len(data[1])
        # label = F.one_hot(data[1]-1, num_classes=230).float().cuda()
        label = (data[1] - 1).type(torch.LongTensor).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        y_pred = net(image, gender)
        y_pred = y_pred.squeeze()
        label = label.squeeze()
        # print(y_pred)
        # print(y_pred, label)
        loss = loss_fn(y_pred, label)
        # backward,calculate gradients
        total_loss = loss + L1_penalty(net, 1e-5)
        total_loss.backward()
        # backward,update parameter
        optimizer.step()
        batch_loss = loss.item()

        training_loss += batch_loss
        total_size += batch_size
    return training_loss / total_size


def evaluate_fn(net, val_loader):
    net.eval()

    global mae_loss
    global val_total_size
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            val_total_size += len(data[1])

            image, gender = data[0]
            image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()

            label = data[1].cuda()

            y_pred = net(image, gender)
            # y_pred = net(image, gender)
            y_pred = torch.argmax(y_pred, dim=1)+1

            y_pred = y_pred.squeeze()
            label = label.squeeze()

            batch_loss = F.l1_loss(y_pred, label, reduction='sum').item()
            # print(batch_loss/len(data[1]))
            mae_loss += batch_loss
    return mae_loss

import time
from models.model import Reconstruct, Regression


def map_fn(flags):
    backbone = Reconstruct().cuda()
    msg = backbone.load_state_dict(torch.load(checkpoint))
    print(msg)
    mymodel = Regression(backbone).cuda()
    train_set, val_set = create_data_loader(train_df, valid_df, train_path, valid_path)
    print(train_set.__len__())
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=flags['batch_size'],
        shuffle=True,
        num_workers=flags['num_workers'],
        drop_last=True,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=flags['batch_size'],
        shuffle=False,
        num_workers=flags['num_workers'],
        pin_memory=True
    )

    global best_loss
    best_loss = float('inf')
    # loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss(reduction='sum')
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    lr = flags['lr']

    wd = 0

    optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr, weight_decay=wd)
    #   optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay = wd)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    ## Trains
    for epoch in range(flags['num_epochs']):
        global training_loss
        training_loss = torch.tensor([0], dtype=torch.float32)
        global total_size
        total_size = torch.tensor([0], dtype=torch.float32)

        global mae_loss
        mae_loss = torch.tensor([0], dtype=torch.float32)
        global val_total_size
        val_total_size = torch.tensor([0], dtype=torch.float32)

        start_time = time.time()
        train_fn(mymodel, train_loader, loss_fn, epoch, optimizer)

        evaluate_fn(mymodel, val_loader)

        train_loss, val_mae = training_loss / total_size, mae_loss / val_total_size
        if val_mae < best_loss:
            best_loss = val_mae
            torch.save(mymodel.state_dict(), '/'.join([save_path, f'{model_name}.bin']))
        print(
            f'training loss is {train_loss}, val loss is {val_mae}, time : {time.time() - start_time}, lr:{optimizer.param_groups[0]["lr"]}')
        scheduler.step()
    print(f'best loss: {best_loss}')

    train_test_dataset = BAAValDataset(train_df, train_path)
    train_test_dataloader = torch.utils.data.DataLoader(
        train_test_dataset,
        batch_size=flags['batch_size'],
        shuffle=False,
        num_workers=flags['num_workers'],
        pin_memory=True
    )

    # save log
    with torch.no_grad():
        train_record = [['label', 'pred']]
        train_record_path = os.path.join(save_path, f"train_result.csv")
        train_length = 0.
        total_loss = 0.
        mymodel.eval()
        for idx, data in enumerate(train_test_dataloader):
            image, gender = data[0]
            image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()

            batch_size = len(data[1])
            label = data[1].cuda()

            y_pred = mymodel(image, gender)

            output = torch.argmax(y_pred, dim=1) + 1

            output = torch.squeeze(output)
            label = torch.squeeze(label)
            for i in range(output.shape[0]):
                train_record.append([label[i].item(), round(output[i].item(), 2)])
            assert output.shape == label.shape, "pred and output isn't the same shape"

            total_loss += F.l1_loss(output, label, reduction='sum').item()
            train_length += batch_size
        print(f"training dataset length :{train_length}")
        print(f'final training loss: {round(total_loss / train_length, 3)}')
        with open(train_record_path, 'w', newline='') as csvfile:
            writer_train = csv.writer(csvfile)
            for row in train_record:
                writer_train.writerow(row)

    with torch.no_grad():
        val_record = [['label', 'pred']]
        val_record_path = os.path.join(save_path, f"val_result.csv")
        val_length = 0.
        val_loss = 0.
        mymodel.eval()
        for idx, data in enumerate(val_loader):
            image, gender = data[0]
            image, gender = image.type(torch.FloatTensor).cuda(), gender.type(torch.FloatTensor).cuda()

            batch_size = len(data[1])
            label = data[1].cuda()

            y_pred = mymodel(image, gender)

            output = torch.argmax(y_pred, dim=1) + 1
            if output.shape[0] != 1:
                output = torch.squeeze(output)
                label = torch.squeeze(label)
            for i in range(output.shape[0]):
                val_record.append([label[i].item(), round(output[i].item(), 2)])
            # assert output.shape == label.shape, "pred and output isn't the same shape"

            val_loss += F.l1_loss(output, label, reduction='sum').item()
            val_length += batch_size
        print(f"valid dataset length :{val_length}")
        print(f'final val loss: {round(val_loss / val_length, 3)}')
        with open(val_record_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in val_record:
                writer.writerow(row)
# torch.save(mymodel.state_dict(), '/'.join([save_path, f'{model_name}.bin']))


if __name__ == "__main__":
    save_path = '../../autodl-tmp/ReconstructionFinetune'
    os.makedirs(save_path, exist_ok=True)
    model_name = f'ReconstructionFinetune'
    checkpoint = '../models/modelsRecord/Reconstruction/Reconstruction_0.bin'

    flags = {}
    flags['lr'] = 5e-4
    flags['batch_size'] = 32
    flags['num_workers'] = 8
    flags['num_epochs'] = 75
    flags['seed'] = 1
    # flags['alpha'] = 0.8

    data_dir = '../../autodl-tmp/FirstRotate/'
    # data_dir = r'E:/code/archive/masked_1K_fold/fold_1'

    train_csv = os.path.join(data_dir, "train.csv")
    train_df = pd.read_csv(train_csv)
    valid_csv = os.path.join(data_dir, "valid.csv")
    valid_df = pd.read_csv(valid_csv)
    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")

    # train_ori_dir = '../../autodl-tmp/ori_4K_fold/'
    # train_ori_dir = '../archive/masked_1K_fold/'
    print(f'{save_path} start')
    map_fn(flags)
