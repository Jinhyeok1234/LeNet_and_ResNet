import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F
from torchmetrics import Accuracy
import csv
import os
import time

class ResidualConv_2(nn.Module):
    def __init__(self, in_channels, out_channels, downsampling):
        super(ResidualConv_2, self).__init__()

        if downsampling==True:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if downsampling==True:
            self.downsampling = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            self.downsampling_bn = nn.BatchNorm2d(out_channels)
        else:
            self.downsampling = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.downsampling is not None:
            x = self.downsampling_bn(self.downsampling(x))
        out = F.relu(out + x)
        return out

class LesNet_34(nn.Module):
    def __init__(self, classes):
        super(LesNet_34, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2_1 = ResidualConv_2(64, 64, False)
        self.conv2_2 = ResidualConv_2(64, 64, False)
        self.conv2_3 = ResidualConv_2(64, 64, False)

        self.conv3_1 = ResidualConv_2(64, 128, True)
        self.conv3_2 = ResidualConv_2(128, 128, False)
        self.conv3_3 = ResidualConv_2(128, 128, False)
        self.conv3_4 = ResidualConv_2(128, 128, False)

        self.conv4_1 = ResidualConv_2(128, 256, True)
        self.conv4_2 = ResidualConv_2(256, 256, False)
        self.conv4_3 = ResidualConv_2(256, 256, False)
        self.conv4_4 = ResidualConv_2(256, 256, False)
        self.conv4_5 = ResidualConv_2(256, 256, False)
        self.conv4_6 = ResidualConv_2(256, 256, False)

        self.conv5_1 = ResidualConv_2(256, 512, True)
        self.conv5_2 = ResidualConv_2(512, 512, False)
        self.conv5_3 = ResidualConv_2(512, 512, False)

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)

        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        return x

class LesNet_18(nn.Module):
    def __init__(self, classes):
        super(LesNet_18, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_1 = ResidualConv_2(64, 64, False)
        self.conv2_2 = ResidualConv_2(64, 64, False)

        self.conv3_1 = ResidualConv_2(64, 128, True)
        self.conv3_2 = ResidualConv_2(128, 128, False)

        self.conv4_1 = ResidualConv_2(128, 256, True)
        self.conv4_2 = ResidualConv_2(256, 256, False)

        self.conv5_1 = ResidualConv_2(256, 512, True)
        self.conv5_2 = ResidualConv_2(512, 512, False)

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)

        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        return x

def set_models(classes):
    # 학습모델 세팅
    models = [
        LesNet_34(classes),
        LesNet_18(classes),
    ]
    loss_fn = nn.CrossEntropyLoss()
    optimizers = [torch.optim.Adam(params=model.parameters(), lr=0.001) for model in models]
    accuracy = Accuracy(task='multiclass', num_classes=classes)
    return models, loss_fn, optimizers, accuracy

def train_and_test(train_dataloader, test_dataloader, models, loss_fn, optimizers, accuracy, device, epochs):
    # 학습 진행
    train_losses = [[] for model in models]
    train_accs = [[] for model in models]
    test_losses = [[] for model in models]
    test_accs = [[] for model in models]
    for i in range(len(models)):
        print(f"model {i: 2d}")
        start_model = time.time()
        for epoch in range(epochs):
            print(f'\tepoch {epoch} starts')
            start_epoch = time.time()

            # train
            print('\t\ttrain start')
            train_loss, train_acc = 0.0, 0.0
            for X, y in train_dataloader:
                X, y = X.to(device), y.to(device)
                models[i].train()
                y_pred = models[i](X)
                loss = loss_fn(y_pred, y)
                train_loss += loss.item()
                acc = accuracy(y_pred, y)
                train_acc += acc.item()
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()
            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)
            train_losses[i].append(train_loss)
            train_accs[i].append(train_acc)

            # test
            print('\t\ttest start')
            test_loss, test_acc = 0.0, 0.0
            models[i].eval()
            with torch.inference_mode():
                for X, y in test_dataloader:
                    X, y = X.to(device), y.to(device)
                    y_pred = models[i](X)
                    loss = loss_fn(y_pred, y)
                    test_loss += loss.item()
                    acc = accuracy(y_pred, y)
                    test_acc += acc.item()
                test_loss /= len(test_dataloader)
                test_acc /= len(test_dataloader)
            test_losses[i].append(test_loss)
            test_accs[i].append(test_acc)

            end_epoch = time.time()
            print(
                f"\tEpoch: {epoch: 2d}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Test loss: {test_loss: .5f}| Test acc: {test_acc: .5f}| Exec time: {(end_epoch - start_epoch): .0f} s")

        end_model = time.time()
        print(
            f"model {i: d} ends. Exec time: {(end_model - start_model): .0f} s")

    return train_accs, train_losses, test_accs, test_losses

def save_result(name, train_accs, train_losses, test_accs, test_losses, epochs):
    # 결과 기록
    epoch = []
    for i in range(epochs):
        epoch.append(i)
    result_dir = './result/' + name
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(os.path.join(result_dir, 'train_accs.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(epoch)
        writer.writerows(train_accs)
    with open(os.path.join(result_dir, 'train_losses.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(epoch)
        writer.writerows(train_losses)
    with open(os.path.join(result_dir, 'test_accs.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(epoch)
        writer.writerows(test_accs)
    with open(os.path.join(result_dir, 'test_losses.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(epoch)
        writer.writerows(test_losses)

if __name__ == '__main__':
    dir = './data'

    # CIFAR-10
    CIFAR10_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    CIFAR10_trainset = datasets.CIFAR10(root=dir, train=True, download=True, transform=CIFAR10_transform)
    CIFAR10_testset = datasets.CIFAR10(root=dir, train=False, download=True, transform=CIFAR10_transform)

    # 데이터로더
    train_dataloader = DataLoader(dataset=CIFAR10_trainset, batch_size=32, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dataset=CIFAR10_testset, batch_size=32, shuffle=False, num_workers=0)

    # 학습모델 세팅
    models, loss_fn, optimizers, accuracy = set_models(10)

    # 장치 세팅
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    accuracy = accuracy.to(device)
    models = [model.to(device) for model in models]

    # 학습 및 저장
    EPOCHS = 100
    train_accs, train_losses, test_accs, test_losses =(
        train_and_test(train_dataloader, test_dataloader, models, loss_fn, optimizers, accuracy, device, EPOCHS))
    save_result("ResNet_CIFAR10", train_accs, train_losses, test_accs, test_losses, EPOCHS)



    # CIFAR-100
    CIFAR100_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ])
    CIFAR100_trainset = datasets.CIFAR100(root=dir, train=True, download=True, transform=CIFAR100_transform)
    CIFAR100_testset = datasets.CIFAR100(root=dir, train=False, download=True, transform=CIFAR100_transform)

    # 데이터로더
    train_dataloader = DataLoader(dataset=CIFAR10_trainset, batch_size=32, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dataset=CIFAR10_testset, batch_size=32, shuffle=False, num_workers=0)

    # 학습모델 세팅
    models, loss_fn, optimizers, accuracy = set_models(10)

    # 장치 세팅
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    accuracy = accuracy.to(device)
    models = [model.to(device) for model in models]

    # 학습 및 저장
    EPOCHS = 100
    train_accs, train_losses, test_accs, test_losses = (
        train_and_test(train_dataloader, test_dataloader, models, loss_fn, optimizers, accuracy, device, EPOCHS))
    save_result("ResNet_CIFAR100", train_accs, train_losses, test_accs, test_losses, EPOCHS)