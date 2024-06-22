import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F
from torchmetrics import Accuracy
from PIL import Image, ImageDraw, ImageFont
import csv
import os
import time

class Subsampling(nn.Module):
    # LeNet-5의 subsampling layer. 학습 가능한 파라미터 w, b 적용
    def __init__(self):
        super(Subsampling, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError('Input should be a 4D tensor')
        x = self.pool(x)
        return self.w * x + self.b

class ConvC3(nn.Module):
    # LeNet-5의 C3 convolutional layer. input을 선택적으로 연결
    selection_table = [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 0],
        [5, 0, 1],
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 0],
        [4, 5, 0, 1],
        [5, 0, 1, 2],
        [0, 1, 3, 4],
        [1, 2, 4, 5],
        [0, 2, 3, 5],
        [0, 1, 2, 3, 4, 5],
    ]
    def __init__(self, kernel_size):
        super(ConvC3, self).__init__()
        self.convs = nn.ModuleList([])
        for selected in self.selection_table:
            self.convs.append(nn.Conv2d(len(selected), 1, kernel_size=kernel_size))

    def forward(self, x):
        output_channels = []
        for i, conv in enumerate(self.convs):
            selected_x = x[:,self.selection_table[i],:,:]
            output = conv(selected_x)
            output_channels.append(output)
        return torch.cat(output_channels, dim=1)

class EuclideanRBF(nn.Module):
    # LeNet-5의 마지막 fully connected layer. 인풋과 파라미터의 거리 계산
    def toBitmap(self, character):
        font = ImageFont.load_default()
        # 빈 이미지 생성
        image = Image.new('1', (7, 12), 1)
        draw = ImageDraw.Draw(image)
        # 문자를 이미지에 그리기 (중앙 정렬)
        draw.text((0, 0), character, font=font, fill=-1)
        # 이미지를 비트맵으로 변환
        bitmap = []
        for y in range(12):
            row = []
            for x in range(7):
                row.append(1 if image.getpixel((x, y)) == 0 else -1)
            bitmap.append(row)
        # 1D tensor로 변환
        tensor = torch.tensor(bitmap, dtype=torch.float32).view(-1)
        return tensor

    def __init__(self):
        super(EuclideanRBF, self).__init__()
        bitmaps = [self.toBitmap(str(i)) for i in range(10)]
        self.weights = nn.Parameter(torch.stack(bitmaps))

    def forward(self, x):
        output = []
        for i in range(x.size(0)):
            diff = x[i] - self.weights
            pow = diff ** 2
            sum = pow.sum(dim=1)
            output.append(sum)
        return torch.stack(output)

class LeNet5_simplified(nn.Module):
    # 기존 LeNet-5와, 단순화된 레이어 비교를 위한 클래스
    # init parameter에 bool값으로 각 레이어 단순화 여부 선택
    def tanH(self, x):
        return 1.7159 * F.tanh(2/3 * x)

    def __init__(self, tanh, pool, conv, fc):
        super(LeNet5_simplified, self).__init__()
        if tanh==True:
            self.tanh = F.tanh
        else:
            self.tanh = self.tanH
        self.conv1 = nn.Conv2d(1,6, kernel_size=5)
        if pool == True:
            self.pool1 = nn.AvgPool2d(kernel_size=2)
        else:
            self.pool1 = Subsampling()
        if conv == True:
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        else:
            self.conv2 = ConvC3(kernel_size=5)
        if pool == True:
            self.pool2 = nn.AvgPool2d(kernel_size=2)
        else:
            self.pool2 = Subsampling()
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84)
        if fc == True:
            self.fc2 = nn.Linear(84, 62)
        else:
            self.fc2 = EuclideanRBF()

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(self.pool1(x))
        x = self.conv2(x)
        x = self.tanh(self.pool2(x))
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

def train_and_test(train_dataloader, test_dataloader, models, loss_fn, optimizers, accuracy, device, epochs):
    # 학습 진행
    train_losses = [[] for model in models]
    train_accs = [[] for model in models]
    test_losses = [[] for model in models]
    test_accs = [[] for model in models]
    for i in range(len(models)):
        print(f"model {i: 2d}")
        start_model = time.time()
        for epoch in range(EPOCHS):
            start_epoch = time.time()

            # train
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
            f"model {i: d} ends. Exec time: {(end_model - start_model): .4f} s")

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

dir = './data'
MNIST_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081), transforms.Pad(2)])

if __name__ == '__main__':
    # MNIST 로드
    MNIST_trainset = datasets.MNIST(root=dir, train=True, download=True, transform=MNIST_transform)
    MNIST_testset = datasets.MNIST(root=dir, train=False, download=True, transform=MNIST_transform)

    # 데이터로더
    train_dataloader = DataLoader(dataset=MNIST_trainset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(dataset=MNIST_testset, batch_size=512, shuffle=False)

    # 학습모델 세팅
    models = [
        LeNet5_simplified(False, False, False, False),
        LeNet5_simplified(False, True, False, False),
        LeNet5_simplified(False, False, True, False),
        LeNet5_simplified(False, False, False, True),
        LeNet5_simplified(False, True, True, False),
        LeNet5_simplified(False, True, False, True),
        LeNet5_simplified(False, False, True, True),
        LeNet5_simplified(False, True, True, True),
        LeNet5_simplified(True, False, False, False),
        LeNet5_simplified(True, True, False, False),
        LeNet5_simplified(True, False, True, False),
        LeNet5_simplified(True, False, False, True),
        LeNet5_simplified(True, True, True, False),
        LeNet5_simplified(True, True, False, True),
        LeNet5_simplified(True, False, True, True),
        LeNet5_simplified(True, True, True, True),
    ]
    loss_fn = nn.CrossEntropyLoss()
    optimizers = [torch.optim.Adam(params=model.parameters(), lr=0.001) for model in models]
    accuracy = Accuracy(task='multiclass', num_classes=10)

    # 장치 세팅
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    accuracy = accuracy.to(device)
    models = [model.to(device) for model in models]

    # 학습 진행 및 저장
    EPOCHS = 10
    train_accs, train_losses, test_accs, test_losses = (
        train_and_test(train_dataloader, test_dataloader, models, loss_fn, optimizers, accuracy, device, EPOCHS))
    save_result('LeNet', train_accs, train_losses, test_accs, test_losses, EPOCHS)


    # 아래는 해당 모델의 EMNIST의 정확도 확인을 위한 코드
    # EMNIST 로드
    EMNIST_trainset = datasets.EMNIST(root=dir, train=True, download=True, transform=MNIST_transform,
                                      split='byclass')
    EMNIST_testset = datasets.EMNIST(root=dir, train=False, download=True, transform=MNIST_transform,
                                     split='byclass')
    train_dataloader = DataLoader(dataset=EMNIST_trainset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(dataset=EMNIST_testset, batch_size=512, shuffle=False)

    # 학습모델 세팅
    models = [
        LeNet5_simplified(False, False, False, True),
        LeNet5_simplified(True, True, True, True),
    ]
    loss_fn = nn.CrossEntropyLoss()
    optimizers = [torch.optim.Adam(params=model.parameters(), lr=0.001) for model in models]
    accuracy = Accuracy(task='multiclass', num_classes=62)

    # 장치 세팅
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    accuracy = accuracy.to(device)
    models = [model.to(device) for model in models]

    # 학습 진행 및 저장
    EPOCHS = 10
    train_accs, train_losses, test_accs, test_losses = (
        train_and_test(train_dataloader, test_dataloader, models, loss_fn, optimizers, accuracy, device, EPOCHS))
    save_result('LeNet_EMNIST', train_accs, train_losses, test_accs, test_losses, EPOCHS)