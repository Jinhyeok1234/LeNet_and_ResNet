import matplotlib.pyplot as plt
import pandas as pd
import os

# 결과 확인을 위한 코드
# 거의 하드코딩되어 있으므로, 저장값이 달라지면 새로 짜야 된다

dir = './result'

def loadResult(filename):
    train_accs = pd.read_csv(os.path.join(dir, filename, 'train_accs.csv'))
    test_accs = pd.read_csv(os.path.join(dir, filename, 'test_accs.csv'))
    train_errors = 1 - train_accs
    test_errors = 1 - test_accs
    return train_errors, test_errors

def showLeNet():
    filename = 'LeNet'
    train_errors, test_errors = loadResult(filename)
    plt.figure(figsize=(10, 8))
    plt.suptitle('LeNet')
    for i in range(16):
        tanh = bool(i // 8)
        j = i % 8
        pool = (j == 1 or j == 4 or j == 5 or j == 7)
        conv = (j == 2 or j == 4 or j == 6 or j == 7)
        fc = (j == 3 or j == 5 or j == 6 or j == 7)
        plt.subplot(1, 2, 1)
        plt.plot(train_errors.columns, train_errors.iloc[i], label=f'{tanh:d}{pool:d}{conv:d}{fc:d}', marker='o',
                 linestyle='-')
        plt.subplot(1, 2, 2)
        plt.plot(test_errors.columns, test_errors.iloc[i], label=f'{tanh:d}{pool:d}{conv:d}{fc:d}', marker='x',
                 linestyle='--')
    plt.subplot(1, 2, 1)
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.grid(True)
    plt.title('train')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.grid(True)
    plt.title('test')
    plt.legend()
    plt.show()
    plt.close()

    for c in range(1,5):
        plt.figure(figsize=(18, 8))
        if c == 1:
            title = 'modify tanh'
        if c == 2:
            title = 'modify pooling layer'
        if c == 3:
            title = 'modify C3 convolution layer'
        if c == 4:
            title = 'modify last fully connected layer'
        plt.suptitle(title)
        for i in range(16):
            tanh = bool(i//8)
            j = i % 8
            pool = (j==1 or j==4 or j==5 or j==7)
            conv = (j==2 or j==4 or j==6 or j==7)
            fc = (j==3 or j==5 or j==6 or j==7)
            if c == 1:
                color = 'b' if tanh else 'r'
            if c == 2:
                color = 'b' if pool else 'r'
            if c == 3:
                color = 'b' if conv else 'r'
            if c == 4:
                color = 'b' if fc else 'r'
            plt.subplot(1,2,1)
            plt.plot(train_errors.columns, train_errors.iloc[i], label=f'{tanh:d}{pool:d}{conv:d}{fc:d}', marker='o', linestyle='-', color=color)
            plt.subplot(1, 2, 2)
            plt.plot(test_errors.columns, test_errors.iloc[i], label=f'{tanh:d}{pool:d}{conv:d}{fc:d}', marker='x', linestyle='--', color=color)
        plt.subplot(1, 2, 1)
        plt.xlabel('epoch')
        plt.ylabel('error')
        plt.grid(True)
        plt.title('train')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.xlabel('epoch')
        plt.ylabel('error')
        plt.grid(True)
        plt.title('test')
        plt.legend(title='blue is modified')
        plt.show()
        plt.close()

def showEMNIST():
    filename = 'LeNet_EMNIST'
    train_errors, test_errors = loadResult(filename)
    plt.figure(figsize=(10, 8))
    for i in range(2):
        plt.plot(train_errors.columns, train_errors.iloc[i], label=f'train_{i}', marker='o', linestyle='-')
        plt.plot(test_errors.columns, test_errors.iloc[i], label=f'test_{i}', marker='x', linestyle='--')
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.grid(True)
    plt.title(filename)
    plt.legend()
    plt.show()
    plt.close()

def showResNet():
    filename = 'ResNet'
    train_errors, test_errors = loadResult(filename)
    plt.figure(figsize=(10, 8))
    for i in range(2):
        plt.plot(train_errors.columns, train_errors.iloc[i], label=f'train_{i}', marker='o', linestyle='-')
        plt.plot(test_errors.columns, test_errors.iloc[i], label=f'test_{i}', marker='x', linestyle='--')
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.grid(True)
    plt.title(filename)
    plt.legend()
    plt.show()
    plt.close()

if __name__ == '__main__':
    #showLeNet()
    #showEMNIST()
    showResNet()