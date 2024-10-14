import matplotlib.pyplot as plt
import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader

# 选择GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loss
train_loss = []
test_loss = []

# 训练集
train_data_set = torchvision.datasets.CIFAR10(root='../dataset', train=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]), download=True)

# 测试集
test_data_set = torchvision.datasets.CIFAR10(root='../dataset', train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]), download=True)

train_data_size = len(train_data_set)
test_data_size = len(test_data_set)

print(f'训练集大小：{train_data_size}')
print(f'测试集大小：{test_data_size}')

train_data_load = DataLoader(dataset=train_data_set, batch_size=64, shuffle=True, drop_last=True)
test_data_load = DataLoader(dataset=test_data_set, batch_size=64, shuffle=True, drop_last=True)


def plot_loss(loss_train, loss_test):
    plt.Figure(figsize=(5, 5))
    plt.plot(loss_train, label='train_loss', alpha=0.5)
    plt.plot(loss_test, label='test_loss', alpha=0.5)
    plt.title('CIFAR10')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=128),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.fc(self.main(x))


mynet = MyNet()
mynet = mynet.to(device)
print(mynet)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

learning_rate = 1e-3
optim = torch.optim.Adam(mynet.parameters(), lr=learning_rate)

train_step = 0
test_step = 0

epoch = 20







if __name__ == '__main__':
    for i in range(epoch):
        print(f'-------------第{i+1}轮训练开始-------------------')
        mynet.train()
        for j, (imgs, targets) in enumerate(train_data_load):

            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = mynet(imgs)
            loss = loss_fn(outputs, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_step += 1
            if train_step % 100 == 0:
                print(f'训练{train_step}次，loss={loss}')
                train_loss.append(loss.cpu().tolist())

        mynet.eval()
        accuracy = 0
        accuracy_total = 0
        with torch.no_grad():
            for j, (imgs, targets) in enumerate(test_data_load):

                imgs = imgs.to(device)
                targets = targets.to(device)

                outputs = mynet(imgs)
                loss = loss_fn(outputs, targets)

                accuracy = (outputs.argmax(axis=1) == targets).sum()
                accuracy_total += accuracy
                test_step += 1
                if test_step % 100 == 0:
                    print(f'测试{test_step}次，loss={loss}')
                    test_loss.append(loss.cpu().tolist())

        print(f'第{i+1}轮训练结束，准确率{accuracy_total/test_data_size}')
        torch.save(mynet, f'CIFAR_10_{i+1}.pth')


    print(train_loss)
    print(test_loss)
    plot_loss(train_loss, test_loss)