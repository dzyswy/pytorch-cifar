import os
import torch
from PIL import Image
import torchvision
from torch import nn
import torch.onnx

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


targets_idx = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

root_dir = 'test_CIFAR_10'
obj_dir = 'test7.png'

img_dir = os.path.join(root_dir, obj_dir)
img = Image.open(img_dir)

pre_process = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.ToTensor()
])

img = pre_process(img)
img = torch.reshape(img, (1, 3, 32, 32))
print(img.shape)

mynet = torch.load('CIFAR_10_20.pth', map_location=torch.device('cpu'))
mynet.eval()

output = mynet(img)
print(targets_idx[output.argmax(axis=1).item()])


# 创建一个随机输入张量
dummy_input = torch.randn(1, 3, 32, 32)

# 指定动态轴，这里我们假设batch size是动态的
dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

# 导出模型
torch.onnx.export(mynet,         # 模型 being run
                  dummy_input,   # 模型输入 (or a tuple for multiple inputs)
                  "CIFAR_10_20.onnx",      # 导出的文件名 (can be a file or file-like object)
                  export_params=True,  # 是否存储模型参数
                  opset_version=12,    # ONNX版本
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=['input'],   # 输入的名称
                  output_names=['output'],  # 输出的名称
                  dynamic_axes=dynamic_axes)

print("模型已导出为ONNX格式")

