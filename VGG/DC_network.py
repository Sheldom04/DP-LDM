import torch
from torch import nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        dim = 24
        self.block1 = nn.Sequential(  # 用一个序列（块）来表示
            nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2**2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=dim * 2**2, out_channels=dim * 2**2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=dim * 2**2, out_channels=dim * 2**2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2**2, out_channels=dim * 2**3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=dim * 2**3, out_channels=dim * 2**3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=dim * 2**3, out_channels=dim * 2**3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2 ** 3, out_channels=dim * 2 ** 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=dim * 2 ** 4, out_channels=dim * 2 ** 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=dim * 2 ** 4, out_channels=dim * 2 ** 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 参数初始化
        for m in self.modules():  # 权重初始化，防止模型的参数随机生成，从而产生不收敛的现象
            if isinstance(m, nn.Conv2d):  # 卷积层参数初始化
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # 使用何恺明初始化，优化参数w
                if m.bias is not None:  # b参数优化，初始置为0，如果有的话
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 全连接层参数初始化
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        return x4


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16()
    noise = torch.randn(1, 3, 128, 128)
    x1 = model.block1(noise)
    x2 = model.block2(x1)
    x3 = model.block3(x2)
    x4 = model.block4(x3)
    print(x1.shape)  # 打印模型信息
    print(x2.shape)  # 打印模型信息
    print(x3.shape)  # 打印模型信息
    print(x4.shape)  # 打印模型信息
