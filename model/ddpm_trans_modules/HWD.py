# 开发者：SHELDOM
# 开发时间: 2024/7/24 16:44

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward


class HWDownsampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HWDownsampling, self).__init__()
        self.wt = DWTForward(J=1, wave='haar', mode='zero')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channel * 4, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x


if __name__ == '__main__':
    downsampling_layer = HWDownsampling(3, 64)
    input_data = torch.rand((1, 3, 64, 64))
    output_data = downsampling_layer(input_data)
    print("Input shape:", output_data.shape)
