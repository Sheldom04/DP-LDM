# 开发者：SHELDOM
# 开发时间: 2024/10/14 14:04
import torch
import torch.nn as nn

class EMA(nn.Module):
    def __init__(self, channels, num_heads, factor=8):
        super(EMA, self).__init__()
        # 设置分组数量，用于特征分组
        self.groups = factor
        self.num_heads = num_heads
        assert channels // (self.groups * self.num_heads) > 0  # 确保分组后的通道数大于0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 为每个头创建独立的GroupNorm、1x1卷积和3x3卷积
        self.gn = nn.ModuleList([
            nn.GroupNorm(channels // (self.groups * self.num_heads), channels // (self.groups * self.num_heads))
            for _ in range(self.num_heads)
        ])
        self.conv1x1 = nn.ModuleList([
            nn.Conv2d(channels // (self.groups * self.num_heads), channels // (self.groups * self.num_heads),
                      kernel_size=1, stride=1, padding=0)
            for _ in range(self.num_heads)
        ])
        self.conv3x3 = nn.ModuleList([
            nn.Conv2d(channels // (self.groups * self.num_heads), channels // (self.groups * self.num_heads),
                      kernel_size=3, stride=1, padding=1)
            for _ in range(self.num_heads)
        ])

    def forward(self, x):
        b, c, h, w = x.size()
        # 对输入特征图进行多头处理
        heads = x.chunk(self.num_heads, dim=1)
        outputs = []

        # 遍历每个头
        for i, head in enumerate(heads):
            group_x = head.reshape(b * self.groups, -1, h, w)  # b*g, c//(g*num_heads), h, w

            # 应用水平和垂直方向的全局平均池化
            x_h = self.pool_h(group_x)
            x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

            # 通过1x1卷积和sigmoid激活函数，获得注意力权重
            hw = self.conv1x1[i](torch.cat([x_h, x_w], dim=2))
            x_h, x_w = torch.split(hw, [h, w], dim=2)

            # 应用GroupNorm和注意力权重调整特征图
            x1 = self.gn[i](group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
            x2 = self.conv3x3[i](group_x)

            # 将特征图通过全局平均池化和softmax进行处理，得到权重
            x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
            x12 = x2.reshape(b * self.groups, c // (self.groups * self.num_heads), -1)  # b*g, c//(g*num_heads), hw
            x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
            x22 = x1.reshape(b * self.groups, c // (self.groups * self.num_heads), -1)  # b*g, c//(g*num_heads), hw

            # 通过矩阵乘法和sigmoid激活获得最终的注意力权重，调整特征图
            weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

            # 将调整后的特征图添加到输出
            output = (group_x * weights.sigmoid()).reshape(b, -1, h, w)
            outputs.append(output)

        # 合并所有头的输出
        return torch.cat(outputs, dim=1)

    # 测试EMA模块
if __name__ == '__main__':
    block = EMA(64, 2).cuda()  # 实例化EMA模块，并移至CUDA设备
    input = torch.rand(1, 64, 64, 64).cuda()  # 创建随机输入数据
    output = block(input)  # 前向传播
    print(output.shape)  # 打印输入和输出的尺寸