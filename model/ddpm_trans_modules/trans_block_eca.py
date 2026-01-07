import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# ECA attention module
class Attention_eca(nn.Module):
    def __init__(self, num_heads, k_size, bias):
        super(Attention_eca, self).__init__()
        self.num_heads = num_heads

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        heads = x.chunk(self.num_heads, dim=1)
        outputs = []
        for head in heads:
            y = self.avg_pool(head)
            y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y)
            out = head * y.expand_as(head)
            outputs.append(out)
        # Two different branches of ECA module
        output = torch.cat(outputs, dim=1)

        return output


# 定义h_sigmoid激活函数，这是一种硬Sigmoid函数
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)  # 使用ReLU6实现

    def forward(self, x):
        return self.relu(x + 3) / 6  # 公式为ReLU6(x+3)/6，模拟Sigmoid激活函数


# 定义h_swish激活函数，这是基于h_sigmoid的Swish函数变体
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)  # 使用上面定义的h_sigmoid

    def forward(self, x):
        return x * self.sigmoid(x)  # 公式为x * h_sigmoid(x)

class Attention_ca(nn.Module):
    def __init__(self, inp, oup, num_heads=4, reduction=16):
        super(Attention_ca, self).__init__()
        self.num_heads = num_heads  # 定义头的数量

        # 定义水平和垂直方向的自适应平均池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平方向
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直方向

        mip = inp // reduction  # 计算中间层的通道数
        inp = inp//self.num_heads
        # 使用多个1x1卷积核来构建多头注意力
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)  # 批归一化
        self.act = h_swish()  # 激活函数

        # 两个1x1卷积，分别对应水平和垂直方向
        self.conv_h = nn.Conv2d(mip, oup//self.num_heads, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup//self.num_heads, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        heads = x.chunk(self.num_heads, dim=1)  #按通道分组
        outputs = []
        for head in heads:
            identity = head  # 保存输入作为残差连接
            n, c, h, w = head.size()  # 获取输入的尺寸
            x_h = self.pool_h(head)  # 水平方向池化
            x_w = self.pool_w(head).permute(0, 1, 3, 2)  # 垂直方向池化并交换维度以适应拼接

            y = torch.cat([x_h, x_w], dim=2)  # 拼接水平和垂直方向的特征
            y = self.conv1(y)  # 通过1x1卷积降维
            y = self.bn1(y)  # 批归一化
            y = self.act(y)  # 激活函数

            x_h, x_w = torch.split(y, [h, w], dim=2)  # 将特征拆分回水平和垂直方向
            x_w = x_w.permute(0, 1, 3, 2)  # 恢复x_w的原始维度

            a_h = self.conv_h(x_h).sigmoid()  # 通过1x1卷积并应用Sigmoid获取水平方向的注意力权重
            a_w = self.conv_w(x_w).sigmoid()  # 通过1x1卷积并应用Sigmoid获取垂直方向的注意力权重

            out = identity * a_w * a_h  # 应用注意力权重到输入特征，并与残差连接相乘
            outputs.append(out)
        output = torch.cat(outputs, dim=1)
        return output  # 返回输出


class Attention_ema(nn.Module):
    def __init__(self, channels, num_heads, factor=8):
        super(Attention_ema, self).__init__()
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
        heads = x.chunk(self.num_heads, dim=1)  # 对通道维度进行分块处理
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
            x1 = self.gn[i](group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # 逐元素相乘的广播规则
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


##########################################################################
class TransformerBlock_eca(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_eca, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_eca(num_heads, 3, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class TransformerBlock_ema(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_ema, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_ema(dim, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class TransformerBlock_ca(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_ca, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_ca(dim, dim, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


if __name__ == '__main__':
    input = torch.zeros([2, 48, 128, 128])
    # model = Restormer()
    # output = model(input)
    model2 = nn.Sequential(*[
        TransformerBlock_ca(dim=int(48), num_heads=2, ffn_expansion_factor=2.66,
                            bias=False, LayerNorm_type='WithBias') for i in range(1)])
    # model3 = Attention_sa(1, 16, 48)
    output2 = model2(input)
    print(output2.shape)
