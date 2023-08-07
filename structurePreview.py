import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.random.manual_seed(10)
torch.manual_seed(10)


def t1():
    x = torch.tensor([
        [-1, -2.0, -3.0],
        [-2.0, -3.0, -4.0]
    ])
    x = torch.rand(8, 3)
    w = nn.Parameter(torch.rand(3, 4))
    bn = nn.BatchNorm1d(num_features=4)
    print(bn.running_mean)
    print(bn.running_var)

    z = x @ w
    z_mean = torch.mean(z, dim=0, keepdim=True)
    z_std = torch.std(z, dim=0, keepdim=True)
    print(z)
    print(z_mean)
    print(z_std)
    print((z - z_mean) / z_std)
    bnz = bn(z)
    print(bnz)
    o = F.sigmoid(bnz)
    print(o)
    print(F.sigmoid(z))

    print(bn.running_mean)
    print(bn.running_var)


def t2():
    z = torch.rand(8, 32, 128, 126) * 0.1 + 5.0  # [N,C,H,W]
    bn = nn.BatchNorm2d(num_features=32)

    print(bn.weight.shape)  # [C]  每个channel对应一个对应的模型参数γ
    print(bn.bias.shape)  # [C] 每个channel对应一个对应的模型参数β
    print(bn.running_mean.shape)  # [C] 计算每个channel的所有数据的均值
    print(bn.running_var.shape)  # [C] 计算每个channel的所有数据的方差

    print(F.sigmoid(z))
    print(F.sigmoid(bn(z)))

    print(bn.running_mean)
    print(bn.running_var)


def t3():
    z = torch.rand(8, 32, 128, 126) * 3.0  # [N,C,H,W]

    # [32] 依赖批次 每个通道计算一个均值  推理用训练过程中平均出来的
    bn_mean = torch.mean(z, dim=(0, 2, 3), keepdim=True)
    bn_norm = nn.BatchNorm2d(num_features=32)
    print(bn_mean.shape)

    # [8] 依赖通道 每个样本计算一个均值  推理用当前样本自身的
    ln_mean = torch.mean(z, dim=(1, 2, 3), keepdim=True)
    ln_norm = nn.LayerNorm(normalized_shape=[1, 2, 3])
    print(ln_mean.shape)
    # print(ln_mean)

    # [8,32] 每个样本的每个feature map计算一个均值 推理用当前样本自身的
    in_mean = torch.mean(z, dim=(2, 3), keepdim=True)
    in_norm = nn.InstanceNorm2d(num_features=32)
    print(in_mean.shape)

    gz = z.reshape(8, 2, 16, 128, 126)  # [N,C,H,W] -> [N,G,NC,H,W]  C = G * NC
    print(gz.shape)
    # 每个样本每组的所有feature map计算一个均值 推理用当前样本自身的
    gz_mean = torch.mean(gz, dim=(2, 3, 4), keepdim=True)
    gb_norm = nn.GroupNorm(num_groups=2, num_channels=32)
    print(gz_mean.shape)
    # print(gz_mean)

    lambda_k = nn.Parameter(torch.tensor([0.5, 0.6, 0.8]))
    weight = F.softmax(lambda_k, dim=0)
    print(weight)
    sn_mean = weight[0] * bn_mean + weight[1] * ln_mean + weight[2] * in_mean
    print(sn_mean.shape)

    print(ln_mean.view(-1))
    print(torch.mean(in_mean, dim=(1, 2, 3)).view(-1))


class BN(nn.Module):
    # note: BN
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        # traing of mean and var
        self.runing_mean = torch.zeros(1, num_features, 1, 1)
        self.runing_var = torch.zeros(1, num_features, 1, 1)
        # a smoothing factor void Divison error
        self.eps = eps
        # using momentum approach update the gama and beta
        self.momentum = momentum
        # be like in torch self.weight
        self.gama = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        # be like in torch self. bais
        self.beta = nn.Parameter(torch.ones(1, num_features, 1, 1))

    def batch_norm(self, x):
        """_summary_

        Args:
            x (_type_): _description_:batch of data

        Returns:
            _type_: _description_
        """
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_var = torch.var(x, dim=1, keepdim=True)
        # update runing_meand,and runing_var
        self.runing_mean = (1-self.momentum)*x_mean + \
            self.momentum*self.runing_mean
        self.runing_var = (1-self.momentum)*x_var+self.momentum*self.runing_var
        mean = self.runing_mean
        var = self.runing_var
        # this the result of bacthNorm is y_hat
        return ((x-x_mean)/np.sqrt(x_var+self.eps))*self.gama+self.beta


class LN(nn.Module):
    # note: LN
    pass
    # note: BN

    def __init__(self, num_feature, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        # traing of mean and var
        self.runing_mean = 0
        self.runing_var = 1
        # a smoothing factor void Divison error
        self.eps = eps
        self.num_feature = num_feature
        # using momentum approach update the gama and beta
        self.momentum = momentum
        # be like in torch self.weight
        self.gama = nn.Parameter(torch.ones(1, num_feature, 1, 1))
        # be like in torch self. bais
        self.beta = nn.Parameter(torch.zeros(1, num_feature, 1, 1))
        # runing_mean,runnig_var
        self.runing_mean = torch.zeros(1, num_feature, 1, 1)
        self.runing_var = torch.zeros(1, num_feature, 1, 1)

    def ln_norm(self, x):
        """_summary_

        Args:
            x (_type_): _description_:batch of data

        Returns:
            _type_: _description_
        """
        x_mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        x_var = torch.var(x, dim=(1, 2, 3), keepdim=True)
        # update runing_meand,and runing_var
        self.runing_mean = (1-self.momentum) * \
            self.runing_mean+self.momentum*x_mean
        self.runing_var = (1-self.momentum) * \
            self.runing_mean+self.momentum*x_var
        x_mean = self.runing_mean
        x_var = self.runing_var
        # this the result of bacthNorm is y_hat
        return ((x-x_mean)/np.sqrt(x_var+self.eps))*self.gama+self.beta


class IN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,) -> None:
        super().__init__()
        # note: IN
        # initial model paramters
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # initial mean and var
        self.running_mean = torch.zeros(1, num_features, 1, 1)
        self.running_var = torch.ones(1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def InNorm(self, x):
        """
        _summary_

        Args:
            sefl (_type_): _description_
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        var = torch.var(x, dim=(2, 3), keepdim=True)
        self.running_mean = (1-self.momentum) * \
            self.running_mean+self.momentum*mean
        self.runnin_var = (1-self.momentum)*self.running_var+self.momentum*var
        mean = self.running_mean
        var = self.running_var
        x_hat = (x-mean)/torch.sqrt(var+self.eps)
        y_hat = self.gamma*x_hat+self.beta
        return y_hat


class GN(nn.Module):
    # note: gn
    def __init__(self, num_features, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        # note: IN
        # initial model paramters
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # initial mean and var
        self.running_mean = torch.zeros(1, num_features, 1, 1)
        self.running_var = torch.ones(1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def GNorm(self, x):
        mean = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        var = torch.var(x, dim=(2, 3, 4), keepdim=True)
        self.running_mean = (1-self.momentum) * \
            self.running_mean+self.momentum*mean
        self.runnin_var = (1-self.momentum)*self.running_var+self.momentum*var
        mean = self.running_mean
        var = self.running_var
        x_hat = (x-mean)/torch.sqrt(var+self.eps)
        y_hat = self.gamma*x_hat+self.beta
        return y_hat


class SN(nn.Module):
    # note: SN
    def __init__(self, num_features, num_groups=2, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps
        # Initialize parameters
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.running_mean = nn.Parameter(torch.ones(num_groups))
        self.running_var = nn.Parameter(torch.ones(num_groups))

    def SNorm(self, x):
        batch_size, num_channels, height, width = x.size()
        num_per_group = num_channels // self.num_groups

        # Reshape input into groups
        x = x.view(batch_size, self.num_groups, num_per_group, height, width)

        # Calculate group-wise mean and variance
        mean = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        var = torch.var(x, dim=(2, 3, 4), keepdim=True)

        # Normalize the input within each group
        x = (x - mean) / torch.sqrt(var + self.eps)

        # Calculate group-wise mean and variance for scaling
        mean = torch.mean(x, dim=(2, 3, 4))
        var = torch.var(x, dim=(2, 3, 4))

        # Calculate the scaling factors based on the mean and variance weights
        scale = self.running_mean.view(1, self.num_groups, 1, 1) * mean + self.running_var.view(
            1, self.num_groups, 1, 1) * torch.sqrt(var + self.eps)

        # Reshape back to the original shape
        x = x.view(batch_size, num_channels, height, width)

        # Apply learned scaling and shifting
        x = x * scale.view(batch_size, num_channels, 1, 1) + self.beta

        # Apply global scaling and shifting
        x = x * self.gamma + self.beta
        return x


# noinspection PyMethodMayBeStatic
class WS(nn.Module):
    # note: SN
    def __init__(self, num_features, eps=1e-8):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = 0

    def forward(self, x):
        # Normalize weight along channels dimension
        weight_mean = self.weight.mean(dim=1, keepdim=True)  # conv2d weight
        weight_std = torch.sqrt(self.weight.var(
            dim=1, keepdim=True) + self.eps)
        normalized_weight = (self.weight - weight_mean) / weight_std

        # Apply weight standardization to the input
        x = x * normalized_weight

        return x


class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.ws = WS(num_features=32)
        conv = nn.Conv2d(in_channels=32, out_channels=3,
                         kernel_size=5)  
        conv_w = conv.weight # 获取卷积的weight
        # conv_new_w = self.ws(conv_w)
        self.weight = nn.Parameter(conv_w)
        self.register_buffer('weight_mean', torch.zeros(1))
        self.register_buffer('weight_std', torch.ones(1))

    def forward(self, x):
        if self.training:
            # Calculate weight mean and standard deviation
            weight_mean = self.weight.mean(
                dim=[1, 2, 3], keepdim=True)  # 卷积的weigth
            weight_std = self.weight.std(dim=[1, 2, 3], keepdim=True)  # 卷积的std

            # Update running weight mean and standard deviation
            self.weight_mean = self.weight_mean * 0.9 + weight_mean * 0.1
            self.weight_std = self.weight_std * 0.9 + weight_std * 0.1

            # Normalize weight
            weight = (self.weight - self.weight_mean) / \
                (self.weight_std + 1e-5)
        else:
            # Normalize weight using running mean and standard deviation
            weight = (self.weight - self.weight_mean) / \
                (self.weight_std + 1e-5)

        return nn.functional.conv2d(x, weight)


def t4():
    conv = nn.Conv2d(in_channels=3, out_channels=32,
                     kernel_size=(5, 5), stride=(1, 1), padding='same')
    gn = nn.GroupNorm(num_groups=2, num_channels=32)
    ws = WS()

    x = torch.rand(8, 3, 256, 256)  # [N,C,H,W]
    # 卷积正常的执行
    x1 = conv(x)  # [8,3,256,256] -> [8,32,256,256]
    # 卷积 + GN
    # [N,C,H,W] [8,3,256,256] -> [8,32,256,256] -> [8,32,256,256]
    x2 = gn(conv(x))
    # 卷积 + GN + SW
    conv_w = conv.weight  # [Cout,Cin,Kh,Kw] [32,3,5,5]
    new_conv_w = ws(conv_w)  # [32,3,5,5] -> [32,3,5,5]
    conv.weight.data.copy_(new_conv_w.data)  # 参数赋值
    # [N,C,H,W] [8,3,256,256] -> [8,32,256,256] -> [8,32,256,256]
    x2 = gn(conv(x))


if __name__ == '__main__':
    # t4()
    # data = np.array([[1, 2],
    #              [1, 3],
    #              [1, 4]]).astype(np.float32)

    # data_torch = torch.from_numpy(data)
    # bn = nn.BatchNorm1d(num_features=2)
    # bn_output = bn(data_torch)
    # print(bn_output)
    # print("#"*20)
    # bns = BN(momentum=0.01, eps=0.001, num_features=2)
    # bns.beta = bn.bias.detach().numpy()
    # bns.gama = bn.weight.detach().numpy()
    # bns_output = bns.batch_norm(data,)
    # print(bns_output)
    print("######## {} ########".format("InstanceNom"))

    data = torch.randn(8, 32, 128, 126)
    # n = nn.InstanceNorm2d(num_features=32)
    # n_output = n(data)
    # print(n_output.view(-1))
    # _in = IN(num_features=32)
    # _in.gamma = n.weight.detach().numpy()
    # _in.beta = n.bias.detach().numpy()
    # output = _in.InNorm(data)
    # print(output.view(-1))
    ws = WSConv2d(in_channels=32, out_channels=3, kernel_size=5)
    output = ws(data)
    print(output)
