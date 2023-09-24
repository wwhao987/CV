class Normalize(nn.Module):
    def __init__(self, c1, num=20):
        super(Normalize, self).__init__()
        self.scale_weight = nn.Parameter(torch.ones(c1) * num)

    def forward(self, x):
        return self.scale_weight.view(1, -1, 1, 1) * F.normalize(x)


class DownSampling(nn.Module):
    """
    下采样：组合卷积和池化下两种方式的下采样
    卷积 + 池化 --> 1*1卷积
    """

    def __init__(self, c1, c2, k=3, p=1):
        super(DownSampling, self).__init__()
        self.cv1 = Conv(c1, c1, k, 2, p)
        self.pool1 = nn.MaxPool2d(k, 2, padding=p)
        self.cv2 = Conv(2 * c1, c2)

    def forward(self, x):
        z1 = self.cv1(x)
        z2 = self.pool1(x)
        z = torch.concat([z1, z2], dim=1)
        z = self.cv2(z)
        return z


class UpSampling(nn.Module):
    """
    上采样操作：合并反卷积 + UpSample --> conv合并通道
    """

    def __init__(self, c1, c2, output_padding=1, mode='nearest'):
        super(UpSampling, self).__init__()
        self.b1 = nn.Upsample(None, 2, mode)
        # noinspection PyTypeChecker
        self.b2 = nn.Sequential(
            nn.ConvTranspose2d(c1, c1, 3, stride=2, padding=1, output_padding=output_padding),
            nn.BatchNorm2d(c1),
            Conv.default_act
        )
        self.c1 = Conv(c1 * 2, c2)

    def forward(self, x):
        z1 = self.b1(x)
        z2 = self.b2(x)
        z = torch.concat([z1, z2], dim=1)
        z = self.c1(z)
        return z


# ---------------------------SE Begin---------------------------
class SE(nn.Module):
    def __init__(self, c1, ratio=16):
        super(SE, self).__init__()
        # c*1*1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class SE_Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.se = SE(c2, 16)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.se(self.cv2(self.cv1(x))) if self.add else self.se(self.cv2(self.cv1(x)))


class C3_SE(nn.Module):
    # CSP Bottleneck with 3 convolutions and 1 SE. by CSDN迪菲赫尔曼
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(SE_Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
# ---------------------------SE End---------------------------


class DropBlock2D(nn.Module):
    def __init__(self, p: float = 0.1, block_size: int = 7, inplace: bool = False):
        super(DropBlock2D, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("DropBlock probability has to be between 0 and 1, "
                             "but got {}".format(p))
        if block_size < 1:
            raise ValueError("DropBlock block size必须大于0.")
        if block_size % 2 != 1:
            raise ValueError("当前代码实现的并不是特别完善，要求drop的区域大小必须是奇数")
        self.p = p
        self.inplace = inplace
        self.block_size = block_size

    # noinspection PyShadowingBuiltins
    def forward(self, input: Tensor) -> Tensor:
        if not self.training:
            return input

        N, C, H, W = input.size()
        mask_h = H - self.block_size + 1
        mask_w = W - self.block_size + 1
        gamma = (self.p * H * W) / ((self.block_size ** 2) * mask_h * mask_w)
        mask_shape = (N, C, mask_h, mask_w)
        # bernoulli:伯努利数据产生器，取值只有两种：0或者1；底层每个点会产生一个随机数，随机数小于等于gamma的，对应位置就是1；否则就是0
        mask = torch.bernoulli(torch.full(mask_shape, gamma, device=input.device))

        mask = F.pad(mask, [self.block_size // 2] * 4, value=0)  # 当前0表示保留，1表示删除
        mask = F.max_pool2d(mask, (self.block_size, self.block_size), (1, 1), self.block_size // 2)

        mask = 1.0 - mask  # 最终的drop mask产生了， 0删除，1保留
        normalize_scale = mask.numel() / (1e-6 + mask.sum())  # 为了保证训练和推理的数据一致性

        if self.inplace:
            input.mul_(mask * normalize_scale)
        else:
            input = input * mask * normalize_scale
        return input