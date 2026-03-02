import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
# from TEA import TEAttention

def hard_sigmoid(x, inplace: bool = False):
    return F.hardsigmoid(x, inplace=inplace)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def gabor_kernel(size, sigma, theta, Lambda, gamma):
    sigma_x = sigma
    sigma_y = sigma / gamma
    psi = 0
    (y, x) = torch.meshgrid(
        torch.arange(-size // 2 + 1, size // 2 + 1),
        torch.arange(-size // 2 + 1, size // 2 + 1),
        indexing='ij'  # 显式指定索引顺序
    )
    x = x.float()
    y = y.float()

    x_theta = x * torch.cos(torch.tensor(theta)) + y * torch.sin(torch.tensor(theta))
    y_theta = -x * torch.sin(torch.tensor(theta)) + y * torch.cos(torch.tensor(theta))

    gb = torch.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * torch.exp(
        2 * math.pi / Lambda * x_theta + psi
    )
    return gb


class TEAttention(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, num_bins=10, stride=1):
        super(TEAttention, self).__init__()
        self.M = num_bins
        self.oup = oup
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(1, oup, kernel_size=1, bias=False)
        self.Position_reinforcement = nn.Sequential(
            nn.BatchNorm2d(oup),
            nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
            nn.BatchNorm2d(oup),
            nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
            nn.BatchNorm2d(oup),
            nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=(1, 1), padding=(0, 4), groups=oup, dilation=2),
            nn.BatchNorm2d(oup),
            nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), groups=oup, dilation=2),
            nn.BatchNorm2d(oup)
        )

        self.fc1 = nn.Linear(2, oup)
        self.phi1 = nn.Conv1d(oup, oup, kernel_size=1)
        self.phi2 = nn.Conv1d(oup, oup, kernel_size=1)
        self.phi3 = nn.Conv1d(oup, oup, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(oup)
        self.bn2 = nn.BatchNorm2d(oup)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch, _, wth, hth = x.size()
        x1 = x

        x_center_1 = self.conv1(x1)
        x_center_final = self.Position_reinforcement(x_center_1)  # Shape: (batch_size, reduced_channels, height//2, width//2)
        # print("x_center_final", x_center_final.shape)

        x_granularity = self.global_avg_pool(x1)  # Shape: (batch_size, reduced_channels, 1, 1)
        # print("x_granularity", x_granularity.shape)
        S = F.cosine_similarity(x_granularity, x1, dim=1)
        # print("S", S.shape)
        # Assuming S_flat is already computed

        S_flat = S.view(batch, -1)  # Shape: (batch_size, height * width)

        max_val = torch.max(S_flat, dim=1, keepdim=True)[0]  # Shape: (batch_size, 1)
        min_val = torch.min(S_flat, dim=1, keepdim=True)[0]  # Shape: (batch_size, 1)

        # Create Level tensor for all batches in one go
        Level = min_val + (max_val - min_val) * torch.linspace(0, 1, self.M, device=x.device).view(1,
                                                                                                   -1)  # Shape: (1, num_bins)
        # Expand S_flat for broadcasting
        S_flat_expanded = S_flat.unsqueeze(2)  # Shape: (batch_size, height * width, 1)
        S_flat_expanded = S_flat_expanded.expand(-1, -1, self.M)  # Shape: (batch_size, height * width, num_bins)
        Level_expanded = Level.unsqueeze(1)  # Shape: (1, 1, num_bins)
        # Now compute the difference
        diff = torch.abs(Level_expanded - S_flat_expanded)  # Shape: (batch_size, height * width, num_bins)
        V = (1 - diff) * (diff < (0.5 / self.M)).float()

        # Compute C_hist
        C_hist = torch.zeros(batch, self.M, 2, device=x.device)
        C_hist[:, :, 0] = V.sum(dim=1) / V.sum(dim=(1, 2)).view(batch, 1)  # Normalize
        C_hist[:, :, 1] = Level.squeeze()  # Use the computed Level

        # print("C_hist调整前", C_hist.shape)
        C_hist = C_hist.view(-1, 2)  # Shape: (batch_size*num_bins, 2)
        # print("C_hist调整后", C_hist.shape)
        C_hist = C_hist.to(device)
        C_hist = self.fc1(C_hist)
        C_hist = C_hist.view(batch, -1, self.M)  # Shape: (batch_size, reduced_channels, num_bins)
        # print("C_hist经过MLP之后", C_hist.shape)

        phi1_D = self.phi1(C_hist)  # Shape: (batch_size, reduced_channels, num_bins)
        # print("phi1_D", phi1_D.shape)
        phi2_D = self.phi2(C_hist)  # Shape: (batch_size, reduced_channels, num_bins)
        # print("phi2_D", phi2_D.shape)

        X = torch.bmm(phi1_D.permute(0, 2, 1), phi2_D)
        X = F.softmax(X, dim=-1)  # Shape: (batch_size, num_bins, num_bins)

        # print("X", X.shape)
        L_prime = self.phi3(C_hist)
        L_prime = torch.bmm(L_prime, X)  # Shape: (batch_size, reduced_channels, num_bins)
        # print("L_prime", L_prime.shape)
        V = V.view(batch, self.M, wth*hth)
        R = torch.bmm(L_prime, V)
        R = R.view(batch, -1,  wth, hth)  # R-Shape: (batch_size, reduced_channels, wth, hth)
        # print("R_prime", R.shape)

        x_texture = self.bn1(R)
        SF = S_flat.view(batch, -1,  wth, hth)
        x_texture_full = self.conv2(SF)
        x_texture_full = self.bn2(x_texture_full)

        x_all = torch.add(x_texture_full, x_texture)
        x_all = torch.add(x_all, x_center_final)
        # print("x_all", x_all.shape)
        x_all = self.sigmoid(x_all)

        return x * x_all.expand_as(x)


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class GaborConvLayer(nn.Module):
    def __init__(self, in_channels, scales, kernel_size, stride):
        super(GaborConvLayer, self).__init__()
        self.scales = scales
        self.kernel_size = kernel_size
        self.filter_channels = scales
        self.stride = stride
        self.in_channels = in_channels
        self.sigma = nn.Parameter(torch.ones(self.filter_channels) * 1.0, requires_grad=True)
        self.lambd = nn.Parameter(torch.ones(self.filter_channels) * 10.0, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(self.filter_channels) * 0.5, requires_grad=True)
        self.basic_kernel = nn.Parameter(torch.randn(scales, kernel_size, kernel_size), requires_grad=True)
        self.gabor_filters = nn.Parameter(torch.zeros(scales, in_channels, kernel_size, kernel_size))
        thetas = [i * math.pi / scales for i in range(scales)]
        for i in range(scales):
            self.gabor_filters.data[i] = gabor_kernel(kernel_size, self.sigma[i], thetas[i], self.lambd[i], self.gamma[i])

    def forward(self, x):
        basic_kernel_expanded = self.basic_kernel.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        combined_filters = basic_kernel_expanded * self.gabor_filters
        return F.conv2d(x, combined_filters, stride=self.stride, padding=self.kernel_size // 2)


class MultiGaborConv(nn.Module):
    def __init__(self, in_channels, out_channels, scales, kernel_size, stride=1):
        super(MultiGaborConv, self).__init__()
        assert in_channels % scales == 0, "in_channels must be divisible by groups"
        assert out_channels % scales == 0, "out_channels must be divisible by groups"
        self.scales = scales
        self.in_channels = in_channels
        self.groups = out_channels // scales
        self.layers = nn.ModuleList()
        for _ in range(self.groups):
            self.layers.append(
                GaborConvLayer(self.in_channels, self.scales,  kernel_size, stride))

    def forward(self, x):
        output_groups = [self.layers[i](x) for i in range(self.groups)]
        return torch.cat(output_groups, dim=1)


class MultiGaborModule(nn.Module):
    def __init__(self, in_channels, out_channels, scales, kernel_size, stride=1, ratio=2, dw_size=3, relu=True):
        super(MultiGaborModule, self).__init__()
        self.stride = stride
        self.gate_fn = nn.Sigmoid()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        self.MGaborConv = MultiGaborConv(in_channels, init_channels, scales, kernel_size, self.stride)
        self.primary_conv = nn.Sequential(
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x = self.MGaborConv(x)
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class MGaborNeck(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, scales, kernel_size, dw_kernel_size=3,
                 stride=1, se_ratio=0., tea_id=None):
        super(MGaborNeck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        has_tea = tea_id is not None and tea_id > 0.
        self.scales = scales
        self.stride = stride
        # Point-wise expansion
        self.ghost1 = MultiGaborModule(in_chs, mid_chs, self.scales, kernel_size)
        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        if has_tea:
            self.tea = TEAttention(mid_chs, mid_chs)
        else:
            self.tea = None
        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        self.ghost2 = MultiGaborModule(mid_chs, out_chs, self.scales, kernel_size)

        # shortcut
        self.shortcut = nn.Sequential()
        if in_chs != out_chs or self.stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        if self.tea is not None:
            x = self.tea(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class MGaborNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.5, dropout=0.2, round_nearest=8):
        super(MGaborNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        # 416,416,3->208,208,16
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = MGaborNeck
        for cfg in self.cfgs:
            layers = []
            # print("-------------------------")
            for dk, ksize, exp_size, c_out, scales, se_ratio, stride, tea_id in cfg:
                # print("=========")
                output_channel = _make_divisible(c_out * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                scales = _make_divisible(scales * width, 2)
                layers.append(block(input_channel, hidden_channel, output_channel, scales, kernel_size=ksize,
                                    dw_kernel_size=dk, stride=stride, se_ratio=se_ratio, tea_id=tea_id))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))
        # 卷积标准化+激活函数
        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        # 根据构建好的block序列模型
        self.blocks = nn.Sequential(*stages)

        output_channel = 1280
        self.output_channel = _make_divisible(output_channel * max(1.0, 0), round_nearest)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(self.output_channel, num_classes)

    def forward(self, x):
        # print("x_input", x.shape)
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


def LGTRNetv1(**kwargs):

    cfgs = [
        # dw_kernel_size, kernel_size, exp_size, c_out, scales, se_ratio, stride, layer_id

        # stage1
        # 208,208,16 -> 208,208,16
        [[3, 5, 16, 16, 4, 0, 1, 1]],
        # stage2
        # 208,208,16 -> 104,104,24-
        [[3, 5, 48, 24, 4, 0, 2, 1]],
        [[3, 5, 72, 24, 4, 0, 1, 0]],
        # stage3(有效特征层)
        # 104,104,24 -> 52,52,40
        [[5, 5, 72, 48, 4, 0, 2, 1]],
        [[5, 5, 128, 48, 8, 0.25, 1, 0]],  # 40
        # stage4(有效特征层)
        # 52,52,40->26,26,80->26,26,112
        [[3, 3, 240, 80, 8, 0, 2, 0]],
        [[3, 3, 224, 80, 8, 0, 1, 0],
         [3, 3, 192, 80, 8, 0, 1, 0],
         [3, 3, 480, 112, 8, 0, 1, 0],
         [3, 3, 672, 112, 8, 0.25, 1, 0]
         ],  # 112
        # stage5(有效特征层)
        # 26,26,112 -> 13,13,160
        [[3, 3, 672, 160, 16, 0, 2, 0]],
        [[3, 3, 960, 160, 16, 0, 1, 0],
         [3, 3, 960, 192, 32, 0, 1, 0],
         [3, 3, 960, 192, 32, 0.25, 1, 0]
         ]
    ]

    return MGaborNet(cfgs, **kwargs)


if __name__ == "__main__":
    from torchsummary import summary
    from thop import clever_format, profile
    ####
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LGTRNetv1().to(device)
    summary(model, input_size=(3, 224, 224))
    input_shape = [224, 224]
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model, (dummy_input, ), verbose=False)
    flops           = flops
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))



def Lgtr_Netv1_15(pretrained=False, progress=True, num_classes=1000):
    model = LGTRNetv1()

    if num_classes != 1000:
        model.classifier = nn.Linear(model.output_channel, num_classes)
    return model

