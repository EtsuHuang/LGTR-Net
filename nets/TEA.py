import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        S_flat = S.view(batch, 1, -1)  # Shape: (batch_size, height * width, 1)
        # print("S_flat", S_flat.shape)

        max_val = torch.max(S_flat, dim=2)[0]
        # print("max_val", max_val.shape)
        min_val = torch.min(S_flat, dim=2)[0]
        # print("min_val", min_val.shape)

        V = torch.zeros(batch, wth*hth, self.M).to(S_flat.device)  # V-Shape: (batch_size, wth*hth//4, num_bins)
        # E_hist = torch.zeros(batch, self.M)
        C_hist = torch.zeros(batch, self.M, 2)  # Shape: (batch_size, num_bins, 2)
        for i in range(batch):
            Level = torch.linspace(float(min_val[i]), float(max_val[i]), self.M).to(S_flat.device)
            # print("Level", Level.shape)
            for j in range(self.M):
                for k in range(wth*hth):
                    diff = torch.abs(Level[j] - S_flat[i, :, k])
                    if diff < (0.5 / self.M):
                        V[i, k, j] = 1 - diff
                    else:
                        V[i, k, j] = 0
                C_hist[i, j, 0] = V[i, :, j].sum() / V[i, :, :].sum(dim=(0, 1))
                C_hist[i, j, 1] = Level[j]
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

if __name__ == "__main__":
    from torchsummary import summary
    from thop import clever_format, profile
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TEAttention(inp=40, oup=40).to(device)
    summary(model, input_size=(40, 64, 64))
    input_shape = [64, 64]
    dummy_input     = torch.randn(1, 40, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model, (dummy_input, ), verbose=False)
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
