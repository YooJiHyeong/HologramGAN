import torch
import torch.nn as nn
import torch.nn.functional as F

from wavelet_modules import WaveUnpool, WavePool
from preset import block_ch


class DownConvReluBN(nn.Module):
    def __init__(self, conv, pool, norm=None, act=None):
        super().__init__()
        self.conv = conv
        self.pool = pool
        self.norm = norm
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        LL, LH, HL, HH = self.pool(out)

        if self.act is not None:
            LL = self.act(LL)

        if self.norm is not None:
            LL = self.norm(LL)

        return LL, LH, HL, HH, out


class UpConvReluBN(nn.Module):
    def __init__(self, pool, conv, norm=None, act=None):
        super().__init__()
        self.pool = pool
        self.conv = conv
        self.norm = norm
        self.act = act

    def forward(self, LL, LH, HL, HH, original=None):
        if self.act is not None:
            LL = self.act(LL)

        out = self.pool(LL, LH, HL, HH, original)
        out = self.conv(out)

        if self.norm is not None:
            out = self.norm(out)

        return out


class GeneratorWavpool(nn.Module):
    def __init__(self, img_ch=2, ch_grow=4):
        super().__init__()

        start_ch = block_ch[0][0]
        self.from_rgb = DownConvReluBN(
            nn.Conv2d(img_ch, start_ch, 3, 1, 1),
            WavePool(start_ch),
            nn.BatchNorm2d(start_ch),
            nn.LeakyReLU(0.2, True)
        )

        self.to_rgb = UpConvReluBN(
            WaveUnpool(start_ch),
            nn.Conv2d(start_ch * 5, img_ch // 2, 3, 1, 1),
            nn.Tanh()
        )

        self.encoder = []
        for in_ch, out_ch in block_ch:
            self.encoder.append(DownConvReluBN(
                                nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                WavePool(out_ch),
                                nn.BatchNorm2d(out_ch),
                                nn.ReLU(True)))

        self.encoder = nn.Sequential(*self.encoder)

        self.decoder = []
        for (out_ch, in_ch) in block_ch[::-1]:
            self.decoder.append(UpConvReluBN(
                WaveUnpool(in_ch),
                nn.Conv2d(in_ch * 5, out_ch, 3, 1, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True)
            ))
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        LL, LH, HL, HH, original = self.from_rgb(x)
        features = [(LH, HL, HH, original)]

        for i, e in enumerate(self.encoder):
            LL, LH, HL, HH, original = e(LL)
            features.append((LH, HL, HH, original))

        for i, d in enumerate(self.decoder):
            LL = d(LL, *features[-i - 1])

        x = self.to_rgb(LL, *features[0])
        return x


if __name__ == "__main__":
    G = GeneratorWavpool()
    print(G)
    img = torch.randn(4, 2, 512, 512)
    print(G(img).shape)
