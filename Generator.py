import torch
import torch.nn as nn

from preset import block_ch


class Generator(nn.Module):
    def __init__(self, img_ch=2, ch_grow=4):
        super().__init__()

        start_ch = block_ch[0][0]
        self.from_rgb = nn.Sequential(
            nn.Conv2d(img_ch, start_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(start_ch)
        )

        self.to_rgb = nn.Sequential(
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(start_ch, img_ch // 2, 3, 1, 1),
            # nn.ConvTranspose2d(start_ch, img_ch // 2, 4, 2, 1),
        )

        self.encoder = []
        for in_ch, out_ch in block_ch:
            self.encoder.append(nn.Sequential(
                                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                                nn.LeakyReLU(0.2, True),
                                nn.BatchNorm2d(out_ch)))

        self.encoder = nn.Sequential(*self.encoder)

        self.decoder = []
        for i, (out_ch, in_ch) in enumerate(block_ch[::-1]):
            if i > 0:
                in_ch *= 2
            self.decoder.append(nn.Sequential(
                                nn.ReLU(True),
                                nn.Upsample(scale_factor=2),
                                nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                # nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                                nn.BatchNorm2d(out_ch)))
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        x = self.from_rgb(x)
        features = [x]

        for e in self.encoder:
            x = e(x)
            features.append(x)

        for i, d in enumerate(self.decoder):
            if i == 0:
                x = d(x)
            else:
                x = d(torch.cat([x, features[-i - 1]], 1))

        x = self.to_rgb(x)

        return x


if __name__ == "__main__":
    G = Generator()
    img = torch.randn(4, 1, 512, 512)
    print(G(img).shape)