import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_ch=1, start_ch=64, kernel_size=4, padding=1, block_num=4):
        super().__init__()

        self.layers = [nn.Conv2d(img_ch * 2, start_ch,
                                 kernel_size=kernel_size, stride=2, padding=padding),
                       nn.LeakyReLU(0.2, True)]

        in_ch, out_ch = start_ch, start_ch * 2
        for i in range(block_num - 1):
            if i == block_num - 2:
                stride = 1
            else:
                stride = 2
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, True)
            ))
            in_ch, out_ch = out_ch, out_ch * 2

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.layers(x)
        return out


if __name__ == "__main__":
    D = Discriminator()
    img = torch.randn(4, 2, 256, 256)
    print(D(img).shape)