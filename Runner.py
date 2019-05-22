import torch

from Generator import Generator
from Discriminator import Discriminator

if __name__ == "__main__":
    G = Generator()
    D = Discriminator()

    x = torch.randn(4, 1, 512, 512)
    x_2 = torch.randn(4, 1, 512, 512)

    g_out = G(x)
    d_out = D(torch.cat([g_out, x_2], 1))

    print(g_out.shape, d_out.shape)