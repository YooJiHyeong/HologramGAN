import torch

from Loader import Loader
from Generator import Generator
from Discriminator import Discriminator

if __name__ == "__main__":
    loader = Loader()
    G = Generator()
    D = Discriminator()