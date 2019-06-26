import os

import torch
import torchvision

from tensorboardX import SummaryWriter


class TensorboardLogger(SummaryWriter):
    def __init__(self, log_path):
        super().__init__(log_path)

    def log_hist(self, net, global_step):
        for n, p in net.named_parameters():
            n = n.replace(".", "/")
            n = "%s.%s" % (net._get_name(), n)
            self.add_histogram(n, p.detach().cpu().clone().numpy(), global_step)

    def log_scalar(self, tag, scalar, global_step):
        # single scalar
        if isinstance(scalar, (int, float)):
            self.add_scalar(tag, scalar, global_step)
        # scalar group
        elif isinstance(scalar, dict):
            self.add_scalars(tag, scalar, global_step)

    def log_image(self, fake_x, x, target, global_step):
        img = torch.cat([fake_x.cpu(), x[:, 0:1, :, :], target, x[:, 1:2, :, :]], 3)
        # img.clamp_(0, 1)
        img = torchvision.utils.make_grid(img, nrow=1)
        self.add_image("fake | x0 | target(x1) | x2", img, global_step)
