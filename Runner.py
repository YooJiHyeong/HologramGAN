import torch
import torch.nn as nn


class Runner():
    def __init__(self, arg, total_step, G, D, train_loader, test_loader, device, tensorboard, lambda_L1=100):
        self.arg = arg
        self.total_step = total_step

        self.G = G
        self.D = D

        self.G_optim = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.D_optim = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.GAN_loss = nn.MSELoss()
        self.L1_loss = nn.L1Loss()
        self.lambda_L1 = lambda_L1

        self.device = device
        self.tensorboard = tensorboard

        self.global_step = 0

    def train(self):
        def _get_target(x, target):
            out = self.D.forward(torch.cat([x, target], 1))
            self.target_true  = torch.tensor(1.0).expand_as(out).to(self.device["target"])
            # print("#####", self.target_true.shape)
            self.target_false = torch.tensor(0.0).expand_as(out).to(self.device["target"])

        x, target, path = next(self.train_loader)
        _get_target(x, target)

        for s in range(self.total_step):
            for i, (x, target, path) in enumerate(self.train_loader):
                self.global_step += 1

                target = target.to(self.device["target"])
                # print("#####", target.shape)
                fake_x = self.train_G(x, target)
                self.train_D(x, target)

                if self.global_step % 100 == 0:
                    print("[%5d / %5d]" % (self.global_step, self.total_step))
                    self.test()
                    
    def train_G(self, x, target):
        fake_x = self.G.forward(x)
        with torch.no_grad():
            fake_y = self.D.forward(torch.cat([x, fake_x.cpu()], 1))

        G_loss = self.GAN_loss(fake_y, self.target_true)
        G_L1_loss = self.L1_loss(fake_x, target) * self.lambda_L1

        G_total_loss = G_loss + G_L1_loss
        # print(1, G_loss.device, G_loss.shape, G_L1_loss.device, G_L1_loss.shape, G_total_loss.device, G_total_loss.shape)

        self.G_optim.zero_grad()
        G_total_loss.backward()
        self.G_optim.step()

        return fake_x

    def train_D(self, x, target, d_iter=1):
        for _ in range(d_iter):
            with torch.no_grad():
                fake_x = self.G.forward(x)
            fake_y = self.D.forward(torch.cat([x, fake_x.cpu()], 1))
            fake_loss = self.GAN_loss(fake_y, self.target_false)

            real_y = self.D.forward(torch.cat([x, target.cpu()], 1))
            real_loss = self.GAN_loss(real_y, self.target_true)

            D_total_loss = fake_loss + real_loss
            # print(2, D_total_loss.device, D_total_loss.shape, fake_loss.device, fake_loss.shape, real_loss.device, real_loss.shape)
            # print(3, fake_x.shape, fake_y.shape)

            self.D_optim.zero_grad()
            D_total_loss.backward()
            self.D_optim.step()

    def test(self):
        with torch.no_grad():
            x, target, path = next(self.test_loader)
            fake_x = self.G.forward(x)
            self.tensorboard.log_image(fake_x, x, target, self.global_step)


if __name__ == "__main__":
    G = Generator()
    D = Discriminator()

    x = torch.randn(4, 1, 512, 512)
    x_2 = torch.randn(4, 1, 512, 512)

    g_out = G(x)
    d_out = D(torch.cat([g_out, x_2], 1))

    print(g_out.shape, d_out.shape)