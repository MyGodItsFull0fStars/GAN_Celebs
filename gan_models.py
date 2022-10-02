from turtle import forward
from torch import Tensor
import torch.nn as nn

from utils import device


def weights_init(model: nn.Module):
    classname = model.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


class Generator(nn.Module):

    def __init__(
        self,
        latent_dim_z: int = 100,
        num_gen_feat: int = 64,
        num_channels: int = 3
    ) -> None:

        super(Generator, self).__init__()

        self.latent_dim_z: int = latent_dim_z
        self.num_gen_feat: int = num_gen_feat
        self.num_channels: int = num_channels

        self.main_model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.latent_dim_z, self.num_gen_feat * \
                               8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.num_gen_feat * 8),
            nn.ReLU(inplace=True),
            # state size (self.num_gen_feat * 8) x 4 x 4
            nn.ConvTranspose2d(
                self.num_gen_feat * 8, self.num_gen_feat * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_gen_feat * 4),
            nn.ReLU(inplace=True),
            # state size (self.num_gen_feat * 4) x 8 x 8
            nn.ConvTranspose2d(
                self.num_gen_feat * 4, self.num_gen_feat * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_gen_feat * 2),
            nn.ReLU(inplace=True),
            # state size (self.num_gen_feat * 2) x 16 x 16
            nn.ConvTranspose2d(
                self.num_gen_feat * 2, self.num_gen_feat, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_gen_feat),
            nn.ReLU(inplace=True),
            # state size (self.num_gen_feat) x 32 x 32
            nn.ConvTranspose2d(self.num_gen_feat, self.num_channels,
                               4, 2, 1, bias=False),
            nn.Tanh()
            # state size (num_channels) x 64 x 64
        )

        self.apply(weights_init)
        
        self.to()

    def forward(self, input: Tensor) -> Tensor:
        return self.main_model(input)


class Discriminator(nn.Module):

    def __init__(
        self,
        num_dis_feat: int = 64,
        num_channels: int = 3,
    ) -> None:
        super(Discriminator, self).__init__()

        self.num_dis_feat: int = num_dis_feat
        self.num_channels: int = num_channels

        self.main_model = nn.Sequential(
            # input is (num_channels) x 64 x 64
            nn.Conv2d(self.num_channels, num_dis_feat, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (num_dis_feat) x 32 x 32
            nn.Conv2d(num_dis_feat, num_dis_feat * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_dis_feat * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (num_dis_feat * 2) x 16 x 16
            nn.Conv2d(num_dis_feat * 2, num_dis_feat * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_dis_feat * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (num_dis_feat * 4) x 8 x 8
            nn.Conv2d(num_dis_feat * 4, num_dis_feat * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_dis_feat * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (num_dis_feat * 8) x 4 x 4
            nn.Conv2d(num_dis_feat * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        self.apply(weights_init)

    def forward(self, input: Tensor) -> Tensor:
        return self.main_model(input)


if __name__ == '__main__':
    gen = Generator()
    dis = Discriminator()
