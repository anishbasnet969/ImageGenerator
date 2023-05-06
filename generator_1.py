import torch
import torch.nn as nn


class StageIGenerator(nn.Module):
    def __init__(self, c_dim, z_dim):
        super(StageIGenerator, self).__init__()

        self.upsampling = nn.Sequential(
            self.upsampling_block(
                c_dim + z_dim,
                192,
                4,
                1,
                0,
            ),
            self.upsampling_block(192, 96, 4, 2, 1),
            self.upsampling_block(96, 48, 4, 2, 1),
            self.upsampling_block(48, 24, 4, 2, 1),
            nn.ConvTranspose2d(24, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def upsampling_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        return self.upsampling(x)
