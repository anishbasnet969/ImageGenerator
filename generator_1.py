import torch
import torch.nn as nn


class StageIGenerator(nn.Module):
    def __init__(self, con_a_size, latent_size):
        super(StageIGenerator, self).__init__()

        self.upsampling = nn.Sequential(
            self.upsampling_block(
                con_a_size + latent_size,
                196,
                4,
                1,
                0,
            ),
            self.upsampling_block(196, 128, 4, 2, 1),
            self.upsampling_block(128, 64, 4, 2, 1),
            self.upsampling_block(64, 24, 4, 2, 1),
            self.upsampling_block(24, 12, 4, 2, 1),
            nn.Conv2d(12, 3, 1),
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


if __name__ == "__main__":
    generator = StageIGenerator(128, 100)
    print(generator(torch.randn(8, 228)).shape)
