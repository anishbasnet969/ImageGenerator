import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels):
        super(ResidualBlock, self).__init__()
        self.layer1 = self._layer_convBN(in_channels, intermediate_channels, 3, 1, 1)
        self.layer2 = self._layer_convBN(
            intermediate_channels, intermediate_channels, 3, 1, 1
        )
        self.layer3 = self._layer_convBN(intermediate_channels, in_channels, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)

        x += identity
        x = self.relu(x)
        return x

    def _layer_convBN(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )


class StageIIGenerator(nn.Module):
    def __init__(self):
        super(StageIIGenerator, self).__init__()
        self.down_sampler = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1),
            nn.LeakyReLU(0.1),
            self.downsampling_block(128, 512, 4, 2, 1),
        )
        self.residual_blocks = self._get_residual_blocks(640, 320)
        self.up_sampler = nn.Sequential(
            self.upsampling_block(640, 320, 4, 2, 1),
            self.upsampling_block(320, 160, 4, 2, 1),
            self.upsampling_block(160, 80, 4, 2, 1),
            nn.ConvTranspose2d(80, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, img_64, c_hat):
        x = self.down_sampler(img_64)
        c_hat = c_hat.reshape(c_hat.shape[0], c_hat.shape[1], 1, 1)
        replicated_fm = c_hat.repeat(1, 1, 16, 16)
        concatenated_fm = torch.cat((x, replicated_fm), dim=1)
        text_img_fm = self.residual_blocks(concatenated_fm)
        img_256 = self.up_sampler(text_img_fm)

        return img_256

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

    def downsampling_block(
        self, in_channels, out_channels, kernel_size, stride, padding
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def _get_residual_blocks(self, in_channels, intermediate_channels):
        return nn.Sequential(
            ResidualBlock(in_channels, intermediate_channels),
            ResidualBlock(in_channels, intermediate_channels),
            ResidualBlock(in_channels, intermediate_channels),
            ResidualBlock(in_channels, intermediate_channels),
        )
