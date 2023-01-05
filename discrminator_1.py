import torch
import torch.nn as nn
import torch.nn.functional as F


class StageIDiscriminator(nn.Module):
    def __init__(self, tem_size, Nd):
        super(StageIDiscriminator, self).__init__()
        self.down_sampler = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.1),
            self.downsampling_block(64, 64 * 2, 4, 2, 1),
            self.downsampling_block(64 * 2, 64 * 4, 4, 2, 1),
            self.downsampling_block(64 * 4, 64 * 8, 4, 2, 1),
        )
        self.compress = nn.Linear(tem_size, Nd)

        # 1x1 conv for channel adjustment
        self.channel_resize = nn.Conv2d(64 * 8 + Nd, (64 * 8 + Nd) / 2, 1, 1, 0)

        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(((64 * 8 + Nd) / 2) * 4 * 4, 1)

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

    def forward(self, img, tem):
        x = self.down_sampler(img)
        compressed_em = self.compress(tem)
        em_to_fm = compressed_em.resize(
            compressed_em.shape[0], compressed_em.shape[1], 1, 1
        )
        replicated_fm = em_to_fm.repeat(1, 1, 4, 4)
        concatenated_fm = torch.cat((x, replicated_fm), dim=1)
        text_img_fm = self.channel_resize(concatenated_fm)
        flattened_vec = self.flatten(text_img_fm)
        score = self.classifier(flattened_vec)
        output = F.sigmoid(score)
        return output
