import torch
import torch.nn as nn

from stage1GAN import textEmbedder, TEM_SIZE
from discriminator_2 import StageIIDiscriminator
from generator_2 import StageIIGenerator
from con_augment import ConditioningAugmentation


class CAGenII(nn.Module):
    def __init__(self, c_dim):
        super(CAGenII, self).__init__()
        self.con_augment = ConditioningAugmentation(TEM_SIZE, 250, 300, c_dim)
        self.gen = StageIIGenerator(c_dim)

    def forward(self, tem, img_64):
        c_hat, mu, sigma = self.con_augment(tem)
        img_256 = self.gen(img_64, c_hat)

        return img_256, mu, sigma


class TextConGeneratorII(nn.Module):
    def __init__(self, c_dim):
        super(TextConGeneratorII, self).__init__()
        self.gen = CAGenII(c_dim)

    def forward(self, desc_tokens, img_64):
        tem = textEmbedder(desc_tokens)
        img_256, mu, sigma = self.gen(tem, img_64)

        return img_256, mu, sigma


class TextAwareDiscriminatorII(nn.Module):
    def __init__(self, Nd):
        super(TextAwareDiscriminatorII, self).__init__()
        self.disc = StageIIDiscriminator(TEM_SIZE, Nd)

    def forward(self, img_256, desc_tokens):
        tem = textEmbedder(desc_tokens)
        output = self.disc(img_256, tem)
        return output
