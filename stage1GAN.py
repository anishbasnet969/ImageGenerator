import torch
import torch.nn as nn

from stage1_train import textEmbedder, TEM_SIZE
from discrminator_1 import StageIDiscriminator
from ca_gen import CAGenI


class TextConGenerator(nn.Module):
    def __init__(self, c_dim):
        super(TextConGenerator, self).__init__()
        self.gen = CAGenI(c_dim)

    def forward(self, desc_tokens, noise):
        tem = textEmbedder(desc_tokens)
        img_output, mu, sigma = self.gen(tem, noise)
        return img_output, mu, sigma


class TextAwareDiscriminator(nn.Module):
    def __init__(self, Nd):
        super(TextAwareDiscriminator, self).__init__()
        self.disc = StageIDiscriminator(TEM_SIZE, Nd)

    def forward(self, img, desc_tokens):
        tem = textEmbedder(desc_tokens)
        output = self.disc(img, tem)
        return output
