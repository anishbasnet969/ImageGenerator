import torch
import torch.nn as nn

from textEmbed import TextEmbeddingLSTM
from preprocessing import *
from discrminator_1 import StageIDiscriminator
from con_augment import ConditioningAugmentation
from generator_1 import StageIGenerator


EMBEDDING_SIZE = 300
TEXT_EMBEDDING_HIDDEN_SIZE = 300
VOCAB_SIZE = len(dataset.vocab)
TEXT_EMBEDDING_NUM_LAYERS = 2
TEM_SIZE = 400

textEmbedder = TextEmbeddingLSTM(
    EMBEDDING_SIZE,
    TEXT_EMBEDDING_HIDDEN_SIZE,
    VOCAB_SIZE,
    TEXT_EMBEDDING_NUM_LAYERS,
    TEM_SIZE,
)


class CAGenI(nn.Module):
    def __init__(self, c_dim):
        super(CAGenI, self).__init__()
        self.con_augment = ConditioningAugmentation(TEM_SIZE, 300, 200, c_dim)
        self.gen = StageIGenerator(c_dim, 100)

    def forward(self, tem, noise):
        c_hat, mu, sigma = self.con_augment(tem)
        lc = torch.cat((c_hat, noise), dim=1)
        img_output = self.gen(lc)
        return img_output, mu, sigma


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
