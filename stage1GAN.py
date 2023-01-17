import torch
import torch.nn as nn

from textEmbed import TextEmbeddingLSTM
from custom_dataloader import *
from discrminator_1 import StageIDiscriminator
from con_augment import ConditioningAugmentation
from generator_1 import StageIGenerator
from preprocess_glove import embedding_layer


EMBEDDING_SIZE = 300
TEXT_EMBEDDING_HIDDEN_SIZE = 300
TEXT_EMBEDDING_NUM_LAYERS = 2
TEM_SIZE = 400

textEmbedder = TextEmbeddingLSTM(
    embedding_layer,
    EMBEDDING_SIZE,
    TEXT_EMBEDDING_HIDDEN_SIZE,
    TEXT_EMBEDDING_NUM_LAYERS,
    TEM_SIZE,
)


class CAGenI(nn.Module):
    def __init__(self, c_dim, z_dim):
        super(CAGenI, self).__init__()
        self.con_augment = ConditioningAugmentation(TEM_SIZE, 300, 200, c_dim)
        self.gen = StageIGenerator(c_dim, z_dim)

    def forward(self, tem, noise):
        c_hat, mu, sigma = self.con_augment(tem)
        print(c_hat.shape)
        lc = torch.cat((noise, c_hat), dim=1)
        img_output = self.gen(lc)
        return img_output, mu, sigma


class TextConGeneratorI(nn.Module):
    def __init__(self, c_dim, z_dim):
        super(TextConGeneratorI, self).__init__()
        self.gen = CAGenI(c_dim, z_dim)

    def forward(self, desc_tokens, noise):
        tem = textEmbedder(desc_tokens)
        img_output, mu, sigma = self.gen(tem, noise)
        return img_output, mu, sigma


class TextAwareDiscriminatorI(nn.Module):
    def __init__(self, Nd):
        super(TextAwareDiscriminatorI, self).__init__()
        self.disc = StageIDiscriminator(TEM_SIZE, Nd)

    def forward(self, img, desc_tokens):
        tem = textEmbedder(desc_tokens)
        output = self.disc(img, tem)
        return output
