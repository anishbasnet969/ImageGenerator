import torch
import torch.nn as nn

from stage1_train import TEM_SIZE

from con_augment import ConditioningAugmentation
from generator_1 import StageIGenerator


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
