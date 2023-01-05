import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditioningAugmentation(nn.Module):
    def __init__(self, tem_size, h1_dim, h2_dim, c_dim):
        super(ConditioningAugmentation, self).__init__()
        self.h1 = nn.Linear(tem_size, h1_dim)
        self.h2 = nn.Linear(h1_dim, h2_dim)
        self.mu = nn.Linear(h2_dim, c_dim)
        self.sigma = nn.Linear(h2_dim, c_dim)

    def encode(self, tem):
        h1 = F.relu(self.h1(tem))
        h2 = F.relu(self.h2(h1))
        mu, sigma = self.mu(h2), self.sigma(h2)
        return mu, sigma

    def forward(self, tem):
        mu, sigma = self.encode(tem)
        epsilon = torch.randn_like(sigma)
        c_hat = mu + sigma * epsilon
        return c_hat, mu, sigma


if __name__ == "__main__":
    pass