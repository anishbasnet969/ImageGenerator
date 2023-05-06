import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditioningAugmentation(nn.Module):
    def __init__(self, tem_size, h_dim, c_dim):
        super(ConditioningAugmentation, self).__init__()
        self.h = nn.Linear(tem_size, h_dim)
        self.mu = nn.Linear(h_dim, c_dim)
        self.sigma = nn.Linear(h_dim, c_dim)

    def encode(self, tem):
        h = F.relu(self.h(tem))
        mu, sigma = self.mu(h), self.sigma(h)
        return mu, sigma

    def forward(self, tem):
        mu, sigma = self.encode(tem)
        epsilon = torch.randn_like(sigma)
        c_hat = mu + sigma * epsilon
        return c_hat, mu, sigma


if __name__ == "__main__":
    pass
