import torch
from torch import nn

class StageIGenerator(nn.Module):
  def __init__(self, con_a_size, latent_size):
    super(StageIGenerator, self).__init__();

    self.upsampling = nn.Sequential(
      nn.ConvTranspose2d(con_a_size+latent_size, 196, 4, 1, 0),
      nn.ReLU(),
      nn.ConvTranspose2d(196, 128, 4, 2, 1),
      nn.ReLU(),
      nn.ConvTranspose2d(128, 64, 4, 2, 1),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 24, 4, 2, 1),
      nn.ReLU(),
      nn.ConvTranspose2d(24, 12, 4, 2, 1),
      nn.ReLU(),
      nn.Conv2d(12, 3, 1)
    )

  def forward(self, x):
    x = x.reshape(x.shape[0], x.shape[1], 1, 1)
    return self.upsampling(x)


if __name__ == "__main__":
  generator = StageIGenerator(128, 100)
  print(generator(torch.randn(8, 228)).shape)
