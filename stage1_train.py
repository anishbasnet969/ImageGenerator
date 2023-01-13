import torch
import torch.nn as nn
import torch.optim as optim

from stage1GAN import TextConGeneratorI, TextAwareDiscriminatorI, textEmbedder
from data_loader import loader, batch_size

torch.autograd.set_detect_anomaly(True)

device = "cpu"

lr = 3e-4
c_dim = 128
z_dim = 100
Nd = 128
num_epochs = 50

disc_1 = TextAwareDiscriminatorI(Nd).to(device)
gen_1 = TextConGeneratorI(c_dim, z_dim).to(device)

opt_disc_1 = optim.Adam(disc_1.parameters(), lr=lr)
opt_gen_1 = optim.Adam(gen_1.parameters(), lr=lr)
opt_text = optim.Adam(textEmbedder.parameters(), lr=lr)
criterion = nn.BCELoss()


for epoch in range(num_epochs):
    for idx, (desc_tokens, real_img) in enumerate(loader):
        real_img = real_img.to(device)
        desc_tokens = desc_tokens.to(device)

        noise = torch.randn(batch_size, z_dim).to(device)
        fake, mu, sigma = gen_1(desc_tokens, noise)
        disc_real = disc_1(real_img, desc_tokens).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc_1(fake, desc_tokens).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = lossD_real + lossD_fake
        opt_disc_1.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc_1.step()

        output = disc_1(fake, desc_tokens).view(-1)
        lossG_fake = criterion(output, torch.ones_like(output))
        kl_div = torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        lossG = lossG_fake + kl_div
        opt_gen_1.zero_grad()
        lossG.backward()
        opt_gen_1.step()

        opt_text.step()
        opt_text.zero_grad()  # 4 am thoughts

        print(
            f"batch:{idx}, epoch:{epoch}, discLoss:{lossD.item()}, genLoss:{lossG.item()}"
        )
