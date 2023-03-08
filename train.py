import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from stage1GAN import TextConGeneratorI, TextAwareDiscriminatorI
from stage2GAN import TextConGeneratorII, TextAwareDiscriminatorII
from utils import batch_size, train_dl_1, train_dl_2, textEmbedder

torch.autograd.set_detect_anomaly(True)

device = "cpu"

lr = 3e-4
c_dim = 128
z_dim = 100
Nd = 128
num_epochs = 50

disc_1 = TextAwareDiscriminatorI(Nd).to(device)
gen_1 = TextConGeneratorI(c_dim, z_dim).to(device)

disc_2 = TextAwareDiscriminatorII(Nd).to(device)
gen_2 = TextConGeneratorII(c_dim).to(device)

opt_disc_1 = optim.Adam(disc_1.parameters(), lr=lr)
opt_gen_1 = optim.Adam(gen_1.parameters(), lr=lr)

opt_text = optim.Adam(textEmbedder.parameters(), lr=lr)

opt_disc_2 = optim.Adam(disc_2.parameters(), lr=lr)
opt_gen_2 = optim.Adam(gen_2.parameters(), lr=lr)

criterion = nn.BCELoss()

writer = SummaryWriter("runs/ImageGen/COCO")


def train_1(models, optimizers, criterion, loader, num_epochs, device=device):
    disc_1, gen_1 = models
    opt_disc_1, opt_gen_1, opt_text = optimizers
    step = 0  # Tensorboard Global Step

    for epoch in range(num_epochs):
        for idx, (desc_tokens, real_img) in enumerate(loader):
            real_img = real_img.to(device)
            desc_tokens = desc_tokens.to(device)

            noise = torch.randn(batch_size, z_dim).to(device)
            fake, mu1, sigma1 = gen_1(desc_tokens, noise)
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
            kl_div = torch.sum(
                1 + torch.log(sigma1.pow(2)) - mu1.pow(2) - sigma1.pow(2)
            )
            lossG = lossG_fake + kl_div
            opt_gen_1.zero_grad()
            lossG.backward()
            opt_gen_1.step()

            opt_text.step()
            opt_text.zero_grad()  # 4 am thoughts

            writer.add_scalar("Discriminator 1 loss", lossD, global_step=step)
            writer.add_scalar("Generator 1 loss", lossG, global_step=step)
            step += 1

            print(
                f"batch:{idx}, epoch:{epoch}, disc1Loss:{lossD.item()}, gen1Loss:{lossG.item()}"
            )


def train_2(models, optimizers, criterion, loader, num_epochs, device=device):
    gen_1, disc_2, gen_2 = models
    opt_disc_2, opt_gen_2 = optimizers
    step = 0  # Tensorboard Global Step

    for epoch in range(num_epochs):
        for idx, (desc_tokens, real_img_256) in enumerate(loader):
            real_img_256 = real_img_256.to(device)
            desc_tokens = desc_tokens.to(device)

            noise = torch.randn(batch_size, z_dim).to(device)
            fake_64 = gen_1(desc_tokens, noise)[0]
            fake_256, mu2, sigma2 = gen_2(desc_tokens, fake_64)
            disc_real = disc_2(real_img_256, desc_tokens).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc_1(fake_256, desc_tokens).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = lossD_real + lossD_fake
            opt_disc_2.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc_2.step()

            output = disc_2(fake_256, desc_tokens).view(-1)
            lossG_fake = criterion(output, torch.ones_like(output))
            kl_div = torch.sum(
                1 + torch.log(sigma2.pow(2)) - mu2.pow(2) - sigma2.pow(2)
            )
            lossG = lossG_fake + kl_div
            opt_gen_2.zero_grad()
            lossG.backward()
            opt_gen_2.step()

            writer.add_scalar("Discriminator 2 loss", lossD, global_step=step)
            writer.add_scalar("Generator 2 loss", lossG, global_step=step)
            step += 1

            print(
                f"batch:{idx}, epoch:{epoch}, disc2Loss:{lossD.item()}, gen2Loss:{lossG.item()}"
            )


train_1(
    models=[disc_1, gen_1],
    optimizers=[opt_disc_1, opt_gen_1, opt_text],
    criterion=criterion,
    loader=train_dl_1,
    num_epochs=num_epochs,
)
