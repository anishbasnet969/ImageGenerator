import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.tensorboard import SummaryWriter
from stage1GAN import TextConGeneratorI, TextAwareDiscriminatorI
from stage2GAN import TextConGeneratorII, TextAwareDiscriminatorII
from utils import batch_size, train_dl_1, train_dl_2, textEmbedder, gradient_penalty

torch.autograd.set_detect_anomaly(True)

device = "cpu"

lr = 1e-4
c_dim = 128
z_dim = 100
Nd = 128
num_epochs = 50
n_critic = 5
lambda_gp = 10

critic_1 = TextAwareDiscriminatorI(Nd).to(device)
gen_1 = TextConGeneratorI(c_dim, z_dim).to(device)

critic_2 = TextAwareDiscriminatorII(Nd).to(device)
gen_2 = TextConGeneratorII(c_dim).to(device)

opt_critic_1 = optim.Adam(critic_1.parameters(), lr=lr)
opt_gen_1 = optim.Adam(gen_1.parameters(), lr=lr)

opt_text = optim.Adam(textEmbedder.parameters(), lr=lr)

opt_critic_2 = optim.Adam(critic_2.parameters(), lr=lr)
opt_gen_2 = optim.Adam(gen_2.parameters(), lr=lr)

fixed_noise = torch.randn(batch_size, z_dim).to(device)
writer = SummaryWriter("runs/ImageGen/COCO")
writer_real = SummaryWriter(f"runs/ImageGen/real")
writer_fake = SummaryWriter(f"runs/ImageGen/fake")


def train_1(models, optimizers, loader, num_epochs, device=device):
    disc_1, gen_1 = models
    opt_disc_1, opt_gen_1, opt_text = optimizers
    step = 0  # Tensorboard Global Step

    for epoch in range(num_epochs):
        for batch_idx, (desc_tokens, real_img) in enumerate(loader):
            real_img = real_img.to(device)
            desc_tokens = desc_tokens.to(device)
            current_batch_size = real_img.shape[0]

            for _ in range(n_critic):
                noise = torch.randn(current_batch_size, z_dim).to(device)
                fake, mu1, sigma1 = gen_1(desc_tokens, noise)
                critic_1_real = critic_1(real_img, desc_tokens).view(-1)
                critic_1_fake = critic_1(fake, desc_tokens).view(-1)
                gp = gradient_penalty(critic_1, real_img, fake, desc_tokens, device)
                loss_critic = -(
                    torch.mean(critic_1_real)
                    - torch.mean(critic_1_fake)
                    + lambda_gp * gp
                )
                opt_critic_1.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic_1.step()

            # noise = torch.randn(batch_size, z_dim).to(device)
            # fake, mu1, sigma1 = gen_1(desc_tokens, noise)
            # disc_real = disc_1(real_img, desc_tokens).view(-1)
            # lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            # disc_fake = disc_1(fake, desc_tokens).view(-1)
            # lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            # lossD = lossD_real + lossD_fake
            # opt_disc_1.zero_grad()
            # lossD.backward(retain_graph=True)
            # opt_disc_1.step()

            output = critic_1(fake, desc_tokens).view(-1)
            lossG_fake = -torch.mean(output)
            kl_div = torch.sum(
                1 + torch.log(sigma1.pow(2)) - mu1.pow(2) - sigma1.pow(2)
            )
            lossG = lossG_fake + kl_div
            opt_gen_1.zero_grad()
            lossG.backward()
            opt_gen_1.step()

            # output = disc_1(fake, desc_tokens).view(-1)
            # lossG_fake = criterion(output, torch.ones_like(output))
            # kl_div = torch.sum(
            #     1 + torch.log(sigma1.pow(2)) - mu1.pow(2) - sigma1.pow(2)
            # )
            # lossG = lossG_fake + kl_div
            # opt_gen_1.zero_grad()
            # lossG.backward()
            # opt_gen_1.step()

            opt_text.step()
            opt_text.zero_grad()  # 4 am thoughts

            if batch_idx > 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = gen_1(desc_tokens, fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(
                        real_img, normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)

                    writer_real.add_image("Real 64*64", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake 64*64", img_grid_fake, global_step=step)

                writer.add_scalar("Critic 1 loss", loss_critic, global_step=step)
                writer.add_scalar("Generator 1 loss", lossG, global_step=step)

                step += 1


def train_2(models, optimizers, loader, num_epochs, device=device):
    gen_1, disc_2, gen_2 = models
    opt_disc_2, opt_gen_2 = optimizers
    step = 0  # Tensorboard Global Step

    for epoch in range(num_epochs):
        for batch_idx, (desc_tokens, real_img_256) in enumerate(loader):
            real_img_256 = real_img_256.to(device)
            desc_tokens = desc_tokens.to(device)
            current_batch_size = real_img_256.shape[0]

            for _ in range(n_critic):
                noise = torch.randn(current_batch_size, z_dim).to(device)
                fake_64 = gen_1(desc_tokens, noise)[0]
                fake_256, mu2, sigma2 = gen_2(desc_tokens, fake_64)
                critic_2_real = critic_1(real_img_256, desc_tokens).view(-1)
                critic_2_fake = critic_1(fake_256, desc_tokens).view(-1)
                gp = gradient_penalty(
                    critic_1, real_img_256, fake_256, desc_tokens, device
                )
                loss_critic = -(
                    torch.mean(critic_2_real)
                    - torch.mean(critic_2_fake)
                    + lambda_gp * gp
                )
                opt_critic_2.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic_2.step()

            # noise = torch.randn(batch_size, z_dim).to(device)
            # fake_64 = gen_1(desc_tokens, noise)[0]
            # fake_256, mu2, sigma2 = gen_2(desc_tokens, fake_64)
            # disc_real = disc_2(real_img_256, desc_tokens).view(-1)
            # lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            # disc_fake = disc_1(fake_256, desc_tokens).view(-1)
            # lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            # lossD = lossD_real + lossD_fake
            # opt_disc_2.zero_grad()
            # lossD.backward(retain_graph=True)
            # opt_disc_2.step()

            output = critic_2(fake_256, desc_tokens).view(-1)
            lossG_fake = -torch.mean(output)
            kl_div = torch.sum(
                1 + torch.log(sigma2.pow(2)) - mu2.pow(2) - sigma2.pow(2)
            )
            lossG = lossG_fake + kl_div
            opt_gen_2.zero_grad()
            lossG.backward()
            opt_gen_2.step()

            # output = disc_2(fake_256, desc_tokens).view(-1)
            # lossG_fake = criterion(output, torch.ones_like(output))
            # kl_div = torch.sum(
            #     1 + torch.log(sigma2.pow(2)) - mu2.pow(2) - sigma2.pow(2)
            # )
            # lossG = lossG_fake + kl_div
            # opt_gen_2.zero_grad()
            # lossG.backward()
            # opt_gen_2.step()

            if batch_idx % 100 == 0 and batch_idx > 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake_64 = gen_1(desc_tokens, fixed_noise)
                    fake = gen_2(desc_tokens, fake_64)
                    img_grid_real = torchvision.utils.make_grid(
                        real_img_256, normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)

                    writer_real.add_image(
                        "Real 256*256", img_grid_real, global_step=step
                    )
                    writer_fake.add_image(
                        "Fake 256*256", img_grid_fake, global_step=step
                    )

                writer.add_scalar("Critic 2 loss", loss_critic, global_step=step)
                writer.add_scalar("Generator 2 loss", lossG, global_step=step)
                step += 1


train_1(
    models=[critic_1, gen_1],
    optimizers=[opt_critic_1, opt_gen_1, opt_text],
    loader=train_dl_1,
    num_epochs=num_epochs,
)
