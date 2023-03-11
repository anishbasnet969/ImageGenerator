import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.tensorboard import SummaryWriter
from textEmbed import TextEmbeddingLSTM
from con_augment import ConditioningAugmentation
from generator_1 import StageIGenerator
from discrminator_1 import StageIDiscriminator
from generator_2 import StageIIGenerator
from discriminator_2 import StageIIDiscriminator
from utils import batch_size, train_dl_1, train_dl_2, embedding_layer, gradient_penalty

torch.autograd.set_detect_anomaly(True)

device = "cpu"

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

lr = 1e-4
c_dim = 128
z_dim = 100
Nd = 128
num_epochs = 50
n_critic = 5
lambda_gp = 10

con_augment_1 = ConditioningAugmentation(TEM_SIZE, 512, 256, 128).to(device)
critic_1 = StageIDiscriminator(TEM_SIZE, Nd).to(device)
gen_1 = StageIGenerator(c_dim, z_dim).to(device)


con_augment_2 = ConditioningAugmentation(TEM_SIZE, 512, 256, 128).to(device)
critic_2 = StageIIDiscriminator(TEM_SIZE, Nd).to(device)
gen_2 = StageIIGenerator(c_dim).to(device)

opt_text = optim.Adam(textEmbedder.parameters(), lr=lr, betas=(0.9, 0.999))

opt_con_augment_1 = optim.Adam(con_augment_1.parameters(), lr=lr, betas=(0.9, 0.999))
opt_critic_1 = optim.Adam(critic_1.parameters(), lr=lr, betas=(0.9, 0.999))
opt_gen_1 = optim.Adam(gen_1.parameters(), lr=lr, betas=(0.9, 0.999))


opt_con_augment_2 = optim.Adam(con_augment_2.parameters(), lr=lr, betas=(0.9, 0.999))
opt_critic_2 = optim.Adam(critic_2.parameters(), lr=lr, betas=(0.9, 0.999))
opt_gen_2 = optim.Adam(gen_2.parameters(), lr=lr, betas=(0.9, 0.999))

fixed_noise = torch.randn(batch_size, z_dim).to(device)
writer = SummaryWriter("runs/ImageGen/COCO")
writer_real = SummaryWriter(f"runs/ImageGen/real")
writer_fake = SummaryWriter(f"runs/ImageGen/fake")


def train_1(models, optimizers, loader, num_epochs, device=device):
    textEmbedder, con_augment_1, critic_1, gen_1 = models
    opt_text, opt_con_augment_1, opt_critic_1, opt_gen_1 = optimizers
    step = 0  # Tensorboard Global Step

    critic_1.train()
    gen_1.train()

    for epoch in range(num_epochs):
        for batch_idx, (desc_tokens, real_img) in enumerate(loader):
            real_img = real_img.to(device)
            desc_tokens = desc_tokens.to(device)
            current_batch_size = real_img.shape[0]
            mismatched_desc_tokens = desc_tokens[torch.randperm(desc_tokens.shape[0])]

            for _ in range(n_critic):
                tem = textEmbedder(desc_tokens)
                c_hat1, mu1, sigma1 = con_augment_1(tem)
                noise = torch.randn(current_batch_size, z_dim).to(device)
                lc_sum = torch.cat((c_hat1, noise), dim=1)
                fake = gen_1(lc_sum)

                critic_1_real = critic_1(real_img, tem).view(-1)

                tem_mismatched = textEmbedder(mismatched_desc_tokens)
                critic_1_mismatched = critic_1(real_img, tem_mismatched).view(-1)

                critic_1_fake = critic_1(fake, tem).view(-1)

                critic_1_negative_samples = torch.cat(
                    (critic_1_mismatched, critic_1_fake), dim=0
                )

                gp = gradient_penalty(critic_1, real_img, fake, tem, device)

                loss_critic = -(
                    torch.mean(critic_1_real)
                    - torch.mean(critic_1_negative_samples)
                    + lambda_gp * gp
                )
                opt_critic_1.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic_1.step()

            output = critic_1(fake, tem).view(-1)
            lossG_fake = -torch.mean(output)
            kl_div = torch.sum(
                1 + torch.log(sigma1.pow(2)) - mu1.pow(2) - sigma1.pow(2)
            )
            lossG = lossG_fake + kl_div
            opt_gen_1.zero_grad()
            lossG.backward()
            opt_gen_1.step()

            opt_text.step()
            opt_text.zero_grad()  # 4 am thoughts

            opt_con_augment_1.step()
            opt_con_augment_1.zero_grad()

            if batch_idx > 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    tem = textEmbedder(desc_tokens)
                    c_hat, mu, sigma = con_augment_1(tem)
                    lc_sum = torch.cat((c_hat, fixed_noise), dim=1)
                    fake = gen_1(lc_sum)
                    img_grid_real = torchvision.utils.make_grid(
                        real_img, normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(fake[0], normalize=True)

                    writer_real.add_image("Real 64*64", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake 64*64", img_grid_fake, global_step=step)

                writer.add_scalar("Critic 1 loss", loss_critic, global_step=step)
                writer.add_scalar("Generator 1 loss", lossG, global_step=step)

                step += 1


def train_2(models, optimizers, loader, num_epochs, device=device):
    textEmbedder, con_augment_1, con_augment_2, gen_1, critic_2, gen_2 = models
    opt_con_augment_2, opt_critic_2, opt_gen_2 = optimizers
    step = 0  # Tensorboard Global Step

    con_augment_2.train()
    critic_2.train()
    gen_2.train()

    for epoch in range(num_epochs):
        for batch_idx, (desc_tokens, real_img_256) in enumerate(loader):
            real_img_256 = real_img_256.to(device)
            desc_tokens = desc_tokens.to(device)
            current_batch_size = real_img_256.shape[0]
            mismatched_desc_tokens = desc_tokens[torch.randperm(desc_tokens.shape[0])]

            for _ in range(n_critic):
                tem = textEmbedder(desc_tokens)
                c_hat1, mu1, sigma1 = con_augment_1(tem)
                noise = torch.randn(current_batch_size, z_dim).to(device)
                lc_sum = torch.cat((c_hat1, noise), dim=1)
                fake_64 = gen_1(lc_sum)

                c_hat2, mu2, sigma2 = con_augment_2(tem)
                fake_256 = gen_2(fake_64, c_hat2)

                critic_2_real = critic_2(real_img_256, tem).view(-1)

                tem_mismatched = textEmbedder(mismatched_desc_tokens)
                critic_2_mismatched = critic_2(real_img_256, tem_mismatched).view(-1)

                critic_2_fake = critic_2(fake_256, tem).view(-1)

                critic_2_negative_samples = torch.cat(
                    (critic_2_mismatched, critic_2_fake), dim=0
                )

                gp = gradient_penalty(critic_2, real_img_256, fake_256, tem, device)

                loss_critic = -(
                    torch.mean(critic_2_real)
                    - torch.mean(critic_2_negative_samples)
                    + lambda_gp * gp
                )
                opt_critic_2.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic_2.step()

            output = critic_2(fake_256, tem).view(-1)
            lossG_fake = -torch.mean(output)
            kl_div = torch.sum(
                1 + torch.log(sigma2.pow(2)) - mu2.pow(2) - sigma2.pow(2)
            )
            lossG = lossG_fake + kl_div
            opt_gen_2.zero_grad()
            lossG.backward()
            opt_gen_2.step()

            opt_con_augment_2.step()
            opt_con_augment_2.zero_grad()

            if batch_idx % 100 == 0 and batch_idx > 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    tem = textEmbedder(desc_tokens)
                    c_hat1, mu1, sigma1 = con_augment_1(tem)
                    noise = torch.randn(current_batch_size, z_dim).to(device)
                    lc_sum = torch.cat((c_hat1, fixed_noise), dim=1)
                    fake_64 = gen_1(lc_sum)

                    c_hat2, mu2, sigma2 = con_augment_2(tem)
                    fake_256 = gen_2(fake_64, c_hat2)

                    img_grid_real = torchvision.utils.make_grid(
                        real_img_256, normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake_256[0], normalize=True
                    )

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
    models=[textEmbedder, con_augment_1, critic_1, gen_1],
    optimizers=[opt_text, opt_con_augment_1, opt_critic_1, opt_gen_1],
    loader=train_dl_1,
    num_epochs=num_epochs,
)
