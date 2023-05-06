import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel
from con_augment import ConditioningAugmentation
from generator_1 import StageIGenerator
from discrminator_1 import StageIDiscriminator
from generator_2 import StageIIGenerator
from discriminator_2 import StageIIDiscriminator
from utils import batch_size, train_dl_1, train_dl_2, gradient_penalty

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

TEM_SIZE = 512

textEncoder = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
projection_head = nn.Linear(768, TEM_SIZE)

lr = 2e-4
c_dim = 128
z_dim = 100
Nd = 128
num_epochs = 400
n_critic = 5
lambda_gp = 10

con_augment_1 = ConditioningAugmentation(TEM_SIZE, 256, 128).to(device)
critic_1 = StageIDiscriminator(TEM_SIZE, Nd).to(device)
gen_1 = StageIGenerator(c_dim, z_dim).to(device)


con_augment_2 = ConditioningAugmentation(TEM_SIZE, 320, 128).to(device)
critic_2 = StageIIDiscriminator(TEM_SIZE, Nd).to(device)
gen_2 = StageIIGenerator(c_dim).to(device)

# opt_text = optim.Adam(textEmbedder.parameters(), lr=lr, betas=(0.9, 0.999))
opt_encoder = optim.AdamW(textEncoder.parameters(), lr=5e-5)
opt_projection_head = optim.Adam(
    projection_head.parameters(), lr=lr, betas=(0.9, 0.999)
)

opt_con_augment_1 = optim.Adam(con_augment_1.parameters(), lr=lr, betas=(0.9, 0.999))
opt_critic_1 = optim.Adam(critic_1.parameters(), lr=lr, betas=(0.9, 0.999))
opt_gen_1 = optim.Adam(gen_1.parameters(), lr=lr, betas=(0.9, 0.999))


opt_con_augment_2 = optim.Adam(con_augment_2.parameters(), lr=lr, betas=(0.9, 0.999))
opt_critic_2 = optim.Adam(critic_2.parameters(), lr=lr, betas=(0.9, 0.999))
opt_gen_2 = optim.Adam(gen_2.parameters(), lr=lr, betas=(0.9, 0.999))

writer = SummaryWriter("runs/ImageGen/COCO")
writer_real = SummaryWriter(f"runs/ImageGen/real")
writer_fake = SummaryWriter(f"runs/ImageGen/fake")


def train_1(models, optimizers, loader, num_epochs, device=device):
    textEncoder, projection_head, con_augment_1, critic_1, gen_1 = models
    (
        opt_encoder,
        opt_projection_head,
        opt_con_augment_1,
        opt_critic_1,
        opt_gen_1,
    ) = optimizers
    step = 0  # Tensorboard Global Step

    textEncoder.train()
    projection_head.train()
    con_augment_1.train()
    critic_1.train()
    gen_1.train()

    for epoch in range(num_epochs):
        for batch_idx, (tokenized_texts, real_img_64) in enumerate(loader):
            real_img_64 = real_img_64.to(device)
            tokenized_texts = {k: v.to(device) for k, v in tokenized_texts.items()}
            current_batch_size = real_img_64.shape[0]
            torch.manual_seed(random.randint(0, 1000))
            mismatched_tokenized_texts = {
                k: v[torch.randperm(current_batch_size)].to(device)
                for k, v in tokenized_texts.items()
            }
            for _ in range(n_critic):
                encoder_outputs = textEncoder(**tokenized_texts)
                cls_hidden_state = encoder_outputs.last_hidden_state[:, 0, :]
                tem = projection_head(cls_hidden_state)
                c_hat1, mu1, sigma1 = con_augment_1(tem)
                torch.manual_seed(random.randint(0, 1000))
                noise = torch.randn(current_batch_size, z_dim).to(device)
                C_g = torch.cat((c_hat1, noise), dim=1)
                fake_64 = gen_1(C_g)

                critic_1_real = critic_1(real_img_64, tem).view(-1)

                mismatched_encoder_outputs = textEncoder(**mismatched_tokenized_texts)
                cls_hidden_state = mismatched_encoder_outputs.last_hidden_state[:, 0, :]
                tem_mismatched = projection_head(cls_hidden_state)
                critic_1_mismatched = critic_1(real_img_64, tem_mismatched).view(-1)

                critic_1_fake = critic_1(fake_64, tem).view(-1)

                critic_1_negative_samples = torch.cat(
                    (critic_1_mismatched, critic_1_fake), dim=0
                )

                gp = gradient_penalty(critic_1, real_img_64, fake_64, tem, device)

                loss_critic = (
                    torch.mean(critic_1_negative_samples)
                    - torch.mean(critic_1_real)
                    + lambda_gp * gp
                )
                opt_critic_1.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic_1.step()

            output = critic_1(fake_64, tem).view(-1)
            lossG_fake = -torch.mean(output)
            kl_div = torch.sum(
                1 + torch.log(sigma1.pow(2)) - mu1.pow(2) - sigma1.pow(2)
            )
            lossG = lossG_fake + kl_div

            lossG.backward()
            opt_gen_1.step()
            opt_gen_1.zero_grad()

            opt_encoder.step()
            opt_encoder.zero_grad()
            opt_projection_head.step()
            opt_projection_head.zero_grad()

            opt_con_augment_1.step()
            opt_con_augment_1.zero_grad()

            if batch_idx > 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    encoder_outputs = textEncoder(**tokenized_texts)
                    cls_hidden_state = encoder_outputs.last_hidden_state[:, 0, :]
                    tem = projection_head(cls_hidden_state)
                    c_hat1, mu1, sigma1 = con_augment_1(tem)
                    torch.manual_seed(456)
                    fixed_noise = torch.randn(current_batch_size, z_dim).to(device)
                    C_g = torch.cat((c_hat1, fixed_noise), dim=1)
                    fake_64 = gen_1(C_g)
                    img_grid_real = torchvision.utils.make_grid(
                        real_img_64, normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake_64[0], normalize=True
                    )

                    writer_real.add_image("Real 64*64", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake 64*64", img_grid_fake, global_step=step)

                writer.add_scalar("Critic 1 loss", loss_critic, global_step=step)
                writer.add_scalar("Generator 1 loss", lossG, global_step=step)

                step += 1


def train_2(models, optimizers, loader, num_epochs, device=device):
    (
        textEncoder,
        projection_head,
        con_augment_1,
        con_augment_2,
        gen_1,
        critic_2,
        gen_2,
    ) = models
    opt_con_augment_2, opt_critic_2, opt_gen_2 = optimizers
    step = 0  # Tensorboard Global Step

    con_augment_2.train()
    critic_2.train()
    gen_2.train()

    for epoch in range(num_epochs):
        for batch_idx, (tokenized_texts, real_img_256) in enumerate(loader):
            real_img_256 = real_img_256.to(device)
            tokenized_texts = {k: v.to(device) for k, v in tokenized_texts.items()}
            current_batch_size = real_img_256.shape[0]
            torch.manual_seed(random.randint(0, 1000))
            mismatched_tokenized_texts = {
                k: v[torch.randperm(current_batch_size)].to(device)
                for k, v in tokenized_texts.items()
            }

            for _ in range(n_critic):
                encoder_outputs = textEncoder(**tokenized_texts)
                cls_hidden_state = encoder_outputs.last_hidden_state[:, 0, :]
                tem = projection_head(cls_hidden_state)
                c_hat1, mu1, sigma1 = con_augment_1(tem)
                torch.manual_seed(random.randint(0, 1000))
                noise = torch.randn(current_batch_size, z_dim).to(device)
                C_g = torch.cat((c_hat1, noise), dim=1)
                fake_64 = gen_1(C_g)

                c_hat2, mu2, sigma2 = con_augment_2(tem)
                fake_256 = gen_2(fake_64, c_hat2)

                critic_2_real = critic_2(real_img_256, tem).view(-1)

                mismatched_encoder_outputs = textEncoder(**mismatched_tokenized_texts)
                cls_hidden_state = mismatched_encoder_outputs.last_hidden_state[:, 0, :]
                tem_mismatched = projection_head(cls_hidden_state)
                critic_2_mismatched = critic_2(real_img_256, tem_mismatched).view(-1)

                critic_2_fake = critic_2(fake_256, tem).view(-1)

                critic_2_negative_samples = torch.cat(
                    (critic_2_mismatched, critic_2_fake), dim=0
                )

                gp = gradient_penalty(critic_2, real_img_256, fake_256, tem, device)

                loss_critic = (
                    torch.mean(critic_2_negative_samples)
                    - torch.mean(critic_2_real)
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
            lossG.backward()
            opt_gen_2.step()
            opt_gen_2.zero_grad()

            opt_con_augment_2.step()
            opt_con_augment_2.zero_grad()

            if batch_idx % 100 == 0 and batch_idx > 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    encoder_outputs = textEncoder(**tokenized_texts)
                    cls_hidden_state = encoder_outputs.last_hidden_state[:, 0, :]
                    tem = projection_head(cls_hidden_state)
                    c_hat1, mu1, sigma1 = con_augment_1(tem)
                    torch.manual_seed(456)
                    fixed_noise = torch.randn(current_batch_size, z_dim).to(device)
                    C_g = torch.cat((c_hat1, fixed_noise), dim=1)
                    fake_64 = gen_1(C_g)

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
    models=[textEncoder, projection_head, con_augment_1, critic_1, gen_1],
    optimizers=[
        opt_encoder,
        opt_projection_head,
        opt_con_augment_1,
        opt_critic_1,
        opt_gen_1,
    ],
    loader=train_dl_1,
    num_epochs=num_epochs,
)
