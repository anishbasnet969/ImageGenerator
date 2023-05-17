import os
import random
import torch
import torchvision

import torch_xla.core.xla_model as xm

from google.cloud import storage
import tempfile

from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty


n_critic = 5
lambda_gp = 10
z_dim = 100


def train_2(
    models,
    optimizers,
    schedulers,
    loader,
    num_epochs,
    device,
    batch_size,
    bucket_name="data-and-checkpoints-bucket",
    start_epoch=0,
    save_dir="./checkpoints/Stage2",
):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    if xm.is_master_ordinal():
        writer_2 = SummaryWriter(f"gs://{bucket_name}/runs/ImageGen/Stage2")
        writer_real_2 = SummaryWriter(f"gs://{bucket_name}/runs/ImageGen/real_2")
        writer_fake_2 = SummaryWriter(f"gs://{bucket_name}/runs/ImageGen/fake_2")

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
    lr_scheduler_con_augment_2, lr_scheduler_critic_2, lr_scheduler_gen_2 = schedulers

    textEncoder.eval()
    for param in textEncoder.parameters():
        param.requires_grad = False
    projection_head.eval()
    for param in projection_head.parameters():
        param.requires_grad = False
    con_augment_1.eval()
    for param in con_augment_1.parameters():
        param.requires_grad = False
    gen_1.eval()
    for param in gen_1.parameters():
        param.requires_grad = False

    blob_1 = bucket.blob("./checkpoint/Stage1/latest_checkpoint_stage1.pth")
    with tempfile.NamedTemporaryFile() as tmp:
        blob.download_to_filename(tmp.name)
        stage1_checkpoint = torch.load(tmp.name)
    textEncoder.load_state_dict(stage1_checkpoint["textEncoder"])
    projection_head.load_state_dict(stage1_checkpoint["projection_head"])
    con_augment_1.load_state_dict(stage1_checkpoint["con_augment_1"])
    gen_1.load_state_dict(stage1_checkpoint["gen_1"])

    checkpoint_path = os.path.join(save_dir, "latest_checkpoint_stage2.pth")
    blob_2 = bucket.blob(checkpoint_path)
    if blob_2.exists():
        with tempfile.NamedTemporaryFile() as tmp:
            blob_2.download_to_filename(tmp.name)
            checkpoint = torch.load(tmp.name)
        start_epoch = checkpoint["epoch"] + 1
        con_augment_2.load_state_dict(checkpoint["con_augment_2"])
        critic_2.load_state_dict(checkpoint["critic_2"])
        gen_2.load_state_dict(checkpoint["gen_2"])
        opt_con_augment_2.load_state_dict(checkpoint["opt_con_augment_2"])
        opt_critic_2.load_state_dict(checkpoint["opt_critic_2"])
        opt_gen_2.load_state_dict(checkpoint["opt_gen_2"])
        lr_scheduler_con_augment_2.load_state_dict(
            checkpoint["lr_scheduler_con_augment_2"]
        )
        lr_scheduler_critic_2.load_state_dict(checkpoint["lr_scheduler_critic_2"])
        lr_scheduler_gen_2.load_state_dict(checkpoint["lr_scheduler_gen_2"])
        print(f"Loaded checkpoint at epoch {start_epoch-1}")

    step = 0  # Tensorboard Global Step

    con_augment_2.train()
    critic_2.train()
    gen_2.train()

    for epoch in range(start_epoch, num_epochs):
        for batch_idx, (tokenized_texts, real_img_256) in enumerate(loader):
            real_img_256 = real_img_256.to(device)
            tokenized_texts = {k: v.to(device) for k, v in tokenized_texts.items()}

            if xm.is_master_ordinal():
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
            else:
                seed = None

            seed = xm.all_reduce(xm.REDUCE_OP_SUM, torch.tensor(seed).to(device))
            generator = torch.Generator(device=device).manual_seed(seed)

            mismatched_tokenized_texts = {
                k: v[torch.randperm(batch_size, generator=generator)].to(device)
                for k, v in tokenized_texts.items()
            }

            for _ in range(n_critic):
                encoder_outputs = textEncoder(**tokenized_texts)
                cls_hidden_state = encoder_outputs.last_hidden_state[:, 0, :]
                tem = projection_head(cls_hidden_state)
                c_hat1, mu1, sigma1 = con_augment_1(tem)
                torch.manual_seed(random.randint(0, 1000))
                noise = torch.randn(batch_size, z_dim, generator=generator).to(device)
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
                xm.optimizer_step(opt_critic_2, barrier=True)

            output = critic_2(fake_256, tem).view(-1)
            lossG_fake = -torch.mean(output)
            kl_div = torch.sum(
                1 + torch.log(sigma2.pow(2)) - mu2.pow(2) - sigma2.pow(2)
            )
            lossG = lossG_fake + kl_div
            lossG.backward()
            xm.optimizer_step(opt_gen_2, barrier=True)
            opt_gen_2.zero_grad()

            xm.optimizer_step(opt_con_augment_2, barrier=True)
            opt_con_augment_2.zero_grad()

            if xm.is_master_ordinal():
                lr_scheduler_critic_2.step()
                lr_scheduler_gen_2.step()
                lr_scheduler_con_augment_2.step()

            if xm.is_master_ordinal() and batch_idx % 100 == 0 and batch_idx > 0:
                xm.master_print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    encoder_outputs = textEncoder(**tokenized_texts)
                    cls_hidden_state = encoder_outputs.last_hidden_state[:, 0, :]
                    tem = projection_head(cls_hidden_state)
                    c_hat1, mu1, sigma1 = con_augment_1(tem)
                    fixed_generator = torch.Generator(device=device).manual_seed(456)
                    fixed_noise = torch.randn(
                        batch_size, z_dim, generator=fixed_generator
                    ).to(device)
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

                    writer_real_2.add_image(
                        "Real 256*256", img_grid_real, global_step=step
                    )
                    writer_fake_2.add_image(
                        "Fake 256*256", img_grid_fake, global_step=step
                    )

                writer_2.add_scalar("Critic 2 loss", loss_critic, global_step=step)
                writer_2.add_scalar("Generator 2 loss", lossG, global_step=step)
                step += 1

        if xm.is_master_ordinal() and epoch % 10 == 0:
            checkpoint = {
                "con_augment_2": con_augment_2.state_dict(),
                "critic_2": critic_2.state_dict(),
                "gen_2": gen_2.state_dict(),
                "opt_con_augment_2": opt_con_augment_2.state_dict(),
                "opt_critic_2": opt_critic_2.state_dict(),
                "opt_gen_2": opt_gen_2.state_dict(),
                "lr_scheduler_con_augment_2": lr_scheduler_con_augment_2.state_dict(),
                "lr_scheduler_critic_2": lr_scheduler_critic_2.state_dict(),
                "lr_scheduler_gen_2": lr_scheduler_gen_2.state_dict(),
                "epoch": epoch,
            }
            checkpoint_epoch_path = f"{save_dir}/epochs/checkpoint_epoch_{epoch}.pth"
            blob_epoch = bucket.blob(checkpoint_epoch_path)
            with tempfile.NamedTemporaryFile() as tmp:
                torch.save(checkpoint, tmp.name)
                blob_epoch.upload_from_filename(tmp.name)
            blob = bucket.blob(checkpoint_path)
            with tempfile.NamedTemporaryFile() as tmp:
                torch.save(checkpoint, tmp.name)
                blob.upload_from_filename(tmp.name)

    writer_2.close()
    writer_real_2.close()
    writer_fake_2.close()
