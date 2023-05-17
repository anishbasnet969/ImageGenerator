import os, sys, time
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


def train_1(
    models,
    optimizers,
    schedulers,
    loader,
    num_epochs,
    device,
    batch_size,
    start_epoch=0,
    bucket_name="data-and-checkpoints-bucket",
    save_dir="./checkpoints/Stage1",
):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    if xm.is_master_ordinal():
        writer_1 = SummaryWriter(f"gs://{bucket_name}/runs/ImageGen/Stage1")
        writer_real_1 = SummaryWriter(f"gs://{bucket_name}/runs/ImageGen/real_1")
        writer_fake_1 = SummaryWriter(f"gs://{bucket_name}/runs/ImageGen/fake_1")

    textEncoder, projection_head, con_augment_1, critic_1, gen_1 = models

    (
        opt_encoder,
        opt_projection_head,
        opt_con_augment_1,
        opt_critic_1,
        opt_gen_1,
    ) = optimizers

    (
        lr_scheduler_encoder,
        lr_scheduler_projection_head,
        lr_scheduler_con_augment_1,
        lr_scheduler_critic_1,
        lr_scheduler_gen_1,
    ) = schedulers

    checkpoint_path = os.path.join(save_dir, "latest_checkpoint_stage1.pth")

    blob = bucket.blob(checkpoint_path)
    if blob.exists():
        with tempfile.NamedTemporaryFile() as tmp:
            blob.download_to_filename(tmp.name)
            checkpoint = torch.load(tmp.name)
        start_epoch = checkpoint["epoch"] + 1
        textEncoder.load_state_dict(checkpoint["textEncoder"])
        projection_head.load_state_dict(checkpoint["projection_head"])
        con_augment_1.load_state_dict(checkpoint["con_augment_1"])
        critic_1.load_state_dict(checkpoint["critic_1"])
        gen_1.load_state_dict(checkpoint["gen_1"])
        opt_encoder.load_state_dict(checkpoint["opt_encoder"])
        opt_projection_head.load_state_dict(checkpoint["opt_projection_head"])
        opt_con_augment_1.load_state_dict(checkpoint["opt_con_augment_1"])
        opt_critic_1.load_state_dict(checkpoint["opt_critic_1"])
        opt_gen_1.load_state_dict(checkpoint["opt_gen_1"])
        lr_scheduler_encoder.load_state_dict(checkpoint["lr_scheduler_encoder"])
        lr_scheduler_projection_head.load_state_dict(
            checkpoint["lr_scheduler_projection_head"]
        )
        lr_scheduler_con_augment_1.load_state_dict(
            checkpoint["lr_scheduler_con_augment_1"]
        )
        lr_scheduler_critic_1.load_state_dict(checkpoint["lr_scheduler_critic_1"])
        lr_scheduler_gen_1.load_state_dict(checkpoint["lr_scheduler_gen_1"])
        print(f"Loaded checkpoint at epoch {start_epoch-1}")

    step = 0  # Tensorboard Global Step

    textEncoder.train()
    projection_head.train()
    con_augment_1.train()
    critic_1.train()
    gen_1.train()

    for epoch in range(start_epoch, num_epochs):
        for batch_idx, (tokenized_texts, real_img_64) in enumerate(loader):
            real_img_64 = real_img_64.to(device)
            tokenized_texts = {k: v.to(device) for k, v in tokenized_texts.items()}
            print("tokenized texts")

            if xm.is_master_ordinal():
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
            else:
                seed = None

            seed_tensor = torch.tensor(seed if seed is not None else 0).to(device)
            seed_tensor = xm.all_reduce("sum", seed_tensor)
            seed = seed_tensor.item()
            generator = torch.Generator().manual_seed(seed)

            mismatched_tokenized_texts = {
                k: v[torch.randperm(batch_size, generator=generator)]
                for k, v in tokenized_texts.items()
            }
            print("mismatched descriptions")

            time1 = time.time()

            for _ in range(n_critic):
                encoder_outputs = textEncoder(**tokenized_texts)
                cls_hidden_state = encoder_outputs.last_hidden_state[:, 0, :]
                tem = projection_head(cls_hidden_state)
                c_hat1, mu1, sigma1 = con_augment_1(tem)
                noise = torch.randn(batch_size, z_dim, generator=generator).to(device)
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

                xm.optimizer_step(opt_critic_1)

            print('-- End of a n_critic Loop')
            print(time.time() - time1)
            time1 = time.time()

            output = critic_1(fake_64, tem).view(-1)
            lossG_fake = -torch.mean(output)
            kl_div = torch.sum(
                1 + torch.log(sigma1.pow(2)) - mu1.pow(2) - sigma1.pow(2)
            )
            lossG = lossG_fake + kl_div

            lossG.backward()
            xm.optimizer_step(opt_gen_1)
            opt_gen_1.zero_grad()

            print('opt_gen_1 zero grad')

            xm.optimizer_step(opt_encoder)
            opt_encoder.zero_grad()
            xm.optimizer_step(opt_projection_head)
            opt_projection_head.zero_grad()

            xm.optimizer_step(opt_con_augment_1)
            opt_con_augment_1.zero_grad()

            print('opt_con_augment_1 zero grad')
            print(time.time() - time1)
            time1 = time.time()


            xm.master_print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                Loss D: {loss_critic:.4f}, loss G: {lossG:.4f}"
            )

            print('after master print')
            print(time.time() - time1)
            time1 = time.time()

            sys.exit()

            if xm.is_master_ordinal():
                lr_scheduler_critic_1.step()
                lr_scheduler_gen_1.step()
                lr_scheduler_encoder.step()
                lr_scheduler_projection_head.step()
                lr_scheduler_con_augment_1.step()

            print('after lr scheduler print')
            print(time.time() - time1)
            time1 = time.time()

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
                    fixed_generator = torch.Generator().manual_seed(456)
                    fixed_noise = torch.randn(
                        batch_size, z_dim, generator=fixed_generator
                    ).to(device)
                    C_g = torch.cat((c_hat1, fixed_noise), dim=1)
                    fake_64 = gen_1(C_g)
                    img_grid_real = torchvision.utils.make_grid(
                        real_img_64, normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake_64[0], normalize=True
                    )

                    writer_real_1.add_image(
                        "Real 64*64", img_grid_real, global_step=step
                    )
                    writer_fake_1.add_image(
                        "Fake 64*64", img_grid_fake, global_step=step
                    )

                writer_1.add_scalar("Critic 1 loss", loss_critic, global_step=step)
                writer_1.add_scalar("Generator 1 loss", lossG, global_step=step)

                step += 1
                
            print(f' batch loop end {batch_idx}')
            sys.exit()

        if xm.is_master_ordinal() and epoch % 10 == 0:
            checkpoint = {
                "textEncoder": textEncoder.state_dict(),
                "projection_head": projection_head.state_dict(),
                "con_augment_1": con_augment_1.state_dict(),
                "critic_1": critic_1.state_dict(),
                "gen_1": gen_1.state_dict(),
                "opt_encoder": opt_encoder.state_dict(),
                "opt_projection_head": opt_projection_head.state_dict(),
                "opt_con_augment_1": opt_con_augment_1.state_dict(),
                "opt_critic_1": opt_critic_1.state_dict(),
                "opt_gen_1": opt_gen_1.state_dict(),
                "lr_scheduler_encoder": lr_scheduler_encoder.state_dict(),
                "lr_scheduler_projection_head": lr_scheduler_projection_head.state_dict(),
                "lr_scheduler_con_augment_1": lr_scheduler_con_augment_1.state_dict(),
                "lr_scheduler_critic_1": lr_scheduler_critic_1.state_dict(),
                "lr_scheduler_gen_1": lr_scheduler_gen_1.state_dict(),
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

    writer_1.close()
    writer_real_1.close()
    writer_fake_1.close()
