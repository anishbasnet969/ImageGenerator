import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.optim.lr_scheduler import StepLR

from transformers import AutoModel
from con_augment import ConditioningAugmentation
from generator_1 import StageIGenerator
from discrminator_1 import StageIDiscriminator
from generator_2 import StageIIGenerator
from discriminator_2 import StageIIDiscriminator
from data_loader import get_loader
from stage_1_train_fn import train_1
from stage_2_train_fn import train_2

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.experimental.pjrt_backend
import torch_xla.experimental.pjrt as pjrt

TEM_SIZE = 512
lr = 1e-3
c_dim = 128
z_dim = 100
Nd = 128
num_epochs = 500
batch_size = 2048

my_transform_1 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

my_transform_2 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


# opt_text = optim.Adam(textEmbedder.parameters(), lr=lr, betas=(0.9, 0.999))


def train_xmp(index):
    print(index)
    device = xm.xla_device()
    dist.init_process_group("xla", init_method="pjrt://")

    torch.manual_seed(42)

    textEncoder = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased").to(device)
    projection_head = nn.Linear(768, TEM_SIZE).to(device)
    con_augment_1 = ConditioningAugmentation(TEM_SIZE, 256, c_dim).to(device)
    critic_1 = StageIDiscriminator(TEM_SIZE, Nd).to(device)
    gen_1 = StageIGenerator(c_dim, z_dim).to(device)
    con_augment_2 = ConditioningAugmentation(TEM_SIZE, 256, c_dim).to(device)
    critic_2 = StageIIDiscriminator(TEM_SIZE, Nd).to(device)
    gen_2 = StageIIGenerator().to(device)

    pjrt.broadcast_master_param(textEncoder)
    pjrt.broadcast_master_param(projection_head)
    pjrt.broadcast_master_param(con_augment_1)
    pjrt.broadcast_master_param(critic_1)
    pjrt.broadcast_master_param(gen_1)
    pjrt.broadcast_master_param(con_augment_2)
    pjrt.broadcast_master_param(critic_2)
    pjrt.broadcast_master_param(gen_2)

    textEncoder = DDP(textEncoder, gradient_as_bucket_view=True)
    projection_head = DDP(projection_head, gradient_as_bucket_view=True)
    con_augment_1 = DDP(con_augment_1, gradient_as_bucket_view=True)
    critic_1 = DDP(critic_1, gradient_as_bucket_view=True)
    gen_1 = DDP(gen_1, gradient_as_bucket_view=True)
    con_augment_2 = DDP(con_augment_2, gradient_as_bucket_view=True)
    critic_2 = DDP(critic_2, gradient_as_bucket_view=True)
    gen_2 = DDP(gen_2, gradient_as_bucket_view=True)

    opt_encoder = optim.AdamW(textEncoder.parameters(), lr=5e-5)
    opt_projection_head = optim.Adam(
        projection_head.parameters(), lr=lr, betas=(0.9, 0.999)
    )
    opt_con_augment_1 = optim.Adam(
        con_augment_1.parameters(), lr=lr, betas=(0.9, 0.999)
    )
    opt_critic_1 = optim.Adam(critic_1.parameters(), lr=lr, betas=(0.9, 0.999))
    opt_gen_1 = optim.Adam(gen_1.parameters(), lr=lr, betas=(0.9, 0.999))

    opt_con_augment_2 = optim.Adam(
        con_augment_2.parameters(), lr=lr, betas=(0.9, 0.999)
    )
    opt_critic_2 = optim.Adam(critic_2.parameters(), lr=lr, betas=(0.9, 0.999))
    opt_gen_2 = optim.Adam(gen_2.parameters(), lr=lr, betas=(0.9, 0.999))

    lr_scheduler_encoder = StepLR(opt_encoder, step_size=100, gamma=0.5)
    lr_scheduler_projection_head = StepLR(opt_projection_head, step_size=100, gamma=0.5)
    lr_scheduler_con_augment_1 = StepLR(opt_con_augment_1, step_size=100, gamma=0.5)
    lr_scheduler_critic_1 = StepLR(opt_critic_1, step_size=100, gamma=0.5)
    lr_scheduler_gen_1 = StepLR(opt_gen_1, step_size=100, gamma=0.5)

    lr_scheduler_con_augment_2 = StepLR(opt_con_augment_2, step_size=100, gamma=0.5)
    lr_scheduler_critic_2 = StepLR(opt_critic_2, step_size=100, gamma=0.5)
    lr_scheduler_gen_2 = StepLR(opt_gen_2, step_size=100, gamma=0.5)

    train_loader_1 = get_loader(
        bucket_name="data-and-checkpoints-bucket",
        root="dataset/train2017",
        ann_file="dataset/annotations/captions_train2017.json",
        transform=my_transform_1,
        batch_size=batch_size,
        shuffle=True,
    )

    print("we are here after the train loader 1 initialization")

    # train_loader_2 = get_loader(
    #     root="dataset/train2017",
    #     ann_file="dataset/annotations/captions_train2017.json",
    #     transform=my_transform_2,
    #     batch_size=batch_size,
    #     shuffle=True,
    # )

    train_1(
        models=[textEncoder, projection_head, con_augment_1, critic_1, gen_1],
        optimizers=[
            opt_encoder,
            opt_projection_head,
            opt_con_augment_1,
            opt_critic_1,
            opt_gen_1,
        ],
        schedulers=[
            lr_scheduler_encoder,
            lr_scheduler_projection_head,
            lr_scheduler_con_augment_1,
            lr_scheduler_critic_1,
            lr_scheduler_gen_1,
        ],
        loader=train_loader_1,
        num_epochs=num_epochs,
        device=device,
    )


if __name__ == "__main__":
    os.environ["PJRT_DEVICE"] = "TPU"
    xmp.spawn(train_xmp, args=(), nprocs=xm.xrt_world_size(), start_method="fork")
