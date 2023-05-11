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

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

torch.autograd.set_detect_anomaly(True)

device = xm.xla_device()

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

textEncoder = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
projection_head = nn.Linear(768, TEM_SIZE)
con_augment_1 = ConditioningAugmentation(TEM_SIZE, 256, c_dim)
critic_1 = StageIDiscriminator(TEM_SIZE, Nd)
gen_1 = StageIGenerator(c_dim, z_dim)
con_augment_2 = ConditioningAugmentation(TEM_SIZE, 320, c_dim)
critic_2 = StageIIDiscriminator(TEM_SIZE, Nd)
gen_2 = StageIIGenerator(c_dim)

textEncoder = xmp.MpModelWrapper(textEncoder).to(device)
projection_head = xmp.MpModelWrapper(projection_head).to(device)
con_augment_1 = xmp.MpModelWrapper(con_augment_1).to(device)
critic_1 = xmp.MpModelWrapper(critic_1).to(device)
gen_1 = xmp.MpModelWrapper(gen_1).to(device)
con_augment_2 = xmp.MpModelWrapper(con_augment_2).to(device)
critic_2 = xmp.MpModelWrapper(critic_2).to(device)
gen_2 = xmp.MpModelWrapper(gen_2).to(device)


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


lr_scheduler_encoder = StepLR(opt_encoder, step_size=100, gamma=0.5)
lr_scheduler_projection_head = StepLR(opt_projection_head, step_size=100, gamma=0.5)
lr_scheduler_con_augment_1 = StepLR(opt_con_augment_1, step_size=100, gamma=0.5)
lr_scheduler_critic_1 = StepLR(opt_critic_1, step_size=100, gamma=0.5)
lr_scheduler_gen_1 = StepLR(opt_gen_1, step_size=100, gamma=0.5)

lr_scheduler_con_augment_2 = StepLR(opt_con_augment_2, step_size=100, gamma=0.5)
lr_scheduler_critic_2 = StepLR(opt_critic_2, step_size=100, gamma=0.5)
lr_scheduler_gen_2 = StepLR(opt_gen_2, step_size=100, gamma=0.5)


def train_1_xmp(rank):
    train_loader_1 = get_loader(
        bucket_name="data-and-checkpoints-bucket",
        root="dataset/train2017",
        ann_file="dataset/annotations/captions_train2017.json",
        transform=my_transform_1,
        batch_size=batch_size,
        shuffle=True,
    )

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


# def train2_xmp(rank):
#     train_loader_2 = get_loader(
#         root="dataset/train2017",
#         ann_file="dataset/annotations/captions_train2017.json",
#         transform=my_transform_2,
#         batch_size=batch_size,
#         shuffle=True,
#     )


if __name__ == "main":
    xmp.spawn(train_1_xmp, args=(), nprocs=32, start_method="fork")
