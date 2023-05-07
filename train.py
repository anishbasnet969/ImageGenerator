import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from transformers import AutoModel
from con_augment import ConditioningAugmentation
from generator_1 import StageIGenerator
from discrminator_1 import StageIDiscriminator
from generator_2 import StageIIGenerator
from discriminator_2 import StageIIDiscriminator
from utils import train_dl_1, train_dl_2
from stage_1_train_fn import train_1, train_2

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

TEM_SIZE = 512

textEncoder = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
projection_head = nn.Linear(768, TEM_SIZE)

lr = 2e-4
c_dim = 128
z_dim = 100
Nd = 128
num_epochs = 300


con_augment_1 = ConditioningAugmentation(TEM_SIZE, 256, c_dim).to(device)
critic_1 = StageIDiscriminator(TEM_SIZE, Nd).to(device)
gen_1 = StageIGenerator(c_dim, z_dim).to(device)


con_augment_2 = ConditioningAugmentation(TEM_SIZE, 320, c_dim).to(device)
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

lr_scheduler_encoder = StepLR(opt_encoder, step_size=100, gamma=0.5)
lr_scheduler_projection_head = StepLR(opt_projection_head, step_size=100, gamma=0.5)
lr_scheduler_con_augment_1 = StepLR(opt_con_augment_1, step_size=100, gamma=0.5)
lr_scheduler_critic_1 = StepLR(opt_critic_1, step_size=100, gamma=0.5)
lr_scheduler_gen_1 = StepLR(opt_gen_1, step_size=100, gamma=0.5)

lr_scheduler_con_augment_2 = StepLR(opt_con_augment_2, step_size=100, gamma=0.5)
lr_scheduler_critic_2 = StepLR(opt_critic_2, step_size=100, gamma=0.5)
lr_scheduler_gen_2 = StepLR(opt_gen_2, step_size=100, gamma=0.5)

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
    loader=train_dl_1,
    num_epochs=num_epochs,
    device=device,
)
