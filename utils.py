import torch
import torchvision.transforms as transforms
from data_loader import get_loader

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

train_loader_1 = get_loader(
    root="./dataset/train2017",
    ann_file="./dataset/annotations/captions_train2017.json",
    transform=my_transform_1,
    batch_size=batch_size,
    shuffle=True,
)

train_loader_2 = get_loader(
    root="./dataset/train2017",
    ann_file="./dataset/annotations/captions_train2017.json",
    transform=my_transform_2,
    batch_size=batch_size,
    shuffle=True,
)

# embedding_layer = torch.nn.Embedding.from_pretrained(
#     dataset.vocab.glove.vectors, freeze=True
# )


def gradient_penalty(critic, real, fake, tem, device):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    mixed_scores = critic(interpolated_images, tem)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty
