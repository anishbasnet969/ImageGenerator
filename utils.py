import torch
import torch.nn as nn
import torchvision.transforms as transforms
from textEmbed import TextEmbeddingLSTM
from custom_dataloader import get_loader

batch_size = 64

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

train_dl_1, dataset = get_loader(
    root="data/train2017",
    ann_file="data/annotations/captions_train2017.json",
    transform=my_transform_1,
    batch_size=batch_size,
)

train_dl_2, _ = get_loader(
    root="data/train2017",
    ann_file="data/annotations/captions_train2017.json",
    transform=my_transform_2,
    batch_size=batch_size,
)

embedding_layer = torch.nn.Embedding.from_pretrained(
    dataset.vocab.glove.vectors, freeze=True
)

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


def gradient_penalty(critic, real, fake, desc_tokens, device):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    mixed_scores = critic(interpolated_images, desc_tokens)

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
