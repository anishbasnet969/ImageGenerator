import torch
import torchvision.transforms as transforms
from textEmbed import TextEmbeddingLSTM
from custom_dataloader import get_loader

batch_size = 64

my_transform_1 = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((64, 64))]
)

my_transform_2 = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((256, 256))]
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

# (desc_tokens, img) = next(iter(train_dl_1))
# print(img)
# print(img.shape)
# print(desc_tokens)
# print(desc_tokens.shape)
