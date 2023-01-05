from textEmbed import TextEmbeddingLSTM
from preprocessing import *


transform = transforms.Compose([transforms.Resize((64, 64)), transforms.PILToTensor()])

loader, dataset = get_loader(
    "data/annotations/captions_train2017.json",
    "data/train2017/",
    transform=transform,
)

EMBEDDING_SIZE = 300
TEXT_EMBEDDING_HIDDEN_SIZE = 300
VOCAB_SIZE = len(dataset.vocab)
TEXT_EMBEDDING_NUM_LAYERS = 2
TEM_SIZE = 400

textEmbedder = TextEmbeddingLSTM(
    EMBEDDING_SIZE,
    TEXT_EMBEDDING_HIDDEN_SIZE,
    VOCAB_SIZE,
    TEXT_EMBEDDING_NUM_LAYERS,
    TEM_SIZE,
)


# def train_GAN(loader, ):


if __name__ == "__main__":
    pass
