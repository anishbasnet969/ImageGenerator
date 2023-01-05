import matplotlib.pyplot as plt

from con_augment import ConditioningAugmentation
from discrminator_1 import StageIDiscriminator
from generator_1 import StageIGenerator
from textEmbed import TextEmbeddingLSTM
from preprocessing import *


EMBEDDING_SIZE = 300
TEXT_EMBEDDING_HIDDEN_SIZE = 128
VOCAB_SIZE = len(spacy_eng.vocab)
TEXT_EMBEDDING_NUM_LAYERS = 2


transform = transforms.Compose([transforms.Resize((64, 64)), transforms.PILToTensor()])

loader = get_loader(
  "coco_train/annotations/captions_train2017.json", "coco_train/train2017/", transform=transform
)

textEmbedder = TextEmbeddingLSTM(EMBEDDING_SIZE, TEXT_EMBEDDING_HIDDEN_SIZE, len(spacy_eng.vocab), TEXT_EMBEDDING_NUM_LAYERS)


# def train_GAN(loader, ):



if __name__ == "__main__":
  pass