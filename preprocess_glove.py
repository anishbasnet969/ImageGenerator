import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe
import spacy

spacy_eng = spacy.load("en_core_web_lg")

glove = GloVe(name="6B", dim=300)

glove.stoi["<UNK>"] = len(glove.vectors)
glove.stoi["<PAD>"] = len(glove.vectors) + 1

unk_emb = torch.mean(glove.vectors, dim=0).view(1, 300)
pad_emb = torch.zeros(1, 300)

glove.vectors = torch.cat((glove.vectors, unk_emb, pad_emb))

VOCAB_SIZE = len(glove.vectors)


def tokenizer_eng(text):
    return [token.text.lower() for token in spacy_eng.tokenizer(text)]


def text_transform(data):
    data = data[:5]
    tokenized_texts = [tokenizer_eng(text) for text in data]

    return [
        [
            glove.stoi[token] if token in glove.stoi else glove.stoi["<UNK>"]
            for token in tokenized_text
        ]
        for tokenized_text in tokenized_texts
    ]


def collate_fn(batch):
    pad_idx = glove.stoi["<PAD>"]
    list_of_texts = [item[1] for item in batch]
    list_of_texts = [
        pad_sequence(texts, batch_first=True, padding_value=pad_idx)
        for texts in list_of_texts
    ]
    imgs = [item[0].unsqueeze(0) for item in batch]
    imgs = torch.cat(imgs, dim=0)

    return imgs, list_of_texts


def get_loader(root, annFile, transform, batch_size=32, shuffle=True):
    dataset = datasets.CocoCaptions(
        root=root,
        annFile=annFile,
        transform=transform,
        target_transform=text_transform,
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )

    return loader
