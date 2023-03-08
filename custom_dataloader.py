import os
import pandas as pd
import json
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchtext.vocab import GloVe
import torchvision.transforms as transforms

spacy_eng = spacy.load("en_core_web_lg")


class Vocabulary:
    def __init__(self):

        # self.itos = {0: "<PAD>", 1: "<UNK>"}
        # self.stoi = {"<PAD>": 0, "<UNK>": 1}
        # self.freq_threshold = freq_threshold

        self.glove = GloVe(name="6B", dim=300)

        self.glove.stoi["<UNK>"] = len(self.glove.vectors)
        self.glove.stoi["<PAD>"] = len(self.glove.vectors) + 1

        self.glove.itos.append("<UNK>")
        self.glove.itos.append("<PAD>")

        unk_emb = torch.mean(self.glove.vectors, dim=0).view(1, 300)
        pad_emb = torch.zeros(1, 300)

        self.glove.vectors = torch.cat((self.glove.vectors, unk_emb, pad_emb))

    def __len__(self):
        return len(self.glove.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.glove.stoi[token]
            if token in self.glove.stoi
            else self.glove.stoi["<UNK>"]
            for token in tokenized_text
        ]


class TexttoImgCOCO(Dataset):
    def __init__(self, root, ann_file, transform=None, freq_threshold=2):
        self.img_dir = root
        self.df = self.get_text_img_df(ann_file)
        self.transform = transform

        self.texts = self.df["caption"]
        self.imgs = self.df["file_name"]

        self.vocab = Vocabulary()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.texts[index]
        img_f = self.imgs[index]
        img = Image.open(os.path.join(self.img_dir, img_f)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_text = torch.tensor(self.vocab.numericalize(text))

        return numericalized_text, img

    def get_text_img_df(self, ann_file):
        with open(ann_file, "r") as f:
            anns = json.load(f)
            imgs = anns["images"]
            texts = anns["annotations"]

        img_df = pd.DataFrame(imgs)[["id", "file_name"]]
        img_df.rename(columns={"id": "image_id"}, inplace=True)
        text_df = pd.DataFrame(texts)[["image_id", "caption"]]
        text_img_df = text_df.merge(img_df, on="image_id")

        return text_img_df


class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        texts = [item[0] for item in batch]
        texts = pad_sequence(texts, batch_first=True, padding_value=self.pad_idx)
        imgs = [item[1].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        return texts, imgs


def get_loader(root, ann_file, transform, batch_size=32, shuffle=True):
    dataset = TexttoImgCOCO(root, ann_file, transform=transform)

    pad_idx = dataset.vocab.glove.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Collate(pad_idx=pad_idx),
    )

    return loader, dataset
