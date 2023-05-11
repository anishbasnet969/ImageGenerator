import os
import pandas as pd
import json
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from transformers import AutoTokenizer
import torchvision.transforms as transforms


class TexttoImgCOCO(Dataset):
    def __init__(self, root, ann_file, transform=None):
        self.img_dir = root
        self.df = self.get_text_img_df(ann_file)
        self.transform = transform

        self.texts = self.df["caption"]
        self.imgs = self.df["file_name"]

        self.tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.texts[index]
        img_f = self.imgs[index]
        img = Image.open(os.path.join(self.img_dir, img_f)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return text, img

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
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        texts = [item[0] for item in batch]
        tokenized_texts = self.tokenizer.batch_encode_plus(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        imgs = [item[1].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        return tokenized_texts, imgs


def get_loader(root, ann_file, transform, batch_size=64, shuffle=True):
    dataset = TexttoImgCOCO(
        root=root,
        ann_file=ann_file,
        transform=transform,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=32,
        rank=xm.get_ordinal(),
        shuffle=shuffle,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=Collate(dataset.tokenizer),
    )

    return pl.MpDeviceLoader(loader, xm.xla_device())
