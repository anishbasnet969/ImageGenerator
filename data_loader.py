import os
import pandas as pd
import json
from PIL import Image
from google.cloud import storage
import io

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
import torch_xla.core.xla_model as xm
import torchvision.transforms as transforms


class TexttoImgCOCO(Dataset):
    def __init__(self, bucket, root, ann_file, transform=None):
        self.bucket = bucket
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

        blob = self.bucket.blob(os.path.join(self.img_dir, img_f))
        img_data = blob.download_as_bytes()

        img = Image.open(io.BytesIO(img_data)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return text, img

    def get_text_img_df(self, ann_file):
        blob = self.bucket.blob(ann_file)
        anns = json.loads(blob.download_as_text())

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


def get_loader(bucket_name, root, ann_file, transform, batch_size=64, shuffle=True):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    dataset = TexttoImgCOCO(
        bucket=bucket,
        root=root,
        ann_file=ann_file,
        transform=transform,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=shuffle,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=Collate(dataset.tokenizer),
        sampler=sampler,
        drop_last=True,
        num_workers=8,
        persistent_workers=False,
        prefetch_factor=16,
    )

    return loader
