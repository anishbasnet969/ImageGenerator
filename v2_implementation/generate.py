import argparse
import math
import random
from urllib.request import urlopen
from tqdm import tqdm
import sys
import os

from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

torch.backends.cudnn.benchmark = False

from CLIP import clip
import kornia.augmentation as K
import numpy as np
import imageio

from PIL import ImageFile, Image, PngImagePlugin, ImageChops

ImageFile.LOAD_TRUNCATED_IMAGES = True

from subprocess import Popen, PIPE
import re

from vision_utils import random_noise_image, random_gradient_image, resample
from utils import ReplaceGrad, ClampWithGrad

import warnings

warnings.filterwarnings("ignore")

default_image_size = 128

parser = argparse.ArgumentParser(description="ImageGenv2 using VQGAN+CLIP")

parser.add_argument(
    "-p", "--prompts", type=str, help="Text prompts", default=None, dest="prompts"
)
parser.add_argument(
    "-i",
    "--iterations",
    type=int,
    help="Number of iterations",
    default=200,
    dest="max_iterations",
)
parser.add_argument(
    "-se",
    "--save_every",
    type=int,
    help="Save image iterations",
    default=20,
    dest="display_freq",
)
parser.add_argument(
    "-s",
    "--size",
    nargs=2,
    type=int,
    help="Image size (width height) (default: %(default)s)",
    default=[default_image_size, default_image_size],
    dest="size",
)
parser.add_argument(
    "-iw",
    "--init_weight",
    type=float,
    help="Initial weight",
    default=0.0,
    dest="init_weight",
)
parser.add_argument(
    "-m",
    "--clip_model",
    type=str,
    help="CLIP model (e.g. ViT-B/32, ViT-B/16)",
    default="ViT-B/32",
    dest="clip_model",
)
parser.add_argument(
    "-conf",
    "--vqgan_config",
    type=str,
    help="VQGAN config",
    default=f"checkpoints/vqgan_imagenet_f16_16384.yaml",
    dest="vqgan_config",
)
parser.add_argument(
    "-ckpt",
    "--vqgan_checkpoint",
    type=str,
    help="VQGAN checkpoint",
    default=f"checkpoints/vqgan_imagenet_f16_16384.ckpt",
    dest="vqgan_checkpoint",
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    help="Learning rate",
    default=0.1,
    dest="step_size",
)
parser.add_argument(
    "-cuts", "--num_cuts", type=int, help="Number of cuts", default=32, dest="cutn"
)
parser.add_argument(
    "-cutp", "--cut_power", type=float, help="Cut power", default=1.0, dest="cut_pow"
)
parser.add_argument("-sd", "--seed", type=int, help="Seed", default=None, dest="seed")
parser.add_argument(
    "-d",
    "--deterministic",
    action="store_true",
    help="Enable cudnn.deterministic?",
    dest="cudnn_determinism",
)
parser.add_argument(
    "-cd",
    "--cuda_device",
    type=str,
    help="Cuda device to use",
    default="cuda:0",
    dest="cuda_device",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="Output image filename",
    default="output.png",
    dest="output",
)

args = parser.parse_args()

if args.cudnn_determinism:
    torch.backends.cudnn.deterministic = True

if not args.cuda_device == "cpu" and not torch.cuda.is_available():
    args.cuda_device = "cpu"

if args.prompts:
    stripped_phrases = args.prompts.strip()
    args.prompts = stripped_phrases.split("|")

replace_grad = ReplaceGrad.apply


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = (
        x.pow(2).sum(dim=-1, keepdim=True)
        + codebook.pow(2).sum(dim=1)
        - 2 * x @ codebook.T
    )
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1.0, stop=float("-inf")):
        super().__init__()
        self.register_buffer("embed", embed)
        self.register_buffer("weight", torch.as_tensor(weight))
        self.register_buffer("stop", torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return (
            self.weight.abs()
            * replace_grad(dists, torch.maximum(dists, self.stop)).mean()
        )


def split_prompt(prompt):
    vals = prompt.rsplit(":", 2)
    vals = vals + ["", "1", "-inf"][len(vals) :]
    return vals[0], float(vals[1]), float(vals[2])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode="border"),
            K.RandomPerspective(0.2, p=0.4),
        )
        self.noise_fac = 0.1

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(
                torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size
            )
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == "taming.models.vqgan.VQModel":
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == "taming.models.cond_transformer.Net2NetTransformer":
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f"unknown model type: {config.model.target}")
    del model.loss
    return model


device = torch.device(args.cuda_device)
model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
jit = False
perceptor = (
    clip.load(args.clip_model, jit=jit)[0].eval().requires_grad_(False).to(device)
)

cut_size = perceptor.visual.input_resolution
f = 2 ** (model.decoder.num_resolutions - 1)


make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)

toksX, toksY = args.size[0] // f, args.size[1] // f
sideX, sideY = toksX * f, toksY * f

e_dim = model.quantize.e_dim
n_toks = model.quantize.n_e
z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

one_hot = F.one_hot(
    torch.randint(n_toks, [toksY * toksX], device=device), n_toks
).float()
z = one_hot @ model.quantize.embedding.weight
z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)

z_orig = z.clone()
z.requires_grad_(True)

pMs = []
normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)

if args.prompts:
    for prompt in args.prompts:
        txt, weight, stop = split_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

opt = optim.Adam([z], lr=args.step_size)

if args.seed is None:
    seed = torch.seed()
else:
    seed = args.seed
torch.manual_seed(seed)
print("Using seed:", seed)


# Vector quantization
def synth(z):
    z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(
        3, 1
    )
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)


@torch.inference_mode()
def checkin(i, losses):
    losses_str = ", ".join(f"{loss.item():g}" for loss in losses)
    tqdm.write(f"i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}")
    out = synth(z)
    info = PngImagePlugin.PngInfo()
    info.add_text("comment", f"{args.prompts}")
    TF.to_pil_image(out[0].cpu()).save(args.output, pnginfo=info)


def ascend_txt():
    global i
    out = synth(z)
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

    result = []

    if args.init_weight:
        # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
        result.append(
            F.mse_loss(z, torch.zeros_like(z_orig))
            * ((1 / torch.tensor(i * 2 + 1)) * args.init_weight)
            / 2
        )

    for prompt in pMs:
        result.append(prompt(iii))

    return result


def train(i):
    opt.zero_grad(set_to_none=True)
    lossAll = ascend_txt()

    checkin(i, lossAll)

    loss = sum(lossAll)
    loss.backward()
    opt.step()

    with torch.inference_mode():
        z.copy_(z.maximum(z_min).minimum(z_max))


i = 0

try:
    with tqdm() as pbar:
        while True:
            train(i)

            if i == args.max_iterations:
                break

            i += 1
            pbar.update()
except KeyboardInterrupt:
    pass
