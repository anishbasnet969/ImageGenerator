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
    "-cutm",
    "--cut_method",
    type=str,
    help="Cut method",
    choices=["original", "updated", "nrupdated", "updatedpooling", "latest"],
    default="latest",
    dest="cut_method",
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
    "-aug",
    "--augments",
    nargs="+",
    action="append",
    type=str,
    choices=["Ji", "Sh", "Gn", "Pe", "Ro", "Af", "Et", "Ts", "Cr", "Er", "Re"],
    help="Enabled augments (latest vut method only)",
    default=[],
    dest="augments",
)
parser.add_argument(
    "-cd",
    "--cuda_device",
    type=str,
    help="Cuda device to use",
    default="cuda:0",
    dest="cuda_device",
)

args = parser.parse_args()

if args.cudnn_determinism:
    torch.backends.cudnn.deterministic = True

if not args.augments:
    args.augments = [["Af", "Pe", "Ji", "Er"]]

if not args.cuda_device == "cpu" and not torch.cuda.is_available():
    args.cuda_device = "cpu"


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def random_noise_image(w, h):
    random_image = Image.fromarray(
        np.random.randint(0, 255, (w, h, 3), dtype=np.dtype("uint8"))
    )
    return random_image


def gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=float)

    for i, (start, stop, is_horizontal) in enumerate(
        zip(start_list, stop_list, is_horizontal_list)
    ):
        result[:, :, i] = gradient_2d(start, stop, width, height, is_horizontal)

    return result


def random_gradient_image(w, h):
    array = gradient_3d(
        w,
        h,
        (0, 0, np.random.randint(0, 255)),
        (
            np.random.randint(1, 255),
            np.random.randint(2, 255),
            np.random.randint(3, 128),
        ),
        (True, False, False),
    )
    random_image = Image.fromarray(np.uint8(array))
    return random_image


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), "reflect")
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), "reflect")
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode="bicubic", align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        (input,) = ctx.saved_tensors
        return (
            grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0),
            None,
            None,
        )


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


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow  # not used with pooling

        # Pick your own augments & their order
        augment_list = []
        for item in args.augments[0]:
            if item == "Ji":
                augment_list.append(
                    K.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7
                    )
                )
            elif item == "Sh":
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == "Gn":
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1.0, p=0.5))
            elif item == "Pe":
                augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
            elif item == "Ro":
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == "Af":
                augment_list.append(
                    K.RandomAffine(
                        degrees=15,
                        translate=0.1,
                        shear=5,
                        p=0.7,
                        padding_mode="zeros",
                        keepdim=True,
                    )
                )  # border, reflection, zeros
            elif item == "Et":
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == "Ts":
                augment_list.append(
                    K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7)
                )
            elif item == "Cr":
                augment_list.append(
                    K.RandomCrop(
                        size=(self.cut_size, self.cut_size),
                        pad_if_needed=True,
                        padding_mode="reflect",
                        p=0.5,
                    )
                )
            elif item == "Er":
                augment_list.append(
                    K.RandomErasing(
                        scale=(0.1, 0.4),
                        ratio=(0.3, 1 / 0.3),
                        same_on_batch=True,
                        p=0.7,
                    )
                )
            elif item == "Re":
                augment_list.append(
                    K.RandomResizedCrop(
                        size=(self.cut_size, self.cut_size),
                        scale=(0.1, 1),
                        ratio=(0.75, 1.333),
                        cropping_mode="resample",
                        p=0.5,
                    )
                )

        self.augs = nn.Sequential(*augment_list)
        self.noise_fac = 0.1
        # self.noise_fac = False

        # Uncomment if you like seeing the list ;)
        # print(augment_list)

        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        cutouts = []

        for _ in range(self.cutn):
            # Use Pooling
            cutout = (self.av_pool(input) + self.max_pool(input)) / 2
            cutouts.append(cutout)

        batch = self.augs(torch.cat(cutouts, dim=0))

        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


class MakeCutoutsPoolingUpdate(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow  # Not used with pooling

        self.augs = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode="border"),
            K.RandomPerspective(0.7, p=0.7),
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((0.1, 0.4), (0.3, 1 / 0.3), same_on_batch=True, p=0.7),
        )

        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []

        for _ in range(self.cutn):
            cutout = (self.av_pool(input) + self.max_pool(input)) / 2
            cutouts.append(cutout)

        batch = self.augs(torch.cat(cutouts, dim=0))

        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# An Nerdy updated version with selectable Kornia augments, but no pooling:
class MakeCutoutsNRUpdate(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = 0.1

        # Pick your own augments & their order
        augment_list = []
        for item in args.augments[0]:
            if item == "Ji":
                augment_list.append(
                    K.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7
                    )
                )
            elif item == "Sh":
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == "Gn":
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1.0, p=0.5))
            elif item == "Pe":
                augment_list.append(K.RandomPerspective(distortion_scale=0.5, p=0.7))
            elif item == "Ro":
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == "Af":
                augment_list.append(
                    K.RandomAffine(
                        degrees=30,
                        translate=0.1,
                        shear=5,
                        p=0.7,
                        padding_mode="zeros",
                        keepdim=True,
                    )
                )  # border, reflection, zeros
            elif item == "Et":
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == "Ts":
                augment_list.append(
                    K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7)
                )
            elif item == "Cr":
                augment_list.append(
                    K.RandomCrop(
                        size=(self.cut_size, self.cut_size),
                        pad_if_needed=True,
                        padding_mode="reflect",
                        p=0.5,
                    )
                )
            elif item == "Er":
                augment_list.append(
                    K.RandomErasing(
                        scale=(0.1, 0.4),
                        ratio=(0.3, 1 / 0.3),
                        same_on_batch=True,
                        p=0.7,
                    )
                )
            elif item == "Re":
                augment_list.append(
                    K.RandomResizedCrop(
                        size=(self.cut_size, self.cut_size),
                        scale=(0.1, 1),
                        ratio=(0.75, 1.333),
                        cropping_mode="resample",
                        p=0.5,
                    )
                )

        self.augs = nn.Sequential(*augment_list)

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


# An updated version with Kornia augments, but no pooling:
class MakeCutoutsUpdate(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
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