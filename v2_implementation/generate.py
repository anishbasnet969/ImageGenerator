from tqdm import tqdm
from load_vqgan import load_vqgan_model

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

torch.backends.cudnn.benchmark = False

from CLIP import clip
from PIL import ImageFile, PngImagePlugin, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

from arg_parser import get_parser
from vision_utils import MakeCutouts, random_gradient_image, random_noise_image
from utils import ReplaceGrad, ClampWithGrad, split_prompt

import warnings

warnings.filterwarnings("ignore")

default_image_size = 128

parser = get_parser(default_image_size=default_image_size)
args = parser.parse_args()

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


device = torch.device(args.cuda_device)
model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
jit = False
perceptor = (
    clip.load(args.clip_model, jit=jit)[0].eval().requires_grad_(False).to(device)
)

cut_size = perceptor.visual.input_resolution
cutn = 32
cut_pow = 1.0
f = 2 ** (model.decoder.num_resolutions - 1)


make_cutouts = MakeCutouts(cut_size, cutn, cut_pow=cut_pow)

toksX, toksY = args.size[0] // f, args.size[1] // f
sideX, sideY = toksX * f, toksY * f

e_dim = model.quantize.e_dim
n_toks = model.quantize.n_e
z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

if args.init_noise == "pixels":
    img = random_noise_image(args.size[0], args.size[1])
    pil_image = img.convert("RGB")
    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
    pil_tensor = TF.to_tensor(pil_image)
    z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
elif args.init_noise == "gradient":
    img = random_gradient_image(args.size[0], args.size[1])
    pil_image = img.convert("RGB")
    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
    pil_tensor = TF.to_tensor(pil_image)
    z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
else:
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
    image_embeddings = perceptor.encode_image(normalize(make_cutouts(out))).float()

    result = []

    for prompt in pMs:
        result.append(prompt(image_embeddings))

    return result


def train(i):
    opt.zero_grad(set_to_none=True)
    lossAll = ascend_txt()

    if i % args.display_freq == 0:
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
