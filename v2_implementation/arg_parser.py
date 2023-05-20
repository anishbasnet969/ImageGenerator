import argparse


def get_parser(default_image_size):
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
        "-sd", "--seed", type=int, help="Seed", default=None, dest="seed"
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
    parser.add_argument(
        "-in",
        "--init_noise",
        type=str,
        help="Initial noise image (pixels or gradient)",
        default=None,
        dest="init_noise",
    )

    return parser
