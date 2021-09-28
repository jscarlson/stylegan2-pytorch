import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
import os


def infer_from_latent(args, g_ema, device, mean_latent, input_latent, mix_n = 5):

    with torch.no_grad():
        g_ema.eval()

        input_intermed = input_latent.unsqueeze(0).to(device)
        sample_z = torch.randn(1, args.latent, device=device)

        _, latents = g_ema(
            [sample_z], 
            truncation=args.truncation, 
            truncation_latent=mean_latent,
            return_latents=True
        )

        input_final = torch.cat([input_intermed[:,:mix_n,:], latents[:,mix_n:,:]], dim=1)

        sample, _ = g_ema(
            [input_final], 
            truncation=args.truncation, 
            truncation_latent=mean_latent,
            input_is_latent=True
        )

        os.makedirs("./infer_from_latent", exist_ok=True)

        utils.save_image(
            sample,
            f"infer_from_latent/{os.path.basename(args.latent_path[:-3])}.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--latent_path",
        required=True,
        type=str,
        help="path to latent pt"
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    proj_res = torch.load(args.latent_path)
    proj_latent = proj_res[os.path.basename(args.latent_path)[:-2] + "png"]["latent"]
    infer_from_latent(args, g_ema, device, mean_latent, proj_latent)
