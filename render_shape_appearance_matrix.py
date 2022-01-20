import argparse
import json
import numpy as np
import os

import torch
from torchvision.utils import save_image
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageFont

import curriculums

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_image(gen, z_s, z_a, **kwargs):
    with torch.no_grad():
        img, depth_map = gen.staged_forward(z_s, z_a, **kwargs)

        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min) * 256
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    return img, depth_map


def make_curriculum(curriculum):
    # verify file system
    curriculum = getattr(curriculums, curriculum, None)
    if curriculum is None:
        raise ValueError(f"{curriculum} is not a valid curriculum")
    curriculum["num_steps"] = curriculum[0]["num_steps"]
    curriculum["psi"] = 0.7
    curriculum["v_stddev"] = 0
    curriculum["h_stddev"] = 0
    curriculum["nerf_noise"] = 0
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}
    return curriculum


def load_generator(model_path):
    generator = torch.load(
        os.path.join(model_path, "generator.pth"), map_location=torch.device(device)
    )
    ema_dict = torch.load(
        os.path.join(model_path, "ema.pth"), map_location=torch.device(device)
    )
    ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    ema.load_state_dict(ema_dict)
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()
    return generator


def make_matrix(
    gen, curriculum, seed, s_num, a_num, img_size
):
    torch.manual_seed(seed)
    curriculum = make_curriculum(curriculum)
    curriculum["img_size"] = img_size
    z_s = torch.randn((s_num, curriculum["latent_dim_s"]), device=device)
    z_a = torch.randn((a_num, curriculum["latent_dim_a"]), device=device)
    canvas = Image.new(
        # channels
        "RGBA",
        (
            # width
            img_size * a_num,
            # height
            img_size * s_num,
        ),
        # fill color
        (255, 255, 255, 255),
    )
    canvas_w, canvas_h = canvas.size
    for z_s_idx in tqdm(range(s_num)):
        for z_a_idx in range(a_num):
            print("Making Image at ({}, {})".format(z_s_idx, z_a_idx))
            img, depth_img = generate_image(gen, z_s[z_s_idx, :].unsqueeze(0), z_a[z_a_idx, :].unsqueeze(0), **curriculum)
            PIL_image = Image.fromarray(np.uint8(img)).convert("RGB")
            # PIL_image.save("{}_{}.png".format(iy, ip))
            canvas.paste(
                PIL_image, (img_size * z_a_idx, img_size * z_s_idx)
            )
    return canvas


def main():
    model_path = "/h/edwardl/pigan/output/5704257/DELAYEDPURGE/"
    curriculum = "CARLA"
    img_size = 64
    s_num = 5
    a_num = 5
    seed = 10
    print("Starting Generation")
    image = make_matrix(
        load_generator(model_path),
        curriculum,
        seed,
        s_num,
        a_num,
        img_size,
    )
    print("Saving Image")
    image.save("./test.png")
    return


if __name__ == "__main__":
    main()
